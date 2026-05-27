/**
 * Rate-Distortion Theory for PD-Separated KV Transfer
 * 
 * 理论基础:
 * - R(D) = min I(KV_compressed; KV_original) s.t. E[d(output)] ≤ D
 * - 在带宽B约束下找最小D
 * - 不等误差保护: 高语义重要性层高精度传输，低语义重要性层低精度+激进剪枝
 * 
 * 核心创新: Semantic Distortion
 * d(x, x') = 1 - similarity(LLM_output(KV_original), LLM_output(KV_compressed))
 * 用生成质量而非L2误差衡量失真
 */

// ============================================
// 类型定义
// ============================================

import { TaskType } from '../core/types.js';

/**
 * R-D配置
 */
export interface RDConfig {
  bandwidthBytesPerMs: number;   // 带宽约束 (bytes/ms)
  maxDistortion: number;         // 最大允许失真 [0-1]
  numLayers: number;             // 模型层数
  taskType: TaskType;            // 任务类型
  sequenceLength: number;        // 序列长度
}

/**
 * 层分配结果
 */
export interface RDLayerAllocation {
  layerIndex: number;
  allocatedRate: number;         // 分配的传输速率 (bits per token)
  precision: number;             // 量化精度 (bits)
  retentionRatio: number;       // 保留比例 [0-1]
  estimatedDistortion: number;  // 估算失真 [0-1]
  semanticImportance: number;   // 语义重要性 [0-1]
  variance: number;             // 层方差 σ²
}

/**
 * R-D最优结果
 */
export interface RDOptimalResult {
  totalRate: number;            // 总传输速率 (bits/token)
  totalDistortion: number;       // 总失真 [0-1]
  layerAllocations: RDLayerAllocation[];
  rdPoint: { rate: number; distortion: number };
  isOptimal: boolean;           // 是否达到理论最优
  bandwidthUsedBytesPerMs: number;
  bandwidthSavingRatio: number;  // 带宽节省比例
}

/**
 * R-D曲线上的点
 */
export interface RDPoint {
  rate: number;                 // 传输速率 (bits/token)
  distortion: number;           // 失真 [0-1]
  quality: number;              // 质量 = 1 - distortion
}

/**
 * 任务类型的基础参数
 */
interface TaskProfile {
  // 各层基础方差 (σ²_l)
  baseVariance: number[];
  // 头部/中部/尾部敏感度
  sensitivity: { head: number; middle: number; tail: number };
  // 精度偏好
  preferredPrecision: number;
}

// ============================================
// 任务Profile定义
// ============================================

const TASK_PROFILES: Record<TaskType, TaskProfile> = {
  math: {
    baseVariance: generateLayerVariance(32, 'math'),
    sensitivity: { head: 0.037, middle: 0.037, tail: 0.02 },
    preferredPrecision: 16
  },
  code: {
    baseVariance: generateLayerVariance(32, 'code'),
    sensitivity: { head: 0.0, middle: 0.0, tail: 0.01 },
    preferredPrecision: 8
  },
  qa: {
    baseVariance: generateLayerVariance(32, 'qa'),
    sensitivity: { head: 0.042, middle: 0.0, tail: 0.02 },
    preferredPrecision: 16
  },
  conversation: {
    baseVariance: generateLayerVariance(32, 'conversation'),
    sensitivity: { head: 0.02, middle: 0.01, tail: 0.01 },
    preferredPrecision: 8
  },
  unknown: {
    baseVariance: generateLayerVariance(32, 'unknown'),
    sensitivity: { head: 0.02, middle: 0.02, tail: 0.02 },
    preferredPrecision: 8
  }
};

/**
 * 生成各层的方差分布
 * 高层KV方差大（语义信息丰富），低层方差小（语法信息冗余）
 */
function generateLayerVariance(numLayers: number, taskType: TaskType): number[] {
  const variances: number[] = [];
  
  for (let i = 0; i < numLayers; i++) {
    const position = i / numLayers; // 0 = 低层, 1 = 高层
    
    // 根据任务类型调整方差分布
    let baseVariance: number;
    switch (taskType) {
      case 'math':
        // 数学: 高层方差大（推理链依赖）
        baseVariance = 0.1 + position * 0.9;
        break;
      case 'code':
        // 代码: 低层方差大（语法结构）
        baseVariance = 0.9 - position * 0.7;
        break;
      case 'qa':
        // QA: 高层方差大（语义理解）
        baseVariance = 0.2 + position * 0.8;
        break;
      case 'conversation':
        // 对话: 均匀分布
        baseVariance = 0.5;
        break;
      default:
        baseVariance = 0.1 + position * 0.8;
    }
    
    variances.push(Math.max(0.01, baseVariance));
  }
  
  return variances;
}

// ============================================
// Rate-Distortion 核心类
// ============================================

export class RateDistortion {
  /**
   * 高斯信源下的R-D函数: R(D) = 0.5 * log₂(σ²/D)
   * 
   * 推导:
   * - KV Cache视为独立高斯信源
   * - 均方误差失真度量
   * - 当 D ≥ σ² 时，R(D) = 0（不传输）
   * - 当 D < σ² 时，R(D) > 0
   * 
   * @param layerVariance 层方差 σ²
   * @param distortion 目标失真 D
   * @returns 所需传输速率 R (bits/token)
   */
  static computeLayerRD(layerVariance: number, distortion: number): number {
    // 如果失真大于等于方差，不需要传输
    if (distortion >= layerVariance) return 0;
    
    // R(D) = 0.5 * log₂(σ²/D) bits per symbol
    const rd = 0.5 * Math.log2(layerVariance / distortion);
    return Math.max(0, rd);
  }
  
  /**
   * 计算层失真: D = σ² * 2^(-2R)
   * 给定传输速率R，计算能达到的最小失真
   * 
   * @param layerVariance 层方差 σ²
   * @param rate 传输速率 R (bits/token)
   * @returns 失真 D
   */
  static computeLayerDistortion(layerVariance: number, rate: number): number {
    if (rate <= 0) return layerVariance;
    // D = σ² * 2^(-2R)
    const distortion = layerVariance * Math.pow(2, -2 * rate);
    return distortion;
  }
  
  /**
   * 计算R-D函数曲线
   * 
   * @param config R-D配置
   * @param distortionRange 失真范围 [0, 0.01, 0.05, 0.1, 0.2, 0.5]
   * @returns R-D曲线上的点
   */
  computeRDFunction(
    config: RDConfig,
    distortionRange: number[]
  ): RDPoint[] {
    const { numLayers, taskType, sequenceLength } = config;
    const profile = TASK_PROFILES[taskType] || TASK_PROFILES.unknown;
    
    const rdPoints: RDPoint[] = [];
    
    for (const targetDistortion of distortionRange) {
      let totalRate = 0;
      
      // 对每层计算R(D)
      for (let i = 0; i < numLayers; i++) {
        const variance = profile.baseVariance[i];
        const rate = RateDistortion.computeLayerRD(variance, targetDistortion * variance);
        totalRate += rate;
      }
      
      rdPoints.push({
        rate: totalRate / numLayers,  // 平均每层速率
        distortion: targetDistortion,
        quality: 1 - targetDistortion
      });
    }
    
    return rdPoints;
  }
  
  /**
   * 给定带宽约束B，求最小D的最优分配
   * 
   * 使用拉格朗日优化:
   * min Σ D_l s.t. Σ R_l ≤ B
   * 拉格朗日: L = Σ D_l + λ(Σ R_l - B)
   * 
   * 最优解条件:
   * - 所有活跃层的 ∂D_l/∂R_l 相等（等边际失真原则）
   * - D_l* = min(σ²_l, λ/(2·ln2))
   * - R_l* = max(0, 0.5·log₂(σ²_l/D_l*))
   * 
   * @param bandwidthBytesPerMs 带宽约束 (bytes/ms)
   * @param numLayers 层数
   * @param taskType 任务类型
   * @param sequenceLength 序列长度
   * @returns 最优分配结果
   */
  minimizeDistortion(
    bandwidthBytesPerMs: number,
    numLayers: number,
    taskType: TaskType,
    sequenceLength: number
  ): RDOptimalResult {
    const profile = TASK_PROFILES[taskType] || TASK_PROFILES.unknown;
    
    // 每token的KV大小 (假设每token 1KB，每层贡献)
    const kvBytesPerTokenPerLayer = 1024 / numLayers;
    
    // 带宽预算转换为 bits/token
    // 带宽bytes/ms -> bits/ms -> bits/token (假设tokenRate = 1000 tokens/s)
    const tokenRate = 1000; // tokens/s
    const bandwidthBitsPerToken = (bandwidthBytesPerMs * 8) / tokenRate;
    
    // 拉格朗日乘子 λ 的搜索范围
    let lambdaLow = 0;
    let lambdaHigh = Math.max(...profile.baseVariance) * 2 * Math.LN2;
    
    // 二分搜索最优 λ
    const maxIterations = 50;
    const tolerance = 1e-6;
    
    let optimalLambda = lambdaHigh;
    
    for (let iter = 0; iter < maxIterations; iter++) {
      const lambdaMid = (lambdaLow + lambdaHigh) / 2;
      
      // 计算当前 λ 下的总速率
      let totalRate = 0;
      for (let i = 0; i < numLayers; i++) {
        const sigma2 = profile.baseVariance[i];
        const D_star = Math.min(sigma2, lambdaMid / (2 * Math.LN2));
        if (D_star < sigma2) {
          const R = 0.5 * Math.log2(sigma2 / D_star);
          totalRate += R;
        }
      }
      
      if (totalRate > bandwidthBitsPerToken) {
        // 速率太大，需要增大 λ（减少失真，提高速率）
        lambdaLow = lambdaMid;
      } else {
        // 速率太小，可以减小 λ（增加失真，减少速率）
        lambdaHigh = lambdaMid;
      }
      
      if (Math.abs(lambdaHigh - lambdaLow) < tolerance) break;
      optimalLambda = lambdaMid;
    }
    
    // 使用最优 λ 计算最终分配
    const layerAllocations = this.computeAllocations(
      optimalLambda,
      profile,
      numLayers,
      taskType
    );
    
    // 计算总失真和总速率
    const totalDistortion = layerAllocations.reduce(
      (sum, a) => sum + a.estimatedDistortion, 0
    ) / numLayers;
    const totalRate = layerAllocations.reduce(
      (sum, a) => sum + a.allocatedRate, 0
    ) / numLayers;
    
    // 计算带宽使用
    const bandwidthUsedBytesPerMs = (totalRate * numLayers * tokenRate) / 8;
    const maxBandwidth = numLayers * kvBytesPerTokenPerLayer * tokenRate;
    const bandwidthSavingRatio = 1 - bandwidthUsedBytesPerMs / maxBandwidth;
    
    return {
      totalRate,
      totalDistortion,
      layerAllocations,
      rdPoint: { rate: totalRate, distortion: totalDistortion },
      isOptimal: totalDistortion < 0.5,  // 简化判断
      bandwidthUsedBytesPerMs,
      bandwidthSavingRatio: Math.max(0, bandwidthSavingRatio)
    };
  }
  
  /**
   * 使用拉格朗日乘子计算各层分配
   */
  private computeAllocations(
    lambda: number,
    profile: TaskProfile,
    numLayers: number,
    taskType: TaskType
  ): RDLayerAllocation[] {
    const allocations: RDLayerAllocation[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      const variance = profile.baseVariance[i];
      
      // 最优失真: D* = min(σ², λ/(2·ln2))
      const D_star = Math.min(variance, lambda / (2 * Math.LN2));
      
      // 归一化失真
      const normalizedDistortion = D_star / variance;
      
      // 传输速率
      let rate: number;
      let precision: number;
      let retention: number;
      
      if (D_star >= variance) {
        // 不传输
        rate = 0;
        precision = 0;
        retention = 0;
      } else {
        // R(D) = 0.5 * log₂(σ²/D*)
        rate = 0.5 * Math.log2(variance / D_star);
        
        // 根据速率推断精度
        // 16-bit: rate ≈ 16, 8-bit: rate ≈ 8, 4-bit: rate ≈ 4
        if (rate >= 12) {
          precision = 16;
          retention = 0.9;
        } else if (rate >= 6) {
          precision = 8;
          retention = 0.6;
        } else {
          precision = 4;
          retention = 0.3;
        }
      }
      
      // 语义重要性（高层更重要，取决于任务类型）
      const semanticImportance = this.computeSemanticImportance(i, numLayers, taskType);
      
      allocations.push({
        layerIndex: i,
        allocatedRate: rate,
        precision,
        retentionRatio: retention,
        estimatedDistortion: normalizedDistortion,
        semanticImportance,
        variance
      });
    }
    
    return allocations;
  }
  
  /**
   * 计算层的语义重要性
   * 根据任务类型，不同层的语义重要性不同
   */
  private computeSemanticImportance(
    layerIndex: number,
    numLayers: number,
    taskType: TaskType
  ): number {
    const position = layerIndex / numLayers;
    
    switch (taskType) {
      case 'math':
        // 数学：高层更重要
        return 0.3 + position * 0.7;
      case 'code':
        // 代码：低层更重要（语法）
        return 1 - position * 0.7;
      case 'qa':
        // QA：高层更重要
        return 0.2 + position * 0.8;
      case 'conversation':
        // 对话：均匀
        return 0.5;
      default:
        return 0.2 + position * 0.6;
    }
  }
  
  /**
   * 不等误差保护 (Unequal Error Protection)
   * 
   * 核心思想:
   * - 高I(Z;Y)的KV层（高语义重要性）→ 高精度传输（FP16）
   * - 低I(Z;Y)的KV层 → 低精度传输（INT4）+ 激进剪枝
   * 
   * 实现: 根据层重要性分配不同的比特预算
   * 
   * @param bandwidthBudget 总带宽预算 (bits/token)
   * @param layerImportance 每层语义重要性 [0-1]
   * @param numLayers 层数
   * @returns 各层分配结果
   */
  unequalErrorProtection(
    bandwidthBudget: number,
    layerImportance: number[],
    numLayers: number
  ): RDLayerAllocation[] {
    const allocations: RDLayerAllocation[] = [];
    
    // 归一化重要性权重
    const totalImportance = layerImportance.reduce((a, b) => a + b, 0);
    const weights = layerImportance.map(i => i / totalImportance);
    
    // 基础精度选项
    const precisionOptions = [
      { bits: 16, name: 'FP16', rate: 16 },
      { bits: 8, name: 'FP8', rate: 8 },
      { bits: 4, name: 'INT4', rate: 4 }
    ];
    
    // 按重要性排序分配
    const sortedIndices = layerImportance
      .map((_, i) => i)
      .sort((a, b) => layerImportance[b] - layerImportance[a]);
    
    let remainingBudget = bandwidthBudget;
    const assignedLayers = new Set<number>();
    
    // 第一遍：高重要性层优先分配高精度
    for (const idx of sortedIndices) {
      if (remainingBudget < 4) break;  // 最小需要4 bits
      
      const importance = layerImportance[idx];
      let precision: number;
      let retention: number;
      
      if (importance > 0.7) {
        // 高重要性: FP16
        precision = 16;
        retention = 0.9;
        remainingBudget -= 16;
      } else if (importance > 0.4) {
        // 中等重要性: FP8
        precision = 8;
        retention = 0.6;
        remainingBudget -= 8;
      } else {
        // 低重要性: 暂时跳过，等第二遍处理
        continue;
      }
      
      assignedLayers.add(idx);
      allocations.push({
        layerIndex: idx,
        allocatedRate: precision,
        precision,
        retentionRatio: retention,
        estimatedDistortion: 1 - retention,
        semanticImportance: importance,
        variance: importance
      });
    }
    
    // 第二遍：低重要性层分配低精度
    for (const idx of sortedIndices) {
      if (assignedLayers.has(idx)) continue;
      if (remainingBudget < 4) {
        // 没有预算了，剪枝掉
        allocations.push({
          layerIndex: idx,
          allocatedRate: 0,
          precision: 0,
          retentionRatio: 0,
          estimatedDistortion: 1,
          semanticImportance: layerImportance[idx],
          variance: layerImportance[idx]
        });
        continue;
      }
      
      const importance = layerImportance[idx];
      
      // INT4 激进剪枝
      precision = 4;
      retention = 0.3;
      remainingBudget -= 4;
      
      allocations.push({
        layerIndex: idx,
        allocatedRate: precision,
        precision,
        retentionRatio: retention,
        estimatedDistortion: 1 - retention,
        semanticImportance: importance,
        variance: importance
      });
    }
    
    // 按原始层顺序排序
    return allocations.sort((a, b) => a.layerIndex - b.layerIndex);
  }
  
  /**
   * 计算信息瓶颈价值 I(Z;Y) - 层重要性度量
   * 
   * 这里用简化的代理度量:
   * - 高层KV对输出的互信息更高
   * - 不同任务类型对不同层的依赖不同
   * 
   * @param layerIndex 层索引
   * @param totalLayers 总层数
   * @param taskType 任务类型
   * @returns 互信息估计值
   */
  static computeMutualInformation(
    layerIndex: number,
    totalLayers: number,
    taskType: TaskType
  ): number {
    const position = layerIndex / totalLayers;
    const profile = TASK_PROFILES[taskType] || TASK_PROFILES.unknown;
    
    // 基础互信息与层位置相关
    let baseMI: number;
    switch (taskType) {
      case 'math':
        baseMI = 0.2 + position * 0.8;  // 高层高互信息
        break;
      case 'code':
        baseMI = 0.8 - position * 0.5;  // 低层高互信息
        break;
      case 'qa':
        baseMI = 0.3 + position * 0.7;
        break;
      default:
        baseMI = 0.5;
    }
    
    return Math.min(1, Math.max(0, baseMI));
  }
  
  /**
   * 生成理论R-D曲线（用于论文图）
   * 
   * @param numLayers 层数
   * @param taskType 任务类型
   * @param numPoints 曲线点数
   * @returns R-D曲线点
   */
  generateTheoreticalRDCurve(
    numLayers: number,
    taskType: TaskType,
    numPoints: number = 20
  ): RDPoint[] {
    const profile = TASK_PROFILES[taskType] || TASK_PROFILES.unknown;
    const points: RDPoint[] = [];
    
    // 平均方差
    const avgVariance = profile.baseVariance.reduce((a, b) => a + b, 0) / numLayers;
    
    // 从高质量(低失真)到低质量(高失真)生成点
    for (let i = 0; i < numPoints; i++) {
      // distortion从0.01到0.99
      const distortion = 0.01 + (i / (numPoints - 1)) * 0.98;
      
      // 计算平均R(D)
      let totalRate = 0;
      for (let l = 0; l < numLayers; l++) {
        totalRate += RateDistortion.computeLayerRD(
          profile.baseVariance[l],
          distortion * profile.baseVariance[l]
        );
      }
      
      const avgRate = totalRate / numLayers;
      
      points.push({
        rate: avgRate,
        distortion,
        quality: 1 - distortion
      });
    }
    
    return points;
  }
}

// ============================================
// 导出
// ============================================

export type { TaskProfile } from './RateDistortion.js';
