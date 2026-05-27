/**
 * 语义Rate-Distortion统一框架 ⭐
 * 
 * IB和R-D的理论统一：
 * - IB: min I(Z;X) - β × I(Z;Y) → 最大化预测信息，最小化传输信息
 * - R-D: min D s.t. R ≤ B → 在带宽约束下最小化失真
 * 
 * 当distortion measure = -log P(Y|Z)/P(Y|X)时，IB和R-D等价。
 * 
 * 统一框架目标函数：
 * R(D) = min I(Z; X) - β × I(Z; Y)
 * s.t. E[d_semantic(Y_original, Y_compressed)] ≤ D
 *      I(Z; X) ≤ B  (bandwidth constraint)
 * 
 * 核心思想：
 * - I(Z;Y)作为重要性度量（IB贡献）- 预测信息越多，层越重要
 * - R(D)作为带宽-质量tradeoff（R-D贡献）- 带宽分配最优
 * - d_semantic作为生成质量失真（统一贡献）- 语义保真度
 */

import {
  CompressionParams,
  CompressionOutput,
  clamp,
  round4,
  ensureRetentionRange,
  TaskType
} from '../core/types.js';
import { ICompressionStrategy, TASK_PROFILES, TaskProfile } from '../compression/CompressionOrchestrator.js';

// ============================================
// 类型定义
// ============================================

// 任务类型
export type { TaskType };

// 阶段类型
export type PhaseType = 'prefill' | 'decode';

// 统一框架配置
export interface SemanticRDConfig {
  // 带宽约束 (bytes/ms)
  bandwidthBytesPerMs: number;
  // 最大失真容忍度 [0-1]
  maxDistortion: number;
  // 任务类型
  taskType: TaskType;
  // 模型层数
  numLayers: number;
  // 序列长度
  sequenceLength: number;
  // 当前阶段
  phase: PhaseType;
  // IB权重参数 β (控制重要性vs压缩的权衡)
  beta: number;
  // 隐藏维度
  hiddenSize?: number;
  // 注意力头数
  numHeads?: number;
}

// 层分配结果
export interface LayerAllocation {
  layerIndex: number;
  // IB重要性：I(Z_l; Y) - 预测信息，越大越重要
  predictiveInformation: number;
  // R-D分配的传输速率 (bytes/ms)
  allocatedRate: number;
  // 精度 (bits)
  precision: number;
  // 保留比例 [0.1-1.0]
  retentionRatio: number;
  // 层的重要性分数 (综合IB和R-D)
  importanceScore: number;
}

// 理论结果
export interface TheoreticalBounds {
  // 理论最小传输速率 (bytes/ms)
  theoreticalMinRate: number;
  // 理论最大质量 [0-1]
  theoreticalMaxQuality: number;
  // Shannon下界
  shannonBound: number;
  // 最优操作点
  optimalOperatingPoint: {
    rate: number;
    distortion: number;
    quality: number;
    beta: number;
  };
}

// 统一框架输出
export interface SemanticRDResult {
  // 理论结果
  theoretical: TheoreticalBounds;
  
  // 层分配
  layerAllocations: LayerAllocation[];
  
  // 统一目标函数值
  unifiedObjective: number;
  
  // Phase-aware输出
  prefillOutput: CompressionOutput;
  decodeOutput: CompressionOutput;
  
  // R-D指标
  achievedRate: number;
  achievedDistortion: number;
  achievedQuality: number;
  
  // IB-RD trade-off参数
  ibContribution: number;      // I(Z;Y) 重要性贡献
  rdContribution: number;     // R(D) 带宽效率贡献
}

// R-D曲线上的点
export interface RDCurvePoint {
  rate: number;
  distortion: number;
  quality: number;
  beta: number;
  operationMode: 'conservative' | 'balanced' | 'aggressive';
}

// ============================================
// 统一框架实现
// ============================================

/**
 * 语义Rate-Distortion统一框架
 * 
 * 核心创新：
 * 1. 将IB的预测信息 I(Z;Y) 作为层重要性度量
 * 2. 将R-D的带宽最优分配融入统一目标
 * 3. 支持Phase-aware差异化（P端激进，D端保守）
 */
export class SemanticRDFramework {
  // 精度常量
  private readonly FULL_PRECISION = 16;
  private readonly HALF_PRECISION = 8;
  private readonly QUARTER_PRECISION = 4;
  
  // IB参数范围
  private readonly MIN_BETA = 0.1;
  private readonly MAX_BETA = 10.0;
  
  // 语义失真权重
  private readonly SEMANTIC_WEIGHT = 0.6;
  private readonly SYNTAX_WEIGHT = 0.2;
  private readonly RETENTION_WEIGHT = 0.2;

  constructor() {
    // 预热
  }

  /**
   * 计算单层的IB重要性：I(Z_l; Y) ≈ H(Y) - H(Y|Z_l)
   * 
   * 理论依据：
   * - 信息瓶颈理论中，I(Z;Y)表示Z对Y的预测能力
   * - 在LLM中，深层对最终输出Y的预测能力更强
   * - 因此，深层的I(Z;Y)应该更大
   * 
   * 近似计算：
   * I(Z_l; Y) ≈ layer_depth_ratio × task_specific_factor
   */
  computeLayerIBImportance(
    layerIndex: number,
    totalLayers: number,
    taskType: TaskType,
    phase: PhaseType
  ): number {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    
    // 层深度比例 [0, 1]，0=浅层，1=深层
    const depthRatio = layerIndex / Math.max(1, totalLayers - 1);
    
    // 任务特定的层重要性
    const [lowImp, midImp, highImp] = profile.layerImportance;
    
    // 根据层位置确定基础重要性
    let taskImportance: number;
    if (depthRatio < 0.33) {
      taskImportance = lowImp;
    } else if (depthRatio < 0.67) {
      taskImportance = midImp;
    } else {
      taskImportance = highImp;
    }
    
    // Phase调整：P端依赖高层信息，D端更均衡
    let phaseFactor: number;
    if (phase === 'prefill') {
      // Prefill阶段，深层信息更重要
      phaseFactor = 0.7 + 0.3 * depthRatio;
    } else {
      // Decode阶段，各层信息都重要
      phaseFactor = 0.85 + 0.15 * depthRatio;
    }
    
    // 综合IB重要性 [0, 1]
    const ibImportance = clamp(taskImportance * phaseFactor, 0, 1);
    
    return round4(ibImportance);
  }

  /**
   * 计算R-D最优带宽分配
   * 
   * 理论依据（Shannon R-D理论）：
   * - 给定失真约束D，最优速率 R(D) = min_{P(z|x): E[d(x,z)]≤D} I(X;Z)
   * - 在实践中，我们使用加权注水算法进行近似
   * 
   * 带宽分配原则：
   * - 高重要性层（高IB）→ 更多带宽（高保留率）
   * - 低重要性层（低IB）→ 较少带宽（低保留率）
   */
  computeOptimalRateAllocation(
    layerIndex: number,
    ibImportance: number,
    totalLayers: number,
    bandwidthBytesPerMs: number,
    beta: number
  ): { rate: number; retention: number } {
    // 所有层的IB重要性归一化
    const normalizedImportance = ibImportance; // 已经在[0,1]范围
    
    // 带宽分配权重 = IB重要性 ^ beta
    // beta > 1: 放大差异（激进）
    // beta < 1: 缩小差异（保守）
    const allocationWeight = Math.pow(normalizedImportance, beta);
    
    // 计算该层的带宽分配
    // 基础带宽 = 总带宽 / 层数
    const baseRate = bandwidthBytesPerMs / totalLayers;
    
    // 加权分配
    const layerRate = baseRate * (0.5 + 0.5 * allocationWeight);
    
    // 计算保留率：保留率越高，需要传输的数据越多
    // 映射到[0.1, 1.0]范围
    const minRetention = 0.1;
    const maxRetention = 1.0;
    const retention = minRetention + (maxRetention - minRetention) * normalizedImportance;
    
    return {
      rate: round4(layerRate),
      retention: round4(retention)
    };
  }

  /**
   * 计算语义失真
   * 
   * 失真度量：
   * d_semantic = w1 × d_meaning + w2 × d_syntax + w3 × d_retention
   * 
   * 其中：
   * - d_meaning: 语义失真（通过IB重要性加权）
   * - d_syntax: 语法失真（低层更重要）
   * - d_retention: 保留率失真
   */
  computeSemanticDistortion(
    layerIndex: number,
    totalLayers: number,
    retention: number,
    taskType: TaskType
  ): number {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    
    // 层深度
    const depthRatio = layerIndex / Math.max(1, totalLayers - 1);
    
    // 语义重要性（高层更重要）
    const meaningWeight = depthRatio;
    
    // 语法重要性（低层更重要，对代码任务尤其明显）
    const syntaxWeight = taskType === 'code' ? 1 - depthRatio : 0.3 * (1 - depthRatio);
    
    // 保留率失真：保留率越低，失真越大
    const retentionDistortion = 1 - retention;
    
    // 综合失真
    const distortion = 
      this.SEMANTIC_WEIGHT * meaningWeight * retentionDistortion +
      this.SYNTAX_WEIGHT * syntaxWeight * retentionDistortion +
      this.RETENTION_WEIGHT * retentionDistortion;
    
    return round4(clamp(distortion, 0, 1));
  }

  /**
   * 计算统一目标函数值
   * 
   * 统一目标函数：
   * L(β) = min I(Z;X) - β × I(Z;Y)
   * 
   * 等价于：
   * min (bandwidth) - β × (importance)
   * 
   * 这个目标函数平衡了：
   * - 压缩率（带宽使用）
   * - 重要性保留（IB贡献）
   */
  computeUnifiedObjective(
    totalRate: number,
    totalImportance: number,
    beta: number
  ): number {
    // 归一化
    const normalizedRate = totalRate / 100; // 假设最大100 bytes/ms
    const normalizedImportance = totalImportance;
    
    // 目标函数
    const objective = normalizedRate - beta * normalizedImportance;
    
    return round4(objective);
  }

  /**
   * 计算最优β的闭式解
   * 
   * 理论推导：
   * β* = argmin_β [I(Z_P;X) + I(Z_D;X)] s.t. D_total ≤ D_max
   * 
   * 在实践中，我们通过以下近似：
   * β* ≈ (R_max / I_max) × (D_max / D_total)
   * 
   * 其中：
   * - R_max: 最大可用带宽
   * - I_max: 最大IB重要性
   * - D_max: 最大失真容忍
   * - D_total: 当前总失真
   */
  deriveOptimalBeta(
    bandwidthBytesPerMs: number,
    maxDistortion: number,
    taskType: TaskType
  ): number {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    
    // 最大IB重要性（任务相关）
    const maxImportance = Math.max(...profile.layerImportance); // 通常是0.9
    
    // 任务压缩偏好影响beta
    let preferenceFactor: number;
    switch (profile.preferredCompression) {
      case 'conservative':
        preferenceFactor = 0.5;  // 低beta，保留质量
        break;
      case 'balanced':
        preferenceFactor = 1.0;
        break;
      case 'aggressive':
        preferenceFactor = 2.0;  // 高beta，允许激进压缩
        break;
    }
    
    // 带宽压力影响beta
    const bandwidthFactor = bandwidthBytesPerMs < 50 ? 1.5 : 
                            bandwidthBytesPerMs < 100 ? 1.0 : 0.7;
    
    // 综合beta
    const beta = preferenceFactor * bandwidthFactor * (maxImportance / maxDistortion);
    
    return round4(clamp(beta, this.MIN_BETA, this.MAX_BETA));
  }

  /**
   * Phase-aware β调整
   * 
   * 理论依据：
   * - P端（prefill）：带宽约束紧，需要更大的β来强制压缩
   * - D端（decode）：质量约束紧，需要更小的β来保留信息
   */
  adjustBetaForPhase(
    baseBeta: number,
    phase: PhaseType,
    bandwidthBytesPerMs: number
  ): number {
    let phaseBeta: number;
    
    if (phase === 'prefill') {
      // P端：更激进的压缩
      // β增大 → 强调带宽最小化
      const bandwidthStress = Math.max(0, 1 - bandwidthBytesPerMs / 100);
      phaseBeta = baseBeta * (1 + 0.5 * bandwidthStress);
    } else {
      // D端：更保守的压缩
      // β减小 → 强调质量保留
      phaseBeta = baseBeta * 0.6;
    }
    
    return round4(clamp(phaseBeta, this.MIN_BETA, this.MAX_BETA));
  }

  /**
   * 统一优化入口
   * 
   * 算法流程：
   * 1. 计算每层的IB重要性 I(Z_l; Y)
   * 2. 根据β和带宽约束，计算R-D最优分配
   * 3. 生成Phase-aware的压缩配置
   * 4. 计算理论界和实际指标
   */
  optimize(config: SemanticRDConfig): SemanticRDResult {
    const {
      bandwidthBytesPerMs,
      maxDistortion,
      taskType,
      numLayers,
      sequenceLength,
      phase
    } = config;
    
    // Step 1: 计算基础β
    const baseBeta = this.deriveOptimalBeta(bandwidthBytesPerMs, maxDistortion, taskType);
    
    // Step 2: Phase-aware β调整
    const adjustedBeta = this.adjustBetaForPhase(baseBeta, phase, bandwidthBytesPerMs);
    
    // Step 3: 计算每层的IB重要性和带宽分配
    const layerAllocations: LayerAllocation[] = [];
    let totalRate = 0;
    let totalImportance = 0;
    
    for (let l = 0; l < numLayers; l++) {
      // IB重要性
      const ibImportance = this.computeLayerIBImportance(l, numLayers, taskType, phase);
      
      // R-D带宽分配
      const { rate, retention } = this.computeOptimalRateAllocation(
        l, ibImportance, numLayers, bandwidthBytesPerMs, adjustedBeta
      );
      
      // 语义失真
      const distortion = this.computeSemanticDistortion(l, numLayers, retention, taskType);
      
      // 精度计算
      const precision = this.computePrecision(ibImportance, distortion, taskType);
      
      // 综合重要性分数
      const importanceScore = ibImportance * adjustedBeta;
      
      layerAllocations.push({
        layerIndex: l,
        predictiveInformation: ibImportance,
        allocatedRate: rate,
        precision,
        retentionRatio: retention,
        importanceScore
      });
      
      totalRate += rate;
      totalImportance += ibImportance;
    }
    
    // Step 4: 计算理论界
    const theoretical = this.computeTheoreticalBounds(
      bandwidthBytesPerMs,
      maxDistortion,
      totalImportance,
      numLayers,
      baseBeta
    );
    
    // Step 5: 生成压缩配置输出
    const prefillOutput = this.generateCompressionOutput(
      layerAllocations,
      numLayers,
      phase === 'prefill' ? 'prefill' : 'decode'
    );
    const decodeOutput = this.generateCompressionOutput(
      layerAllocations,
      numLayers,
      'decode'
    );
    
    // Step 6: 计算统一目标函数
    const unifiedObjective = this.computeUnifiedObjective(totalRate, totalImportance / numLayers, adjustedBeta);
    
    // Step 7: 计算实际R-D指标
    const achievedDistortion = layerAllocations.reduce(
      (sum, l) => sum + l.predictiveInformation * (1 - l.retentionRatio), 0
    ) / numLayers;
    
    const achievedQuality = 1 - achievedDistortion;
    const achievedRate = totalRate;
    
    return {
      theoretical,
      layerAllocations,
      unifiedObjective,
      prefillOutput,
      decodeOutput,
      achievedRate,
      achievedDistortion,
      achievedQuality,
      ibContribution: totalImportance / numLayers,
      rdContribution: achievedQuality / (achievedRate + 1e-6)
    };
  }

  /**
   * 计算精度
   */
  private computePrecision(
    ibImportance: number,
    distortion: number,
    taskType: TaskType
  ): number {
    // 精度根据重要性和失真动态调整
    // 高重要性 + 低失真容忍 → 高精度
    // 低重要性 + 高失真容忍 → 低精度
    
    if (ibImportance > 0.7 && distortion < 0.3) {
      return this.FULL_PRECISION;  // FP16
    } else if (ibImportance > 0.4) {
      return this.HALF_PRECISION;  // FP8
    } else {
      return this.QUARTER_PRECISION; // INT4
    }
  }

  /**
   * 生成压缩配置输出
   */
  private generateCompressionOutput(
    layerAllocations: LayerAllocation[],
    totalLayers: number,
    phase: 'prefill' | 'decode'
  ): CompressionOutput {
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    for (const alloc of layerAllocations) {
      if (phase === 'prefill') {
        // P端：更激进的压缩
        const pRetention = phase === 'prefill' ? 
          alloc.retentionRatio * 0.9 : // P端保留稍低
          alloc.retentionRatio;
        pLayerRetention.push(ensureRetentionRange(pRetention));
        dLayerRetention.push(ensureRetentionRange(alloc.retentionRatio));
        
        // P端精度：K8V4（标准压缩）
        pKeyPrecision.push(this.HALF_PRECISION);
        pValuePrecision.push(this.QUARTER_PRECISION);
      } else {
        // D端：更保守的压缩
        pLayerRetention.push(ensureRetentionRange(alloc.retentionRatio * 0.95));
        dLayerRetention.push(ensureRetentionRange(alloc.retentionRatio * 1.05));
        
        pKeyPrecision.push(this.HALF_PRECISION);
        pValuePrecision.push(this.QUARTER_PRECISION);
      }
      
      // D端精度：根据重要性分配
      if (alloc.predictiveInformation > 0.7) {
        dKeyPrecision.push(this.FULL_PRECISION);
        dValuePrecision.push(this.HALF_PRECISION);
      } else if (alloc.predictiveInformation > 0.4) {
        dKeyPrecision.push(this.HALF_PRECISION);
        dValuePrecision.push(this.QUARTER_PRECISION);
      } else {
        dKeyPrecision.push(this.QUARTER_PRECISION);
        dValuePrecision.push(this.QUARTER_PRECISION);
      }
    }
    
    const avgRetention = pLayerRetention.reduce((a, b) => a + b, 0) / totalLayers;
    const pPrecisionRatio = pKeyPrecision.reduce((a, b) => a + b, 0) / (totalLayers * this.FULL_PRECISION);
    const avgCompressionRatio = avgRetention * pPrecisionRatio;
    
    return {
      strategy: `IB-RD-Unified-${phase}`,
      totalLayers,
      pLayerRetention,
      dLayerRetention,
      pKeyPrecision,
      pValuePrecision,
      dKeyPrecision,
      dValuePrecision,
      avgCompressionRatio: ensureRetentionRange(avgCompressionRatio),
      estimatedBandwidthSaving: ensureRetentionRange(1 - avgCompressionRatio)
    };
  }

  /**
   * 计算理论界
   */
  private computeTheoreticalBounds(
    bandwidthBytesPerMs: number,
    maxDistortion: number,
    totalImportance: number,
    numLayers: number,
    beta: number
  ): TheoreticalBounds {
    // Shannon下界：R(D) >= h(X) - h(X|Z)
    // 在实践中，我们使用近似
    const shannonBound = Math.log2(bandwidthBytesPerMs + 1) / numLayers;
    
    // 理论最小速率：在maxDistortion约束下
    const theoreticalMinRate = bandwidthBytesPerMs * (1 - maxDistortion);
    
    // 理论最大质量
    const theoreticalMaxQuality = Math.min(1, totalImportance / numLayers + (1 - maxDistortion));
    
    // 最优操作点：在R-D曲线上平衡点
    const optimalRate = theoreticalMinRate * 1.2; // 略高于最小值以保证质量
    const optimalDistortion = maxDistortion * 0.8;
    const optimalQuality = 1 - optimalDistortion;
    
    return {
      theoreticalMinRate: round4(theoreticalMinRate),
      theoreticalMaxQuality: round4(Math.min(1, theoreticalMaxQuality)),
      shannonBound: round4(shannonBound),
      optimalOperatingPoint: {
        rate: round4(optimalRate),
        distortion: round4(optimalDistortion),
        quality: round4(optimalQuality),
        beta: round4(beta)
      }
    };
  }

  /**
   * 生成完整R-D曲线
   */
  generateRDCurve(
    config: Omit<SemanticRDConfig, 'maxDistortion' | 'bandwidthBytesPerMs'>,
    rateRange: number[],
    bandwidthBytesPerMs: number
  ): RDCurvePoint[] {
    const points: RDCurvePoint[] = [];
    
    for (const rate of rateRange) {
      // 对于每个速率，找到最优的β
      for (const beta of [0.5, 1.0, 2.0, 5.0]) {
        const result = this.optimize({
          ...config,
          bandwidthBytesPerMs: rate,
          maxDistortion: 0.5,
          beta
        });
        
        points.push({
          rate: result.achievedRate,
          distortion: result.achievedDistortion,
          quality: result.achievedQuality,
          beta,
          operationMode: beta < 1 ? 'conservative' : beta < 2 ? 'balanced' : 'aggressive'
        });
      }
    }
    
    // 去重并排序
    const uniquePoints = Array.from(
      new Map(points.map(p => [`${p.rate}-${p.beta}`, p])).values()
    ).sort((a, b) => a.rate - b.rate);
    
    return uniquePoints;
  }

  /**
   * Phase-aware统一优化
   * 
   * 核心思想：
   * - P端（prefill）：β大，带宽约束紧 → 激进压缩
   * - D端（decode）：β小，质量约束紧 → 保守压缩
   * 
   * 总目标：min [L_P(β_P) + L_D(β_D)] s.t. D_P + D_D ≤ D_max
   */
  phaseAwareOptimize(
    taskType: TaskType,
    numLayers: number,
    bandwidthBytesPerMs: number,
    sloLatencyMs: number = 1000
  ): {
    prefill: SemanticRDResult;
    decode: SemanticRDResult;
    unifiedObjective: number;
    totalDistortion: number;
    totalRate: number;
  } {
    // P端优化
    const prefillResult = this.optimize({
      bandwidthBytesPerMs,
      maxDistortion: 0.4, // P端失真容忍较低
      taskType,
      numLayers,
      sequenceLength: 2048,
      phase: 'prefill',
      beta: this.deriveOptimalBeta(bandwidthBytesPerMs, 0.4, taskType) * 1.3
    });
    
    // D端优化
    const decodeResult = this.optimize({
      bandwidthBytesPerMs: bandwidthBytesPerMs * 0.7, // D端可用带宽较少
      maxDistortion: 0.3, // D端失真容忍更低
      taskType,
      numLayers,
      sequenceLength: 512,
      phase: 'decode',
      beta: this.deriveOptimalBeta(bandwidthBytesPerMs, 0.3, taskType) * 0.7
    });
    
    // 总目标函数
    const unifiedObjective = 
      prefillResult.unifiedObjective * 0.6 + 
      decodeResult.unifiedObjective * 0.4;
    
    // 总失真和总速率
    const totalDistortion = 
      prefillResult.achievedDistortion * 0.6 + 
      decodeResult.achievedDistortion * 0.4;
    
    const totalRate = 
      prefillResult.achievedRate * 0.6 + 
      decodeResult.achievedRate * 0.4;
    
    return {
      prefill: prefillResult,
      decode: decodeResult,
      unifiedObjective: round4(unifiedObjective),
      totalDistortion: round4(totalDistortion),
      totalRate: round4(totalRate)
    };
  }
}

// 导出单例访问函数
export const semanticRDFramework = new SemanticRDFramework();

// 导出便捷函数
export function optimizeSemanticRD(config: SemanticRDConfig): SemanticRDResult {
  return semanticRDFramework.optimize(config);
}

export function phaseAwareOptimize(
  taskType: TaskType,
  numLayers: number,
  bandwidthBytesPerMs: number,
  sloLatencyMs?: number
): ReturnType<SemanticRDFramework['phaseAwareOptimize']> {
  return semanticRDFramework.phaseAwareOptimize(taskType, numLayers, bandwidthBytesPerMs, sloLatencyMs);
}
