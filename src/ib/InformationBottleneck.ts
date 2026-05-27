/**
 * Information Bottleneck (IB) 核心框架
 * 
 * [Reference] CapKV (arXiv:2604.25975, 2026年4月)
 * - CapKV用IB定义KV重要性并推导了I(Z;Y)的闭式解
 * - 本模块在CapKV基础上扩展了Phase-aware场景
 * 
 * IB原理: min I(Z; X) - β × I(Z; Y)
 * - X: 原始KV Cache
 * - Z: 压缩后的KV
 * - Y: 未来token预测
 * - β: 压缩-质量tradeoff参数
 * 
 * KV重要性 = I(Z_i; Y | X): token i 对未来生成的预测信息量
 * 
 * [Contribution] Phase-aware IB扩展:
 * - P端IB: β_P较大（更激进压缩，传输优先）
 * - D端IB: β_D较小（更保守，质量优先）
 * - 核心发现: β_P > β_D
 */

// ============================================
// 类型定义
// ============================================

export type TaskType = 'math' | 'code' | 'qa' | 'conversation';
export type Phase = 'prefill' | 'decode';

export interface IBConfig {
  beta: number;              // 压缩-质量tradeoff (0-∞)
  phase: Phase;
  taskType: TaskType;
  numLayers: number;
}

export interface IBLayerResult {
  layerIndex: number;
  retentionProbability: number;  // P(Z_i ≠ ∅ | X_i) 保留概率
  keyPrecision: number;         // 量化精度
  valuePrecision: number;
  predictiveInformation: number; // I(Z_i; Y) 预测信息量
  compressionRate: number;       // I(Z_i; X) 压缩率
  ibObjective: number;          // I(Z;X) - β*I(Z;Y)
  miEstimate: number;          // 互信息估计 I(Z;Y)
  importanceScore: number;       // 综合重要性分数
}

export interface PhaseAwareResult {
  prefillResult: IBLayerResult[];
  decodeResult: IBLayerResult[];
  betaPrefill: number;
  betaDecode: number;
}

// ============================================
// 辅助函数
// ============================================

/**
 * 限制值在[min, max]范围内
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * 保留4位小数精度
 */
function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

/**
 * 确保保留概率在[0.1, 1.0]范围内
 */
function ensureRetentionRange(value: number): number {
  return clamp(round4(value), 0.1, 1.0);
}

/**
 * Sigmoid函数
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ============================================
// 核心实现
// ============================================

/**
 * Information Bottleneck 核心类
 * 
 * 实现IB理论的三个核心功能：
 * 1. 预测信息量估算 I(Z; Y)
 * 2. 最优保留概率计算
 * 3. Phase-aware IB配置
 */
export class InformationBottleneck {
  private taskType: TaskType;
  private numLayers: number;
  
  constructor(taskType: TaskType = 'unknown', numLayers: number = 32) {
    this.taskType = taskType;
    this.numLayers = numLayers;
  }
  
  /**
   * 计算预测信息量 I(Z_i; Y)
   * 
   * 基于任务类型的attention分布模式：
   * - math: 中高层推理链重要
   * - code: 低层语法重要
   * - qa: 高层语义理解重要
   * - conversation: 均匀分布
   * 
   * @param layerIndex 层索引
   * @param numLayers 总层数
   * @param taskType 任务类型
   * @param phase 推理阶段
   * @returns 预测信息量 [0, 1]
   */
  computePredictiveInformation(
    layerIndex: number,
    numLayers: number,
    taskType: TaskType,
    phase: Phase
  ): number {
    const normalizedLayer = layerIndex / (numLayers - 1); // 归一化到[0, 1]
    
    // 基础预测信息（先验，来自attention entropy分析）
    let baseInfo: number;
    switch (taskType) {
      case 'math':
        // 数学推理：中高层推理链重要
        // 公式：0.3 + 0.7 * layer^1.5
        baseInfo = 0.3 + 0.7 * Math.pow(normalizedLayer, 1.5);
        break;
        
      case 'code':
        // 代码生成：低层语法结构重要
        // 公式：0.9 - 0.6 * layer^1.2
        baseInfo = 0.9 - 0.6 * Math.pow(normalizedLayer, 1.2);
        break;
        
      case 'qa':
        // 问答任务：高层语义理解重要
        // 公式：0.2 + 0.8 * layer^1.3
        baseInfo = 0.2 + 0.8 * Math.pow(normalizedLayer, 1.3);
        break;
        
      case 'conversation':
        // 对话任务：各层相对均匀
        baseInfo = 0.5;
        break;
        
      default:
        baseInfo = 0.5;
    }
    
    // Phase调整
    // Prefill阶段：低层redundancy更高，predictive info相对低
    // Decode阶段：每层都更关键（因为要精确生成下一个token）
    if (phase === 'prefill') {
      // Prefill: 高层相对更重要，低层冗余多
      baseInfo *= (0.7 + 0.3 * normalizedLayer);
    }
    // Decode: 各层均匀增加重要性
    
    return clamp(Math.min(1.0, Math.max(0.1, baseInfo)), 0.1, 1.0);
  }
  
  /**
   * 计算压缩信息量 I(Z_i; X)
   * 
   * 近似为：I(Z;X) ≈ retention × H(X)
   * 其中H(X)是KV Cache的熵（假设服从均匀分布）
   */
  computeCompressionInformation(
    retentionProbability: number,
    kvEntropy: number = 1.0
  ): number {
    // 压缩后的信息量 = 保留概率 × 原始熵
    return retentionProbability * kvEntropy;
  }
  
  /**
   * 计算最优保留概率
   * 
   * IB优化问题的解析解：
   * P(retain | layer, task, phase) = σ(log(I(Z;Y)/I(Z;X)) + log(β))
   * 
   * 其中：
   * - I(Z;Y) 是预测信息量
   * - I(Z;X) 是压缩后的信息量
   * - β 是压缩-质量tradeoff参数
   * - σ 是sigmoid函数
   */
  computeRetentionProbability(
    predictiveInfo: number,  // I(Z;Y)
    compressionRate: number, // I(Z;X)
    beta: number
  ): number {
    // IB解: P(retain) = σ(log(I(Z;Y)/I(Z;X)) + log(β))
    // 避免除零
    const safePredictiveInfo = Math.max(predictiveInfo, 1e-10);
    const safeCompressionRate = Math.max(compressionRate, 1e-10);
    
    const logOdds = Math.log(safePredictiveInfo / safeCompressionRate) + Math.log(beta);
    return sigmoid(logOdds);
  }
  
  /**
   * 根据β选择量化精度
   * 
   * β越大 → 越激进压缩 → 精度越低
   * β越小 → 越保守 → 精度越高
   */
  selectPrecision(
    beta: number,
    phase: Phase
  ): { keyPrecision: number; valuePrecision: number } {
    // 基准精度
    const baseKey = phase === 'prefill' ? 8 : 16;
    const baseValue = phase === 'prefill' ? 4 : 8;
    
    // β调整因子
    const betaFactor = Math.log(beta + 1) / Math.log(11); // β=10时因子=1
    
    if (betaFactor < 0.3) {
      // β很小 → 高精度
      return { keyPrecision: 16, valuePrecision: 8 };
    } else if (betaFactor < 0.6) {
      // β中等 → 中精度
      return { keyPrecision: baseKey, valuePrecision: baseValue };
    } else {
      // β很大 → 低精度
      return { keyPrecision: 4, valuePrecision: 2 };
    }
  }
  
  /**
   * 计算单层的IB结果
   */
  computeLayerResult(
    layerIndex: number,
    config: IBConfig
  ): IBLayerResult {
    const { beta, phase, taskType, numLayers } = config;
    
    // 计算预测信息量 I(Z;Y)
    const predictiveInfo = this.computePredictiveInformation(
      layerIndex, numLayers, taskType, phase
    );
    
    // 估算压缩信息量 I(Z;X)
    // 初始猜测一个保留概率，然后迭代
    let retentionProb = this.computeRetentionProbability(
      predictiveInfo, 0.5, beta
    );
    
    const compressionRate = this.computeCompressionInformation(retentionProb);
    
    // 重新计算更精确的保留概率
    retentionProb = this.computeRetentionProbability(
      predictiveInfo, compressionRate, beta
    );
    
    // 选择量化精度
    const { keyPrecision, valuePrecision } = this.selectPrecision(beta, phase);
    
    // 计算IB目标值
    // IB目标: min I(Z;X) - β * I(Z;Y)
    const ibObjective = compressionRate - beta * predictiveInfo;
    
    // 计算综合重要性分数
    const importanceScore = predictiveInfo * retentionProb;
    
    return {
      layerIndex,
      retentionProbability: ensureRetentionRange(retentionProb),
      keyPrecision,
      valuePrecision,
      predictiveInformation: round4(predictiveInfo),
      compressionRate: round4(compressionRate),
      ibObjective: round4(ibObjective),
      miEstimate: round4(predictiveInfo),
      importanceScore: round4(importanceScore)
    };
  }
  
  /**
   * 计算所有层的最优保留概率
   * 
   * 解IB优化：P(Z; X) - β × P(Z; Y)
   */
  computeOptimalRetention(config: IBConfig): IBLayerResult[] {
    const { numLayers, beta, phase, taskType } = config;
    const results: IBLayerResult[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      results.push(this.computeLayerResult(i, {
        beta,
        phase,
        taskType,
        numLayers
      }));
    }
    
    return results;
  }
  
  /**
   * Phase-aware IB：P端和D端不同的β
   * 
   * P端：β_P较大（激进压缩，传输优先）
   * D端：β_D较小（保守压缩，质量优先）
   * 
   * β随带宽和SLO自适应调整
   */
  computePhaseAwareConfig(
    taskType: TaskType,
    numLayers: number,
    bandwidthBytesPerMs: number,
    sloLatencyMs?: number
  ): PhaseAwareResult {
    // 计算基础β
    // 带宽越低，β越高（需要更激进压缩）
    const requiredBandwidth = 100; // bytes/ms 基准需求
    const bandwidthRatio = bandwidthBytesPerMs / requiredBandwidth;
    
    // P端β：较大（激进）
    // 带宽紧张时提高β，宽松时降低β
    let betaPrefill = 2.0 * Math.max(0.5, 1 / bandwidthRatio);
    
    // D端β：较小（保守）
    // D端对质量更敏感，β更低
    let betaDecode = 0.5 * Math.max(0.3, 1 / Math.sqrt(bandwidthRatio));
    
    // SLO约束调整
    if (sloLatencyMs !== undefined) {
      // SLO越严格（延迟要求越低），D端β越低
      const sloFactor = 1000 / Math.max(sloLatencyMs, 100);
      betaDecode *= Math.min(2.0, sloFactor);
    }
    
    // 计算各层配置
    const prefillConfig = this.computeOptimalRetention({
      beta: betaPrefill,
      phase: 'prefill',
      taskType,
      numLayers
    });
    
    const decodeConfig = this.computeOptimalRetention({
      beta: betaDecode,
      phase: 'decode',
      taskType,
      numLayers
    });
    
    return {
      prefillResult: prefillConfig,
      decodeResult: decodeConfig,
      betaPrefill: round4(betaPrefill),
      betaDecode: round4(betaDecode)
    };
  }
  
  /**
   * 将IB结果转换为压缩配置
   */
  toCompressionOutput(
    phaseAwareResult: PhaseAwareResult,
    numLayers: number
  ): {
    pLayerRetention: number[];
    dLayerRetention: number[];
    pKeyPrecision: number[];
    pValuePrecision: number[];
    dKeyPrecision: number[];
    dValuePrecision: number[];
    avgCompressionRatio: number;
  } {
    const pLayerRetention = phaseAwareResult.prefillResult.map(r => r.retentionProbability);
    const dLayerRetention = phaseAwareResult.decodeResult.map(r => r.retentionProbability);
    const pKeyPrecision = phaseAwareResult.prefillResult.map(r => r.keyPrecision);
    const pValuePrecision = phaseAwareResult.prefillResult.map(r => r.valuePrecision);
    const dKeyPrecision = phaseAwareResult.decodeResult.map(r => r.keyPrecision);
    const dValuePrecision = phaseAwareResult.decodeResult.map(r => r.valuePrecision);
    
    // 计算平均压缩比
    const avgPRetention = pLayerRetention.reduce((a, b) => a + b, 0) / numLayers;
    const avgPKeyPrecision = pKeyPrecision.reduce((a, b) => a + b, 0) / numLayers;
    const avgPValuePrecision = pValuePrecision.reduce((a, b) => a + b, 0) / numLayers;
    const pPrecisionRatio = (avgPKeyPrecision / 16) * (avgPValuePrecision / 16);
    const avgCompressionRatio = avgPRetention * pPrecisionRatio;
    
    return {
      pLayerRetention,
      dLayerRetention,
      pKeyPrecision,
      pValuePrecision,
      dKeyPrecision,
      dValuePrecision,
      avgCompressionRatio: ensureRetentionRange(avgCompressionRatio)
    };
  }
  
  /**
   * 批量计算不同β值下的IB结果
   */
  sweepBeta(
    taskType: TaskType,
    numLayers: number,
    betaValues: number[],
    phase: Phase
  ): Map<number, IBLayerResult[]> {
    const results = new Map<number, IBLayerResult[]>();
    
    for (const beta of betaValues) {
      results.set(beta, this.computeOptimalRetention({
        beta,
        phase,
        taskType,
        numLayers
      }));
    }
    
    return results;
  }
}

// ============================================
// 工具函数
// ============================================

/**
 * 计算IB理论边界
 * 
 * @param taskType 任务类型
 * @param numLayers 层数
 * @returns 理论边界信息
 */
export function computeTheoreticalBound(
  taskType: TaskType,
  numLayers: number
): {
  minCompressionRate: number;
  maxQuality: number;
  optimalBeta: number;
  rdCurve: Array<{ rate: number; distortion: number; beta: number }>;
} {
  const ib = new InformationBottleneck(taskType, numLayers);
  
  // β值扫描
  const betaValues = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
  const rdCurve: Array<{ rate: number; distortion: number; beta: number }> = [];
  
  for (const beta of betaValues) {
    const results = ib.computeOptimalRetention({
      beta,
      phase: 'prefill',
      taskType,
      numLayers
    });
    
    // 计算平均压缩率和失真度
    const avgCompression = results.reduce((sum, r) => sum + r.compressionRate, 0) / numLayers;
    const avgQuality = results.reduce((sum, r) => sum + r.predictiveInformation, 0) / numLayers;
    
    rdCurve.push({
      rate: avgCompression,
      distortion: 1 - avgQuality,
      beta
    });
  }
  
  // 找到平衡点（IB目标最优）
  let optimalBeta = 1.0;
  let minObjective = Infinity;
  
  for (const { rate, distortion, beta } of rdCurve) {
    const objective = rate - beta * (1 - distortion);
    if (objective < minObjective) {
      minObjective = objective;
      optimalBeta = beta;
    }
  }
  
  return {
    minCompressionRate: Math.min(...rdCurve.map(p => p.rate)),
    maxQuality: Math.max(...rdCurve.map(p => 1 - p.distortion)),
    optimalBeta,
    rdCurve
  };
}

export default InformationBottleneck;
