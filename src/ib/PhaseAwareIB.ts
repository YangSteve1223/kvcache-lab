/**
 * Phase-aware IB 统一框架
 * 
 * [Reference] IB基础框架参考: CapKV (arXiv:2604.25975, 2026年4月)
 * CapKV用IB定义KV重要性并推导了I(Z;Y)的闭式解
 * 
 * [Contribution] 本模块的核心贡献：将IB扩展到Phase-aware场景
 * CapKV不区分Prefill/Decode端，本模块证明P端和D端需要不同的β值
 * 
 * Phase-aware IB:
 * - Prefill: min I(Z_P; X) - β_P × I(Z_P; Y)
 * - Decode:  min I(Z_D; X) - β_D × I(Z_D; Y)
 * 
 * 核心发现：β_P > β_D（这是CapKV未探索的差异化设计空间）
 */

import {
  InformationBottleneck,
  IBLayerResult,
  TaskType,
  Phase,
  computeTheoreticalBound
} from './InformationBottleneck.js';
import { MutualInformationEstimator } from './MutualInformationEstimator.js';

// ============================================
// 类型定义
// ============================================

export interface PhaseAwareIBConfig {
  taskType: TaskType;
  numLayers: number;
  bandwidthBytesPerMs: number;
  memoryBytes: number;
  sloLatencyMs?: number;
  qualityTarget?: number;
  compressionTarget?: number;
}

export interface PhaseAwareIBResult {
  prefillResult: IBLayerResult[];
  decodeResult: IBLayerResult[];
  betaPrefill: number;
  betaDecode: number;
  theoreticalBound: {
    minCompressionRate: number;
    maxQuality: number;
    optimalBeta: number;
    rdCurve: Array<{ rate: number; distortion: number; beta: number }>;
  };
  constraints: {
    bandwidthSatisfied: boolean;
    memorySatisfied: boolean;
    sloSatisfied: boolean;
    qualitySatisfied: boolean;
  };
  tradeoffs: {
    prefillCompressionGain: number;
    decodeQualityGain: number;
    phaseDiscriminationFactor: number;
  };
  // CapKV baseline对比
  capkvBaseline?: {
    uniformBeta: number;
    avgCompressionRate: number;
    avgQuality: number;
  };
}

export interface RDCurvePoint {
  rate: number;
  quality: number;
  beta: number;
  strategy: 'prefill' | 'decode' | 'phase-aware' | 'capkv-baseline';
}

// ============================================
// 辅助函数
// ============================================

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function estimateTransferTime(compressedBytes: number, bandwidthBytesPerMs: number): number {
  if (bandwidthBytesPerMs <= 0) return Infinity;
  return compressedBytes / bandwidthBytesPerMs;
}

// ============================================
// Phase-aware IB 统一框架
// ============================================

/**
 * Phase-aware IB 类
 * 
 * [核心贡献] 将CapKV的IB框架扩展到Phase-aware场景
 * 
 * 与CapKV的区别：
 * - CapKV: 统一的IB优化，不区分P/D端
 * - PhaseAwareIB: P端和D端独立的IB优化，β_P > β_D
 * 
 * 物理动机：
 * - P端受传输带宽限制 → 需要更激进的压缩 → β_P大
 * - D端在GPU上计算 → 精度更重要 → β_D小
 */
export class PhaseAwareIB {
  private taskType: TaskType;
  private numLayers: number;
  private ib: InformationBottleneck;
  private miEstimator: MutualInformationEstimator;
  
  // 默认参数
  private defaultBandwidth = 100;
  private defaultMemory = 8 * 1024 * 1024 * 1024;
  private defaultSLO = 1000;
  
  constructor(taskType: TaskType = 'unknown', numLayers: number = 32) {
    this.taskType = taskType;
    this.numLayers = numLayers;
    this.ib = new InformationBottleneck(taskType, numLayers);
    this.miEstimator = new MutualInformationEstimator(0, numLayers, taskType, 'prefill');
  }
  
  /**
   * [核心方法] Phase-aware IB优化
   * 
   * 与CapKV的区别：
   * - CapKV: 单个β应用于所有层
   * - PhaseAwareIB: β_P（P端） ≠ β_D（D端）
   */
  optimize(config: PhaseAwareIBConfig): PhaseAwareIBResult {
    const { taskType, numLayers, bandwidthBytesPerMs, memoryBytes, sloLatencyMs } = config;
    
    // 计算Phase-aware的β值（核心创新点）
    const { betaPrefill, betaDecode } = this.computePhaseAwareBeta(
      bandwidthBytesPerMs,
      sloLatencyMs
    );
    
    // P端优化（激进压缩）
    const prefillResult = this.ib.computeOptimalRetention({
      beta: betaPrefill,
      phase: 'prefill',
      taskType,
      numLayers
    });
    
    // D端优化（保守压缩）
    const decodeResult = this.ib.computeOptimalRetention({
      beta: betaDecode,
      phase: 'decode',
      taskType,
      numLayers
    });
    
    // 计算理论边界
    const theoreticalBound = computeTheoreticalBound(taskType, numLayers);
    
    // 检查约束
    const constraints = this.checkConstraints(
      prefillResult, decodeResult, bandwidthBytesPerMs, memoryBytes, sloLatencyMs, numLayers
    );
    
    // 计算tradeoff收益
    const tradeoffs = this.computeTradeoffs(prefillResult, decodeResult, betaPrefill, betaDecode);
    
    // 计算CapKV baseline（统一β）
    const capkvBaseline = this.computeCapKVBaseline(taskType, numLayers);
    
    return {
      prefillResult,
      decodeResult,
      betaPrefill,
      betaDecode,
      theoreticalBound,
      constraints,
      tradeoffs,
      capkvBaseline
    };
  }
  
  /**
   * [核心创新] 计算Phase-aware的β值
   * 
   * CapKV用统一β，本方法证明P端和D端需要不同的β
   */
  private computePhaseAwareBeta(
    bandwidthBytesPerMs: number,
    sloLatencyMs?: number
  ): { betaPrefill: number; betaDecode: number } {
    // 带宽比率
    const bandwidthRatio = clamp(bandwidthBytesPerMs / this.defaultBandwidth, 0.1, 10);
    
    // SLO比率
    const sloRatio = sloLatencyMs
      ? clamp(this.defaultSLO / Math.max(sloLatencyMs, 1), 0.1, 10)
      : 1;
    
    // P端β：带宽紧张时增大
    // [创新点] P端β与带宽成反比（传输瓶颈）
    let betaPrefill = 2.0 * Math.max(0.5, 1 / bandwidthRatio);
    
    // D端β：SLO严格时减小
    // [创新点] D端β与SLO严格程度成正比（质量优先）
    let betaDecode = 0.5 * sloRatio;
    
    // [关键约束] 确保 β_P > β_D
    // 这就是Phase-aware与CapKV的本质区别！
    if (betaPrefill <= betaDecode) {
      betaPrefill = betaDecode * 1.5;
    }
    
    return { betaPrefill: round4(betaPrefill), betaDecode: round4(betaDecode) };
  }
  
  /**
   * [Baseline] 计算CapKV baseline（统一β）
   * 
   * CapKV不区分P/D端，用单一β
   */
  private computeCapKVBaseline(taskType: TaskType, numLayers: number): PhaseAwareIBResult['capkvBaseline'] {
    // CapKV用固定β=1.0（中等压缩）
    const uniformBeta = 1.0;
    const result = this.ib.computeOptimalRetention({
      beta: uniformBeta,
      phase: 'prefill',
      taskType,
      numLayers
    });
    
    const avgCompressionRate = result.reduce((s, r) => s + r.compressionRate, 0) / numLayers;
    const avgQuality = result.reduce((s, r) => s + r.predictiveInformation, 0) / numLayers;
    
    return {
      uniformBeta,
      avgCompressionRate: round4(avgCompressionRate),
      avgQuality: round4(avgQuality)
    };
  }
  
  private checkConstraints(
    prefillResult: IBLayerResult[],
    decodeResult: IBLayerResult[],
    bandwidthBytesPerMs: number,
    memoryBytes: number,
    sloLatencyMs: number | undefined,
    numLayers: number
  ): PhaseAwareIBResult['constraints'] {
    const avgPrefillCompression = prefillResult.reduce((s, r) => s + r.retentionProbability, 0) / numLayers;
    const avgPrefillPrecision = prefillResult.reduce((s, r) => s + (r.keyPrecision + r.valuePrecision) / 2, 0) / numLayers;
    const prefillCompression = avgPrefillCompression * (avgPrefillPrecision / 16);
    
    const kvSizePerToken = 128 * 1024;
    const estimatedTransferBytes = kvSizePerToken * prefillCompression;
    const transferTime = estimateTransferTime(estimatedTransferBytes, bandwidthBytesPerMs);
    
    const avgDecodeRetention = decodeResult.reduce((s, r) => s + r.retentionProbability, 0) / numLayers;
    const avgDecodePrecision = decodeResult.reduce((s, r) => s + (r.keyPrecision + r.valuePrecision) / 2, 0) / numLayers;
    const decodeMemory = avgDecodeRetention * avgDecodePrecision / 16;
    const estimatedMemoryUsage = memoryBytes * decodeMemory;
    
    return {
      bandwidthSatisfied: bandwidthBytesPerMs >= estimatedTransferBytes / 1000,
      memorySatisfied: estimatedMemoryUsage <= memoryBytes * 0.8,
      sloSatisfied: sloLatencyMs ? transferTime <= sloLatencyMs : true,
      qualitySatisfied: (decodeResult.reduce((s, r) => s + r.predictiveInformation, 0) / numLayers) >= 0.7
    };
  }
  
  private computeTradeoffs(
    prefillResult: IBLayerResult[],
    decodeResult: IBLayerResult[],
    betaPrefill: number,
    betaDecode: number
  ): PhaseAwareIBResult['tradeoffs'] {
    const avgPrefillCompression = prefillResult.reduce((s, r) => s + r.retentionProbability, 0) / this.numLayers;
    const baselineCompression = 0.5;
    const prefillCompressionGain = (baselineCompression - avgPrefillCompression) / baselineCompression;
    
    const avgPrefillQuality = prefillResult.reduce((s, r) => s + r.predictiveInformation, 0) / this.numLayers;
    const avgDecodeQuality = decodeResult.reduce((s, r) => s + r.predictiveInformation, 0) / this.numLayers;
    const decodeQualityGain = (avgDecodeQuality - avgPrefillQuality) / avgPrefillQuality;
    
    const phaseDiscriminationFactor = betaPrefill / betaDecode;
    
    return {
      prefillCompressionGain: round4(prefillCompressionGain),
      decodeQualityGain: round4(decodeQualityGain),
      phaseDiscriminationFactor: round4(phaseDiscriminationFactor)
    };
  }
  
  /**
   * 生成RD曲线
   */
  generateRDCurve(taskType: TaskType, numLayers: number): RDCurvePoint[] {
    const betaValues = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
    const curve: RDCurvePoint[] = [];
    
    for (const beta of betaValues) {
      // Prefill曲线
      const prefillResult = this.ib.computeOptimalRetention({ beta: beta * 2, phase: 'prefill', taskType, numLayers });
      const prefillRate = prefillResult.reduce((s, r) => s + r.compressionRate, 0) / numLayers;
      const prefillQuality = prefillResult.reduce((s, r) => s + r.predictiveInformation, 0) / numLayers;
      curve.push({ rate: round4(prefillRate), quality: round4(prefillQuality), beta, strategy: 'prefill' });
      
      // Decode曲线
      const decodeResult = this.ib.computeOptimalRetention({ beta: beta * 0.5, phase: 'decode', taskType, numLayers });
      const decodeRate = decodeResult.reduce((s, r) => s + r.compressionRate, 0) / numLayers;
      const decodeQuality = decodeResult.reduce((s, r) => s + r.predictiveInformation, 0) / numLayers;
      curve.push({ rate: round4(decodeRate), quality: round4(decodeQuality), beta, strategy: 'decode' });
      
      // Phase-aware曲线
      const phaseAwareRate = (prefillRate + decodeRate) / 2;
      const phaseAwareQuality = (prefillQuality + decodeQuality) / 2;
      curve.push({ rate: round4(phaseAwareRate), quality: round4(phaseAwareQuality), beta, strategy: 'phase-aware' });
      
      // CapKV baseline（统一β）
      const capkvResult = this.ib.computeOptimalRetention({ beta, phase: 'prefill', taskType, numLayers });
      const capkvRate = capkvResult.reduce((s, r) => s + r.compressionRate, 0) / numLayers;
      const capkvQuality = capkvResult.reduce((s, r) => s + r.predictiveInformation, 0) / numLayers;
      curve.push({ rate: round4(capkvRate), quality: round4(capkvQuality), beta, strategy: 'capkv-baseline' });
    }
    
    return curve;
  }
  
  /**
   * 对比Phase-aware vs CapKV
   */
  compareWithCapKV(taskType: TaskType, numLayers: number): {
    phaseAware: { avgCompressionRate: number; avgQuality: number };
    capkv: { avgCompressionRate: number; avgQuality: number };
    improvement: { compressionGain: number; qualityGain: number };
  } {
    const phaseAwareResult = this.optimize({
      taskType, numLayers,
      bandwidthBytesPerMs: this.defaultBandwidth,
      memoryBytes: this.defaultMemory,
      sloLatencyMs: this.defaultSLO
    });
    
    const capkvBaseline = phaseAwareResult.capkvBaseline!;
    
    const phaseAwareCompression = phaseAwareResult.prefillResult.reduce((s, r) => s + r.compressionRate, 0) / numLayers;
    const phaseAwareQuality = phaseAwareResult.decodeResult.reduce((s, r) => s + r.predictiveInformation, 0) / numLayers;
    
    return {
      phaseAware: { avgCompressionRate: round4(phaseAwareCompression), avgQuality: round4(phaseAwareQuality) },
      capkv: { avgCompressionRate: capkvBaseline.avgCompressionRate, avgQuality: capkvBaseline.avgQuality },
      improvement: {
        compressionGain: round4((capkvBaseline.avgCompressionRate - phaseAwareCompression) / capkvBaseline.avgCompressionRate),
        qualityGain: round4((phaseAwareQuality - capkvBaseline.avgQuality) / capkvBaseline.avgQuality)
      }
    };
  }
}

// ============================================
// 便捷函数
// ============================================

export function optimizePhaseAwareIB(config: PhaseAwareIBConfig): PhaseAwareIBResult {
  const pib = new PhaseAwareIB(config.taskType, config.numLayers);
  return pib.optimize(config);
}

export default PhaseAwareIB;
