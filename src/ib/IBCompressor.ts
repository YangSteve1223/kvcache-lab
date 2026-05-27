/**
 * PhaseAwareIBCompressor - Phase-aware IB驱动的压缩策略
 * 
 * [Reference] IB基础参考: CapKV (arXiv:2604.25975)
 * [Contribution] 核心创新: Phase-aware β选择（P端 ≠ D端）
 */

import type { CompressionParams, CompressionOutput, ICompressionStrategy, TaskType } from '../core/types.js';
import { clamp, round4, ensureRetentionRange } from '../core/types.js';
import { InformationBottleneck, IBLayerResult, TaskType as IBTaskType, Phase } from './InformationBottleneck.js';
import { PhaseAwareIB, PhaseAwareIBConfig, PhaseAwareIBResult } from './PhaseAwareIB.js';

// ============================================
// 常量定义
// ============================================

const P_KEY_PRECISION = 8;      // P端Key精度: FP8
const P_VALUE_PRECISION = 4;    // P端Value精度: INT4
const D_KEY_PRECISION = 16;     // D端Key精度: FP16
const D_VALUE_PRECISION = 8;    // D端Value精度: FP8

// ============================================
// PhaseAwareIBCompressor
// ============================================

/**
 * PhaseAwareIBCompressor - Phase-aware IB驱动的压缩策略
 * 
 * [核心创新] 不同于CapKV的统一β，本策略为P端和D端选择不同的β
 * - P端 β_P: 较大（激进压缩，传输优先）
 * - D端 β_D: 较小（保守压缩，质量优先）
 * 
 * 这解决了CapKV未探索的问题：P端和D端的物理约束不同
 */
export class PhaseAwareIBCompressor implements ICompressionStrategy {
  readonly name = 'PhaseAwareIBCompressor';
  readonly type = 'phase-aware-ib';
  
  private ib: InformationBottleneck;
  private pib: PhaseAwareIB;
  
  constructor() {
    this.ib = new InformationBottleneck('unknown', 32);
    this.pib = new PhaseAwareIB('unknown', 32);
  }
  
  /**
   * 计算Phase-aware IB压缩配置
   * 
   * [核心] 利用Phase-aware IB为每层计算最优的保留率和精度
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, bandwidthBytesPerMs, memoryBytes, sloLatencyMs, taskType } = params;
    
    // 转换任务类型
    const ibTaskType = taskType as IBTaskType;
    
    // 创建Phase-aware IB配置
    const config: PhaseAwareIBConfig = {
      taskType: ibTaskType,
      numLayers: totalLayers,
      bandwidthBytesPerMs,
      memoryBytes: memoryBytes || 8 * 1024 * 1024 * 1024,
      sloLatencyMs
    };
    
    // 执行Phase-aware IB优化
    const result = this.pib.optimize(config);
    
    // 转换为压缩配置
    return this.toCompressionOutput(result, totalLayers);
  }
  
  /**
   * 将PhaseAwareIBResult转换为CompressionOutput
   */
  private toCompressionOutput(result: PhaseAwareIBResult, totalLayers: number): CompressionOutput {
    const pLayerRetention = result.prefillResult.map(r => r.retentionProbability);
    const dLayerRetention = result.decodeResult.map(r => r.retentionProbability);
    
    // P端精度：基于β_P自适应选择
    const pKeyPrecision = result.prefillResult.map(r => r.keyPrecision);
    const pValuePrecision = result.prefillResult.map(r => r.valuePrecision);
    
    // D端精度：基于β_D自适应选择
    const dKeyPrecision = result.decodeResult.map(r => r.keyPrecision);
    const dValuePrecision = result.decodeResult.map(r => r.valuePrecision);
    
    // 计算平均压缩比
    const avgPRetention = pLayerRetention.reduce((a, b) => a + b, 0) / totalLayers;
    const avgPKeyPrec = pKeyPrecision.reduce((a, b) => a + b, 0) / totalLayers;
    const avgPValuePrec = pValuePrecision.reduce((a, b) => a + b, 0) / totalLayers;
    const pPrecisionRatio = (avgPKeyPrec / 16) * (avgPValuePrec / 16);
    const avgCompressionRatio = avgPRetention * pPrecisionRatio;
    
    // 带宽节省
    const estimatedBandwidthSaving = 1 - avgCompressionRatio;
    
    return {
      strategy: this.name,
      totalLayers,
      pLayerRetention: pLayerRetention.map(ensureRetentionRange),
      dLayerRetention: dLayerRetention.map(ensureRetentionRange),
      pKeyPrecision,
      pValuePrecision,
      dKeyPrecision,
      dValuePrecision,
      avgCompressionRatio: ensureRetentionRange(avgCompressionRatio),
      estimatedBandwidthSaving: ensureRetentionRange(estimatedBandwidthSaving)
    };
  }
  
  /**
   * 预估质量影响
   * 
   * Phase-aware IB的质量预估考虑：
   * - D端的保留率通常高于P端
   * - β_D < β_P 确保D端质量
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgPRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    const avgDRetention = config.dLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    
    // D端权重更高（因为D端决定生成质量）
    const quality = avgPRetention * 0.35 + avgDRetention * 0.65;
    
    // 根据任务类型调整
    let taskBonus = 1.0;
    switch (taskType) {
      case 'math': taskBonus = 0.95; break;  // 数学对精度更敏感
      case 'code': taskBonus = 0.90; break;  // 代码对语法精度要求高
      case 'qa': taskBonus = 1.0; break;
      case 'conversation': taskBonus = 1.05; break;  // 对话对质量要求稍低
    }
    
    return ensureRetentionRange(quality * taskBonus);
  }
  
  /**
   * 获取当前配置的β值
   */
  getBetaValues(taskType: IBTaskType, numLayers: number, bandwidth: number, slo?: number): {
    betaPrefill: number;
    betaDecode: number;
    phaseDiscriminationFactor: number;
  } {
    const result = this.pib.optimize({
      taskType,
      numLayers,
      bandwidthBytesPerMs: bandwidth,
      memoryBytes: 8 * 1024 * 1024 * 1024,
      sloLatencyMs: slo
    });
    
    return {
      betaPrefill: result.betaPrefill,
      betaDecode: result.betaDecode,
      phaseDiscriminationFactor: result.tradeoffs.phaseDiscriminationFactor
    };
  }
  
  /**
   * 生成对比报告
   */
  generateComparisonReport(taskType: IBTaskType, numLayers: number): string {
    const result = this.pib.compareWithCapKV(taskType, numLayers);
    
    return `
=== Phase-aware IB vs CapKV Baseline ===

任务类型: ${taskType}
层数: ${numLayers}

【Phase-aware IB】
  压缩率: ${(result.phaseAware.avgCompressionRate * 100).toFixed(1)}%
  质量: ${(result.phaseAware.avgQuality * 100).toFixed(1)}%

【CapKV Baseline】(统一β)
  压缩率: ${(result.capkv.avgCompressionRate * 100).toFixed(1)}%
  质量: ${(result.capkv.avgQuality * 100).toFixed(1)}%

【改进】
  压缩增益: +${(result.improvement.compressionGain * 100).toFixed(1)}%
  质量增益: +${(result.improvement.qualityGain * 100).toFixed(1)}%

【关键洞察】
  CapKV使用统一的β，无法同时优化P端和D端
  Phase-aware IB通过β_P ≠ β_D，在传输和质量间取得更好平衡
`;
  }
}

// ============================================
// CapKV Baseline Compressor
// ============================================

/**
 * CapKVBaselineCompressor - 简化版CapKV（用于对比）
 * 
 * [Reference] CapKV (arXiv:2604.25975)
 * 统一β，不区分P/D端
 */
export class CapKVBaselineCompressor implements ICompressionStrategy {
  readonly name = 'CapKVBaseline';
  readonly type = 'capkv';
  
  private ib: InformationBottleneck;
  private fixedBeta: number;
  
  constructor(fixedBeta: number = 1.0) {
    this.ib = new InformationBottleneck('unknown', 32);
    this.fixedBeta = fixedBeta;
  }
  
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, taskType } = params;
    
    const ibTaskType = taskType as IBTaskType;
    
    // CapKV：统一β应用于所有层
    const results = this.ib.computeOptimalRetention({
      beta: this.fixedBeta,
      phase: 'prefill',
      taskType: ibTaskType,
      numLayers: totalLayers
    });
    
    // 统一精度（CapKV风格）
    const keyPrecision = results.map(() => 8);
    const valuePrecision = results.map(() => 4);
    
    const retentions = results.map(r => r.retentionProbability);
    const avgRetention = retentions.reduce((a, b) => a + b, 0) / totalLayers;
    const avgCompressionRatio = avgRetention * 0.25; // 8/16 * 4/16
    
    return {
      strategy: this.name,
      totalLayers,
      pLayerRetention: retentions.map(ensureRetentionRange),
      dLayerRetention: retentions.map(ensureRetentionRange), // CapKV不区分P/D
      pKeyPrecision: keyPrecision,
      pValuePrecision: valuePrecision,
      dKeyPrecision: keyPrecision,
      dValuePrecision: valuePrecision,
      avgCompressionRatio: ensureRetentionRange(avgCompressionRatio),
      estimatedBandwidthSaving: ensureRetentionRange(1 - avgCompressionRatio)
    };
  }
  
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    return ensureRetentionRange(avgRetention * 0.95);
  }
}

// ============================================
// 导出
// ============================================

// 已在类定义处导出: PhaseAwareIBCompressor, CapKVBaselineCompressor
