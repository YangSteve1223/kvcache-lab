/**
 * PD感知压缩策略 - 核心创新
 * P端(传输前)和D端(接收后)差异化压缩
 * 
 * P端策略：
 * - 低层 (Layer < L/3): 激进保留率 0.2-0.4，信息冗余度高
 * - 中层 (L/3 ≤ Layer < 2L/3): 中等保留率 0.4-0.6
 * - 高层 (Layer ≥ 2L/3): 保守保留率 0.6-0.8，语义信息关键
 * - Key精度: FP8(8bit), Value精度: INT4(4bit) — 参考LeanKV的K8V4
 * 
 * D端策略：
 * - 所有层保留率不低于0.6
 * - Key精度: FP16(16bit), Value精度: FP8(8bit) — D端对精度更敏感
 * - 近期token保留率更高（滑动窗口机制）
 */

import {
  CompressionParams,
  CompressionOutput,
  ensureRetentionRange,
  clamp
} from '../../core/types.js';
import { ICompressionStrategy } from '../CompressionOrchestrator.js';

// 精度常量
const P_KEY_PRECISION = 8;      // P端Key精度: FP8
const P_VALUE_PRECISION = 4;    // P端Value精度: INT4 (LeanKV K8V4)
const D_KEY_PRECISION = 16;     // D端Key精度: FP16
const D_VALUE_PRECISION = 8;    // D端Value精度: FP8
const FULL_PRECISION = 16;     // FP16全精度

// P端保留率范围（按层分布）
const P_LOWER_RETENTION = 0.2;    // 低层最小保留率
const P_LOWER_RETENTION_MAX = 0.4; // 低层最大保留率
const P_MIDDLE_RETENTION = 0.4;   // 中层保留率
const P_MIDDLE_RETENTION_MAX = 0.6;
const P_UPPER_RETENTION = 0.6;    // 高层最小保留率
const P_UPPER_RETENTION_MAX = 0.8;

// D端保留率范围
const D_MIN_RETENTION = 0.6;      // D端最小保留率

/**
 * PDAwareCompression - PD感知压缩策略
 * 实现P端和D端的差异化压缩配置
 */
export class PDAwareCompression implements ICompressionStrategy {
  readonly name = 'PDAwareCompression';
  readonly type = 'pd-aware';

  /**
   * 计算PD感知压缩配置
   * 根据带宽约束调整P端激进程度，根据SLO约束调整D端保守程度
   * 
   * v2: 改进带宽感知
   * - 使用实际KV传输需求计算带宽压力，而非硬编码100 bytes/ms
   * - 高带宽时不压缩（保留率=1, FP16）
   * - 低带宽时渐进压缩（保留率+精度同时调整）
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { 
      totalLayers, 
      totalTokens,
      bandwidthBytesPerMs, 
      sloLatencyMs = 1000,
    } = params;
    
    // v2: 计算实际带宽压力
    // KV总大小 = totalTokens × kvBytesPerToken × totalLayers
    // kvBytesPerToken ≈ 2(K+V) × headDim × 2bytes(FP16) = 512 bytes/token/layer
    const kvBytesPerTokenPerLayer = 2 * 128 * 2; // headDim=128, FP16
    const kvTotalBytes = totalTokens * kvBytesPerTokenPerLayer * totalLayers;
    
    // SLO内可传输的最大KV量
    const maxTransferInSLO = bandwidthBytesPerMs * sloLatencyMs;
    
    // 带宽压力 = 1 - 可传输/需传输，0=无压力，1=极端压力
    let bandwidthPressure: number;
    if (kvTotalBytes <= maxTransferInSLO) {
      bandwidthPressure = 0;  // 带宽充足，无需压缩
    } else {
      bandwidthPressure = clamp(1 - maxTransferInSLO / kvTotalBytes, 0, 1);
    }
    
    // 激进程度 = 带宽压力
    const aggressiveness = bandwidthPressure;
    
    // SLO严格程度因子: sloLatency越小，越严格
    // 假设100ms为基准SLO
    const baseSlo = 100; // ms
    const sloStrictness = clamp(1 - sloLatencyMs / baseSlo, 0, 1);
    
    // 计算三层边界
    const layerBound1 = Math.floor(totalLayers / 3);
    const layerBound2 = Math.floor((2 * totalLayers) / 3);
    
    // 初始化数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    for (let i = 0; i < totalLayers; i++) {
      // 带宽充足时，不压缩（保留率=1.0, FP16）
      let pRetention: number;
      if (aggressiveness === 0) {
        // 无带宽压力：完整传输，不压缩
        pRetention = 1.0;
      } else if (i < layerBound1) {
        // 低层: 最激进
        const base = P_LOWER_RETENTION;
        const range = P_LOWER_RETENTION_MAX - P_LOWER_RETENTION;
        pRetention = base + range * (1 - aggressiveness);
      } else if (i < layerBound2) {
        // 中层: 中等
        const base = P_MIDDLE_RETENTION;
        const range = P_MIDDLE_RETENTION_MAX - P_MIDDLE_RETENTION;
        pRetention = base + range * (1 - aggressiveness * 0.5);
      } else {
        // 高层: 保守
        const base = P_UPPER_RETENTION;
        const range = P_UPPER_RETENTION_MAX - P_UPPER_RETENTION;
        pRetention = base + range * (1 - aggressiveness * 0.3);
      }
      pLayerRetention.push(ensureRetentionRange(pRetention));
      
      // 计算D端保留率（不低于0.6 + SLO调整）
      // 近期层（后半部分）保留率更高
      const recencyBoost = i >= layerBound2 ? 0.2 : 0;
      const sloAdjustment = sloStrictness * 0.1;
      let dRetention = Math.max(D_MIN_RETENTION, pRetention + recencyBoost - sloAdjustment);
      dLayerRetention.push(ensureRetentionRange(dRetention));
      
      // P端精度: 根据带宽压力动态调整
      // 压力低: K16V16 (FP16), 压力中: K8V8, 压力高: K8V4
      let pKeyPrec: number;
      let pValuePrec: number;
      if (aggressiveness < 0.3) {
        pKeyPrec = FULL_PRECISION;   // FP16
        pValuePrec = FULL_PRECISION; // FP16
      } else if (aggressiveness < 0.7) {
        pKeyPrec = P_KEY_PRECISION;  // FP8
        pValuePrec = P_VALUE_PRECISION + 4; // INT8
      } else {
        pKeyPrec = P_KEY_PRECISION;  // FP8
        pValuePrec = P_VALUE_PRECISION; // INT4
      }
      pKeyPrecision.push(pKeyPrec);
      pValuePrecision.push(pValuePrec);
      
      // D端精度: K16V8（D端对精度更敏感）
      dKeyPrecision.push(D_KEY_PRECISION);
      dValuePrecision.push(D_VALUE_PRECISION);
    }
    
    // 计算平均压缩比
    // P端: 保留率 * 精度压缩比(8/16 * 4/16 = 0.125)
    const pPrecisionRatio = (P_KEY_PRECISION / 16) * (P_VALUE_PRECISION / 16);
    const avgRetentionP = pLayerRetention.reduce((a, b) => a + b, 0) / totalLayers;
    const avgCompressionRatio = avgRetentionP * pPrecisionRatio;
    
    // 带宽节省
    const estimatedBandwidthSaving = 1 - avgCompressionRatio;
    
    return {
      strategy: this.name,
      totalLayers,
      pLayerRetention,
      dLayerRetention,
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
   * PD感知压缩通过差异化处理保持质量
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgPRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    const avgDRetention = config.dLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    
    // P端和D端的加权质量
    const quality = avgPRetention * 0.4 + avgDRetention * 0.6;
    return ensureRetentionRange(quality);
  }
}
