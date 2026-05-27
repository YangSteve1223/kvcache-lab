/**
 * 均匀压缩策略
 * 所有层使用统一保留率，根据显存压力动态调整
 */

import {
  CompressionParams,
  CompressionOutput,
  ensureRetentionRange,
  clamp
} from '../../core/types.js';
import { ICompressionStrategy } from '../CompressionOrchestrator.js';

// 精度常量
const FULL_PRECISION = 16;
const REDUCED_PRECISION = 8;

/**
 * UniformCompression - 均匀压缩策略
 * 根据显存压力动态计算统一保留率，Key和Value使用相同精度
 */
export class UniformCompression implements ICompressionStrategy {
  readonly name = 'UniformCompression';
  readonly type = 'uniform';

  /**
   * 计算均匀压缩配置
   * 保留率公式: retention = max(0.3, 1 - (memoryUsage - gpuMemory * 0.8) / (gpuMemory * 0.2))
   * 显存使用超过80%时开始压缩，保留率最低0.3
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, gpuMemoryBytes, currentMemoryUsage } = params;
    
    // 计算显存压力
    const memoryUsageRatio = currentMemoryUsage / gpuMemoryBytes;
    
    // 计算保留率：超过80%时开始压缩
    // 当显存使用达到100%时，保留率降至0.3（最低）
    let retention: number;
    if (memoryUsageRatio <= 0.8) {
      // 显存使用低于80%，不压缩
      retention = 1.0;
    } else {
      // 显存压力公式: 1 - (usage - 0.8) / 0.2
      retention = 1 - (memoryUsageRatio - 0.8) / 0.2;
      retention = clamp(retention, 0.3, 1.0);
    }
    
    retention = ensureRetentionRange(retention);
    
    // 初始化数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    // 所有层使用相同配置
    for (let i = 0; i < totalLayers; i++) {
      pLayerRetention.push(retention);
      dLayerRetention.push(retention);
      // Key和Value使用相同精度
      pKeyPrecision.push(REDUCED_PRECISION);
      pValuePrecision.push(REDUCED_PRECISION);
      dKeyPrecision.push(FULL_PRECISION);
      dValuePrecision.push(REDUCED_PRECISION);
    }
    
    // 计算压缩比: 保留率 * 精度压缩比
    // P端: 精度从16bit压缩到8bit = 0.5
    const precisionRatioP = REDUCED_PRECISION / FULL_PRECISION;
    const avgCompressionRatio = retention * precisionRatioP;
    
    // 带宽节省 = 1 - 压缩后大小 / 原始大小
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
   * 均匀压缩对所有任务类型影响相同
   * 保留率越高，质量影响越小
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    return ensureRetentionRange(avgRetention);
  }
}
