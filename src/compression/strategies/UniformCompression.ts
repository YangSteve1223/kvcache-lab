/**
 * 均匀压缩策略 v2
 * 修复：增加带宽感知
 * 
 * v1只看显存压力，PD分离场景下带宽是核心约束
 * v2同时考虑带宽约束和显存压力，取更严格的保留率
 */

import {
  CompressionParams,
  CompressionOutput,
  ensureRetentionRange,
  clamp
} from '../../core/types.js';
import { ICompressionStrategy } from '../CompressionOrchestrator.js';

const FULL_PRECISION = 16;
const REDUCED_PRECISION = 8;

export class UniformCompression implements ICompressionStrategy {
  readonly name = 'UniformCompression';
  readonly type = 'uniform';

  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, gpuMemoryBytes, currentMemoryUsage, bandwidthBytesPerMs, sloLatencyMs = 1000 } = params;
    
    // 1. 显存压力维度
    const memoryUsageRatio = currentMemoryUsage / gpuMemoryBytes;
    let memoryRetention: number;
    if (memoryUsageRatio <= 0.8) {
      memoryRetention = 1.0;
    } else {
      memoryRetention = 1 - (memoryUsageRatio - 0.8) / 0.2;
      memoryRetention = clamp(memoryRetention, 0.3, 1.0);
    }
    
    // 2. 带宽压力维度
    // KV总大小 = totalTokens × kvBytesPerTokenPerLayer × totalLayers
    // kvBytesPerTokenPerLayer ≈ 2(K+V) × headDim × 2bytes(FP16) = 512 bytes/token/layer
    const kvBytesPerTokenPerLayer = 2 * 128 * 2; // headDim=128, FP16
    const totalTokens = params.totalTokens;
    const kvTransferNeeded = totalTokens * kvBytesPerTokenPerLayer * totalLayers;
    const maxTransferInSLO = bandwidthBytesPerMs * sloLatencyMs;
    
    let bandwidthRetention: number;
    if (kvTransferNeeded <= maxTransferInSLO) {
      bandwidthRetention = 1.0;  // 带宽足够，不需要压缩
    } else {
      bandwidthRetention = clamp(maxTransferInSLO / kvTransferNeeded, 0.2, 1.0);
    }
    
    // 3. 取两个维度中更严格的保留率
    const retention = ensureRetentionRange(Math.min(memoryRetention, bandwidthRetention));
    
    // 初始化数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    for (let i = 0; i < totalLayers; i++) {
      pLayerRetention.push(retention);
      dLayerRetention.push(retention);
      pKeyPrecision.push(REDUCED_PRECISION);
      pValuePrecision.push(REDUCED_PRECISION);
      dKeyPrecision.push(FULL_PRECISION);
      dValuePrecision.push(REDUCED_PRECISION);
    }
    
    const precisionRatioP = REDUCED_PRECISION / FULL_PRECISION;
    const avgCompressionRatio = retention * precisionRatioP;
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

  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    return ensureRetentionRange(avgRetention);
  }
}
