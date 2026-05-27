/**
 * IB-RD统一压缩策略 ⭐
 * 
 * 集成了：
 * - IB的重要度度量（I(Z;Y)）
 * - R-D的带宽最优分配
 * - Phase-aware差异化（P端激进，D端保守）
 * 
 * 差异化说明：
 * - RDKV (arXiv:2605.08317): 单节点bit allocation，不考虑传输
 * - 本框架: PD分离场景的传输最优分配 + Unequal Error Protection
 * - Semantic Distortion: RDKV用attention失真，本框架用生成质量
 */

import {
  CompressionParams,
  CompressionOutput,
  ICompressionStrategy,
  ensureRetentionRange,
  clamp
} from '../core/types.js';
import {
  SemanticRDFramework,
  semanticRDFramework,
  SemanticRDConfig,
  SemanticRDResult
} from './SemanticRDFramework.js';

/**
 * IB-RD统一压缩策略
 */
export class IBRDCompressor implements ICompressionStrategy {
  readonly name = 'IB-RD-Compressor';
  readonly type = 'ib-rd-unified';
  
  private framework: SemanticRDFramework;
  private cache: Map<string, SemanticRDResult> = new Map();
  private readonly CACHE_TTL = 60000;

  constructor() {
    this.framework = semanticRDFramework;
  }

  private getCacheKey(params: CompressionParams): string {
    return `${params.taskType}-${params.totalLayers}-${params.bandwidthBytesPerMs}-${params.sloLatencyMs || 'default'}`;
  }

  computeConfig(params: CompressionParams): CompressionOutput {
    const { bandwidthBytesPerMs, sloLatencyMs = 1000 } = params;
    
    const cacheKey = this.getCacheKey(params);
    const cached = this.cache.get(cacheKey);
    if (cached && (Date.now() - (cached as any)._cacheTime) < this.CACHE_TTL) {
      return cached.prefillOutput;
    }
    
    const phase = (sloLatencyMs < 500 || bandwidthBytesPerMs < 50) ? 'prefill' : 'decode';
    
    const result = this.framework.optimize({
      bandwidthBytesPerMs,
      maxDistortion: this.estimateMaxDistortion(bandwidthBytesPerMs, sloLatencyMs),
      taskType: taskType as any,
      numLayers: params.totalLayers,
      sequenceLength: params.totalTokens,
      phase,
      beta: this.framework['deriveOptimalBeta'](bandwidthBytesPerMs, 0.4, params.taskType)
    } as SemanticRDConfig);
    
    (result as any)._cacheTime = Date.now();
    this.cache.set(cacheKey, result);
    
    return phase === 'prefill' ? result.prefillOutput : result.decodeOutput;
  }

  private estimateMaxDistortion(bandwidthBytesPerMs: number, sloLatencyMs: number): number {
    let distortion = 0.3;
    if (bandwidthBytesPerMs < 30) distortion += 0.2;
    else if (bandwidthBytesPerMs < 60) distortion += 0.1;
    else if (bandwidthBytesPerMs > 150) distortion -= 0.1;
    if (sloLatencyMs < 300) distortion += 0.15;
    else if (sloLatencyMs > 2000) distortion -= 0.1;
    return clamp(distortion, 0.1, 0.6);
  }

  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const { totalLayers, pLayerRetention, dLayerRetention } = config;
    
    const avgPRetention = pLayerRetention.reduce((a, b) => a + b, 0) / totalLayers;
    const avgDRetention = dLayerRetention.reduce((a, b) => a + b, 0) / totalLayers;
    const avgKeyPrecision = config.dKeyPrecision.reduce((a, b) => a + b, 0) / totalLayers;
    const avgValuePrecision = config.dValuePrecision.reduce((a, b) => a + b, 0) / totalLayers;
    const precisionLoss = 1 - (avgKeyPrecision / 16 + avgValuePrecision / 16) / 2;
    
    const profile = this.getTaskProfile(taskType);
    const ibWeight = Math.max(...profile.layerImportance);
    
    const quality = avgPRetention * 0.40 + avgDRetention * 0.35 + (1 - precisionLoss) * 0.15 + ibWeight * 0.10;
    return ensureRetentionRange(clamp(quality, 0, 1));
  }

  private getTaskProfile(taskType: string): any {
    const profiles: Record<string, any> = {
      math: { layerImportance: [0.3, 0.8, 0.9] },
      code: { layerImportance: [0.9, 0.5, 0.3] },
      qa: { layerImportance: [0.4, 0.5, 0.8] },
      conversation: { layerImportance: [0.5, 0.5, 0.5] }
    };
    return profiles[taskType] || profiles.conversation;
  }

  getPhaseAwareConfig(params: CompressionParams): { prefill: CompressionOutput; decode: CompressionOutput } {
    const result = this.framework.phaseAwareOptimize(
      params.taskType as any,
      params.totalLayers,
      params.bandwidthBytesPerMs,
      params.sloLatencyMs
    );
    return { prefill: result.prefill.prefillOutput, decode: result.decode.decodeOutput };
  }

  clearCache(): void { this.cache.clear(); }
}

// 导出单例
export const ibrdCompressor = new IBRDCompressor();
