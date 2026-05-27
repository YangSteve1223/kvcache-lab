/**
 * 无压缩策略 - Baseline对照组
 * 不进行任何压缩，所有层保留率1.0，精度16bit
 */

import {
  CompressionParams,
  CompressionOutput,
  ensureRetentionRange
} from '../../core/types.js';
import { ICompressionStrategy } from '../CompressionOrchestrator.js';

// 精度常量
const FULL_PRECISION = 16;

/**
 * NoneCompression - 无压缩策略
 * 用于作为baseline对照组，验证压缩效果
 */
export class NoneCompression implements ICompressionStrategy {
  readonly name = 'NoneCompression';
  readonly type = 'none';

  /**
   * 计算无压缩配置
   * 所有层完全保留，精度为FP16
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers } = params;
    
    // 初始化数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    // 所有层保留率1.0，精度16bit
    for (let i = 0; i < totalLayers; i++) {
      pLayerRetention.push(ensureRetentionRange(1.0));
      dLayerRetention.push(ensureRetentionRange(1.0));
      pKeyPrecision.push(FULL_PRECISION);
      pValuePrecision.push(FULL_PRECISION);
      dKeyPrecision.push(FULL_PRECISION);
      dValuePrecision.push(FULL_PRECISION);
    }
    
    // 无压缩，压缩比为1.0（表示无压缩）
    const avgCompressionRatio = 1.0;
    // 带宽节省为0（无压缩）
    const estimatedBandwidthSaving = 0;
    
    return {
      strategy: this.name,
      totalLayers,
      pLayerRetention,
      dLayerRetention,
      pKeyPrecision,
      pValuePrecision,
      dKeyPrecision,
      dValuePrecision,
      avgCompressionRatio,
      estimatedBandwidthSaving
    };
  }

  /**
   * 预估质量影响
   * 无压缩，质量影响为1.0（最高）
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    return 1.0;
  }
}
