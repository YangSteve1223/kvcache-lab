/**
 * PD-Task-Aware联合压缩策略
 * 
 * 继承PDAware和TaskAware的优点：
 * - P端基础策略来自PD-Aware（低层激进、高层保守）
 * - 在P端基础上叠加Task-Aware的层重要性权重调整
 * - D端基础策略来自PD-Aware（保守保留）
 * - 在D端基础上叠加Task-Aware的精度分配
 * - 新增：带宽自适应调整
 */

import {
  CompressionParams,
  CompressionOutput,
  ensureRetentionRange,
  clamp
} from '../../core/types.js';
import { ICompressionStrategy, TASK_PROFILES, TaskProfile } from '../CompressionOrchestrator.js';
import { PDAwareCompression } from './PDAwareCompression.js';

// 精度常量
const FULL_PRECISION = 16;
const HALF_PRECISION = 8;
const QUARTER_PRECISION = 4;

/**
 * PDTaskAwareCompression - PD-Task-Aware联合压缩策略
 * 结合PD感知和任务感知的优点，实现更优的压缩配置
 */
export class PDTaskAwareCompression implements ICompressionStrategy {
  readonly name = 'PDTaskAwareCompression';
  readonly type = 'pd-aware-task-aware';
  
  // 引用PD感知压缩计算基础配置
  private pdAware = new PDAwareCompression();
  
  // 任务调整强度参数
  private readonly ALPHA = 0.3;
  // 带宽自适应参数
  private readonly BANDWIDTH_ALPHA = 0.2;
  // 默认所需带宽 (bytes/ms)
  private readonly REQUIRED_BANDWIDTH = 100;

  /**
   * 计算层权重
   * 根据任务类型返回每层的相对重要性权重 [0.2, 1.0]
   */
  private getTaskLayerWeights(totalLayers: number, taskType: string): number[] {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    const { layerImportance } = profile;
    
    // 计算层边界
    const layerBound1 = Math.floor(totalLayers / 3);
    const layerBound2 = Math.floor((2 * totalLayers) / 3);
    
    const weights: number[] = [];
    for (let i = 0; i < totalLayers; i++) {
      let segmentIndex: 0 | 1 | 2;
      if (i < layerBound1) {
        segmentIndex = 0;
      } else if (i < layerBound2) {
        segmentIndex = 1;
      } else {
        segmentIndex = 2;
      }
      
      // 将[0.3-0.9]映射到[0.2-1.0]
      const baseWeight = layerImportance[segmentIndex];
      const weight = 0.2 + (baseWeight - 0.3) * (0.8 / 0.6);
      weights.push(clamp(weight, 0.2, 1.0));
    }
    
    return weights;
  }

  /**
   * 计算压缩配置
   * 
   * Step 1: 获取PD-Aware基础配置
   * Step 2: 获取Task-Aware的层权重
   * Step 3: P端叠加任务调整
   * Step 4: D端叠加任务精度调整
   * Step 5: 自适应带宽约束
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { 
      totalLayers, 
      bandwidthBytesPerMs,
      sloLatencyMs = 1000
    } = params;
    
    // Step 1: 获取PD-Aware基础配置
    const pdBase = this.pdAware.computeConfig(params);
    
    // Step 2: 获取Task-Aware的层权重
    const taskWeights = this.getTaskLayerWeights(totalLayers, params.taskType);
    
    // Step 5: 自适应带宽约束
    // 如果带宽低，P端更激进（alpha增大）
    const bandwidthFactor = clamp(1 - bandwidthBytesPerMs / this.REQUIRED_BANDWIDTH, 0, 1);
    const adjustedAlpha = this.ALPHA + bandwidthFactor * this.BANDWIDTH_ALPHA;
    
    // Step 3: P端叠加任务调整
    // 保留率 = pdBase * (1 + alpha * (taskWeight - 0.5))
    // 任务重要层（weight>0.5）保留更多，不重要层（weight<0.5）保留更少
    const pLayerRetention: number[] = pdBase.pLayerRetention.map((r, l) => {
      const taskWeight = taskWeights[l];
      // 相对于中心值0.5的偏差，乘以alpha调整强度
      const adjustment = 1 + adjustedAlpha * (taskWeight - 0.5);
      const adjusted = r * adjustment;
      return ensureRetentionRange(adjusted);
    });
    
    // 获取任务profile用于精度调整
    const profile: TaskProfile = TASK_PROFILES[params.taskType] || TASK_PROFILES['conversation'];
    
    // Step 4: D端叠加任务精度调整
    // D端精度分配基于任务重要性：
    // - 高重要性层（weight > 0.7）: Key FP16 + Value FP8
    // - 中重要性层（0.4 < weight <= 0.7）: Key FP8 + Value INT4
    // - 低重要性层（weight <= 0.4）: Key INT4 + Value INT4
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    for (let l = 0; l < totalLayers; l++) {
      const taskWeight = taskWeights[l];
      
      let keyPrec: number;
      let valuePrec: number;
      
      if (taskWeight > 0.7) {
        // 高重要性层：最高精度
        keyPrec = FULL_PRECISION;
        valuePrec = HALF_PRECISION;
      } else if (taskWeight > 0.4) {
        // 中重要性层：中等精度
        keyPrec = HALF_PRECISION;
        valuePrec = QUARTER_PRECISION;
      } else {
        // 低重要性层：最低精度
        keyPrec = QUARTER_PRECISION;
        valuePrec = QUARTER_PRECISION;
      }
      
      dKeyPrecision.push(keyPrec);
      dValuePrecision.push(valuePrec);
    }
    
    // D端保留率在PD基础上有小幅任务调整
    const dLayerRetention: number[] = pdBase.dLayerRetention.map((r, l) => {
      const taskWeight = taskWeights[l];
      // SLO约束调整：如果SLO紧，D端更保守
      const sloFactor = clamp(1 - sloLatencyMs / 100, 0, 0.2);
      const adjustment = 1 + adjustedAlpha * 0.3 * (taskWeight - 0.5) - sloFactor;
      const adjusted = r * adjustment;
      return ensureRetentionRange(adjusted);
    });
    
    // P端精度：使用任务profile指定的精度（已在PD中定义为K8V4）
    const pKeyPrecision = pdBase.pKeyPrecision;
    const pValuePrecision = pdBase.pValuePrecision;
    
    // 计算平均压缩比
    // P端: 保留率 * 精度压缩比
    const pPrecisionRatio = pKeyPrecision.reduce((sum, p) => sum + p, 0) / (totalLayers * FULL_PRECISION);
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
   * 联合策略通过P/D差异化和任务感知来维持质量
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    
    // 计算各层质量加权
    const avgPRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    const avgDRetention = config.dLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    
    // D端精度加权（相对于FP16）
    const avgDKeyPrecision = config.dKeyPrecision.reduce((a, b) => a + b, 0) / config.totalLayers;
    const avgDValuePrecision = config.dValuePrecision.reduce((a, b) => a + b, 0) / config.totalLayers;
    const dPrecisionQuality = (avgDKeyPrecision / FULL_PRECISION + avgDValuePrecision / FULL_PRECISION) / 2;
    
    // 综合质量评估
    // P端保留贡献40%，D端保留贡献40%，D端精度贡献20%
    const quality = avgPRetention * 0.4 + avgDRetention * 0.4 + dPrecisionQuality * 0.2;
    
    return ensureRetentionRange(quality);
  }
}
