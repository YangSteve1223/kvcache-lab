/**
 * 任务感知压缩策略 - 核心创新
 * 在PDAwareCompression基础上增加任务感知层预算分配
 * 
 * 核心思想：
 * - 不同任务类型对不同层的依赖程度不同
 * - 例如: 数学推理依赖中高层，代码语法依赖低层
 * - 根据任务profile调整各层保留率
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

/**
 * TaskAwareCompression - 任务感知压缩策略
 * 继承PDAwareCompression框架，增加任务感知层预算分配
 */
export class TaskAwareCompression implements ICompressionStrategy {
  readonly name = 'TaskAwareCompression';
  readonly type = 'task-aware';
  
  // 引用PD感知压缩计算基础配置
  private pdAware = new PDAwareCompression();

  /**
   * 计算任务感知压缩配置
   * 1. 先获取PD感知基础配置
   * 2. 根据任务profile调整各层保留率
   * 
   * 调整逻辑：
   * - 任务重要层根据 preferredCompression 决定保留程度
   * - conservative: 任务重要层保留更多
   * - aggressive: 非重要层更激进压缩，重要层保持
   * - balanced: 均匀调整
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, taskType } = params;
    
    // 获取任务profile
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    
    // 获取PD感知基础配置
    const pdBase = this.pdAware.computeConfig(params);
    
    // 计算层边界
    const layerBound1 = Math.floor(totalLayers / 3);
    const layerBound2 = Math.floor((2 * totalLayers) / 3);
    
    // 初始化新数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    // 计算各段的相对重要性因子（相对于平均值的倍数）
    const avgImportance = (profile.layerImportance[0] + profile.layerImportance[1] + profile.layerImportance[2]) / 3;
    const importanceFactors = profile.layerImportance.map(imp => imp / avgImportance);
    
    // 预计算各段的保留率调整基数
    // 策略调整范围
    const strategyRanges = {
      'conservative': { base: 0.5, scale: 0.5 },   // 0.5 - 1.0
      'balanced': { base: 0.4, scale: 0.6 },       // 0.4 - 1.0
      'aggressive': { base: 0.2, scale: 0.8 },     // 0.2 - 1.0
    };
    const range = strategyRanges[profile.preferredCompression];
    
    for (let i = 0; i < totalLayers; i++) {
      // 确定该层属于哪一段
      let segmentIndex: 0 | 1 | 2;
      if (i < layerBound1) {
        segmentIndex = 0;
      } else if (i < layerBound2) {
        segmentIndex = 1;
      } else {
        segmentIndex = 2;
      }
      
      const importanceFactor = importanceFactors[segmentIndex];
      const pdBaseRetention = pdBase.pLayerRetention[i];
      
      // P端调整: 根据任务重要性和压缩偏好调整保留率
      // 核心思想：使用 importanceFactor 缩放保留率范围
      // importanceFactor > 1 -> 更接近范围上限
      // importanceFactor < 1 -> 更接近范围下限
      const scaleFactor = range.base + importanceFactor * range.scale;
      let pRetention = pdBaseRetention * clamp(scaleFactor, 0.1, 1.5);
      
      // 对于激进策略，特别处理：确保低层比高层有更高的保留率
      if (profile.preferredCompression === 'aggressive') {
        // 任务重要层保持较高保留率
        // 任务非重要层降低保留率
        if (importanceFactor > 1.0) {
          // 重要层: 保持或提高
          pRetention = Math.max(pRetention, pdBaseRetention * 0.9);
        } else {
          // 非重要层: 降低
          pRetention = pdBaseRetention * 0.7;
        }
      }
      
      pLayerRetention.push(ensureRetentionRange(pRetention));
      
      // D端调整: 不低于PD基础和任务要求
      // D端对任务重要性更敏感
      const minDRetention = Math.max(pdBase.dLayerRetention[i], 0.6);
      const dRetention = minDRetention * (0.7 + importanceFactor * 0.3);
      dLayerRetention.push(ensureRetentionRange(dRetention));
      
      // P端精度: 使用profile指定的精度
      pKeyPrecision.push(profile.keyPrecision);
      pValuePrecision.push(profile.valuePrecision);
      
      // D端精度: D端对精度更敏感，使用更高精度
      dKeyPrecision.push(FULL_PRECISION);
      dValuePrecision.push(Math.max(profile.valuePrecision, HALF_PRECISION));
    }
    
    // 计算压缩比
    const pPrecisionRatio = (profile.keyPrecision / 16) * (profile.valuePrecision / 16);
    const avgRetentionP = pLayerRetention.reduce((a, b) => a + b, 0) / totalLayers;
    const avgCompressionRatio = avgRetentionP * pPrecisionRatio;
    
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
   * 任务感知压缩通过保留关键层来维持质量
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    
    // 计算加权质量
    const avgPRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    const avgDRetention = config.dLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    
    // 根据任务类型加权
    // 保守任务更依赖高层保留率
    const highLayerWeight = profile.preferredCompression === 'conservative' ? 0.7 : 0.5;
    const quality = avgPRetention * 0.3 + avgDRetention * 0.4 + highLayerWeight * (1 - avgPRetention) * 0.3;
    
    return ensureRetentionRange(quality);
  }
}

// 重新导出TASK_PROFILES以保持向后兼容
export { TASK_PROFILES };
export type { TaskProfile };
