/**
 * Semantic Distortion - 语义失真度量
 * 
 * 核心创新:
 * 不使用传统的L2误差或MSE来衡量压缩失真
 * 而是用生成质量来衡量失真:
 * 
 * d(x, x') = 1 - similarity(LLM_output(KV_original), LLM_output(KV_compressed))
 * 
 * 语义失真优势:
 * 1. 直接反映对用户的影响
 * 2. 捕捉高层语义变化
 * 3. 与任务评估指标相关
 */

import { TaskType } from '../core/types.js';
import { RateDistortion } from './RateDistortion.js';

/**
 * 语义失真配置
 */
export interface SemanticDistortionConfig {
  taskType: TaskType;
  compressionType: 'quantization' | 'pruning' | 'mixed';
  layerIndex: number;
  totalLayers: number;
  compressionRatio: number;  // 压缩比 [0-1]
}

/**
 * 语义失真估算结果
 */
export interface SemanticDistortionResult {
  distortion: number;        // 语义失真 [0-1]
  qualityScore: number;      // 质量分数 = 1 - distortion
  l2Distortion?: number;     // 可选：L2失真（用于对比）
  semanticGap: number;       // 语义差距（质量下降幅度）
}

/**
 * 语义R-D曲线点
 */
export interface SemanticRDPoint {
  rate: number;              // 传输速率 (bits/token)
  distortion: number;        // 语义失真 [0-1]
  quality: number;            // 语义质量 [0-1]
  l2Distortion: number;      // L2失真（对比用）
  compressionRatio: number;  // 压缩比
}

/**
 * 任务类型的语义敏感性参数
 */
interface TaskSemanticProfile {
  // 对精度损失的敏感度
  precisionSensitivity: number;
  // 对剪枝损失的敏感度
  pruningSensitivity: number;
  // 各层位置敏感度 [低层, 中层, 高层]
  layerSensitivity: [number, number, number];
  // 基础质量
  baseQuality: number;
}

/**
 * 任务语义Profile库
 */
const SEMANTIC_PROFILES: Record<TaskType, TaskSemanticProfile> = {
  math: {
    precisionSensitivity: 0.15,  // 对量化敏感
    pruningSensitivity: 0.2,     // 对剪枝较敏感
    layerSensitivity: [0.3, 0.6, 0.9],  // 高层更敏感
    baseQuality: 0.95
  },
  code: {
    precisionSensitivity: 0.05,  // 对量化不敏感
    pruningSensitivity: 0.1,    // 对剪枝较不敏感
    layerSensitivity: [0.8, 0.5, 0.3],  // 低层更敏感
    baseQuality: 0.92
  },
  qa: {
    precisionSensitivity: 0.12,
    pruningSensitivity: 0.18,
    layerSensitivity: [0.4, 0.5, 0.8],  // 高层更敏感
    baseQuality: 0.93
  },
  conversation: {
    precisionSensitivity: 0.08,
    pruningSensitivity: 0.12,
    layerSensitivity: [0.5, 0.5, 0.5],  // 均匀敏感
    baseQuality: 0.90
  },
  unknown: {
    precisionSensitivity: 0.1,
    pruningSensitivity: 0.15,
    layerSensitivity: [0.4, 0.5, 0.6],
    baseQuality: 0.90
  }
};

/**
 * SemanticDistortion - 语义失真度量器
 */
export class SemanticDistortion {
  /**
   * 估算语义失真
   * 
   * 基于压缩比、层位置和任务类型估算压缩对生成质量的影响
   * 
   * @param compressionRatio 压缩比 [0-1], 1表示无压缩
   * @param layerIndex 层索引
   * @param numLayers 总层数
   * @param taskType 任务类型
   * @param compressionType 压缩类型
   * @returns 语义失真估算结果
   */
  estimateSemanticDistortion(
    compressionRatio: number,
    layerIndex: number,
    numLayers: number,
    taskType: TaskType,
    compressionType: 'quantization' | 'pruning' | 'mixed' = 'mixed'
  ): SemanticDistortionResult {
    const profile = SEMANTIC_PROFILES[taskType] || SEMANTIC_PROFILES.unknown;
    
    // 计算层位置 (0=低层, 1=高层)
    const layerPosition = layerIndex / numLayers;
    
    // 计算位置敏感度
    let layerSensitivity: number;
    if (layerPosition < 0.33) {
      layerSensitivity = profile.layerSensitivity[0];
    } else if (layerPosition < 0.66) {
      layerSensitivity = profile.layerSensitivity[1];
    } else {
      layerSensitivity = profile.layerSensitivity[2];
    }
    
    // 计算压缩损失
    const compressionLoss = 1 - compressionRatio;
    
    // 选择敏感度
    let sensitivity: number;
    switch (compressionType) {
      case 'quantization':
        sensitivity = profile.precisionSensitivity;
        break;
      case 'pruning':
        sensitivity = profile.pruningSensitivity;
        break;
      case 'mixed':
      default:
        // 混合压缩的敏感度是加权和
        sensitivity = (profile.precisionSensitivity + profile.pruningSensitivity) / 2;
    }
    
    // 语义失真 = 压缩损失 × 敏感度 × 位置敏感度
    const distortion = compressionLoss * sensitivity * layerSensitivity;
    
    // 限制在[0, 1]范围内
    const clampedDistortion = Math.min(1, Math.max(0, distortion));
    
    // 质量分数
    const qualityScore = Math.max(0, 1 - clampedDistortion);
    
    // 语义差距（相对于无压缩状态的下降）
    const semanticGap = profile.baseQuality - qualityScore;
    
    // 估算L2失真（用于对比）
    const l2Distortion = compressionLoss * compressionLoss * 0.5;
    
    return {
      distortion: clampedDistortion,
      qualityScore,
      l2Distortion: Math.min(1, l2Distortion),
      semanticGap
    };
  }
  
  /**
   * 构建语义R-D曲线
   * 
   * 与传统R-D曲线不同，这里的distortion是语义失真而非L2误差
   * 
   * @param config R-D配置
   * @param ratePoints 速率点数组
   * @returns 语义R-D曲线点
   */
  buildSemanticRDCurve(
    config: {
      numLayers: number;
      taskType: TaskType;
    },
    ratePoints: number[]
  ): SemanticRDPoint[] {
    const { numLayers, taskType } = config;
    const profile = SEMANTIC_PROFILES[taskType] || SEMANTIC_PROFILES.unknown;
    const points: SemanticRDPoint[] = [];
    
    for (const rate of ratePoints) {
      // 根据速率估算平均压缩比
      // 假设速率16对应compressionRatio=1.0, 速率4对应compressionRatio=0.25
      const compressionRatio = Math.max(0.1, Math.min(1, rate / 16));
      
      // 计算总语义失真
      let totalSemanticDistortion = 0;
      let totalL2Distortion = 0;
      
      for (let i = 0; i < numLayers; i++) {
        const layerDistortion = this.estimateSemanticDistortion(
          compressionRatio,
          i,
          numLayers,
          taskType,
          'mixed'
        );
        
        // 加权平均（各层权重相等）
        totalSemanticDistortion += layerDistortion.distortion;
        totalL2Distortion += layerDistortion.l2Distortion || 0;
      }
      
      const avgSemanticDistortion = totalSemanticDistortion / numLayers;
      const avgL2Distortion = totalL2Distortion / numLayers;
      
      points.push({
        rate,
        distortion: avgSemanticDistortion,
        quality: 1 - avgSemanticDistortion,
        l2Distortion: avgL2Distortion,
        compressionRatio
      });
    }
    
    return points;
  }
  
  /**
   * 对比语义失真和L2失真
   * 
   * 说明为什么语义失真更适合评估压缩质量
   * 
   * @param config 配置
   * @returns 对比结果
   */
  compareDistortionMetrics(
    config: {
      compressionRatio: number;
      layerIndex: number;
      numLayers: number;
      taskType: TaskType;
    }
  ): {
    semanticDistortion: number;
    l2Distortion: number;
    difference: number;
    interpretation: string;
  } {
    const semanticResult = this.estimateSemanticDistortion(
      config.compressionRatio,
      config.layerIndex,
      config.numLayers,
      config.taskType,
      'mixed'
    );
    
    // L2失真直接用平方差
    const compressionLoss = 1 - config.compressionRatio;
    const l2Distortion = compressionLoss * compressionLoss;
    
    const difference = Math.abs(semanticResult.distortion - l2Distortion);
    
    let interpretation: string;
    if (semanticResult.distortion > l2Distortion) {
      interpretation = '语义失真大于L2失真，说明压缩对语义的影响超过对数值精度的影响';
    } else if (semanticResult.distortion < l2Distortion) {
      interpretation = '语义失真小于L2失真，说明数值精度损失比语义损失更严重';
    } else {
      interpretation = '两种失真度量相等';
    }
    
    return {
      semanticDistortion: semanticResult.distortion,
      l2Distortion,
      difference,
      interpretation
    };
  }
  
  /**
   * 计算语义R-D效率
   * 
   * 语义R-D效率 = (质量改善) / (速率增加)
   * 用于评估不同压缩策略的效率
   * 
   * @param basePoint 基准点
   * @param newPoint 新点
   * @returns 效率分数
   */
  computeSemanticRDEfficiency(
    basePoint: SemanticRDPoint,
    newPoint: SemanticRDPoint
  ): number {
    const qualityGain = newPoint.quality - basePoint.quality;
    const rateIncrease = newPoint.rate - basePoint.rate;
    
    if (rateIncrease === 0) return 0;
    
    return qualityGain / rateIncrease;
  }
  
  /**
   * 生成任务特定的语义敏感度图
   * 
   * @param taskType 任务类型
   * @param numLayers 层数
   * @returns 每层的敏感度值
   */
  generateLayerSensitivityMap(
    taskType: TaskType,
    numLayers: number
  ): number[] {
    const profile = SEMANTIC_PROFILES[taskType] || SEMANTIC_PROFILES.unknown;
    const sensitivities: number[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      const position = i / numLayers;
      
      let sensitivity: number;
      if (position < 0.33) {
        sensitivity = profile.layerSensitivity[0];
      } else if (position < 0.66) {
        sensitivity = profile.layerSensitivity[1];
      } else {
        sensitivity = profile.layerSensitivity[2];
      }
      
      sensitivities.push(sensitivity);
    }
    
    return sensitivities;
  }
  
  /**
   * 评估混合压缩的语义影响
   * 
   * @param quantizationRatio 量化压缩比 [0-1]
   * @param pruningRatio 剪枝压缩比 [0-1]
   * @param taskType 任务类型
   * @returns 混合压缩的语义失真
   */
  evaluateMixedCompression(
    quantizationRatio: number,
    pruningRatio: number,
    taskType: TaskType
  ): SemanticDistortionResult {
    const profile = SEMANTIC_PROFILES[taskType] || SEMANTIC_PROFILES.unknown;
    
    // 量化失真
    const quantDistortion = (1 - quantizationRatio) * profile.precisionSensitivity;
    
    // 剪枝失真
    const pruneDistortion = (1 - pruningRatio) * profile.pruningSensitivity;
    
    // 混合失真（非线性组合，假设存在交互效应）
    const interactionEffect = quantDistortion * pruneDistortion * 0.5;
    const totalDistortion = quantDistortion + pruneDistortion + interactionEffect;
    
    // 限制在[0, 1]范围内
    const clampedDistortion = Math.min(1, Math.max(0, totalDistortion));
    
    return {
      distortion: clampedDistortion,
      qualityScore: 1 - clampedDistortion,
      semanticGap: profile.baseQuality - (1 - clampedDistortion)
    };
  }
  
  /**
   * 生成语义失真容忍曲线
   * 
   * 给定一个可接受的语义失真范围，计算所需的最小速率
   * 
   * @param maxAcceptableDistortion 最大可接受失真 [0-1]
   * @param taskType 任务类型
   * @param numLayers 层数
   * @returns 各层的最小速率要求
   */
  generateDistortionToleranceCurve(
    maxAcceptableDistortion: number,
    taskType: TaskType,
    numLayers: number
  ): { layerIndex: number; minRate: number; importance: number }[] {
    const profile = SEMANTIC_PROFILES[taskType] || SEMANTIC_PROFILES.unknown;
    const results: { layerIndex: number; minRate: number; importance: number }[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      const position = i / numLayers;
      
      // 层敏感度
      let layerSensitivity: number;
      if (position < 0.33) {
        layerSensitivity = profile.layerSensitivity[0];
      } else if (position < 0.66) {
        layerSensitivity = profile.layerSensitivity[1];
      } else {
        layerSensitivity = profile.layerSensitivity[2];
      }
      
      // 平均敏感度
      const avgSensitivity = (profile.precisionSensitivity + profile.pruningSensitivity) / 2;
      const combinedSensitivity = avgSensitivity * layerSensitivity;
      
      // 最小压缩比 = 1 - (maxDistortion / (sensitivity * combinedSensitivity))
      const minCompressionRatio = Math.max(0.1, 
        1 - (maxAcceptableDistortion / combinedSensitivity)
      );
      
      // 最小速率 (假设速率16对应compressionRatio=1)
      const minRate = minCompressionRatio * 16;
      
      // 重要性（用于排序）
      const importance = RateDistortion.computeMutualInformation(i, numLayers, taskType);
      
      results.push({
        layerIndex: i,
        minRate: Math.max(0, minRate),
        importance
      });
    }
    
    return results.sort((a, b) => b.importance - a.importance);
  }
}
