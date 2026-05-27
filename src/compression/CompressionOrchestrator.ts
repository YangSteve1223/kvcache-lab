/**
 * 插件式压缩编排器
 * 支持注册、选择、对比压缩策略
 * 
 * 类型定义统一从 ../core/types.ts 导入
 */

import {
  CompressionParams,
  CompressionOutput,
  ICompressionStrategy,
  clamp,
  round4,
  ensureRetentionRange
} from '../core/types.js';

// 重新导出ICompressionStrategy以供策略文件使用
export type { ICompressionStrategy };

/**
 * 任务Profile定义
 * 基于实验数据定义各任务类型的层重要性分布
 */
export interface TaskProfile {
  // 各层重要性权重 [低层重要性, 中层重要性, 高层重要性]
  layerImportance: [number, number, number];
  // 各层敏感性
  headSensitivity: number;
  middleSensitivity: number;
  // 压缩偏好: 'conservative' | 'balanced' | 'aggressive'
  preferredCompression: 'conservative' | 'balanced' | 'aggressive';
  // 精度偏好
  keyPrecision: number;
  valuePrecision: number;
}

/**
 * 任务Profile库
 * 基于实验2数据定义各任务类型的特征
 */
export const TASK_PROFILES: Record<string, TaskProfile> = {
  // 数学推理任务: 依赖中高层（推理链关键）
  math: {
    layerImportance: [0.3, 0.8, 0.9],
    headSensitivity: 0.037,
    middleSensitivity: 0.037,
    preferredCompression: 'conservative',
    keyPrecision: 16,
    valuePrecision: 8,
  },
  
  // 代码任务: 依赖低层（代码语法在低层）
  code: {
    layerImportance: [0.9, 0.5, 0.3],
    headSensitivity: 0.0,
    middleSensitivity: 0.0,
    preferredCompression: 'aggressive',
    keyPrecision: 8,
    valuePrecision: 4, // INT4
  },
  
  // 问答任务: 依赖高层（高层语义）
  qa: {
    layerImportance: [0.4, 0.5, 0.8],
    headSensitivity: 0.042,
    middleSensitivity: 0.0,
    preferredCompression: 'balanced',
    keyPrecision: 16,
    valuePrecision: 8,
  },
  
  // 对话任务: 均匀分布
  conversation: {
    layerImportance: [0.5, 0.5, 0.5],
    headSensitivity: 0.02,
    middleSensitivity: 0.01,
    preferredCompression: 'balanced',
    keyPrecision: 8,
    valuePrecision: 8,
  },
};

/**
 * CompressionOrchestrator - 压缩编排器
 * 支持注册、选择、对比压缩策略
 */
export class CompressionOrchestrator {
  private strategies: Map<string, ICompressionStrategy> = new Map();

  /**
   * 注册压缩策略
   */
  registerStrategy(strategy: ICompressionStrategy): void {
    this.strategies.set(strategy.name, strategy);
  }

  /**
   * 根据参数自动选择最优策略
   * 策略选择逻辑:
   * - 无压缩需求(bandwidth充足 + 显存充足) -> NoneCompression
   * - 有任务类型 + PD-Task联合可用 -> PDTaskAwareCompression（联合策略最优）
   * - 有任务类型 -> TaskAwareCompression
   * - 有压缩需求 -> PDAwareCompression
   * - 默认 -> UniformCompression
   */
  selectStrategy(params: CompressionParams): ICompressionStrategy {
    const { bandwidthBytesPerMs, gpuMemoryBytes, currentMemoryUsage, taskType } = params;
    
    // 计算资源压力
    const memoryUsageRatio = currentMemoryUsage / gpuMemoryBytes;
    
    // 无压缩需求：带宽充足(>150) 且 显存充足(<80%)
    if (bandwidthBytesPerMs > 150 && memoryUsageRatio < 0.8) {
      const noneStrategy = this.strategies.get('NoneCompression');
      if (noneStrategy) return noneStrategy;
    }
    
    // 有任务类型，优先选择联合策略（PD-Task-Aware）
    if (taskType && this.strategies.has('PDTaskAwareCompression')) {
      return this.strategies.get('PDTaskAwareCompression')!;
    }
    
    // 有任务类型，选择任务感知策略
    if (taskType && this.strategies.has('TaskAwareCompression')) {
      return this.strategies.get('TaskAwareCompression')!;
    }
    
    // 有压缩需求，选择PD感知策略
    if (this.strategies.has('PDAwareCompression')) {
      return this.strategies.get('PDAwareCompression')!;
    }
    
    // 默认选择均匀压缩
    const uniformStrategy = this.strategies.get('UniformCompression');
    if (uniformStrategy) return uniformStrategy;
    
    // 如果没有任何注册的策略，抛出错误
    const firstStrategy = this.strategies.values().next().value;
    if (!firstStrategy) {
      throw new Error('No compression strategy registered');
    }
    return firstStrategy;
  }

  /**
   * 计算最优压缩配置
   * 使用选中策略计算压缩配置
   */
  computeOptimalConfig(params: CompressionParams): CompressionOutput {
    const selectedStrategy = this.selectStrategy(params);
    return selectedStrategy.computeConfig(params);
  }

  /**
   * 对比所有已注册策略
   * 返回各策略的配置对比
   */
  compareStrategies(params: CompressionParams): Map<string, CompressionOutput> {
    const comparisons = new Map<string, CompressionOutput>();
    
    for (const [name, strategy] of this.strategies) {
      try {
        const config = strategy.computeConfig(params);
        comparisons.set(name, config);
      } catch (error) {
        console.warn(`Strategy ${name} failed: ${error}`);
      }
    }
    
    return comparisons;
  }

  /**
   * 获取已注册策略列表
   */
  getRegisteredStrategies(): string[] {
    return Array.from(this.strategies.keys());
  }
}

// 导出类型和工具函数供外部使用
export {
  clamp,
  round4,
  ensureRetentionRange
};
