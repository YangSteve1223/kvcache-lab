/**
 * 层预算分配器 - 基于任务感知的KV Cache资源分配
 * 参考 DynamicKV 和 PyramidKV 的思想
 * 
 * 类型定义统一从 ../core/types.ts 导入
 */

import {
  TaskType,
  PrecisionType,
  LayerBudget,
  BudgetConstraints,
  ProfileType
} from '../core/types.js';

/**
 * 生成金字塔型权重分布
 * @param layers 层数
 * @param type 分布类型
 *   - inverted: 低层权重低，高层权重高（适合数学推理、QA）
 *   - normal: 低层权重重，高层权重低（适合代码）
 *   - flat: 均匀分布
 */
export function generatePyramid(layers: number, type: ProfileType): number[] {
  const weights: number[] = new Array(layers);
  
  switch (type) {
    case 'inverted': {
      // 低层0.3 -> 高层0.9，线性插值
      for (let i = 0; i < layers; i++) {
        const t = i / (layers - 1);
        weights[i] = 0.3 + 0.6 * t;
      }
      break;
    }
    case 'normal': {
      // 低层0.9 -> 高层0.3，线性插值
      for (let i = 0; i < layers; i++) {
        const t = i / (layers - 1);
        weights[i] = 0.9 - 0.6 * t;
      }
      break;
    }
    case 'flat': {
      // 均匀分布
      for (let i = 0; i < layers; i++) {
        weights[i] = 0.5;
      }
      break;
    }
  }
  
  return weights;
}

// 任务Profile映射
export const TASK_PROFILES: Record<TaskType, number[]> = {
  // 数学推理：中高层重要（推理链在高层）
  math: generatePyramid(32, 'inverted'),
  // 代码：低层重要（语法解析）
  code: generatePyramid(32, 'normal'),
  // QA：高层重要（语义理解）
  qa: generatePyramid(32, 'inverted'),
  // 对话：均匀
  conversation: generatePyramid(32, 'flat'),
  // 未知类型：均匀
  unknown: generatePyramid(32, 'flat'),
};

/**
 * 精度分配策略
 * 基于层重要性分配不同的数值精度
 */
export function getPrecisionForLayer(
  importance: number,
  keyPrecision: number,
  valuePrecision: number
): { keyPrecision: number; valuePrecision: number } {
  // 高重要性层：Key FP16 + Value FP8
  if (importance >= 0.7) {
    return { keyPrecision: 16, valuePrecision: 8 };
  }
  // 中重要性层：Key FP8 + Value INT4
  if (importance >= 0.5) {
    return { keyPrecision: 8, valuePrecision: 4 };
  }
  // 低重要性层：Key INT4 + Value INT4
  return { keyPrecision: 4, valuePrecision: 4 };
}

/**
 * 层预算分配器类
 */
export class LayerBudgetAllocator {
  private constraints: BudgetConstraints;
  private layerWeights: number[];
  private alpha: number; // 调整因子，控制权重差异的影响程度
  
  constructor(constraints: BudgetConstraints, alpha = 0.5) {
    this.constraints = constraints;
    this.alpha = alpha;
    
    // 获取对应任务的权重分布
    const profile = TASK_PROFILES[constraints.taskType];
    
    // 如果层数不匹配，重新生成
    if (profile.length !== constraints.totalLayers) {
      const profileType = this.getProfileType(constraints.taskType);
      this.layerWeights = generatePyramid(constraints.totalLayers, profileType);
    } else {
      this.layerWeights = [...profile];
    }
  }
  
  /**
   * 获取任务对应的profile类型
   */
  private getProfileType(taskType: TaskType): ProfileType {
    switch (taskType) {
      case 'math':
      case 'qa':
        return 'inverted';
      case 'code':
        return 'normal';
      default:
        return 'flat';
    }
  }
  
  /**
   * 计算总权重
   */
  private sumWeights(): number {
    return this.layerWeights.reduce((sum, w) => sum + w, 0);
  }
  
  /**
   * 计算平均权重
   */
  private avgWeight(): number {
    return this.sumWeights() / this.layerWeights.length;
  }
  
  /**
   * 计算单层的基础KV大小（字节）
   * 假设：KV Cache = 2 * hidden_size * sequence_length * 2(bytes for FP16)
   *       hidden_size = num_heads * head_dim (假设head_dim = 64)
   */
  private calculateMaxLayerSize(): number {
    const { hiddenSize, numHeads, sequenceLength } = this.constraints;
    // 每个token的KV大小: 2 * hidden_size * 2(bytes FP16)
    const kvPerToken = 2 * hiddenSize * 2;
    // 整层的KV大小
    return kvPerToken * sequenceLength;
  }
  
  /**
   * 分配层预算
   * 核心算法：
   * 1. 计算每层基础预算
   * 2. 基于权重调整
   * 3. 归一化确保总预算不变
   */
  allocate(): LayerBudget {
    const { totalMemoryBytes, totalLayers } = this.constraints;
    const maxLayerSize = this.calculateMaxLayerSize();
    const avgWeight = this.avgWeight();
    
    // 1. 计算每层基础预算
    const baseBudget = totalMemoryBytes / totalLayers;
    
    // 2. 基于权重调整
    const adjustedBudgets: number[] = new Array(totalLayers);
    for (let i = 0; i < totalLayers; i++) {
      // adjustedBudget[l] = baseBudget * (1 + alpha * (w[l] - avg(w)))
      const adjustment = 1 + this.alpha * (this.layerWeights[i] - avgWeight);
      adjustedBudgets[i] = baseBudget * adjustment;
    }
    
    // 3. 归一化确保总和 = totalMemoryBytes
    const adjustedSum = adjustedBudgets.reduce((sum, b) => sum + b, 0);
    const normalizedBudgets: number[] = adjustedBudgets.map(
      b => (b / adjustedSum) * totalMemoryBytes
    );
    
    // 4. 计算每层保留率
    const retentionRatios: number[] = normalizedBudgets.map(
      b => Math.min(1, b / maxLayerSize)
    );
    
    // 5. 计算每层的精度
    const keyPrecisions: number[] = new Array(totalLayers);
    const valuePrecisions: number[] = new Array(totalLayers);
    
    for (let i = 0; i < totalLayers; i++) {
      const precision = getPrecisionForLayer(
        this.layerWeights[i],
        16, // 默认FP16
        16  // 默认FP16
      );
      keyPrecisions[i] = precision.keyPrecision;
      valuePrecisions[i] = precision.valuePrecision;
    }
    
    return {
      totalLayers,
      retentionRatios,
      keyPrecisions,
      valuePrecisions,
      totalBudgetBytes: totalMemoryBytes,
      perLayerBudgetBytes: normalizedBudgets,
    };
  }
  
  /**
   * 获取层权重
   */
  getLayerWeights(): number[] {
    return [...this.layerWeights];
  }
  
  /**
   * 获取任务类型
   */
  getTaskType(): TaskType {
    return this.constraints.taskType;
  }
}

// 导出便捷函数
export function allocateLayerBudget(constraints: BudgetConstraints): LayerBudget {
  const allocator = new LayerBudgetAllocator(constraints);
  return allocator.allocate();
}
