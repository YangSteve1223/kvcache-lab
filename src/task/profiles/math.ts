/**
 * 数学任务Profile - 基于DynamicKV论文发现
 * 
 * 论文发现：
 * - 数学推理在Layer 16-28（高层）attention集中
 * - 头部token（公式定义）重要
 * - 中间token（推导步骤）重要
 * - 尾部token相对不重要
 */

// 32层模型的重要性权重
// 分布：低层0.3 -> 高层0.9，线性递增
export const MATH_LAYER_WEIGHTS: number[] = [
  // Layer 0-7: 低层（词汇、基础语法）- 低权重
  0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44,
  // Layer 8-15: 中低层（句法结构）- 中低权重
  0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.72, 0.76,
  // Layer 16-23: 中高层（语义理解、推理）- 中高权重
  0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
  // Layer 24-31: 高层（复杂推理、答案生成）- 高权重
  0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95,
];

// Token敏感度配置
export const MATH_TOKEN_SENSITIVITY = {
  // 头部token（问题定义、公式）- 高敏感度
  head: {
    retentionRatio: 1.0,      // 完全保留
    quantizationBits: 16,     // FP16精度
  },
  // 中间token（推导步骤）- 中敏感度
  middle: {
    retentionRatio: 0.8,      // 80%保留
    quantizationBits: 8,      // INT8精度
  },
  // 尾部token（答案）- 低敏感度
  tail: {
    retentionRatio: 0.5,      // 50%保留
    quantizationBits: 4,      // INT4精度
  },
};

// 推荐压缩配置
export const MATH_COMPRESSION_PREFERENCE = {
  // 早期层（0-7）：低压缩，保持基础语义
  earlyLayers: {
    retentionRatio: 0.4,
    precision: { key: 8, value: 8 },  // INT8
    description: '词汇级压缩，允许较高压缩比',
  },
  // 中期层（8-15）：中等压缩
  middleLayers: {
    retentionRatio: 0.7,
    precision: { key: 16, value: 8 }, // Key FP16, Value INT8
    description: '句法级压缩，Key保持高精度',
  },
  // 后期层（16-31）：低压缩，保持推理关键信息
  lateLayers: {
    retentionRatio: 0.95,
    precision: { key: 16, value: 16 }, // 全FP16
    description: '推理级保留，Key和Value都保持高精度',
  },
};

// 数学任务特征
export const MATH_TASK_FEATURES = {
  // 数学推理的关键token类型
  importantTokens: [
    '数字', '运算符', '等号', '变量', '函数名',
    '公式', '推导步骤', '证明关键词',
  ],
  // 可压缩的token类型
  compressibleTokens: [
    '标点符号', '空格', '换行符', '注释',
  ],
  // 典型数学任务的层重要性模式
  patterns: {
    // 计算类：中间层重要
    calculation: 'middle-heavy',
    // 证明类：高层重要
    proof: 'top-heavy',
    // 求解类：中高层重要
    solving: 'upper-middle-heavy',
  },
};

export default {
  MATH_LAYER_WEIGHTS,
  MATH_TOKEN_SENSITIVITY,
  MATH_COMPRESSION_PREFERENCE,
  MATH_TASK_FEATURES,
};
