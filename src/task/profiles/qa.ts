/**
 * QA任务Profile - 基于问答系统特点
 * 
 * 发现：
 * - 高层语义理解重要（Layer 16-28）
 * - System prompt和问题部分不可压缩
 * - 参考文档中间部分可压缩
 */

// 32层模型的重要性权重
// 分布：低层0.3 -> 高层0.8，线性递增
export const QA_LAYER_WEIGHTS: number[] = [
  // Layer 0-7: 低层（词汇匹配）- 低权重
  0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44,
  // Layer 8-15: 中低层（句法解析）- 中低权重
  0.48, 0.52, 0.56, 0.60, 0.64, 0.68, 0.70, 0.72,
  // Layer 16-23: 中高层（语义理解）- 中高权重
  0.74, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83,
  // Layer 24-31: 高层（答案生成、推理）- 高权重
  0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91,
];

// Token敏感度配置
export const QA_TOKEN_SENSITIVITY = {
  // System prompt和问题 - 高敏感度
  systemPrompt: {
    retentionRatio: 1.0,
    quantizationBits: 16,
    description: '系统指令必须完整保留',
  },
  // 问题本身 - 高敏感度
  question: {
    retentionRatio: 0.95,
    quantizationBits: 16,
    description: '问题内容保持高精度',
  },
  // 参考文档头部 - 中敏感度
  documentHead: {
    retentionRatio: 0.7,
    quantizationBits: 8,
    description: '文档开头可能包含关键信息',
  },
  // 参考文档中间 - 低敏感度（可压缩）
  documentMiddle: {
    retentionRatio: 0.4,
    quantizationBits: 4,
    description: '中间部分重复信息多，可高压缩',
  },
  // 参考文档尾部 - 中敏感度
  documentTail: {
    retentionRatio: 0.6,
    quantizationBits: 8,
    description: '结尾可能包含答案线索',
  },
};

// 推荐压缩配置
export const QA_COMPRESSION_PREFERENCE = {
  // 早期层（0-7）：词汇匹配层
  earlyLayers: {
    retentionRatio: 0.4,
    precision: { key: 8, value: 4 }, // INT8/INT4
    description: '词汇级压缩',
  },
  // 中期层（8-15）：句法解析层
  middleLayers: {
    retentionRatio: 0.6,
    precision: { key: 8, value: 8 }, // INT8
    description: '句法级压缩',
  },
  // 后期层（16-31）：语义理解层
  lateLayers: {
    retentionRatio: 0.85,
    precision: { key: 16, value: 8 }, // Key FP16, Value INT8
    description: '语义级保留',
  },
};

// QA任务特征
export const QA_TASK_FEATURES = {
  // 不可压缩的关键部分
  criticalSections: [
    'system_prompt',
    'question',
    'answer_format',
    'constraints',
  ],
  // 可压缩的部分
  compressibleSections: [
    'document_body',
    'examples',
    'explanations',
    'repetitive_content',
  ],
  // 问答类型
  questionTypes: {
    // 事实类：需要精确匹配
    factual: { sensitivity: 'high', earlyWeight: 0.4, lateWeight: 0.9 },
    // 解释类：需要语义理解
    explanatory: { sensitivity: 'medium', earlyWeight: 0.35, lateWeight: 0.85 },
    // 推理类：需要复杂推理
    reasoning: { sensitivity: 'high', earlyWeight: 0.3, lateWeight: 0.95 },
    // 生成类：需要创意生成
    generative: { sensitivity: 'medium', earlyWeight: 0.35, lateWeight: 0.8 },
  },
};

// Attention模式配置
export const ATTENTION_PATTERN = {
  // 问题到上下文的attention - 高权重
  queryToContext: {
    pattern: 'sparse',
    topkRatio: 0.2,
    description: '只关注最相关的上下文token',
  },
  // 上下文到问题的attention - 中权重
  contextToQuery: {
    pattern: 'dense',
    topkRatio: 0.5,
    description: '更多关注问题token',
  },
  // 层间attention - 根据层位置调整
  layerAttention: {
    lowLayers: 'local',      // 低层关注局部信息
    midLayers: 'moderate',  // 中层混合关注
    highLayers: 'global',   // 高层关注全局信息
  },
};

export default {
  QA_LAYER_WEIGHTS,
  QA_TOKEN_SENSITIVITY,
  QA_COMPRESSION_PREFERENCE,
  QA_TASK_FEATURES,
  ATTENTION_PATTERN,
};
