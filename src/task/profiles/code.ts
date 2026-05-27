/**
 * 代码任务Profile - 基于代码解析特点
 * 
 * 发现：
 * - 代码在低层（Layer 0-12）attention集中（语法解析）
 * - 关键词/Syntax token重要
 * - 注释/空行可压缩
 * - 函数定义、类定义等结构token高敏感度
 */

// 32层模型的重要性权重
// 分布：低层0.9 -> 高层0.3，线性递减
export const CODE_LAYER_WEIGHTS: number[] = [
  // Layer 0-7: 低层（词汇、基础语法）- 高权重（语法解析关键）
  0.95, 0.93, 0.91, 0.89, 0.87, 0.85, 0.83, 0.81,
  // Layer 8-15: 中低层（句法结构）- 中权重
  0.77, 0.73, 0.69, 0.65, 0.61, 0.57, 0.53, 0.49,
  // Layer 16-23: 中高层（语义理解）- 中低权重
  0.45, 0.42, 0.39, 0.36, 0.33, 0.30, 0.28, 0.26,
  // Layer 24-31: 高层（复杂逻辑、生成）- 低权重
  0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17,
];

// Token敏感度配置
export const CODE_TOKEN_SENSITIVITY = {
  // 头部token（import、class、def等关键字）- 高敏感度
  head: {
    retentionRatio: 1.0,      // 完全保留
    quantizationBits: 16,     // FP16精度
  },
  // 中间token（函数体、逻辑）- 中敏感度
  middle: {
    retentionRatio: 0.7,      // 70%保留
    quantizationBits: 8,      // INT8精度
  },
  // 尾部token（注释、空行）- 低敏感度
  tail: {
    retentionRatio: 0.3,      // 30%保留（注释可压缩）
    quantizationBits: 4,      // INT4精度
  },
};

// 推荐压缩配置
export const CODE_COMPRESSION_PREFERENCE = {
  // 早期层（0-7）：低压缩，保持语法解析能力
  earlyLayers: {
    retentionRatio: 0.9,
    precision: { key: 16, value: 16 }, // 全FP16
    description: '语法解析关键层，保持高精度',
  },
  // 中期层（8-15）：中等压缩
  middleLayers: {
    retentionRatio: 0.6,
    precision: { key: 16, value: 8 }, // Key FP16, Value INT8
    description: '句法级压缩',
  },
  // 后期层（16-31）：高压缩，语义信息已提取
  lateLayers: {
    retentionRatio: 0.4,
    precision: { key: 8, value: 4 }, // Key INT8, Value INT4
    description: '语义级压缩，可使用低精度',
  },
};

// 代码任务特征
export const CODE_TASK_FEATURES = {
  // 代码中重要的token类型
  importantTokens: [
    '关键字', '标识符', '操作符', '括号', '类型名',
    '函数名', '类名', '方法名', '变量名',
  ],
  // 可压缩的token类型
  compressibleTokens: [
    '注释', '空格', '缩进', '换行符', '重复代码',
  ],
  // 编程语言偏好
  languagePatterns: {
    // Python：缩进敏感
    python: { indentSensitive: true, earlyLayerWeight: 0.95 },
    // JavaScript：括号敏感
    javascript: { indentSensitive: false, earlyLayerWeight: 0.90 },
    // Java：类型敏感
    java: { indentSensitive: false, earlyLayerWeight: 0.88 },
    // Go：语法简洁
    go: { indentSensitive: false, earlyLayerWeight: 0.85 },
  },
};

// 语法结构优先级
export const SYNTAX_PRIORITY = {
  // 高优先级：必须保留
  high: [
    'import', 'from', 'class', 'def', 'function', 'const', 'let', 'var',
    'return', 'if', 'else', 'for', 'while', 'switch', 'try', 'catch',
    'public', 'private', 'protected', 'static', 'async', 'await',
  ],
  // 中优先级：可部分压缩
  medium: [
    '=', '+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=',
    '&&', '||', '!', '&', '|', '^', '~', '<<', '>>',
  ],
  // 低优先级：可高压缩
  low: [
    ';', ',', '.', ':', '(', ')', '[', ']', '{', '}',
    '空格', '换行', '缩进', '注释', 'docstring',
  ],
};

export default {
  CODE_LAYER_WEIGHTS,
  CODE_TOKEN_SENSITIVITY,
  CODE_COMPRESSION_PREFERENCE,
  CODE_TASK_FEATURES,
  SYNTAX_PRIORITY,
};
