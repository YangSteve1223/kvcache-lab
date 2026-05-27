// 压缩质量模型
// 基于实验数据建模：任务类型对不同位置(头/中/尾)压缩的敏感度

import { 
  CompressionConfig, 
  TaskType,
  LayerCompressionConfig 
} from './types.ts';

/**
 * 任务类型敏感度配置
 * 数值表示该任务对该位置压缩损失的敏感程度
 * head/middle/tail 分别对应KV的头部、中间、尾部位置
 */
const TASK_SENSITIVITY: Record<TaskType, { head: number; middle: number; tail: number }> = {
  math:         { head: 0.037, middle: 0.037, tail: 0.02 },
  code:         { head: 0.00, middle: 0.00, tail: 0.01 },
  qa:           { head: 0.042, middle: 0.00, tail: 0.02 },
  conversation: { head: 0.02, middle: 0.01, tail: 0.01 },
  unknown:      { head: 0.02, middle: 0.02, tail: 0.02 }
};

/**
 * 位置权重：头部KV权重最高，尾部最低
 */
const POSITION_WEIGHTS = {
  head: 0.5,    // 头部50%
  middle: 0.3,  // 中间30%
  tail: 0.2     // 尾部20%
};

/**
 * 获取token在KV中的相对位置类型
 * @param tokenIndex token索引
 * @param totalTokens 总token数
 */
function getPositionType(tokenIndex: number, totalTokens: number): 'head' | 'middle' | 'tail' {
  const ratio = tokenIndex / totalTokens;
  if (ratio < 0.2) return 'head';
  if (ratio > 0.8) return 'tail';
  return 'middle';
}

/**
 * 计算单层压缩的质量影响
 * @param layerConfig 层压缩配置
 * @param position 位置类型
 * @param taskType 任务类型
 */
function computeLayerQualityImpact(
  layerConfig: LayerCompressionConfig,
  position: 'head' | 'middle' | 'tail',
  taskType: TaskType
): number {
  // 压缩率 = 1 - retentionRatio
  const compressionRate = 1 - layerConfig.retentionRatio;
  
  // 精度损失因子 (假设精度从32bit压缩到更低的bits)
  const precisionLoss = layerConfig.valuePrecision < 32 ? 
    (32 - layerConfig.valuePrecision) / 32 : 0;
  
  // 基础质量损失 = 压缩率 × 精度损失
  const baseLoss = compressionRate * precisionLoss;
  
  // 敏感度权重
  const sensitivity = TASK_SENSITIVITY[taskType] || TASK_SENSITIVITY.unknown;
  const sensitivityFactor = sensitivity[position];
  
  // 位置权重
  const positionWeight = POSITION_WEIGHTS[position];
  
  // 综合影响 = 基础损失 × 敏感度 × 位置权重
  return baseLoss * sensitivityFactor * positionWeight;
}

/**
 * 估计整体质量影响
 * @param compression 压缩配置
 * @param taskType 任务类型
 * @returns 质量影响分数 (0-1)，0表示无影响，1表示完全损坏
 */
export function estimateQualityImpact(
  compression: CompressionConfig,
  taskType: TaskType
): number {
  // 无压缩策略
  if (compression.strategy === 'none') {
    return 0;
  }

  let totalImpact = 0;
  const pLayers = compression.pLayers || [];
  
  for (const layerConfig of pLayers) {
    const layerPosition = getPositionType(layerConfig.layerIndex, layerConfig.totalLayers);
    const impact = computeLayerQualityImpact(layerConfig, layerPosition, taskType);
    totalImpact += impact;
  }

  // D端压缩对质量影响较小（已经过P端处理）
  const dLayers = compression.dLayers || [];
  for (const layerConfig of dLayers) {
    const layerPosition = getPositionType(layerConfig.layerIndex, layerConfig.totalLayers);
    const impact = computeLayerQualityImpact(layerConfig, layerPosition, taskType);
    // D端影响减半
    totalImpact += impact * 0.5;
  }

  // 确保在0-1范围内
  return Math.min(1, Math.max(0, totalImpact));
}

/**
 * 根据压缩配置计算质量分数
 * @param compression 压缩配置
 * @param taskType 任务类型
 * @returns 质量分数 (0-1)
 */
export function computeQualityScore(
  compression: CompressionConfig | null,
  taskType: TaskType
): number {
  if (!compression) {
    return 1.0;  // 无压缩时质量为满分
  }
  
  const impact = estimateQualityImpact(compression, taskType);
  return MathUtils.round(1 - impact, 4);
}

/**
 * 创建默认的统一压缩配置
 * @param totalLayers 总层数
 * @param retentionRatio 保留比例
 */
export function createUniformCompression(
  totalLayers: number,
  retentionRatio: number = 0.5,
  valuePrecision: number = 8
): CompressionConfig {
  const pLayers: LayerCompressionConfig[] = [];
  
  for (let i = 0; i < totalLayers; i++) {
    pLayers.push({
      layerIndex: i,
      totalLayers,
      retentionRatio,
      keyPrecision: valuePrecision,
      valuePrecision
    });
  }
  
  return {
    strategy: 'uniform',
    pLayers,
    dLayers: []  // D端不额外压缩
  };
}

/**
 * 创建PD-aware压缩配置
 * P端：头部高保真，尾部低精度
 * D端：全部中等保真
 */
export function createPDAwareCompression(
  totalLayers: number,
  headRetention: number = 0.8,
  tailRetention: number = 0.3,
  valuePrecision: number = 8
): CompressionConfig {
  const pLayers: LayerCompressionConfig[] = [];
  
  for (let i = 0; i < totalLayers; i++) {
    const position = i / totalLayers;
    // 越靠尾部，保留比例越低
    const retentionRatio = position < 0.3 ? headRetention : 
                           position > 0.7 ? tailRetention : 
                           (headRetention + tailRetention) / 2;
    
    pLayers.push({
      layerIndex: i,
      totalLayers,
      retentionRatio,
      keyPrecision: valuePrecision,
      valuePrecision
    });
  }
  
  // D端配置：全部中等保真
  const dLayers: LayerCompressionConfig[] = pLayers.map(config => ({
    ...config,
    retentionRatio: 0.6
  }));
  
  return {
    strategy: 'pd-aware',
    pLayers,
    dLayers
  };
}

/**
 * 创建任务自适应的压缩配置
 * 根据任务类型调整压缩策略
 */
export function createTaskAwareCompression(
  totalLayers: number,
  taskType: TaskType
): CompressionConfig {
  let headRetention: number;
  let tailRetention: number;
  
  // 根据任务类型调整保留比例
  switch (taskType) {
    case 'math':
      headRetention = 0.9;  // 数学任务头部很重要
      tailRetention = 0.4;
      break;
    case 'code':
      headRetention = 0.85;
      tailRetention = 0.5;   // 代码任务中间部分也重要
      break;
    case 'qa':
      headRetention = 0.9;   // QA头部关键
      tailRetention = 0.3;
      break;
    case 'conversation':
      headRetention = 0.7;
      tailRetention = 0.5;
      break;
    default:
      headRetention = 0.7;
      tailRetention = 0.4;
  }
  
  const baseConfig = createPDAwareCompression(
    totalLayers, 
    headRetention, 
    tailRetention
  );
  
  return {
    ...baseConfig,
    strategy: 'task-aware',
    taskType
  };
}

/**
 * 计算压缩配置的平均保留比例
 */
export function computeAverageRetention(compression: CompressionConfig | null): number {
  if (!compression) return 1.0;
  
  const layers = compression.pLayers || [];
  if (layers.length === 0) return 1.0;
  
  const sum = layers.reduce((acc, layer) => acc + layer.retentionRatio, 0);
  return MathUtils.round(sum / layers.length, 4);
}

// 导入MathUtils
import { MathUtils } from './utils.ts';
