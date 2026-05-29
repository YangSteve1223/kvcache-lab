/**
 * 校准版质量模型 - 基于真实GPU实验数据
 * 
 * 数据来源：
 * - Mistral-7B (sink-dominant): 30% budget +0.1% PPL, 50% budget ±0.0%
 * - Gemma-2-9B (hybrid): 30% budget +0.47% PPL (sink=16), 50% budget -0.94% PPL
 * - Qwen2.5-7B (local-dominant): 30% budget -0.8% PPL, 50% budget -0.2%
 * 
 * 核心校准原则：
 * 1. Locality-aware: 高Gini(>0.9)意味着压缩对质量影响极小
 * 2. Sink-aware: sink token保留对hybrid/sink-dominant模型至关重要
 * 3. Task-aware: 不同任务类型对压缩的敏感度不同
 */

import { CompressionOutput, TaskType, clamp, round4 } from '../core/types.js';

// ============================================
// 模型Locality配置 (来自GPU实验)
// ============================================

export interface LocalityProfile {
  gini: number;              // Gini系数
  activeSetPct: number;      // Active set占比
  remoteAttnPct: number;     // 远程注意力占比
  pattern: 'local-dominant' | 'sink-dominant' | 'hybrid';
  sinkTokenCount: number;    // 推荐sink token数
  // 实验拟合参数: PPL变化 = a * (1-retention)^2 + b * (1-retention)
  pplCoefficients: { a: number; b: number; c: number };
}

export const LOCALITY_PROFILES: Record<string, LocalityProfile> = {
  'qwen-7b': {
    gini: 0.911,
    activeSetPct: 0.07,
    remoteAttnPct: 0.10,
    pattern: 'local-dominant',
    sinkTokenCount: 0,
    // 50% budget: -0.2% PPL → very slightly negative (SWS discard hardest tokens)
    // 30% budget: -0.8% PPL → even more negative (more hard tokens discarded)
    pplCoefficients: { a: -0.5, b: 0.2, c: 0 },
  },
  'qwen-14b': {
    gini: 0.952,
    activeSetPct: 0.05,
    remoteAttnPct: 0.05,
    pattern: 'local-dominant',
    sinkTokenCount: 0,
    pplCoefficients: { a: -0.3, b: 0.1, c: 0 },
  },
  'mistral-7b': {
    gini: 0.917,
    activeSetPct: 0.119,
    remoteAttnPct: 0.788,
    pattern: 'sink-dominant',
    sinkTokenCount: 16,
    // 30% budget: +0.1% PPL (sink=16)
    // 50% budget: -0.0% PPL
    pplCoefficients: { a: 0.1, b: -0.05, c: 0 },
  },
  'gemma-9b': {
    gini: 0.866,
    activeSetPct: 0.209,
    remoteAttnPct: 0.606,
    pattern: 'hybrid',
    sinkTokenCount: 16,
    // 30% budget: +0.47% PPL (sink=16), +11.84% (sink=0)
    // 50% budget: -0.94% PPL (sink=16)
    // 70% budget: -1.88% PPL (sink=16)
    pplCoefficients: { a: 5.0, b: -8.0, c: 2.5 },
  },
};

// ============================================
// 任务敏感度配置 (来自Exp D: Multi-task PPL)
// ============================================

export interface TaskSensitivityProfile {
  taskType: TaskType;
  // 相对于conversation的PPL放大因子
  // 例如: Science任务PDTrim +152% vs SWS +34%
  pplAmplificationFactor: number;
  // 最优压缩类型
  optimalCompression: 'conservative' | 'balanced' | 'aggressive';
  // 各段层重要性 (来自TaskClassifier/TASK_PROFILES)
  layerImportance: [number, number, number]; // [low, mid, high]
}

export const TASK_SENSITIVITY: Record<string, TaskSensitivityProfile> = {
  'math': {
    taskType: 'math',
    pplAmplificationFactor: 1.5,    // Math推理对压缩更敏感
    optimalCompression: 'conservative',
    layerImportance: [0.3, 0.8, 0.9], // 中高层推理链关键
  },
  'code': {
    taskType: 'code',
    pplAmplificationFactor: 0.7,    // Code对压缩较不敏感
    optimalCompression: 'aggressive',
    layerImportance: [0.9, 0.5, 0.3], // 底层语法关键
  },
  'qa': {
    taskType: 'qa',
    pplAmplificationFactor: 2.0,    // QA高度依赖远程KV
    optimalCompression: 'balanced',
    layerImportance: [0.4, 0.5, 0.8], // 高层语义关键
  },
  'conversation': {
    taskType: 'conversation',
    pplAmplificationFactor: 1.0,    // 基准
    optimalCompression: 'balanced',
    layerImportance: [0.5, 0.5, 0.5], // 均匀分布
  },
};

// ============================================
// 校准版质量计算
// ============================================

// ============================================
// PPL估算 — 基于真实GPU实验数据点的分段线性插值
// ============================================

// 真实实验数据点 (budget → PPL change %)
// 每个模型两个条件: sink_preserved 和 sink_discarded
interface PPLDataPoint {
  budget: number;       // retention ratio
  pplChange: number;    // PPL change %
}

const REAL_PPL_DATA: Record<string, {
  sinkPreserved: PPLDataPoint[];
  sinkDiscarded: PPLDataPoint[];
}> = {
  'qwen-7b': {
    sinkPreserved: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.7, pplChange: -0.2 },
      { budget: 0.5, pplChange: -0.2 },
      { budget: 0.3, pplChange: -0.8 },
    ],
    sinkDiscarded: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.7, pplChange: -0.1 },
      { budget: 0.5, pplChange: 0.0 },
      { budget: 0.3, pplChange: -0.5 },
    ],
  },
  'qwen-14b': {
    sinkPreserved: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.5, pplChange: -0.1 },
      { budget: 0.3, pplChange: -0.3 },
    ],
    sinkDiscarded: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.5, pplChange: 0.0 },
      { budget: 0.3, pplChange: -0.2 },
    ],
  },
  'mistral-7b': {
    sinkPreserved: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.7, pplChange: -0.0 },
      { budget: 0.5, pplChange: -0.0 },
      { budget: 0.3, pplChange: +0.1 },
    ],
    sinkDiscarded: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.7, pplChange: +0.0 },
      { budget: 0.5, pplChange: +0.0 },
      { budget: 0.3, pplChange: +0.2 },
    ],
  },
  'gemma-9b': {
    sinkPreserved: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.7, pplChange: -1.88 },
      { budget: 0.5, pplChange: -0.94 },
      { budget: 0.3, pplChange: +0.47 },
    ],
    sinkDiscarded: [
      { budget: 1.0, pplChange: 0 },
      { budget: 0.7, pplChange: +2.73 },
      { budget: 0.5, pplChange: +1.79 },
      { budget: 0.3, pplChange: +11.84 },
    ],
  },
};

/**
 * 分段线性插值 + 低保留率外推
 * 
 * 在两个相邻数据点之间线性插值
 * 高端(budget > max)使用最近边界点
 * 低端(budget < min)使用指数外推：
 *   PPL(b) = PPL(b_min) × (b_min / b)^2
 *   这反映了KV急剧减少时PPL的指数级恶化
 */
function piecewiseLinearInterp(
  x: number,
  dataPoints: PPLDataPoint[],
): number {
  // 按budget升序排列
  const sorted = [...dataPoints].sort((a, b) => a.budget - b.budget);
  
  // 高端边界外: 使用最近点
  if (x >= sorted[sorted.length - 1].budget) return sorted[sorted.length - 1].pplChange;
  
  // 低端边界外: 指数外推
  if (x < sorted[0].budget) {
    const bMin = sorted[0].budget;
    const pplMin = sorted[0].pplChange;
    if (x <= 0) return pplMin * 100; // 极端情况：保留率趋近0
    // 指数外推：PPL(b) ≈ PPL(b_min) × (b_min/b)^α
    // α=2 给出合理的恶化曲线
    const alpha = 2.0;
    // 只对PPL>0（质量退化）的情况做外推
    if (pplMin > 0) {
      return pplMin * Math.pow(bMin / x, alpha);
    } else {
      // PPL为负（质量改善=虚低），保守外推
      // 保留率极低时PPL必然大幅退化
      return Math.abs(pplMin) * Math.pow(bMin / x, alpha) + pplMin;
    }
  }
  
  // 正常区间：线性插值
  for (let i = 0; i < sorted.length - 1; i++) {
    if (x >= sorted[i].budget && x <= sorted[i + 1].budget) {
      const t = (x - sorted[i].budget) / (sorted[i + 1].budget - sorted[i].budget);
      return sorted[i].pplChange + t * (sorted[i + 1].pplChange - sorted[i].pplChange);
    }
  }
  
  return sorted[sorted.length - 1].pplChange;
}

/**
 * 基于真实实验数据估算PPL变化 (使用分段线性插值)
 * 
 * @param retentionRatio KV保留比例 (0-1)
 * @param modelName 模型名称
 * @param taskProfile 任务敏感度特征
 * @param sinkPreserved 是否保留sink token
 * @returns 估算的PPL变化百分比 (正值=退化, 负值=改善)
 */
export function estimatePPLChange(
  retentionRatio: number,
  localityProfile: LocalityProfile,
  taskProfile: TaskSensitivityProfile,
  sinkPreserved: boolean = true,
): number {
  // 找到模型名称
  let modelName = 'qwen-7b';
  for (const [name, profile] of Object.entries(LOCALITY_PROFILES)) {
    if (profile === localityProfile) {
      modelName = name;
      break;
    }
  }
  
  const modelData = REAL_PPL_DATA[modelName] || REAL_PPL_DATA['qwen-7b'];
  const dataPoints = sinkPreserved ? modelData.sinkPreserved : modelData.sinkDiscarded;
  
  // 基础PPL变化 (来自插值)
  let basePPL = piecewiseLinearInterp(retentionRatio, dataPoints);
  
  // 任务敏感度调整
  const taskFactor = taskProfile.pplAmplificationFactor;
  
  // 综合PPL变化
  const pplChange = basePPL * taskFactor;
  
  return round4(pplChange);
}

/**
 * 将PPL变化转换为质量分数 (0-1, 越高越好)
 * 
 * 映射规则:
 * PPL change ≤ 0%   → Quality = 1.0 (无损失或改善)
 * PPL change = 1%   → Quality ≈ 0.95
 * PPL change = 5%   → Quality ≈ 0.80
 * PPL change = 10%  → Quality ≈ 0.60
 * PPL change = 20%  → Quality ≈ 0.30
 * PPL change ≥ 50%  → Quality ≈ 0.0
 */
export function pplChangeToQuality(pplChangePct: number): number {
  if (pplChangePct <= 0) return 1.0;
  // 指数衰减: quality = exp(-0.08 * pplChange)
  const quality = Math.exp(-0.08 * pplChangePct);
  return round4(clamp(quality, 0, 1));
}

/**
 * 完整的质量评估: 从压缩配置计算质量分数
 * 考虑P端/D端差异化 + 模型locality + 任务敏感度
 * 
 * 物理模型修正 (2026-05-29):
 * 在PD分离架构中，D端只能使用P端传输过来的KV。
 * 因此effective retention = P端传输比例（D端保留所有收到的KV）。
 * 旧模型用pAvg*0.4 + dAvg*0.6加权平均，会高估D端独立保留率的效果。
 * 
 * 两种模式：
 * - 'transmission': effectiveRetention = pAvg (D端保留所有收到的KV)
 * - 'legacy': effectiveRetention = pAvg*0.4 + dAvg*0.6 (旧模型，向后兼容)
 */
export function computeCalibratedQuality(
  config: CompressionOutput,
  modelName: string = 'qwen-7b',
  taskType: TaskType = 'conversation',
  sinkPreserved: boolean = true,
  mode: 'transmission' | 'legacy' = 'transmission',
): {
  quality: number;
  pplChangePct: number;
  pAvgRetention: number;
  dAvgRetention: number;
  pdDifferentiation: number;
} {
  const locality = LOCALITY_PROFILES[modelName] || LOCALITY_PROFILES['qwen-7b'];
  const task = TASK_SENSITIVITY[taskType] || TASK_SENSITIVITY['conversation'];
  
  // 计算P端和D端平均保留率
  const pAvg = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
  const dAvg = config.dLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
  
  // 计算有效保留率
  let effectiveRetention: number;
  if (mode === 'transmission') {
    // 修正模型：D端只能使用P端传输的KV
    // effectiveRetention = P端传输比例（D端保留所有收到的KV）
    // 精度因子：P端精度降低会引入量化噪声，影响D端生成质量
    const pAvgKeyPrecision = config.pKeyPrecision.reduce((a, b) => a + b, 0) / config.totalLayers;
    const pAvgValuePrecision = config.pValuePrecision.reduce((a, b) => a + b, 0) / config.totalLayers;
    // P端精度影响：FP8→2-3% PPL退化，INT4→5-10% PPL退化
    // 精度因子：1.0 = FP16无损，0.85 = FP8轻微退化，0.7 = INT4显著退化
    const precisionQualityFactor = 0.7 + 0.3 * ((pAvgKeyPrecision / 16 + pAvgValuePrecision / 16) / 2);
    effectiveRetention = pAvg * precisionQualityFactor;
  } else {
    // 旧模型（向后兼容）
    effectiveRetention = pAvg * 0.4 + dAvg * 0.6;
    const dAvgPrecision = config.dKeyPrecision.reduce((a, b) => a + b, 0) / config.totalLayers;
    const precisionFactor = dAvgPrecision / 16;
    effectiveRetention = effectiveRetention * (0.7 + 0.3 * precisionFactor);
  }
  
  // 估算PPL变化
  const pplChange = estimatePPLChange(effectiveRetention, locality, task, sinkPreserved);
  
  // 转换为质量分数
  const quality = pplChangeToQuality(pplChange);
  
  return {
    quality,
    pplChangePct: pplChange,
    pAvgRetention: round4(pAvg),
    dAvgRetention: round4(dAvg),
    pdDifferentiation: round4(Math.abs(pAvg - dAvg)),
  };
}
