/**
 * Quality-Constrained Bandwidth Minimization (QCBM) 压缩策略
 * v2: 自适应精度 + 质量约束带宽最小化
 * 
 * 优化目标：
 *   Minimize: P端传输带宽 Σ_l r[l] × precision_ratio[l]
 *   Subject to: Quality(config) ≥ qualityTarget
 *   Decision: {r[l], key_prec[l], val_prec[l]} for each layer l
 * 
 * v2新增：自适应精度选择
 * - 精度配置表：K16V16, K16V8, K8V8, K8V4, K4V4
 * - 精度越高→带宽成本越大但质量越好
 * - QCBM自动选择满足质量约束的最低精度组合
 * 
 * 物理模型（2026-05-29）：
 * - P端决定传输哪些KV及精度
 * - D端接收P端传输的所有KV
 * - effectiveRetention = pAvg × precisionQualityFactor
 * - 带宽成本 = Σ pRetention[l] × (keyBits[l]/16) × (valBits[l]/16)
 */

import {
  CompressionParams,
  CompressionOutput,
  ensureRetentionRange,
  clamp,
  round4,
  TaskType
} from '../../core/types.js';
import { ICompressionStrategy, TASK_PROFILES, TaskProfile } from '../CompressionOrchestrator.js';
import {
  computeCalibratedQuality,
  LOCALITY_PROFILES,
  TASK_SENSITIVITY
} from '../../core/CalibratedQualityModel.js';

// 精度常量
const PRECISIONS = {
  FP16: 16,
  FP8: 8,
  INT4: 4,
} as const;

// 精度配置：从高到低排列（带宽成本递减，质量递减）
interface PrecisionOption {
  name: string;
  keyBits: number;
  valueBits: number;
  // 带宽成本因子 = (key/16) × (val/16)
  bandwidthFactor: number;
  // 质量因子：1.0=无损，越低质量退化越大
  qualityFactor: number;
}

const PRECISION_OPTIONS: PrecisionOption[] = [
  { name: 'K16V16', keyBits: 16, valueBits: 16, bandwidthFactor: 1.0,    qualityFactor: 1.0 },
  { name: 'K16V8',  keyBits: 16, valueBits: 8,  bandwidthFactor: 0.5,    qualityFactor: 0.925 },
  { name: 'K8V8',   keyBits: 8,  valueBits: 8,  bandwidthFactor: 0.25,   qualityFactor: 0.85 },
  { name: 'K8V4',   keyBits: 8,  valueBits: 4,  bandwidthFactor: 0.125,  qualityFactor: 0.8125 },
  { name: 'K4V4',   keyBits: 4,  valueBits: 4,  bandwidthFactor: 0.0625, qualityFactor: 0.625 },
];

// QCBM配置
interface QCBMConfig {
  qualityTarget: number;
  minRetention: number;
  maxRetention: number;
  binarySearchSteps: number;
  modelName: string;
}

const DEFAULT_QCBM_CONFIG: QCBMConfig = {
  qualityTarget: 0.95,
  minRetention: 0.1,
  maxRetention: 1.0,
  binarySearchSteps: 25,
  modelName: 'qwen-7b',
};

export class QualityConstrainedCompression implements ICompressionStrategy {
  readonly name = 'QualityConstrainedCompression';
  readonly type = 'quality-constrained';
  
  private config: QCBMConfig;

  constructor(config: Partial<QCBMConfig> = {}) {
    this.config = { ...DEFAULT_QCBM_CONFIG, ...config };
  }

  private getTaskLayerWeights(totalLayers: number, taskType: string): number[] {
    const profile: TaskProfile = TASK_PROFILES[taskType] || TASK_PROFILES['conversation'];
    const { layerImportance } = profile;
    
    const b1 = Math.floor(totalLayers / 3);
    const b2 = Math.floor((2 * totalLayers) / 3);
    
    const weights: number[] = [];
    for (let i = 0; i < totalLayers; i++) {
      const seg = i < b1 ? 0 : i < b2 ? 1 : 2;
      const base = layerImportance[seg];
      weights.push(clamp(0.2 + (base - 0.3) * (0.8 / 0.6), 0.2, 1.0));
    }
    return weights;
  }

  /**
   * 按任务权重分配各层保留率，确保总budget守恒
   */
  private allocateLayerRetention(
    totalLayers: number,
    totalBudget: number,
    taskWeights: number[],
    minRetention: number,
    maxRetention: number
  ): number[] {
    const sumWeights = taskWeights.reduce((a, b) => a + b, 0);
    let retentions = taskWeights.map(w => totalBudget * (w / sumWeights));
    
    for (let iter = 0; iter < 10; iter++) {
      const clamped = retentions.map(r => clamp(r, minRetention, maxRetention));
      const totalClamped = clamped.reduce((a, b) => a + b, 0);
      const deficit = totalBudget - totalClamped;
      
      if (Math.abs(deficit) < 0.01 * totalLayers) { retentions = clamped; break; }
      
      if (deficit > 0) {
        const canInc = clamped.map((c, i) => ({ c, i })).filter(x => x.c < maxRetention);
        if (canInc.length === 0) break;
        const incSum = canInc.reduce((s, x) => s + taskWeights[x.i], 0);
        retentions = clamped.map((c, i) => 
          canInc.some(x => x.i === i) ? c + deficit * (taskWeights[i] / incSum) : c
        );
      } else {
        const canDec = clamped.map((c, i) => ({ c, i })).filter(x => x.c > minRetention);
        if (canDec.length === 0) break;
        const decSum = canDec.reduce((s, x) => s + taskWeights[x.i], 0);
        retentions = clamped.map((c, i) => 
          canDec.some(x => x.i === i) ? c + deficit * (taskWeights[i] / decSum) : c
        );
      }
    }
    return retentions.map(r => ensureRetentionRange(clamp(r, minRetention, maxRetention)));
  }

  /**
   * 评估给定配置的质量
   */
  private evaluateQuality(
    pRetention: number[],
    totalLayers: number,
    taskType: string,
    modelName: string,
    precisionOption: PrecisionOption
  ): number {
    const pKeyPrec = new Array(totalLayers).fill(precisionOption.keyBits);
    const pValPrec = new Array(totalLayers).fill(precisionOption.valueBits);
    
    const tempConfig: CompressionOutput = {
      strategy: this.name,
      totalLayers,
      pLayerRetention: pRetention,
      dLayerRetention: new Array(totalLayers).fill(1.0),
      pKeyPrecision: pKeyPrec,
      pValuePrecision: pValPrec,
      dKeyPrecision: new Array(totalLayers).fill(PRECISIONS.FP16),
      dValuePrecision: new Array(totalLayers).fill(PRECISIONS.FP8),
      avgCompressionRatio: 0,
      estimatedBandwidthSaving: 0,
    };
    
    return computeCalibratedQuality(tempConfig, modelName, taskType as TaskType, true, 'transmission').quality;
  }

  /**
   * 计算P端传输带宽成本
   */
  private computeBandwidthCost(
    pRetention: number[],
    precisionOption: PrecisionOption
  ): number {
    const avgR = pRetention.reduce((a, b) => a + b, 0) / pRetention.length;
    return avgR * precisionOption.bandwidthFactor;
  }

  /**
   * 核心算法：自适应精度 + 质量约束带宽最小化
   * 
   * Step 1: 遍历精度配置（从低到高）
   * Step 2: 对每种精度，二分搜索最小满足质量约束的budget
   * Step 3: 选择带宽成本最低的（精度, budget）组合
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, bandwidthBytesPerMs, sloLatencyMs = 1000, taskType } = params;
    
    const taskWeights = this.getTaskLayerWeights(totalLayers, taskType);
    const modelName = this.config.modelName;
    const qualityTarget = this.config.qualityTarget;
    
    // 带宽压力微调（保守：只在极低带宽时轻微放宽）
    const bwPressure = clamp(1 - bandwidthBytesPerMs / 100, 0, 1);
    const adjustedQT = qualityTarget - bwPressure * 0.02;
    
    // 遍历精度配置，找到最优组合
    let bestBandwidthCost = Infinity;
    let bestPRetention: number[] | null = null;
    let bestPrecision = PRECISION_OPTIONS[3]; // default K8V4
    let bestQuality = 0;
    
    for (const precOption of PRECISION_OPTIONS) {
      // 二分搜索：在此精度下，找满足质量约束的最小budget
      let low = totalLayers * this.config.minRetention;
      let high = totalLayers * this.config.maxRetention;
      let foundPRetention: number[] | null = null;
      let foundQuality = 0;
      
      for (let step = 0; step < this.config.binarySearchSteps; step++) {
        const mid = (low + high) / 2;
        const pRetention = this.allocateLayerRetention(
          totalLayers, mid, taskWeights, this.config.minRetention, this.config.maxRetention
        );
        const quality = this.evaluateQuality(pRetention, totalLayers, taskType, modelName, precOption);
        
        if (quality >= adjustedQT) {
          high = mid;
          foundPRetention = pRetention;
          foundQuality = quality;
        } else {
          low = mid;
        }
      }
      
      if (!foundPRetention) continue; // 此精度无法满足质量约束
      
      // 计算此精度+retention组合的带宽成本
      const bwCost = this.computeBandwidthCost(foundPRetention, precOption);
      
      if (bwCost < bestBandwidthCost) {
        bestBandwidthCost = bwCost;
        bestPRetention = foundPRetention;
        bestPrecision = precOption;
        bestQuality = foundQuality;
      }
    }
    
    // 如果所有精度都无法满足约束，使用最高精度+最大budget
    if (!bestPRetention) {
      bestPrecision = PRECISION_OPTIONS[0]; // K16V16
      bestPRetention = this.allocateLayerRetention(
        totalLayers, totalLayers * this.config.maxRetention, taskWeights,
        this.config.minRetention, this.config.maxRetention
      );
      bestQuality = this.evaluateQuality(bestPRetention, totalLayers, taskType, modelName, bestPrecision);
    }
    
    // D端保留率：独立于P端，用于D端内存管理
    const sloStrictness = clamp(1 - sloLatencyMs / 200, 0, 1);
    const dLayerRetention = bestPRetention.map((r, i) => {
      let dR = Math.max(r, 0.6);
      if (i >= Math.floor((2 * totalLayers) / 3)) dR = Math.max(dR, 0.7);
      dR += sloStrictness * 0.1;
      return ensureRetentionRange(Math.min(dR, 1.0));
    });
    
    const pKeyPrecision = new Array(totalLayers).fill(bestPrecision.keyBits);
    const pValuePrecision = new Array(totalLayers).fill(bestPrecision.valueBits);
    
    const bandwidthSaving = 1 - bestBandwidthCost;
    
    return {
      strategy: this.name,
      totalLayers,
      pLayerRetention: bestPRetention,
      dLayerRetention,
      pKeyPrecision,
      pValuePrecision,
      dKeyPrecision: new Array(totalLayers).fill(PRECISIONS.FP16),
      dValuePrecision: new Array(totalLayers).fill(PRECISIONS.FP8),
      avgCompressionRatio: ensureRetentionRange(bestBandwidthCost),
      estimatedBandwidthSaving: ensureRetentionRange(bandwidthSaving),
    };
  }

  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    return computeCalibratedQuality(config, this.config.modelName, taskType as TaskType, true, 'transmission').quality;
  }

  /**
   * 搜索质量-带宽Pareto曲线
   */
  searchQualityBudgetCurve(
    params: CompressionParams,
    qualityTargets: number[] = [0.80, 0.85, 0.90, 0.95, 0.99, 1.00]
  ): Array<{
    qualityTarget: number;
    avgPRetention: number;
    quality: number;
    bandwidthSaving: number;
    precision: string;
  }> {
    const results: Array<{
      qualityTarget: number;
      avgPRetention: number;
      quality: number;
      bandwidthSaving: number;
      precision: string;
    }> = [];
    
    for (const qt of qualityTargets) {
      const qcbm = new QualityConstrainedCompression({ ...this.config, qualityTarget: qt });
      const config = qcbm.computeConfig(params);
      const avgPR = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
      const quality = qcbm.estimateQualityImpact(config, params.taskType);
      
      results.push({
        qualityTarget: qt,
        avgPRetention: round4(avgPR),
        quality: round4(quality),
        bandwidthSaving: round4(config.estimatedBandwidthSaving),
        precision: `${config.pKeyPrecision[0]}K${config.pValuePrecision[0]}V`,
      });
    }
    return results;
  }
}
