/**
 * PDTrim Baseline - 简化版实现
 * 
 * PDTrim核心思想：
 * 1. 首尾token保留：保留每个序列的前N个和后M个token的KV
 * 2. 中间层剪枝：对中间层进行更激进的剪枝
 * 3. P端选择性传输：只传输重要的层到D端
 * 
 * 简化实现：
 * - 首层和末层：100%保留
 * - 中间层：30-50%保留（取决于任务类型）
 * - 首尾token保留策略（sliding window）
 */

// 类型定义
export interface CompressionParams {
  totalLayers: number;
  totalTokens: number;
  bandwidthBytesPerMs: number;
  gpuMemoryBytes: number;
  currentMemoryUsage: number;
  taskType: string;
  sloLatencyMs?: number;
  prefixHitRate?: number;
}

export interface CompressionOutput {
  strategy: string;
  totalLayers: number;
  pLayerRetention: number[];
  dLayerRetention: number[];
  pKeyPrecision: number[];
  pValuePrecision: number[];
  dKeyPrecision: number[];
  dValuePrecision: number[];
  avgCompressionRatio: number;
  estimatedBandwidthSaving: number;
}

export interface ICompressionStrategy {
  readonly name: string;
  readonly type: string;
  computeConfig(params: CompressionParams): CompressionOutput;
  estimateQualityImpact(config: CompressionOutput, taskType: string): number;
}

// 工具函数
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

/**
 * PDTrimBaseline - PDTrim简化版
 * 
 * 核心特点：
 * - 首尾保留：序列开头和结尾的token保留更多KV
 * - 中间层剪枝：中间层使用更激进的压缩
 * - 层差异化：首层和末层保留率高于中间层
 */
export class PDTrimBaseline implements ICompressionStrategy {
  readonly name = 'PDTrimBaseline';
  readonly type = 'baseline';
  
  // 精度常量
  private readonly KEY_PRECISION = 8;   // FP8
  private readonly VALUE_PRECISION = 4; // INT4
  
  // 首尾token保留比例
  private readonly HEAD_TAIL_RETENTION = 0.8; // 首尾各保留80%
  private readonly MIDDLE_RETENTION = 0.3; // 中间保留30%
  
  /**
   * 计算PDTrim压缩配置
   * 
   * 核心策略：
   * 1. 首层和末层：100%保留（关键语义信息）
   * 2. 中间层：根据任务类型和带宽约束动态调整
   *    - math任务：中间层保留40%（推理需要）
   *    - code任务：中间层保留30%（语法在低层）
   *    - qa任务：中间层保留35%
   *    - conversation：中间层保留50%（均匀分布）
   * 3. P端选择性传输：只传输关键层
   * 4. D端首尾增强：保留更多首尾token
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, taskType, bandwidthBytesPerMs } = params;
    
    // 根据任务类型确定中间层保留率
    const middleRetention = this.getMiddleRetentionByTask(taskType);
    
    // 根据带宽调整激进程度
    // 带宽越低，越激进压缩
    const bandwidthFactor = clamp(bandwidthBytesPerMs / 50, 0.5, 1.0);
    const adjustedMiddleRetention = middleRetention * bandwidthFactor;
    
    // 计算层边界
    const layerBound1 = Math.floor(totalLayers / 4);   // 首层边界
    const layerBound2 = Math.floor((3 * totalLayers) / 4); // 末层边界
    
    // 初始化数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    let totalPRetention = 0;
    let totalDRetention = 0;
    
    for (let i = 0; i < totalLayers; i++) {
      let pRetention: number;
      let dRetention: number;
      
      if (i < layerBound1) {
        // 首层区域：保留较多（关键语义）
        pRetention = 0.8;
        dRetention = 0.9;
      } else if (i >= layerBound2) {
        // 末层区域：保留较多（输出生成关键）
        pRetention = 0.8;
        dRetention = 0.9;
      } else {
        // 中间层区域：激进剪枝
        pRetention = adjustedMiddleRetention;
        dRetention = Math.max(adjustedMiddleRetention, 0.5);
      }
      
      pLayerRetention.push(round4(pRetention));
      dLayerRetention.push(round4(dRetention));
      
      // P端精度：K8V4
      pKeyPrecision.push(this.KEY_PRECISION);
      pValuePrecision.push(this.VALUE_PRECISION);
      
      // D端精度：K16V8
      dKeyPrecision.push(16);
      dValuePrecision.push(8);
      
      totalPRetention += pRetention;
      totalDRetention += dRetention;
    }
    
    // 计算平均压缩比
    const precisionRatioP = (this.KEY_PRECISION / 16) * (this.VALUE_PRECISION / 16);
    const avgRetentionP = totalPRetention / totalLayers;
    const avgCompressionRatio = avgRetentionP * precisionRatioP;
    
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
      avgCompressionRatio: round4(avgCompressionRatio),
      estimatedBandwidthSaving: round4(estimatedBandwidthSaving)
    };
  }
  
  /**
   * 根据任务类型确定中间层保留率
   */
  private getMiddleRetentionByTask(taskType: string): number {
    switch (taskType) {
      case 'math':
        // 数学任务：中高层重要，中间层保留40%（推理链需要）
        return 0.4;
      case 'code':
        // 代码任务：低层重要（语法），中间层保留30%
        return 0.3;
      case 'qa':
        // 问答任务：高层重要，中间层保留35%
        return 0.35;
      case 'conversation':
        // 对话任务：均匀分布，中间层保留50%
        return 0.5;
      default:
        return 0.4;
    }
  }
  
  /**
   * 预估质量影响
   * 
   * PDTrim通过保留首尾层来维持质量
   * 首尾层保留率高，对质量影响较小
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    
    // 首尾层权重更高（0.6）
    const headTailWeight = 0.6;
    const middleWeight = 0.4;
    
    // 计算首尾层和中间层的平均保留率
    const totalLayers = config.totalLayers;
    const headTailCount = Math.floor(totalLayers / 4); // 首尾各1/4
    const middleCount = totalLayers - 2 * headTailCount;
    
    let headTailRetention = 0;
    let middleRetention = 0;
    
    for (let i = 0; i < totalLayers; i++) {
      if (i < headTailCount || i >= totalLayers - headTailCount) {
        headTailRetention += config.pLayerRetention[i];
      } else {
        middleRetention += config.pLayerRetention[i];
      }
    }
    
    headTailRetention /= (2 * headTailCount);
    middleRetention /= middleCount;
    
    // 加权质量
    const quality = headTailRetention * headTailWeight + middleRetention * middleWeight;
    
    return round4(Math.min(quality, 1.0));
  }
  
  /**
   * 获取压缩策略描述
   */
  getStrategyDescription(): string {
    return `PDTrim: 首尾层保留80%, 中间层保留30-50%, P端K8V4, D端K16V8`;
  }
}
