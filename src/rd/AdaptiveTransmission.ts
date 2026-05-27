/**
 * Adaptive Transmission - 自适应传输调度
 * 
 * 功能:
 * 1. 根据网络状况动态调整压缩策略
 * 2. 高语义重要性层优先传输
 * 3. 传输优先级排序
 * 4. 延迟预估
 * 
 * 核心算法:
 * - 优先级队列: 语义重要性高的层先传
 * - 带宽分配: 根据当前带宽动态调整压缩比
 * - 时序调度: 最小化TTFT
 */

import { TaskType, ServingRequest } from '../core/types.js';
import { RateDistortion } from './RateDistortion.js';
import { SemanticDistortion } from './SemanticDistortion.js';

/**
 * 网络状况
 */
export interface NetworkCondition {
  bandwidthBytesPerMs: number;   // 当前带宽
  latencyMs: number;             // 网络延迟
  jitterMs: number;              // 抖动
  packetLossRate: number;        // 丢包率 [0-1]
  stability: number;             // 稳定性 [0-1], 1表示完全稳定
}

/**
 * 传输优先级项
 */
export interface TransmissionPriorityItem {
  layerIndex: number;
  priority: number;              // 优先级分数 [0-1]
  semanticImportance: number;    // 语义重要性
  estimatedTransferTimeMs: number;  // 预计传输时间
  estimatedQualityImpact: number;    // 对质量的影响
  sizeBytes: number;             // 数据大小
  isCritical: boolean;           // 是否关键层
}

/**
 * 传输调度结果
 */
export interface TransmissionSchedule {
  transmissionOrder: number[];   // 传输顺序（层索引）
  layerConfigs: {
    layerIndex: number;
    retentionRatio: number;
    precision: number;
    estimatedTransferTime: number;
  }[];
  totalEstimatedTimeMs: number;  // 总预计时间
  totalSizeBytes: number;        // 总数据大小
  qualityScore: number;          // 预计质量分数
  compressionRatio: number;      // 总压缩比
}

/**
 * SLO约束
 */
export interface SLOConstraint {
  maxTTFTMs: number;            // 最大TTFT
  maxE2ELatencyMs: number;       // 最大端到端延迟
  minQualityScore: number;       // 最小质量分数
}

/**
 * 自适应传输器
 */
export class AdaptiveTransmission {
  private semanticDistortion: SemanticDistortion;
  
  constructor() {
    this.semanticDistortion = new SemanticDistortion();
  }
  
  /**
   * 调度传输
   * 
   * 根据当前网络状况和SLO约束，生成最优传输调度
   */
  schedule(
    currentBandwidth: number,
    pendingRequests: ServingRequest[],
    sloConstraints: Map<string, SLOConstraint>,
    numLayers: number = 32,
    taskType: TaskType = 'unknown'
  ): {
    compressionConfig: { retentionRatio: number[]; precision: number[] };
    transmissionOrder: number[];
    estimatedLatencyMs: number;
    qualityScore: number;
    meetsSLO: boolean;
  } {
    if (pendingRequests.length === 0) {
      return {
        compressionConfig: { retentionRatio: Array(numLayers).fill(1), precision: Array(numLayers).fill(16) },
        transmissionOrder: Array.from({ length: numLayers }, (_, i) => i),
        estimatedLatencyMs: 0,
        qualityScore: 1,
        meetsSLO: true
      };
    }
    
    const layerImportance = this.computeLayerImportance(numLayers, taskType);
    const priorityItems = this.computeTransmissionPriority(currentBandwidth, numLayers, layerImportance);
    const transmissionOrder = priorityItems.sort((a, b) => b.priority - a.priority).map(item => item.layerIndex);
    const compressionConfig = this.computeCompressionConfig(currentBandwidth, layerImportance, numLayers);
    const estimatedLatencyMs = this.estimateLatency(currentBandwidth, compressionConfig, numLayers);
    const qualityScore = this.estimateQualityScore(compressionConfig, numLayers, taskType);
    const meetsSLO = this.checkSLOCompliance(estimatedLatencyMs, qualityScore, sloConstraints);
    
    return { compressionConfig, transmissionOrder, estimatedLatencyMs, qualityScore, meetsSLO };
  }
  
  /** 计算层重要性 */
  computeLayerImportance(numLayers: number, taskType: TaskType): number[] {
    return Array.from({ length: numLayers }, (_, i) => 
      RateDistortion.computeMutualInformation(i, numLayers, taskType)
    );
  }
  
  /** 计算传输优先级 */
  computeTransmissionPriority(bandwidth: number, numLayers: number, layerImportance: number[]): TransmissionPriorityItem[] {
    const baseSizePerLayer = 1024 * 1024;
    return layerImportance.map((importance, i) => {
      const sizeBytes = baseSizePerLayer * (1 - importance * 0.5);
      const estimatedTransferTimeMs = sizeBytes / bandwidth;
      const priority = 0.6 * importance + 0.2 * (1 - estimatedTransferTimeMs / 100) + 0.2 * (1 - sizeBytes / baseSizePerLayer);
      return {
        layerIndex: i,
        priority: Math.min(1, Math.max(0, priority)),
        semanticImportance: importance,
        estimatedTransferTimeMs,
        estimatedQualityImpact: importance,
        sizeBytes,
        isCritical: importance > 0.6
      };
    });
  }
  
  /** 排序传输优先级 */
  computeTransmissionPriorityOrder(layerImportance: number[]): number[] {
    return layerImportance.map((importance, index) => ({ importance, index }))
      .sort((a, b) => b.importance - a.importance).map(item => item.index);
  }
  
  /** 计算压缩配置 */
  computeCompressionConfig(bandwidth: number, layerImportance: number[], numLayers: number): { retentionRatio: number[]; precision: number[] } {
    const highBandwidthThreshold = 1024 * 1024;
    const midBandwidthThreshold = 1024;
    let baseRetention: number, basePrecision: number;
    
    if (bandwidth >= highBandwidthThreshold) { baseRetention = 0.9; basePrecision = 16; }
    else if (bandwidth >= midBandwidthThreshold) { baseRetention = 0.6; basePrecision = 8; }
    else { baseRetention = 0.3; basePrecision = 4; }
    
    const retentionRatio: number[] = [];
    const precision: number[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      const importance = layerImportance[i];
      if (importance > 0.7) {
        retentionRatio.push(Math.min(1, baseRetention + 0.2));
        precision.push(Math.min(16, basePrecision + 8));
      } else if (importance > 0.4) {
        retentionRatio.push(baseRetention);
        precision.push(basePrecision);
      } else {
        retentionRatio.push(Math.max(0.1, baseRetention - 0.3));
        precision.push(Math.max(4, basePrecision - 4));
      }
    }
    return { retentionRatio, precision };
  }
  
  private estimateLatency(bandwidth: number, config: { retentionRatio: number[]; precision: number[] }, numLayers: number): number {
    const baseSizePerLayer = 1024 * 1024;
    let totalSize = 0;
    for (let i = 0; i < numLayers; i++) {
      const precisionRatio = config.precision[i] / 16;
      totalSize += baseSizePerLayer * config.retentionRatio[i] * precisionRatio;
    }
    return totalSize / bandwidth;
  }
  
  private estimateQualityScore(config: { retentionRatio: number[]; precision: number[] }, numLayers: number, taskType: TaskType): number {
    let totalQuality = 0;
    for (let i = 0; i < numLayers; i++) {
      const compressionRatio = config.retentionRatio[i] * (config.precision[i] / 16);
      const result = this.semanticDistortion.estimateSemanticDistortion(compressionRatio, i, numLayers, taskType, 'mixed');
      totalQuality += result.qualityScore;
    }
    return totalQuality / numLayers;
  }
  
  private checkSLOCompliance(estimatedLatencyMs: number, qualityScore: number, sloConstraints: Map<string, SLOConstraint>): boolean {
    if (sloConstraints.size === 0) return true;
    for (const constraint of sloConstraints.values()) {
      if (estimatedLatencyMs > constraint.maxTTFTMs) return false;
      if (qualityScore < constraint.minQualityScore) return false;
    }
    return true;
  }
}
