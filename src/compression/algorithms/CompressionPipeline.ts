/**
 * 混合压缩Pipeline
 * 
 * 结合剪枝和量化两种压缩技术，实现端到端的KV Cache压缩
 * 流程：输入 → 剪枝 → 量化 → 压缩后数据
 */

import { MathUtils } from '../../core/utils.js';
import { KVQuantizer, QuantizationConfig, QuantizationType } from './KVQuantizer.js';
import { AttentionPruner, PruningConfig, PruningResult, PruningStrategy } from './AttentionPruner.js';

/**
 * 混合压缩配置
 */
export interface HybridCompressionConfig {
  // 剪枝配置
  pruneStrategy: PruningStrategy;
  retentionRatio: number;       // 全局保留比例
  headRetentionRatio: number;  // 头部保留比例（head_tail策略用）
  tailRetentionRatio: number;  // 尾部保留比例（head_tail策略用）
  
  // 量化配置
  quantizationType: QuantizationType;
  quantizationBlockSize: number; // 量化分块大小
}

/**
 * 压缩Pipeline结果
 */
export interface PipelineResult {
  // 原始大小
  originalSizeBytes: number;
  
  // 压缩后大小
  compressedSizeBytes: number;
  
  // 压缩比
  compressionRatio: number;
  
  // 预估SNR（量化质量）
  estimatedSNR: number;
  
  // 预估质量影响
  qualityImpact: number;
  
  // 各阶段详情
  pruningResult: PruningResult;
  quantizationRatio: number;
  
  // 传输时间估算
  estimatedTransferTimeMs: number;
  bandwidthBytesPerMs: number;
}

/**
 * KV Cache压缩Pipeline
 */
export class CompressionPipeline {
  private quantizer: KVQuantizer;
  private pruner: AttentionPruner;
  
  constructor() {
    this.quantizer = new KVQuantizer();
    this.pruner = new AttentionPruner();
  }
  
  /**
   * 执行完整压缩流程
   * 
   * @param tokenCount token数量
   * @param numHeads 注意力头数
   * @param headDim 每头维度
   * @param numLayers 层数
   * @param taskType 任务类型
   * @param config 压缩配置
   * @param bandwidthBytesPerMs 带宽 (bytes/ms)
   */
  compress(
    tokenCount: number,
    numHeads: number,
    headDim: number,
    numLayers: number,
    taskType: string,
    config: HybridCompressionConfig,
    bandwidthBytesPerMs: number = 1024 * 1024 * 100 // 100MB/s
  ): PipelineResult {
    // Step 1: 生成模拟的attention分数
    const attentionScores = this.pruner.generateSimulatedAttentionScores(
      tokenCount,
      taskType,
      42
    );
    
    // Step 2: 执行剪枝
    const pruningConfig: PruningConfig = {
      strategy: config.pruneStrategy,
      retentionRatio: config.retentionRatio,
      headRetentionRatio: config.headRetentionRatio,
      tailRetentionRatio: config.tailRetentionRatio
    };
    
    const pruningResult = this.pruner.executePrune(
      tokenCount,
      attentionScores,
      pruningConfig,
      taskType,
      42
    );
    
    // Step 3: 量化（剪枝后的token数）
    const retainedTokenCount = pruningResult.retainedIndices.length;
    const quantConfig = KVQuantizer.createConfig(
      config.quantizationType,
      config.quantizationBlockSize
    );
    
    // 计算原始大小（FP16）
    const originalElements = tokenCount * numHeads * headDim * 2 * numLayers;
    const originalSizeBytes = originalElements * 2; // FP16 = 2 bytes
    
    // 计算量化后大小（基于保留的token数）
    const compressedSizeBytes = this.quantizer.computeCompressedSize(
      retainedTokenCount,
      numHeads,
      headDim,
      numLayers,
      quantConfig
    );
    
    // 计算总体压缩比
    const compressionRatio = compressedSizeBytes / originalSizeBytes;
    
    // 估算SNR
    const estimatedSNR = this.quantizer.estimateSNR(quantConfig);
    
    // 估算质量影响
    const qualityImpact = this.estimateQualityImpact(
      pruningResult.estimatedImpact,
      estimatedSNR,
      taskType
    );
    
    // 计算传输时间
    const estimatedTransferTimeMs = compressedSizeBytes / bandwidthBytesPerMs;
    
    return {
      originalSizeBytes,
      compressedSizeBytes,
      compressionRatio: MathUtils.round(compressionRatio, 4),
      estimatedSNR,
      qualityImpact: MathUtils.round(qualityImpact, 4),
      pruningResult,
      quantizationRatio: compressedSizeBytes / (retainedTokenCount * numHeads * headDim * 2 * numLayers * 2),
      estimatedTransferTimeMs: MathUtils.round(estimatedTransferTimeMs, 4),
      bandwidthBytesPerMs
    };
  }
  
  /**
   * 仅执行剪枝（不量化）
   */
  pruneOnly(
    tokenCount: number,
    numHeads: number,
    headDim: number,
    numLayers: number,
    taskType: string,
    config: HybridCompressionConfig
  ): { 
    pruningResult: PruningResult; 
    compressionRatio: number;
    originalSizeBytes: number;
    compressedSizeBytes: number;
  } {
    // 生成attention分数
    const attentionScores = this.pruner.generateSimulatedAttentionScores(
      tokenCount,
      taskType,
      42
    );
    
    // 执行剪枝
    const pruningConfig: PruningConfig = {
      strategy: config.pruneStrategy,
      retentionRatio: config.retentionRatio,
      headRetentionRatio: config.headRetentionRatio,
      tailRetentionRatio: config.tailRetentionRatio
    };
    
    const pruningResult = this.pruner.executePrune(
      tokenCount,
      attentionScores,
      pruningConfig,
      taskType,
      42
    );
    
    // 计算大小
    const retainedTokenCount = pruningResult.retainedIndices.length;
    const originalElements = tokenCount * numHeads * headDim * 2 * numLayers;
    const originalSizeBytes = originalElements * 2;
    const compressedSizeBytes = retainedTokenCount * numHeads * headDim * 2 * numLayers * 2;
    
    return {
      pruningResult,
      compressionRatio: MathUtils.round(comtainedTokenCount / tokenCount, 4),
      originalSizeBytes,
      compressedSizeBytes
    };
  }
  
  /**
   * 仅执行量化（不剪枝）
   */
  quantizeOnly(
    tokenCount: number,
    numHeads: number,
    headDim: number,
    numLayers: number,
    config: HybridCompressionConfig
  ): { 
    quantizationRatio: number;
    estimatedSNR: number;
    originalSizeBytes: number;
    compressedSizeBytes: number;
  } {
    const quantConfig = KVQuantizer.createConfig(
      config.quantizationType,
      config.quantizationBlockSize
    );
    
    // 计算大小
    const originalElements = tokenCount * numHeads * headDim * 2 * numLayers;
    const originalSizeBytes = originalElements * 2;
    const compressedSizeBytes = this.quantizer.computeCompressedSize(
      tokenCount,
      numHeads,
      headDim,
      numLayers,
      quantConfig
    );
    
    return {
      quantizationRatio: MathUtils.round(compressedSizeBytes / originalSizeBytes, 4),
      estimatedSNR: this.quantizer.estimateSNR(quantConfig),
      originalSizeBytes,
      compressedSizeBytes
    };
  }
  
  /**
   * 估算总体质量影响
   * 
   * 综合考虑剪枝影响和量化SNR
   */
  private estimateQualityImpact(
    pruningImpact: number,
    quantizationSNR: number,
    taskType: string
  ): number {
    // 量化SNR转质量损失
    // SNR越高，质量损失越小
    // SNR=∞ → 0%损失, SNR=12dB (INT2) → 较高损失
    let quantizationLoss: number;
    
    if (quantizationSNR === Infinity) {
      quantizationLoss = 0;
    } else if (quantizationSNR >= 40) {
      quantizationLoss = 0.02; // FP8/INT8
    } else if (quantizationSNR >= 24) {
      quantizationLoss = 0.05; // INT4
    } else {
      quantizationLoss = 0.15; // INT2
    }
    
    // 任务类型敏感度
    const taskSensitivity: Record<string, number> = {
      math: 1.2,
      code: 0.8,
      qa: 1.0,
      conversation: 0.6,
      unknown: 1.0
    };
    
    const sensitivity = taskSensitivity[taskType.toLowerCase()] || 1.0;
    
    // 综合影响
    const totalImpact = (pruningImpact + quantizationLoss * sensitivity) / 2;
    
    return Math.min(1, Math.max(0, totalImpact));
  }
  
  /**
   * 创建默认配置
   */
  static createDefaultConfig(): HybridCompressionConfig {
    return {
      pruneStrategy: 'head_tail',
      retentionRatio: 0.5,
      headRetentionRatio: 0.9,
      tailRetentionRatio: 0.7,
      quantizationType: 'int8',
      quantizationBlockSize: 64
    };
  }
  
  /**
   * 创建预定义配置
   */
  static createConfig(
    qualityLevel: 'high' | 'medium' | 'low' | 'extreme'
  ): HybridCompressionConfig {
    switch (qualityLevel) {
      case 'high':
        return {
          pruneStrategy: 'importance',
          retentionRatio: 0.8,
          headRetentionRatio: 0.95,
          tailRetentionRatio: 0.9,
          quantizationType: 'fp8',
          quantizationBlockSize: 64
        };
      case 'medium':
        return {
          pruneStrategy: 'head_tail',
          retentionRatio: 0.6,
          headRetentionRatio: 0.85,
          tailRetentionRatio: 0.75,
          quantizationType: 'int8',
          quantizationBlockSize: 64
        };
      case 'low':
        return {
          pruneStrategy: 'head_tail',
          retentionRatio: 0.4,
          headRetentionRatio: 0.7,
          tailRetentionRatio: 0.6,
          quantizationType: 'int4',
          quantizationBlockSize: 128
        };
      case 'extreme':
        return {
          pruneStrategy: 'random',
          retentionRatio: 0.2,
          headRetentionRatio: 0.5,
          tailRetentionRatio: 0.5,
          quantizationType: 'int2',
          quantizationBlockSize: 256
        };
    }
  }
}
