/**
 * Rate-Distortion Theory Module
 * 
 * Phase-aware Information Bottleneck + Rate-Distortion Unified Framework
 * 
 * 核心创新:
 * 1. Phase-aware IB: P端和D端独立分配重要性
 * 2. Semantic Distortion: 用生成质量而非L2误差
 * 3. 统一优化: IB重要性 + R-D带宽分配
 */

// 核心R-D框架
export {
  RateDistortion,
  type RDConfig,
  type RDLayerAllocation,
  type RDOptimalResult,
  type RDPoint,
  type TaskProfile
} from './RateDistortion.js';

// 语义失真度量
export {
  SemanticDistortion,
  type SemanticDistortionConfig,
  type SemanticDistortionResult,
  type SemanticRDPoint
} from './SemanticDistortion.js';

// 自适应传输调度
export {
  AdaptiveTransmission,
  type NetworkCondition,
  type TransmissionPriorityItem,
  type TransmissionSchedule,
  type SLOConstraint
} from './AdaptiveTransmission.js';

// R-D驱动的压缩策略
export {
  RDCompressor,
  CapKVLikeCompressor,
  RDKVLikeCompressor,
  type RDCompressorConfig
} from './RDCompressor.js';
