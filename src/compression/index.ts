/**
 * 压缩模块导出
 * 
 * 类型定义统一从 ../core/types.ts 导入
 */

// 核心类型和接口
export {
  clamp,
  round4,
  ensureRetentionRange,
} from '../core/types.js';

export type {
  CompressionParams,
  CompressionOutput,
  ICompressionStrategy,
} from '../core/types.js';

// 编排器
export { CompressionOrchestrator } from './CompressionOrchestrator.js';
export type { TaskProfile } from './CompressionOrchestrator.js';

// 压缩策略
export { NoneCompression } from './strategies/NoneCompression.js';
export { UniformCompression } from './strategies/UniformCompression.js';
export { PDAwareCompression } from './strategies/PDAwareCompression.js';
export { TaskAwareCompression, TASK_PROFILES } from './strategies/TaskAwareCompression.js';
export { PDTaskAwareCompression } from './strategies/PDTaskAwareCompression.js';

// ============================================
// 真实压缩算法模块
// ============================================
export {
  KVQuantizer,
  AttentionPruner,
  CompressionPipeline
} from './algorithms/index.js';

export type {
  QuantizationType,
  QuantizationConfig,
  QuantizationResult,
  QuantizationMetadata,
  PruningStrategy,
  PruningConfig,
  PruningResult,
  HybridCompressionConfig,
  PipelineResult
} from './algorithms/index.js';
