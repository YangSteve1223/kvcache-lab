// ============================================
// 核心模块导出
// ============================================

// 类型定义（统一来源）
export * from './types.ts';

// 工具函数
export * from './utils.ts';

// 压缩质量模型
export {
  estimateQualityImpact,
  computeQualityScore,
  createUniformCompression,
  createPDAwareCompression,
  createTaskAwareCompression,
  computeAverageRetention
} from './QualityModel.ts';

// KV缓存管理器
export { KVCacheManager } from './KVCacheManager.ts';
export type { CacheStats } from './KVCacheManager.ts';

// PD分离模拟器
export { PDSimulator, DEFAULT_CONFIG } from './PDSimulator.ts';
