/**
 * Baseline压缩策略导出
 * 
 * 提供KVServe和PDTrim两种Baseline实现
 */

// KVServe Baseline
export { KVServeBaseline } from './KVServeBaseline.js';
export type { CompressionParams as KVServeParams, CompressionOutput as KVServeOutput } from './KVServeBaseline.js';

// PDTrim Baseline
export { PDTrimBaseline } from './PDTrimBaseline.js';
export type { CompressionParams as PDTrimParams, CompressionOutput as PDTrimOutput } from './PDTrimBaseline.js';

// SLO感知路由
export { SLOAwareRouter, checkSLOCompliance, calculateSLOSatisfactionRate } from '../scheduling/SLOAwareRouter.js';
export type { SLOConfig, StrategyEstimate, RoutingDecision, CompressionParams } from '../scheduling/SLOAwareRouter.js';
