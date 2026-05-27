/**
 * kvcache-lab 统一入口
 * 
 * 导出所有模块的公共接口
 */

// 核心模块
export * from './core/index.js';

// 压缩模块
export * from './compression/index.js';

// 任务感知模块
// 注意：TASK_PROFILES已在compression模块导出，这里不再重复导出
export {
  TaskClassifier,
  classifyTask,
  classifyTaskBatch,
} from './task/TaskClassifier.js';

export type {
  ClassificationResult,
  TaskClassifierOptions,
} from './core/types.js';

export {
  LayerBudgetAllocator,
  allocateLayerBudget,
  generatePyramid,
  getPrecisionForLayer,
} from './task/LayerBudgetAllocator.js';

export type {
  PrecisionType,
  LayerBudget,
  BudgetConstraints,
  ProfileType,
} from './core/types.js';

// Profile 导出（从task模块）
export {
  MATH_LAYER_WEIGHTS,
  MATH_TOKEN_SENSITIVITY,
  MATH_COMPRESSION_PREFERENCE,
  MATH_TASK_FEATURES,
} from './task/profiles/math.js';

export {
  CODE_LAYER_WEIGHTS,
  CODE_TOKEN_SENSITIVITY,
  CODE_COMPRESSION_PREFERENCE,
  CODE_TASK_FEATURES,
  SYNTAX_PRIORITY,
} from './task/profiles/code.js';

export {
  QA_LAYER_WEIGHTS,
  QA_TOKEN_SENSITIVITY,
  QA_COMPRESSION_PREFERENCE,
  QA_TASK_FEATURES,
  ATTENTION_PATTERN,
} from './task/profiles/qa.js';

// ============================================
// Agents模块 - KV Memory OS
// ============================================

export {
  CommunicationAgent,
  communicationAgent,
  computeTransmissionAwareScores,
  computeCongestionAwareLatency,
  getBetaCoefficient,
} from './agents/CommunicationAgent.js';

export type {
  CommunicationAgentInput,
  CommunicationState,
  TokenLocation,
  CongestionLevel,
} from './agents/CommunicationAgent.js';

export {
  PlacementAgent,
  placementAgent,
} from './agents/PlacementAgent.js';

export type {
  PlacementAgentInput,
  PlacementState,
  ReusePrediction,
  MemoryUtilizationInfo,
  MemoryCapacityInfo,
  MigrationItem,
  LayerMemoryUtilization,
} from './agents/PlacementAgent.js';

export {
  SemanticAgent,
} from './agents/SemanticAgent.js';

export type {
  SemanticAgentInput,
  SemanticRegion,
  SemanticRegionType,
  SemanticState,
  RegionTemperature,
} from './agents/SemanticAgent.js';

export {
  ReuseAgent,
} from './agents/ReuseAgent.js';

export type {
  ReuseAgentInput,
  TokenReusePrediction,
  LayerReusePrediction,
  ReuseState,
} from './agents/ReuseAgent.js';

export {
  InMemoryGlobalStateStore,
} from './agents/index.js';

export type {
  GlobalStateStore,
} from './agents/index.js';

// ============================================
// Runtime模块 - KV Memory OS核心
// ============================================

export * from './runtime/index.js';

// ============================================
// IB模块 - Information Bottleneck
// ============================================

export * from './ib/index.js';

// ============================================
// RD模块 - Rate-Distortion
// ============================================

export * from './rd/index.js';

// ============================================
// Unified模块 - IB+RD统一框架
// ============================================

export * from './unified/index.js';

// ============================================
// Scheduling模块 - SLO感知路由
// ============================================

export {
  SLOAwareRouter,
} from './scheduling/SLOAwareRouter.js';

// ============================================
// Baselines模块 - 对标基线
// ============================================

export {
  KVServeBaseline,
} from './baselines/KVServeBaseline.js';

export {
  PDTrimBaseline,
} from './baselines/PDTrimBaseline.js';
