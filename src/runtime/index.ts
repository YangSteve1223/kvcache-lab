/**
 * Runtime 模块 - 全局状态存储 + 运行时调度器
 * 
 * 这是 Runtime KV Memory OS 的核心模块
 * 
 * 架构：
 * ┌─────────────────────────────────────────────────────┐
 * │           Global State Store (唯一数据源)           │
 * └─────────────────────────────────────────────────────┘
 *    ↑ 写入           ↑ 写入           ↑ 写入           ↑ 写入
 * ┌─────────┐ ┌─────────┐ ┌────────────┐ ┌─────────┐
 * │Semantic │ │  Reuse  │ │Communication│ │Placement│
 * │ Agent   │ │  Agent  │ │   Agent     │ │ Agent   │ 
 * └─────────┘ └─────────┘ └────────────┘ └─────────┘
 *                      ↓ 读取
 *               ┌─────────────────┐
 *               │  Runtime        │
 *               │  Scheduler      │
 *               └─────────────────┘
 *                      ↓ 决策
 *               ┌─────────────────┐
 *               │   Decision      │
 *               │   Execution     │
 *               └─────────────────┘
 */

// Global State Store
export {
  GlobalStateStore,
  createGlobalStateStore,
  mergeAgentContributions
} from './GlobalState.js';

// 类型导出
export type {
  // 语义状态
  SemanticRegion,
  SemanticState,
  
  // 重用状态
  TokenReusePrediction,
  LayerReusePrediction,
  ReuseState,
  
  // 通信状态
  CongestionLevel,
  CommunicationState,
  
  // 放置状态
  KVLocation,
  MemoryUtilization,
  MigrationTask,
  PlacementState,
  
  // 决策类型
  KVTransferTask,
  KVCompressionTask,
  KVMigrationTask,
  SchedulerDecision,
  
  // 全局状态
  SystemTaskType,
  GlobalKVState
} from './GlobalState.js';

// Runtime Scheduler
export {
  RuntimeScheduler,
  createRuntimeScheduler,
  createSchedulerWithDefaults
} from './RuntimeScheduler.js';

// 类型导出
export type {
  SchedulerWeights,
  SchedulerConstraints,
  SchedulerConfig,
  DecisionContext,
  TokenScore,
  LayerScore
} from './RuntimeScheduler.js';
