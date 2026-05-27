/**
 * Agents模块导出
 * 
 * 提供KV Memory OS的智能Agent集合
 * 
 * Agent架构：
 * - 每个Agent只向Global State Store写状态，不直接通信
 * - Agent具有清晰的输入/输出/objective
 * 
 * 当前Agent列表：
 * - CommunicationAgent: 评估KV访问的通信成本
 * - PlacementAgent: 管理KV在存储层级的放置和迁移
 * - SemanticAgent: 识别当前generation的语义状态
 * - ReuseAgent: 预测KV的reuse distance和reuse probability
 */

// ============================================
// Communication Agent导出
// ============================================

export {
  CommunicationAgent,
  communicationAgent,
  computeTransmissionAwareScores,
  computeCongestionAwareLatency,
  getBetaCoefficient,
} from './CommunicationAgent.js';

export type {
  CommunicationAgentInput,
  CommunicationState,
  TokenLocation,
  CongestionLevel,
} from './CommunicationAgent.js';

// ============================================
// Placement Agent导出
// ============================================

export {
  PlacementAgent,
  placementAgent,
} from './PlacementAgent.js';

export type {
  PlacementAgentInput,
  PlacementState,
  ReusePrediction,
  MemoryUtilizationInfo,
  MemoryCapacityInfo,
  MigrationItem,
  LayerMemoryUtilization,
} from './PlacementAgent.js';

// ============================================
// Semantic Agent导出
// ============================================

export {
  SemanticAgent,
} from './SemanticAgent.js';

export type {
  SemanticAgentInput,
  SemanticRegion,
  SemanticRegionType,
  SemanticState,
  RegionTemperature,
} from './SemanticAgent.js';

// ============================================
// Reuse Agent导出
// ============================================

export {
  ReuseAgent,
} from './ReuseAgent.js';

export type {
  ReuseAgentInput,
  TokenReusePrediction,
  LayerReusePrediction,
  ReuseState,
} from './ReuseAgent.js';

// ============================================
// Global State Store接口
// ============================================

/**
 * Global State Store - 全局状态存储
 * 
 * 所有Agent的状态都写入这里，供其他Agent读取
 * 这是Agent之间解耦通信的唯一方式
 */
export interface GlobalStateStore {
  // 通信相关状态
  communicationState?: CommunicationState;
  
  // 放置相关状态
  placementState?: PlacementState;
  
  // 语义状态
  semanticState?: SemanticState;
  
  // Reuse状态
  reuseState?: ReuseState;
}

// ============================================
// InMemory Global State Store实现
// ============================================

import type { CommunicationState } from './CommunicationAgent.js';
import type { PlacementState } from './PlacementAgent.js';
import type { SemanticState } from './SemanticAgent.js';
import type { ReuseState } from './ReuseAgent.js';

/**
 * 内存实现的Global State Store
 * 
 * 用于单进程场景
 */
export class InMemoryGlobalStateStore implements GlobalStateStore {
  communicationState?: CommunicationState;
  placementState?: PlacementState;
  semanticState?: SemanticState;
  reuseState?: ReuseState;
  
  // 语义状态操作
  writeSemanticState(state: SemanticState): void {
    this.semanticState = state;
  }
  
  readSemanticState(): SemanticState | undefined {
    return this.semanticState;
  }
  
  // Reuse状态操作
  writeReuseState(state: ReuseState): void {
    this.reuseState = state;
  }
  
  readReuseState(): ReuseState | undefined {
    return this.reuseState;
  }
  
  // 清空所有状态
  clear(): void {
    this.communicationState = undefined;
    this.placementState = undefined;
    this.semanticState = undefined;
    this.reuseState = undefined;
  }
}

// ============================================
// Agent Factory
// ============================================

/**
 * 创建Communication Agent实例
 */
export function createCommunicationAgent(): CommunicationAgent {
  return new CommunicationAgent();
}

/**
 * 创建Placement Agent实例
 */
export function createPlacementAgent(kvBytesPerToken?: number): PlacementAgent {
  return new PlacementAgent(kvBytesPerToken);
}

/**
 * 创建Semantic Agent实例
 */
export function createSemanticAgent(): SemanticAgent {
  return new SemanticAgent();
}

/**
 * 创建Reuse Agent实例
 */
export function createReuseAgent(): ReuseAgent {
  return new ReuseAgent();
}
