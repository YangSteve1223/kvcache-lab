/**
 * Global State Store - 所有Agent写入这里，Scheduler读取
 * 
 * 核心设计原则：
 * 1. 任何Agent不直接通信，只通过Global State交换信息
 * 2. Global Runtime Scheduler读取所有状态做最终决策
 * 3. 每个Agent有清晰的输入/输出/objective
 * 
 * OS概念映射：
 * | OS概念      | KV系统对应              |
 * |-------------|-------------------------|
 * | Working Set | Semantic Region         |
 * | Cache Repl. | Predictive Eviction     |
 * | Virtual Mem | Hierarchical KV         |
 * | Page Migr.  | KV Transfer             |
 * | Mem Access  | Transmission-Aware Attn |
 */

import { clamp, round4 } from '../core/types.js';

// ============================================
// 语义状态 - Agent 1: Semantic Agent
// ============================================

/**
 * 语义区域 - 表示一组语义相近的token
 */
export interface SemanticRegion {
  id: number;
  name: string;
  tokenIndices: number[];
  importance: number;           // 0-1，语义重要性
  coherence: number;            // 0-1，区域内coherence
  queryRelevance: number;       // 0-1，与当前查询的相关性
  layerCoverage: number[];      // 每层覆盖度 [0-1]
}

/**
 * 语义状态 - 描述当前推理的语义上下文
 */
export interface SemanticState {
  activeRegions: SemanticRegion[];       // 当前活跃语义区域
  workingSetTokens: number[];            // 工作集token索引
  reasoningFocus: string;                // 推理焦点 ('induction' | 'deduction' | 'retrieval' | 'generation')
  generationProgress: number;            // 生成进度 0-1
  taskPhase: 'prefill' | 'decode';       // 当前阶段
  queryEmbedding?: number[];             // 查询embedding（用于相关性计算）
  attentionSinkTokens: number[];         // Attention sink位置（通常不驱逐）
}

// ============================================
// 重用状态 - Agent 2: Reuse Agent
// ============================================

/**
 * Token重用预测
 */
export interface TokenReusePrediction {
  tokenIndex: number;
  reuseDistance: number;               // 预测reuse distance
  reuseProbability: number;           // 预测reuse概率 [0-1]
  confidence: number;                  // 预测置信度 [0-1]
  temporalPattern: 'temporal' | 'spatial' | 'random';
}

/**
 * 层级别重用预测
 */
export interface LayerReusePrediction {
  layerIndex: number;
  avgReuseDistance: number;
  hotTokens: number[];                 // 预计近期被访问的top-K token
  accessFrequency: number;             // 该层访问频率
  importanceScore: number;             // 该层重要性评分
}

/**
 * 重用状态 - 描述token/层级别的重用模式预测
 */
export interface ReuseState {
  tokenPredictions: Map<number, TokenReusePrediction>;
  layerPredictions: Map<number, LayerReusePrediction>;
  lastAccessTime: Map<number, number>;  // token索引 -> 上次访问时间
  accessCount: Map<number, number>;    // token索引 -> 访问次数
  reuseDistanceDistribution: number[]; // reuse distance直方图
}

// ============================================
// 通信状态 - Agent 3: Communication Agent
// ============================================

/**
 * 拥塞级别
 */
export type CongestionLevel = 'low' | 'medium' | 'high';

/**
 * 通信状态 - 描述网络传输成本和带宽利用
 */
export interface CommunicationState {
  tokenAccessCosts: Map<number, number>;       // token索引 -> 访问成本(ms)
  layerAccessCosts: Map<number, number>;        // 层索引 -> 平均访问成本
  bandwidthUtilization: number;                // 带宽利用率 0-1
  congestionLevel: CongestionLevel;
  estimatedTransferLatency: number;             // 预估传输延迟 (ms)
  availableBandwidthBytesPerMs: number;        // 当前可用带宽
  totalBandwidthBytesPerMs: number;            // 总带宽
  pendingTransfers: number;                    // 待传输KV块数量
}

// ============================================
// 放置状态 - Agent 4: Placement Agent
// ============================================

/**
 * KV存储位置
 */
export type KVLocation = 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted';

/**
 * 内存层级使用情况
 */
export interface MemoryUtilization {
  gpuHBM: number;    // 0-1, GPU HBM使用率
  cpuRAM: number;    // 0-1, CPU RAM使用率
  remote: number;    // 0-1, 远程存储使用率
  compressed: number; // 0-1, 压缩存储使用率
}

/**
 * 待迁移的KV条目
 */
export interface MigrationTask {
  tokenId: number;
  layerIndex: number;
  from: KVLocation;
  to: KVLocation;
  priority: number;        // 0-1, 优先级
  estimatedTransferMs: number;
  sizeBytes: number;
}

/**
 * 放置状态 - 描述KV块的物理位置和迁移队列
 */
export interface PlacementState {
  tokenLocations: Map<number, KVLocation>;     // token索引 -> 当前位置
  layerLocations: Map<number, KVLocation>;     // 层索引 -> 当前位置
  memoryUtilization: MemoryUtilization;
  migrationQueue: MigrationTask[];             // 按priority降序排列
  kvSizes: Map<number, number>;                 // token索引 -> KV大小(bytes)
  lastMigrationTime: number;                    // 上次迁移时间戳
}

// ============================================
// Scheduler决策
// ============================================

/**
 * KV传输任务
 */
export interface KVTransferTask {
  layer: number;
  tokens: number[];
  priority: number;
  estimatedLatencyMs: number;
}

/**
 * KV压缩任务
 */
export interface KVCompressionTask {
  layer: number;
  tokens: number[];
  targetRatio: number;      // 目标压缩比
  compressionType: 'none' | 'uniform' | 'selective';
}

/**
 * KV迁移任务
 */
export interface KVMigrationTask {
  tokenId: number;
  layer: number;
  from: KVLocation;
  to: KVLocation;
  priority: number;
}

/**
 * Scheduler决策 - Runtime Scheduler的输出
 * 
 * 描述了对KV cache的统一管理决策
 */
export interface SchedulerDecision {
  // 保留/驱逐决策
  retainTokens: number[];              // 保留的token
  evictTokens: number[];               // 驱逐的token
  
  // 传输/预取决策
  transmitKV: KVTransferTask[];        // 需要传输的KV
  prefetchKV: KVTransferTask[];        // 预取的KV
  
  // 压缩决策
  compressKV: KVCompressionTask[];     // 压缩的KV
  
  // 迁移决策
  migrateKV: KVMigrationTask[];         // 迁移的KV
  
  // 决策元信息
  objective: number;                    // 目标函数值（越高越好）
  qualityEstimate: number;              // 预估质量 [0-1]
  latencyEstimate: number;             // 预估延迟 (ms)
  memoryEstimate: number;              // 预估内存使用 (bytes)
  
  // 决策解释
  reasoning: string;
  agentContributions: {
    semantic: number;      // 各Agent对该决策的贡献度
    reuse: number;
    communication: number;
    placement: number;
  };
}

// ============================================
// 完整全局状态
// ============================================

/**
 * 任务类型
 */
export type SystemTaskType = 'math' | 'code' | 'qa' | 'conversation' | 'summarization' | 'unknown';

/**
 * 完整全局状态
 */
export interface GlobalKVState {
  // 元信息
  timestamp: number;
  currentStep: number;
  taskType: SystemTaskType;
  
  // 各Agent状态
  semantic: SemanticState;
  reuse: ReuseState;
  communication: CommunicationState;
  placement: PlacementState;
  
  // Scheduler的决策结果
  decision: SchedulerDecision | null;
  
  // 配置
  config: {
    maxMemoryBytes: number;
    bandwidthBytesPerMs: number;
    sloLatencyMs: number;
  };
}

// ============================================
// 全局状态存储
// ============================================

/**
 * GlobalStateStore - 全局状态存储
 * 
 * 核心特性：
 * 1. 线程安全（单线程JS环境下为同步操作）
 * 2. 支持快照（用于ablation对比）
 * 3. 支持增量更新
 * 4. 完整的操作日志
 */
export class GlobalStateStore {
  private state: GlobalKVState;
  private snapshots: GlobalKVState[] = [];
  private updateLog: Array<{
    timestamp: number;
    agent: 'semantic' | 'reuse' | 'communication' | 'placement' | 'scheduler';
    field: string;
    prevValue: unknown;
    newValue: unknown;
  }> = [];
  
  constructor(config?: Partial<GlobalKVState['config']>) {
    const defaultConfig: GlobalKVState['config'] = {
      maxMemoryBytes: 16 * 1024 * 1024 * 1024,  // 16GB
      bandwidthBytesPerMs: 100,                   // 100MB/s
      sloLatencyMs: 1000,                         // 1s SLO
      ...config
    };
    
    this.state = {
      timestamp: Date.now(),
      currentStep: 0,
      taskType: 'unknown',
      semantic: this.createInitialSemanticState(),
      reuse: this.createInitialReuseState(),
      communication: this.createInitialCommunicationState(),
      placement: this.createInitialPlacementState(),
      decision: null,
      config: defaultConfig
    };
  }
  
  // ========== 初始化辅助方法 ==========
  
  private createInitialSemanticState(): SemanticState {
    return {
      activeRegions: [],
      workingSetTokens: [],
      reasoningFocus: 'retrieval',
      generationProgress: 0,
      taskPhase: 'prefill',
      attentionSinkTokens: []
    };
  }
  
  private createInitialReuseState(): ReuseState {
    return {
      tokenPredictions: new Map(),
      layerPredictions: new Map(),
      lastAccessTime: new Map(),
      accessCount: new Map(),
      reuseDistanceDistribution: []
    };
  }
  
  private createInitialCommunicationState(): CommunicationState {
    return {
      tokenAccessCosts: new Map(),
      layerAccessCosts: new Map(),
      bandwidthUtilization: 0,
      congestionLevel: 'low',
      estimatedTransferLatency: 0,
      availableBandwidthBytesPerMs: 100,
      totalBandwidthBytesPerMs: 100,
      pendingTransfers: 0
    };
  }
  
  private createInitialPlacementState(): PlacementState {
    return {
      tokenLocations: new Map(),
      layerLocations: new Map(),
      memoryUtilization: {
        gpuHBM: 0,
        cpuRAM: 0,
        remote: 0,
        compressed: 0
      },
      migrationQueue: [],
      kvSizes: new Map(),
      lastMigrationTime: 0
    };
  }
  
  // ========== Agent写入接口 ==========
  
  /**
   * 更新语义状态 (Semantic Agent)
   */
  updateSemantic(state: Partial<SemanticState>): void {
    const prevValue = { ...this.state.semantic };
    Object.assign(this.state.semantic, state);
    this.logUpdate('semantic', 'partial', prevValue, this.state.semantic);
    this.state.timestamp = Date.now();
  }
  
  /**
   * 更新重用状态 (Reuse Agent)
   */
  updateReuse(state: Partial<ReuseState>): void {
    const prevValue = { 
      tokenPredictions: new Map(this.state.reuse.tokenPredictions),
      layerPredictions: new Map(this.state.reuse.layerPredictions)
    };
    
    if (state.tokenPredictions) {
      state.tokenPredictions.forEach((v, k) => {
        this.state.reuse.tokenPredictions.set(k, v);
      });
    }
    if (state.layerPredictions) {
      state.layerPredictions.forEach((v, k) => {
        this.state.reuse.layerPredictions.set(k, v);
      });
    }
    if (state.lastAccessTime) {
      state.lastAccessTime.forEach((v, k) => {
        this.state.reuse.lastAccessTime.set(k, v);
      });
    }
    if (state.accessCount) {
      state.accessCount.forEach((v, k) => {
        this.state.reuse.accessCount.set(k, v);
      });
    }
    if (state.reuseDistanceDistribution) {
      this.state.reuse.reuseDistanceDistribution = state.reuseDistanceDistribution;
    }
    
    this.logUpdate('reuse', 'partial', prevValue, 'updated');
    this.state.timestamp = Date.now();
  }
  
  /**
   * 更新通信状态 (Communication Agent)
   */
  updateCommunication(state: Partial<CommunicationState>): void {
    const prevValue = { ...this.state.communication };
    
    if (state.tokenAccessCosts) {
      state.tokenAccessCosts.forEach((v, k) => {
        this.state.communication.tokenAccessCosts.set(k, v);
      });
    }
    if (state.layerAccessCosts) {
      state.layerAccessCosts.forEach((v, k) => {
        this.state.communication.layerAccessCosts.set(k, v);
      });
    }
    Object.assign(this.state.communication, {
      bandwidthUtilization: state.bandwidthUtilization ?? this.state.communication.bandwidthUtilization,
      congestionLevel: state.congestionLevel ?? this.state.communication.congestionLevel,
      estimatedTransferLatency: state.estimatedTransferLatency ?? this.state.communication.estimatedTransferLatency,
      availableBandwidthBytesPerMs: state.availableBandwidthBytesPerMs ?? this.state.communication.availableBandwidthBytesPerMs,
      totalBandwidthBytesPerMs: state.totalBandwidthBytesPerMs ?? this.state.communication.totalBandwidthBytesPerMs,
      pendingTransfers: state.pendingTransfers ?? this.state.communication.pendingTransfers
    });
    
    this.logUpdate('communication', 'partial', prevValue, this.state.communication);
    this.state.timestamp = Date.now();
  }
  
  /**
   * 更新放置状态 (Placement Agent)
   */
  updatePlacement(state: Partial<PlacementState>): void {
    const prevValue = { ...this.state.placement };
    
    if (state.tokenLocations) {
      state.tokenLocations.forEach((v, k) => {
        this.state.placement.tokenLocations.set(k, v);
      });
    }
    if (state.layerLocations) {
      state.layerLocations.forEach((v, k) => {
        this.state.placement.layerLocations.set(k, v);
      });
    }
    if (state.memoryUtilization) {
      Object.assign(this.state.placement.memoryUtilization, state.memoryUtilization);
    }
    if (state.migrationQueue) {
      this.state.placement.migrationQueue = state.migrationQueue;
    }
    if (state.kvSizes) {
      state.kvSizes.forEach((v, k) => {
        this.state.placement.kvSizes.set(k, v);
      });
    }
    if (state.lastMigrationTime !== undefined) {
      this.state.placement.lastMigrationTime = state.lastMigrationTime;
    }
    
    this.logUpdate('placement', 'partial', prevValue, this.state.placement);
    this.state.timestamp = Date.now();
  }
  
  // ========== Scheduler写入接口 ==========
  
  /**
   * 更新调度决策 (Runtime Scheduler)
   */
  updateDecision(decision: SchedulerDecision): void {
    const prevDecision = this.state.decision;
    this.state.decision = decision;
    this.logUpdate('scheduler', 'decision', prevDecision, decision);
    this.state.timestamp = Date.now();
    this.state.currentStep++;
  }
  
  // ========== 读取接口 ==========
  
  /**
   * 读取完整状态
   */
  getState(): GlobalKVState {
    return { ...this.state };
  }
  
  /**
   * 读取特定Agent状态
   */
  getSemanticState(): SemanticState {
    return { ...this.state.semantic };
  }
  
  getReuseState(): ReuseState {
    return {
      ...this.state.reuse,
      tokenPredictions: new Map(this.state.reuse.tokenPredictions),
      layerPredictions: new Map(this.state.reuse.layerPredictions),
      lastAccessTime: new Map(this.state.reuse.lastAccessTime),
      accessCount: new Map(this.state.reuse.accessCount),
      reuseDistanceDistribution: [...this.state.reuse.reuseDistanceDistribution]
    };
  }
  
  getCommunicationState(): CommunicationState {
    return {
      ...this.state.communication,
      tokenAccessCosts: new Map(this.state.communication.tokenAccessCosts),
      layerAccessCosts: new Map(this.state.communication.layerAccessCosts)
    };
  }
  
  getPlacementState(): PlacementState {
    return {
      ...this.state.placement,
      tokenLocations: new Map(this.state.placement.tokenLocations),
      layerLocations: new Map(this.state.placement.layerLocations),
      migrationQueue: [...this.state.placement.migrationQueue],
      kvSizes: new Map(this.state.placement.kvSizes)
    };
  }
  
  /**
   * 获取当前决策
   */
  getDecision(): SchedulerDecision | null {
    return this.state.decision;
  }
  
  /**
   * 获取元信息
   */
  getMeta(): Pick<GlobalKVState, 'timestamp' | 'currentStep' | 'taskType' | 'config'> {
    return {
      timestamp: this.state.timestamp,
      currentStep: this.state.currentStep,
      taskType: this.state.taskType,
      config: { ...this.state.config }
    };
  }
  
  // ========== 快照与恢复 ==========
  
  /**
   * 创建快照（用于ablation）
   */
  snapshot(): GlobalKVState {
    const snapshot: GlobalKVState = {
      ...this.state,
      semantic: { ...this.state.semantic },
      reuse: {
        ...this.state.reuse,
        tokenPredictions: new Map(this.state.reuse.tokenPredictions),
        layerPredictions: new Map(this.state.reuse.layerPredictions),
        lastAccessTime: new Map(this.state.reuse.lastAccessTime),
        accessCount: new Map(this.state.reuse.accessCount),
        reuseDistanceDistribution: [...this.state.reuse.reuseDistanceDistribution]
      },
      communication: {
        ...this.state.communication,
        tokenAccessCosts: new Map(this.state.communication.tokenAccessCosts),
        layerAccessCosts: new Map(this.state.communication.layerAccessCosts)
      },
      placement: {
        ...this.state.placement,
        tokenLocations: new Map(this.state.placement.tokenLocations),
        layerLocations: new Map(this.state.placement.layerLocations),
        migrationQueue: [...this.state.placement.migrationQueue],
        kvSizes: new Map(this.state.placement.kvSizes)
      },
      decision: this.state.decision ? { ...this.state.decision } : null,
      config: { ...this.state.config }
    };
    
    this.snapshots.push(snapshot);
    return snapshot;
  }
  
  /**
   * 恢复到指定快照
   */
  restore(snapshotIndex: number): boolean {
    if (snapshotIndex < 0 || snapshotIndex >= this.snapshots.length) {
      return false;
    }
    
    const snapshot = this.snapshots[snapshotIndex];
    this.state = JSON.parse(JSON.stringify(snapshot));
    
    // 恢复Map对象
    this.state.reuse.tokenPredictions = new Map(snapshot.reuse.tokenPredictions);
    this.state.reuse.layerPredictions = new Map(snapshot.reuse.layerPredictions);
    this.state.reuse.lastAccessTime = new Map(snapshot.reuse.lastAccessTime);
    this.state.reuse.accessCount = new Map(snapshot.reuse.accessCount);
    this.state.communication.tokenAccessCosts = new Map(snapshot.communication.tokenAccessCosts);
    this.state.communication.layerAccessCosts = new Map(snapshot.communication.layerAccessCosts);
    this.state.placement.tokenLocations = new Map(snapshot.placement.tokenLocations);
    this.state.placement.layerLocations = new Map(snapshot.placement.layerLocations);
    this.state.placement.kvSizes = new Map(snapshot.placement.kvSizes);
    
    return true;
  }
  
  /**
   * 获取快照数量
   */
  getSnapshotCount(): number {
    return this.snapshots.length;
  }
  
  // ========== 批量操作 ==========
  
  /**
   * 重置所有状态
   */
  reset(): void {
    this.snapshot(); // 保存重置前状态
    this.state = {
      timestamp: Date.now(),
      currentStep: 0,
      taskType: 'unknown',
      semantic: this.createInitialSemanticState(),
      reuse: this.createInitialReuseState(),
      communication: this.createInitialCommunicationState(),
      placement: this.createInitialPlacementState(),
      decision: null,
      config: this.state.config
    };
  }
  
  /**
   * 设置任务类型
   */
  setTaskType(taskType: SystemTaskType): void {
    this.state.taskType = taskType;
    this.state.timestamp = Date.now();
  }
  
  /**
   * 更新配置
   */
  updateConfig(config: Partial<GlobalKVState['config']>): void {
    Object.assign(this.state.config, config);
    this.state.timestamp = Date.now();
  }
  
  // ========== 辅助方法 ==========
  
  private logUpdate(
    agent: 'semantic' | 'reuse' | 'communication' | 'placement' | 'scheduler',
    field: string,
    prevValue: unknown,
    newValue: unknown
  ): void {
    this.updateLog.push({
      timestamp: Date.now(),
      agent,
      field,
      prevValue,
      newValue
    });
  }
  
  /**
   * 获取更新日志
   */
  getUpdateLog(): typeof this.updateLog {
    return [...this.updateLog];
  }
  
  /**
   * 清除更新日志
   */
  clearUpdateLog(): void {
    this.updateLog = [];
  }
  
  /**
   * 获取内存压力评分 (0-1, 越高越紧张)
   */
  getMemoryPressure(): number {
    const { memoryUtilization } = this.state.placement;
    // 加权平均，GPU HBM权重最高
    return (
      memoryUtilization.gpuHBM * 0.5 +
      memoryUtilization.cpuRAM * 0.25 +
      memoryUtilization.remote * 0.15 +
      memoryUtilization.compressed * 0.1
    );
  }
  
  /**
   * 获取带宽压力评分 (0-1, 越高越紧张)
   */
  getBandwidthPressure(): number {
    return this.state.communication.bandwidthUtilization;
  }
}

// ============================================
// 工具函数
// ============================================

/**
 * 创建默认的GlobalStateStore
 */
export function createGlobalStateStore(
  config?: Partial<GlobalKVState['config']>
): GlobalStateStore {
  return new GlobalStateStore(config);
}

/**
 * 合并多个Agent状态的贡献度
 */
export function mergeAgentContributions(
  contributions: SchedulerDecision['agentContributions']
): number {
  const weights = { semantic: 0.3, reuse: 0.3, communication: 0.2, placement: 0.2 };
  return (
    contributions.semantic * weights.semantic +
    contributions.reuse * weights.reuse +
    contributions.communication * weights.communication +
    contributions.placement * weights.placement
  );
}
