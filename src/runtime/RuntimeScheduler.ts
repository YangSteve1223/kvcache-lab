/**
 * Global Runtime Scheduler - 读取所有Agent状态，做最终决策
 * 
 * 统一目标函数：
 * max Quality - λ₁×Latency - λ₂×Memory - λ₃×TransferCost
 * 
 * 它不做具体计算，只做决策：
 * - 哪些token保留/驱逐
 * - 哪些KV传输/预取
 * - 哪些KV压缩/迁移
 * 
 * 架构设计：
 * ┌─────────────────────────────────────────────────────┐
 * │           Global State Store (唯一数据源)           │
 * │  ┌─────────┐ ┌─────────┐ ┌────────────┐ ┌─────────┐ │
 * │  │Semantic │ │  Reuse  │ │Communication│ │Placement│ │
 * │  │ Agent   │ │  Agent  │ │   Agent     │ │ Agent   │ │
 * │  └────┬────┘ └────┬────┘ └─────┬──────┘ └────┬────┘ │
 * └───────┼──────────┼─────────────┼─────────────┼──────┘
 *         │          │             │             │
 *         └──────────┴──────┬──────┴─────────────┘
 *                           │
 *                    ┌──────▼──────┐
 *                    │   Runtime   │
 *                    │  Scheduler  │
 *                    └──────┬──────┘
 *                           │
 *                    ┌──────▼──────┐
 *                    │  Decision   │
 *                    │  Execution  │
 *                    └─────────────┘
 */

import {
  GlobalStateStore,
  GlobalKVState,
  SemanticState,
  ReuseState,
  CommunicationState,
  PlacementState,
  SchedulerDecision,
  TokenReusePrediction,
  KVTransferTask,
  KVCompressionTask,
  KVMigrationTask,
  SemanticRegion,
  KVLocation
} from './GlobalState.js';

// ============================================
// 配置定义
// ============================================

/**
 * Scheduler权重配置
 */
export interface SchedulerWeights {
  qualityWeight: number;      // λ₁: 质量权重
  latencyWeight: number;      // λ₂: 延迟权重
  memoryWeight: number;       // λ₃: 内存权重
  transferWeight: number;     // λ₄: 传输权重
}

/**
 * Scheduler约束配置
 */
export interface SchedulerConstraints {
  maxLatencyMs: number;        // SLO约束
  maxMemoryBytes: number;      // 内存约束
  bandwidthBytesPerMs: number; // 带宽约束
  maxEvictionRatio: number;    // 最大驱逐比例 [0-1]
}

/**
 * Scheduler配置
 */
export interface SchedulerConfig {
  weights: SchedulerWeights;
  constraints: SchedulerConstraints;
  
  // 调参系数
  alpha: number;  // 语义重要性系数
  beta: number;   // 重用概率系数
  gamma: number;  // 访问成本系数
  delta: number;  // 内存压力系数
  
  // 自适应配置
  enableAdaptiveWeights: boolean;
  adaptationIntervalMs: number;
  adaptationSpeed: number;  // [0-1]
}

/**
 * 决策上下文
 */
export interface DecisionContext {
  state: GlobalKVState;
  currentTime: number;
  requestId?: string;
}

// ============================================
// 统一打分结果
// ============================================

/**
 * Token评分
 */
export interface TokenScore {
  tokenIndex: number;
  score: number;
  components: {
    semantic: number;
    reuse: number;
    communication: number;
    placement: number;
  };
  reason: string;
}

/**
 * 层评分
 */
export interface LayerScore {
  layerIndex: number;
  score: number;
  importance: number;
  hotTokenCount: number;
}

// ============================================
// Runtime Scheduler
// ============================================

/**
 * RuntimeScheduler - 全局运行时调度器
 * 
 * 核心职责：
 * 1. 读取各Agent写入Global State的状态
 * 2. 统一目标函数优化
 * 3. 生成最终调度决策
 * 4. 支持ablation（禁用特定Agent）
 */
export class RuntimeScheduler {
  private store: GlobalStateStore;
  private config: SchedulerConfig;
  private enabledAgents: Set<'semantic' | 'reuse' | 'communication' | 'placement'>;
  private lastAdaptationTime: number = 0;
  
  /**
   * 默认配置
   */
  static readonly DEFAULT_CONFIG: SchedulerConfig = {
    weights: {
      qualityWeight: 1.0,
      latencyWeight: 0.5,
      memoryWeight: 0.3,
      transferWeight: 0.4
    },
    constraints: {
      maxLatencyMs: 1000,
      maxMemoryBytes: 16 * 1024 * 1024 * 1024,
      bandwidthBytesPerMs: 100,
      maxEvictionRatio: 0.5
    },
    alpha: 0.3,
    beta: 0.4,
    gamma: 0.2,
    delta: 0.1,
    enableAdaptiveWeights: true,
    adaptationIntervalMs: 1000,
    adaptationSpeed: 0.1
  };
  
  constructor(store: GlobalStateStore, config?: Partial<SchedulerConfig>) {
    this.store = store;
    this.config = { ...RuntimeScheduler.DEFAULT_CONFIG, ...config };
    this.enabledAgents = new Set(['semantic', 'reuse', 'communication', 'placement']);
  }
  
  // ========== Agent控制 ==========
  
  /**
   * 禁用特定Agent（用于ablation）
   */
  disableAgent(agent: 'semantic' | 'reuse' | 'communication' | 'placement'): void {
    this.enabledAgents.delete(agent);
  }
  
  /**
   * 启用特定Agent
   */
  enableAgent(agent: 'semantic' | 'reuse' | 'communication' | 'placement'): void {
    this.enabledAgents.add(agent);
  }
  
  /**
   * 获取启用的Agent列表
   */
  getEnabledAgents(): string[] {
    return Array.from(this.enabledAgents);
  }
  
  /**
   * 启用所有Agent
   */
  enableAllAgents(): void {
    this.enabledAgents = new Set(['semantic', 'reuse', 'communication', 'placement']);
  }
  
  // ========== 核心调度方法 ==========
  
  /**
   * 执行调度 - 主入口
   */
  schedule(context?: Partial<DecisionContext>): SchedulerDecision {
    const state = this.store.getState();
    
    // Step 1: 自适应权重调整
    if (this.config.enableAdaptiveWeights) {
      this.adaptWeights(state);
    }
    
    // Step 2: 获取各Agent状态
    const workingSet = state.semantic.workingSetTokens;
    const reusePredictions = state.reuse.tokenPredictions;
    const accessCosts = state.communication.tokenAccessCosts;
    const locations = state.placement.tokenLocations;
    const memoryPressure = this.store.getMemoryPressure();
    const bandwidthPressure = this.store.getBandwidthPressure();
    
    // Step 3: 统一打分
    const tokenScores = this.computeUnifiedScores(
      workingSet,
      reusePredictions,
      accessCosts,
      locations,
      memoryPressure,
      state.semantic
    );
    
    // Step 4: 约束优化
    const decision = this.optimizeUnderConstraints(
      tokenScores,
      state
    );
    
    // Step 5: 更新全局状态
    this.store.updateDecision(decision);
    
    return decision;
  }
  
  // ========== 统一打分 ==========
  
  /**
   * 计算统一分数
   * 
   * score[i] = α × I_semantic(i) + β × P_reuse(i) - γ × C_access(i) + δ × P_memory(i)
   * 
   * - α: 语义重要性系数
   * - β: 重用概率系数
   * - γ: 访问成本系数
   * - δ: 内存压力系数
   */
  private computeUnifiedScores(
    workingSet: number[],
    reusePredictions: Map<number, TokenReusePrediction>,
    accessCosts: Map<number, number>,
    locations: Map<number, KVLocation>,
    memoryPressure: number,
    semanticState: SemanticState
  ): TokenScore[] {
    const scores: TokenScore[] = [];
    
    // 收集attention sink
    const attentionSinkSet = new Set(semanticState.attentionSinkTokens);
    
    // 计算访问成本的最大值和平均值（用于归一化）
    const costValues = Array.from(accessCosts.values());
    const maxCost = costValues.length > 0 ? Math.max(...costValues) : 1;
    const avgCost = costValues.length > 0 
      ? costValues.reduce((a, b) => a + b, 0) / costValues.length 
      : 1;
    
    for (const tokenIndex of workingSet) {
      // Attention sink 强制高分
      if (attentionSinkSet.has(tokenIndex)) {
        scores.push({
          tokenIndex,
          score: 1.0,
          components: { semantic: 1, reuse: 1, communication: 1, placement: 1 },
          reason: 'attention_sink'
        });
        continue;
      }
      
      // 语义重要性 (0-1)
      let semanticScore = 0;
      if (this.enabledAgents.has('semantic')) {
        semanticScore = this.computeSemanticScore(tokenIndex, semanticState);
      } else {
        semanticScore = 0.5; // 默认中等
      }
      
      // 重用概率 (0-1)
      let reuseScore = 0;
      if (this.enabledAgents.has('reuse')) {
        const prediction = reusePredictions.get(tokenIndex);
        reuseScore = prediction ? prediction.reuseProbability : 0;
      } else {
        reuseScore = 0.5;
      }
      
      // 访问成本 (0-1, 成本越低分数越高)
      let communicationScore = 0;
      if (this.enabledAgents.has('communication')) {
        const cost = accessCosts.get(tokenIndex) ?? avgCost;
        communicationScore = 1 - (cost / maxCost);
      } else {
        communicationScore = 0.5;
      }
      
      // 放置分数 (0-1, 本地存储分数更高)
      let placementScore = 0;
      if (this.enabledAgents.has('placement')) {
        const location = locations.get(tokenIndex);
        placementScore = location === 'gpu_hbm' ? 1.0 : 
                        location === 'cpu_ram' ? 0.7 :
                        location === 'compressed' ? 0.5 : 0.3;
      } else {
        placementScore = 0.5;
      }
      
      // 内存压力惩罚 (0-1, 压力越高，低优先级token越容易被驱逐)
      const memoryPenalty = memoryPressure;
      
      // 统一分数计算
      const score = 
        this.config.alpha * semanticScore +
        this.config.beta * reuseScore +
        this.config.gamma * communicationScore +
        this.config.delta * placementScore;
      
      // 考虑内存压力调整
      const adjustedScore = memoryPressure > 0.8 
        ? score * (1 - memoryPressure * 0.5) 
        : score;
      
      // 生成原因说明
      const reason = this.generateScoreReason(tokenIndex, {
        semantic: semanticScore,
        reuse: reuseScore,
        communication: communicationScore,
        placement: placementScore
      });
      
      scores.push({
        tokenIndex,
        score: Math.round(adjustedScore * 10000) / 10000,
        components: {
          semantic: semanticScore,
          reuse: reuseScore,
          communication: communicationScore,
          placement: placementScore
        },
        reason
      });
    }
    
    // 按分数降序排列
    return scores.sort((a, b) => b.score - a.score);
  }
  
  /**
   * 计算语义重要性分数
   */
  private computeSemanticScore(tokenIndex: number, state: SemanticState): number {
    // 在active regions中的token分数更高
    let regionScore = 0;
    for (const region of state.activeRegions) {
      if (region.tokenIndices.includes(tokenIndex)) {
        regionScore = Math.max(regionScore, region.importance * region.queryRelevance);
      }
    }
    
    // query relevance越高，分数越高
    const queryRelevance = regionScore;
    
    // generation progress影响
    // prefill阶段：早期token重要
    // decode阶段：最新token重要
    let phaseBonus = 0;
    if (state.taskPhase === 'prefill') {
      // prefilling越往前越重要
      const position = state.workingSetTokens.indexOf(tokenIndex);
      phaseBonus = position < 10 ? 0.1 : 0;
    } else {
      // decoding最新token最重要
      const lastToken = state.workingSetTokens[state.workingSetTokens.length - 1];
      phaseBonus = tokenIndex === lastToken ? 0.2 : 0;
    }
    
    return Math.min(1, queryRelevance + phaseBonus);
  }
  
  /**
   * 生成分数原因说明
   */
  private generateScoreReason(
    tokenIndex: number,
    components: Record<string, number>
  ): string {
    const reasons: string[] = [];
    
    if (components.semantic > 0.7) {
      reasons.push('高语义相关性');
    } else if (components.semantic < 0.3) {
      reasons.push('低语义相关性');
    }
    
    if (components.reuse > 0.7) {
      reasons.push('高重用概率');
    } else if (components.reuse < 0.3) {
      reasons.push('低重用概率');
    }
    
    if (components.placement === 1.0) {
      reasons.push('本地GPU存储');
    } else if (components.placement < 0.5) {
      reasons.push('远程存储');
    }
    
    return reasons.length > 0 ? reasons.join(', ') : '中等优先级';
  }
  
  // ========== 约束优化 ==========
  
  /**
   * 在约束下优化决策
   * 
   * 目标：最大化 objective = quality - λ₁×latency - λ₂×memory - λ₃×transfer
   */
  private optimizeUnderConstraints(
    tokenScores: TokenScore[],
    state: GlobalKVState
  ): SchedulerDecision {
    const { constraints, weights } = this.config;
    
    // 计算当前内存使用
    const currentMemoryUsage = this.calculateCurrentMemoryUsage(state);
    const availableMemory = constraints.maxMemoryBytes - currentMemoryUsage;
    
    // 预算token数量（基于内存约束）
    const avgKVSizeBytes = 1024; // 假设每个token平均1KB
    const maxTokensToRetain = Math.floor(availableMemory / avgKVSizeBytes);
    
    // 最大驱逐数量
    const maxEvictions = Math.floor(
      tokenScores.length * constraints.maxEvictionRatio
    );
    
    // 按分数排序，选择要保留的token
    // 分数高于阈值的保留，低于阈值的驱逐
    const scoreThreshold = this.calculateThreshold(tokenScores, maxEvictions);
    
    const retainTokens: number[] = [];
    const evictTokens: number[] = [];
    
    for (const ts of tokenScores) {
      if (ts.score >= scoreThreshold && retainTokens.length < maxTokensToRetain) {
        retainTokens.push(ts.tokenIndex);
      } else {
        evictTokens.push(ts.tokenIndex);
      }
    }
    
    // 生成传输任务（对于非本地token）
    const transmitKV = this.generateTransferTasks(retainTokens, state);
    
    // 生成预取任务
    const prefetchKV = this.generatePrefetchTasks(retainTokens, state);
    
    // 生成压缩任务
    const compressKV = this.generateCompressionTasks(retainTokens, state);
    
    // 生成迁移任务
    const migrateKV = this.generateMigrationTasks(retainTokens, evictTokens, state);
    
    // 计算预估指标
    const { qualityEstimate, latencyEstimate, memoryEstimate } = this.estimateMetrics(
      retainTokens,
      evictTokens,
      transmitKV,
      compressKV,
      state
    );
    
    // 计算目标函数值
    const objective = 
      weights.qualityWeight * qualityEstimate -
      weights.latencyWeight * (latencyEstimate / constraints.maxLatencyMs) -
      weights.memoryWeight * (memoryEstimate / constraints.maxMemoryBytes) -
      weights.transferWeight * (transmitKV.length / 100);
    
    return {
      retainTokens,
      evictTokens,
      transmitKV,
      prefetchKV,
      compressKV,
      migrateKV,
      objective: Math.round(objective * 10000) / 10000,
      qualityEstimate,
      latencyEstimate,
      memoryEstimate,
      reasoning: this.generateDecisionReasoning(retainTokens, evictTokens, transmitKV),
      agentContributions: this.computeAgentContributions(tokenScores)
    };
  }
  
  /**
   * 计算驱逐阈值
   */
  private calculateThreshold(scores: TokenScore[], maxEvictions: number): number {
    if (scores.length <= maxEvictions) {
      return 0; // 保留所有
    }
    
    // 选择第maxEvictions个token的分数作为阈值
    const sortedScores = scores.map(s => s.score).sort((a, b) => b - a);
    return sortedScores[maxEvictions] ?? 0;
  }
  
  /**
   * 计算当前内存使用
   */
  private calculateCurrentMemoryUsage(state: GlobalKVState): number {
    let totalBytes = 0;
    
    for (const [tokenId, size] of state.placement.kvSizes) {
      if (state.placement.tokenLocations.get(tokenId) !== 'evicted') {
        totalBytes += size;
      }
    }
    
    return totalBytes;
  }
  
  /**
   * 生成传输任务
   */
  private generateTransferTasks(
    retainTokens: number[],
    state: GlobalKVState
  ): KVTransferTask[] {
    const tasks: KVTransferTask[] = [];
    const locations = state.placement.tokenLocations;
    
    // 按层分组需要传输的token
    const layerGroups = new Map<number, number[]>();
    
    for (const tokenId of retainTokens) {
      const location = locations.get(tokenId);
      
      if (location !== 'gpu_hbm') {
        // 需要传输的token
        // TODO: 根据实际层信息分组
        const layer = 0; // 简化：所有token归为layer 0
        if (!layerGroups.has(layer)) {
          layerGroups.set(layer, []);
        }
        layerGroups.get(layer)!.push(tokenId);
      }
    }
    
    for (const [layer, tokens] of layerGroups) {
      const estimatedLatencyMs = this.estimateTransferLatency(tokens.length, state);
      
      tasks.push({
        layer,
        tokens,
        priority: this.computeTransferPriority(tokens, state),
        estimatedLatencyMs
      });
    }
    
    return tasks;
  }
  
  /**
   * 生成预取任务
   */
  private generatePrefetchTasks(
    retainTokens: number[],
    state: GlobalKVState
  ): KVTransferTask[] {
    // 预取策略：根据reuse预测，提前预取可能被访问的token
    const prefetchTokens: number[] = [];
    
    if (this.enabledAgents.has('reuse')) {
      for (const [tokenId, prediction] of state.reuse.tokenPredictions) {
        // 预测reuse distance较小且当前不在本地的token
        if (
          prediction.reuseDistance < 10 &&
          prediction.reuseProbability > 0.5 &&
          !retainTokens.includes(tokenId)
        ) {
          prefetchTokens.push(tokenId);
        }
      }
    }
    
    if (prefetchTokens.length === 0) {
      return [];
    }
    
    return [{
      layer: 0,
      tokens: prefetchTokens,
      priority: 0.5,
      estimatedLatencyMs: this.estimateTransferLatency(prefetchTokens.length, state)
    }];
  }
  
  /**
   * 生成压缩任务
   */
  private generateCompressionTasks(
    retainTokens: number[],
    state: GlobalKVState
  ): KVCompressionTask[] {
    const tasks: KVCompressionTask[] = [];
    const memoryPressure = this.store.getMemoryPressure();
    
    // 内存压力大时，增加压缩任务
    if (memoryPressure > 0.7) {
      // 对中间层进行压缩
      const totalLayers = state.reuse.layerPredictions.size || 32;
      const middleLayers = Math.floor(totalLayers * 0.4);
      const startLayer = Math.floor(totalLayers * 0.2);
      
      for (let layer = startLayer; layer < startLayer + middleLayers; layer++) {
        tasks.push({
          layer,
          tokens: retainTokens,
          targetRatio: memoryPressure > 0.9 ? 0.5 : 0.7,
          compressionType: 'selective'
        });
      }
    }
    
    return tasks;
  }
  
  /**
   * 生成迁移任务
   */
  private generateMigrationTasks(
    retainTokens: number[],
    evictTokens: number[],
    state: GlobalKVState
  ): KVMigrationTask[] {
    const tasks: KVMigrationTask[] = [];
    
    // 驱逐的token：移到compressed或evicted
    for (const tokenId of evictTokens) {
      const currentLocation = state.placement.tokenLocations.get(tokenId);
      
      if (currentLocation !== 'evicted') {
        tasks.push({
          tokenId,
          layer: 0,
          from: currentLocation || 'gpu_hbm',
          to: 'compressed',
          priority: 0.8
        });
      }
    }
    
    // 保留的token：确保在合适的位置
    for (const tokenId of retainTokens) {
      const currentLocation = state.placement.tokenLocations.get(tokenId);
      
      // 如果在remote且频繁访问，迁移到本地
      if (currentLocation === 'remote_gpu') {
        const reuseScore = state.reuse.tokenPredictions.get(tokenId)?.reuseProbability ?? 0;
        
        if (reuseScore > 0.7) {
          tasks.push({
            tokenId,
            layer: 0,
            from: 'remote_gpu',
            to: 'gpu_hbm',
            priority: 0.9
          });
        }
      }
    }
    
    return tasks;
  }
  
  /**
   * 计算传输优先级
   */
  private computeTransferPriority(
    tokens: number[],
    state: GlobalKVState
  ): number {
    if (!this.enabledAgents.has('reuse')) {
      return 0.5;
    }
    
    let totalReuseProb = 0;
    let count = 0;
    
    for (const tokenId of tokens) {
      const prediction = state.reuse.tokenPredictions.get(tokenId);
      if (prediction) {
        totalReuseProb += prediction.reuseProbability;
        count++;
      }
    }
    
    return count > 0 ? totalReuseProb / count : 0.5;
  }
  
  /**
   * 估算传输延迟
   */
  private estimateTransferLatency(
    tokenCount: number,
    state: GlobalKVState
  ): number {
    const kvBytesPerToken = 1024;
    const bytesToTransfer = tokenCount * kvBytesPerToken;
    const bandwidth = state.communication.availableBandwidthBytesPerMs;
    
    return bytesToTransfer / bandwidth;
  }
  
  /**
   * 估算指标
   */
  private estimateMetrics(
    retainTokens: number[],
    evictTokens: number[],
    transmitKV: KVTransferTask[],
    compressKV: KVCompressionTask[],
    state: GlobalKVState
  ): { qualityEstimate: number; latencyEstimate: number; memoryEstimate: number } {
    // 质量估算：保留率越高，质量越高
    const retentionRatio = retainTokens.length / 
      (retainTokens.length + evictTokens.length || 1);
    const qualityEstimate = Math.round(retentionRatio * 10000) / 10000;
    
    // 延迟估算
    let latencyEstimate = 0;
    for (const task of transmitKV) {
      latencyEstimate += task.estimatedLatencyMs;
    }
    for (const task of compressKV) {
      // 压缩延迟：每token约0.1ms
      latencyEstimate += task.tokens.length * 0.1;
    }
    
    // 内存估算
    let memoryEstimate = 0;
    for (const tokenId of retainTokens) {
      memoryEstimate += state.placement.kvSizes.get(tokenId) ?? 1024;
    }
    for (const task of compressKV) {
      // 压缩后内存
      memoryEstimate += (task.tokens.length * (task.targetRatio * 1024));
    }
    
    return {
      qualityEstimate,
      latencyEstimate: Math.round(latencyEstimate * 100) / 100,
      memoryEstimate
    };
  }
  
  /**
   * 生成决策原因说明
   */
  private generateDecisionReasoning(
    retainTokens: number[],
    evictTokens: number[],
    transmitKV: KVTransferTask[]
  ): string {
    const parts: string[] = [];
    
    parts.push(`保留${retainTokens.length}个token`);
    
    if (evictTokens.length > 0) {
      parts.push(`驱逐${evictTokens.length}个token`);
    }
    
    if (transmitKV.length > 0) {
      const totalTokens = transmitKV.reduce((sum, t) => sum + t.tokens.length, 0);
      parts.push(`传输${totalTokens}个token的KV`);
    }
    
    return parts.join('; ');
  }
  
  /**
   * 计算各Agent贡献度
   */
  private computeAgentContributions(scores: TokenScore[]): SchedulerDecision['agentContributions'] {
    if (scores.length === 0) {
      return { semantic: 0.25, reuse: 0.25, communication: 0.25, placement: 0.25 };
    }
    
    const contributions = { semantic: 0, reuse: 0, communication: 0, placement: 0 };
    const totalWeight = { semantic: 0, reuse: 0, communication: 0, placement: 0 };
    
    for (const score of scores) {
      const { semantic, reuse, communication, placement } = score.components;
      const total = semantic + reuse + communication + placement;
      
      if (total > 0) {
        contributions.semantic += semantic / total;
        contributions.reuse += reuse / total;
        contributions.communication += communication / total;
        contributions.placement += placement / total;
        
        totalWeight.semantic++;
        totalWeight.reuse++;
        totalWeight.communication++;
        totalWeight.placement++;
      }
    }
    
    // 归一化
    const count = scores.length;
    return {
      semantic: Math.round((contributions.semantic / count) * 100) / 100,
      reuse: Math.round((contributions.reuse / count) * 100) / 100,
      communication: Math.round((contributions.communication / count) * 100) / 100,
      placement: Math.round((contributions.placement / count) * 100) / 100
    };
  }
  
  // ========== 自适应权重调整 ==========
  
  /**
   * 自适应权重调整
   * 
   * 根据系统状态动态调整权重：
   * - 带宽紧张 → 增大transferWeight
   * - 内存紧张 → 增大memoryWeight
   * - SLO紧 → 增大latencyWeight
   */
  private adaptWeights(state: GlobalKVState): void {
    const now = Date.now();
    
    if (now - this.lastAdaptationTime < this.config.adaptationIntervalMs) {
      return;
    }
    
    this.lastAdaptationTime = now;
    
    const { weights, constraints } = this.config;
    const speed = this.config.adaptationSpeed;
    
    // 内存压力调整
    const memoryPressure = this.store.getMemoryPressure();
    if (memoryPressure > 0.8) {
      // 内存紧张，提高内存权重
      const target = weights.memoryWeight * 2;
      weights.memoryWeight = weights.memoryWeight + (target - weights.memoryWeight) * speed;
    }
    
    // 带宽压力调整
    const bandwidthPressure = this.store.getBandwidthPressure();
    if (bandwidthPressure > 0.8) {
      // 带宽紧张，提高传输权重
      const target = weights.transferWeight * 1.5;
      weights.transferWeight = weights.transferWeight + (target - weights.transferWeight) * speed;
    }
    
    // SLO紧迫度调整
    const currentLatency = state.communication.estimatedTransferLatency;
    const sloRatio = currentLatency / constraints.maxLatencyMs;
    if (sloRatio > 0.8) {
      // SLO紧迫，提高延迟权重
      const target = weights.latencyWeight * 1.5;
      weights.latencyWeight = weights.latencyWeight + (target - weights.latencyWeight) * speed;
    }
    
    // 归一化权重（确保总和不变）
    const totalWeight = 
      weights.qualityWeight +
      weights.latencyWeight +
      weights.memoryWeight +
      weights.transferWeight;
    
    if (totalWeight > 2) {
      const scale = 2 / totalWeight;
      weights.qualityWeight *= scale;
      weights.latencyWeight *= scale;
      weights.memoryWeight *= scale;
      weights.transferWeight *= scale;
    }
  }
  
  // ========== 配置更新 ==========
  
  /**
   * 更新配置
   */
  updateConfig(config: Partial<SchedulerConfig>): void {
    Object.assign(this.config, config);
  }
  
  /**
   * 获取当前配置
   */
  getConfig(): SchedulerConfig {
    return { ...this.config };
  }
  
  /**
   * 重置为默认配置
   */
  resetConfig(): void {
    this.config = { ...RuntimeScheduler.DEFAULT_CONFIG };
  }
  
  // ========== 统计方法 ==========
  
  /**
   * 获取调度统计
   */
  getStats(): {
    totalSchedules: number;
    avgRetainCount: number;
    avgEvictCount: number;
    avgTransmitCount: number;
    enabledAgents: string[];
  } {
    const state = this.store.getState();
    const decision = state.decision;
    
    return {
      totalSchedules: state.currentStep,
      avgRetainCount: decision?.retainTokens.length ?? 0,
      avgEvictCount: decision?.evictTokens.length ?? 0,
      avgTransmitCount: decision?.transmitKV.reduce((sum, t) => sum + t.tokens.length, 0) ?? 0,
      enabledAgents: this.getEnabledAgents()
    };
  }
}

// ============================================
// 工厂函数
// ============================================

/**
 * 创建RuntimeScheduler
 */
export function createRuntimeScheduler(
  store: GlobalStateStore,
  config?: Partial<SchedulerConfig>
): RuntimeScheduler {
  return new RuntimeScheduler(store, config);
}

/**
 * 创建带默认配置的Scheduler
 */
export function createSchedulerWithDefaults(store: GlobalStateStore): RuntimeScheduler {
  return new RuntimeScheduler(store, RuntimeScheduler.DEFAULT_CONFIG);
}
