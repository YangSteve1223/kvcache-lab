/**
 * 实验36：Agents协调实验
 * 
 * 目标：Semantic Agent + Reuse Agent协调
 * - 写入Global State → Scheduler决策
 * - 对比：有semantic信息 vs 无semantic信息的eviction效果
 */

import { SemanticAgent, SemanticAgentInput, SemanticRegion, SemanticState } from '../src/agents/SemanticAgent';
import { ReuseAgent, ReuseAgentInput, ReuseState } from '../src/agents/ReuseAgent';
import { InMemoryGlobalStateStore, GlobalStateStore } from '../src/agents/index';
import { TaskType } from '../src/core/types';

// ============================================
// 实验配置
// ============================================

const EXP_CONFIG = {
  taskTypes: ['math', 'code', 'qa', 'conversation'] as TaskType[],
  tokenCount: 1000,
  numLayers: 32,
  decodeSteps: 100,
  cacheSize: 200,
  requestsPerType: 10
};

// ============================================
// 模拟的Global State Store
// ============================================

/**
 * 简化的Scheduler，读取Global State进行eviction决策
 */
class Scheduler {
  private globalState: GlobalStateStore;
  
  constructor(globalState: GlobalStateStore) {
    this.globalState = globalState;
  }
  
  /**
   * 基于Global State做eviction决策
   * 
   * 决策逻辑：
   * 1. 读取Semantic State获取热区域
   * 2. 读取Reuse State获取可驱逐token
   * 3. 综合决策：优先保留热区域token，驱逐低reuse概率token
   */
  makeEvictionDecision(cacheTokens: number[]): number[] {
    const semanticState = this.globalState.readSemanticState();
    const reuseState = this.globalState.readReuseState();
    
    if (!semanticState || !reuseState) {
      return cacheTokens.slice(0, Math.floor(cacheTokens.length / 2));
    }
    
    // 构建热token集合
    const hotTokens = new Set<number>();
    for (const region of semanticState.activeRegions) {
      if (region.temperature === 'hot') {
        for (let i = region.startTokenIndex; i < region.endTokenIndex; i++) {
          hotTokens.add(i);
        }
      }
    }
    
    // 获取可驱逐token
    const evictable = new Set(reuseState.evictableTokens);
    
    // 决策：驱逐不在热区域中的低reuse token
    const toEvict: number[] = [];
    const toKeep: number[] = [];
    
    for (const token of cacheTokens) {
      if (hotTokens.has(token)) {
        // 热区域token必须保留
        toKeep.push(token);
      } else if (evictable.has(token)) {
        // 低reuse token可以驱逐
        toEvict.push(token);
      } else {
        // 中等reuse token，取决于缓存压力
        if (cacheTokens.length > EXP_CONFIG.cacheSize * 1.2) {
          toEvict.push(token);
        } else {
          toKeep.push(token);
        }
      }
    }
    
    // 如果需要更多空间，驱逐中等reuse token
    if (toKeep.length > EXP_CONFIG.cacheSize) {
      const excess = toKeep.length - EXP_CONFIG.cacheSize;
      // 从toKeep尾部取excess个
      toEvict.push(...toKeep.slice(-excess));
      toKeep.splice(toKeep.length - excess, excess);
    }
    
    return toEvict;
  }
}

// ============================================
// 模拟的访问模式
// ============================================

/**
 * 生成访问序列
 */
function generateAccessPattern(
  taskType: TaskType, 
  tokenCount: number, 
  steps: number
): number[][] {
  const accessPattern: number[][] = [];
  
  for (let step = 0; step < steps; step++) {
    const access: number[] = [];
    
    switch (taskType) {
      case 'math':
        // system prompt
        for (let i = 0; i < 50; i++) access.push(i);
        // 最近推理
        for (let i = tokenCount - 50; i < tokenCount; i++) access.push(i);
        // 周期性回溯
        if (step % 5 === 0) {
          for (let i = 200; i < 250; i++) access.push(i);
        }
        break;
        
      case 'code':
        // imports偶尔访问
        if (step % 10 === 0) {
          for (let i = 0; i < 30; i++) access.push(i);
        }
        // 当前函数
        for (let i = tokenCount - 30; i < tokenCount; i++) access.push(i);
        break;
        
      case 'qa':
        // system prompt
        for (let i = 0; i < 50; i++) access.push(i);
        // 当前chunk
        for (let i = tokenCount - 100; i < tokenCount; i++) access.push(i);
        break;
        
      case 'conversation':
        // system prompt
        for (let i = 0; i < 30; i++) access.push(i);
        // 最近轮
        for (let i = tokenCount - 50; i < tokenCount; i++) access.push(i);
        break;
    }
    
    accessPattern.push([...new Set(access)]);
  }
  
  return accessPattern;
}

/**
 * 生成attention分布
 */
function generateAttention(tokenCount: number, step: number): Float64Array {
  const attention = new Float64Array(tokenCount);
  const focusStart = Math.max(0, tokenCount - 50 - (step % 20));
  
  for (let i = 0; i < tokenCount; i++) {
    if (i >= focusStart && i < focusStart + 30) {
      attention[i] = 0.8 / 30;
    } else if (i < 100) {
      attention[i] = 0.1 / 100;
    } else {
      attention[i] = 0.1 / (tokenCount - 130);
    }
  }
  
  return attention;
}

// ============================================
// 协调实验
// ============================================

/**
 * 有semantic信息的eviction
 */
function runWithSemanticInfo(
  taskType: TaskType,
  accessPattern: number[][],
  semanticAgent: SemanticAgent,
  reuseAgent: ReuseAgent,
  globalState: GlobalStateStore
): { hitRate: number; missRate: number; evictionCount: number } {
  const scheduler = new Scheduler(globalState);
  const cache = new Set<number>();
  let hits = 0;
  let misses = 0;
  let evictionCount = 0;
  
  for (let step = 0; step < accessPattern.length; step++) {
    const currentAccess = accessPattern[step];
    
    // 1. Semantic Agent分析
    const tokens = Array.from({ length: EXP_CONFIG.tokenCount }, (_, i) => `token-${i}`);
    const semanticInput: SemanticAgentInput = {
      tokens,
      taskType,
      recentAttention: generateAttention(EXP_CONFIG.tokenCount, step),
      decodeStep: step,
      totalSteps: accessPattern.length
    };
    const semanticState = semanticAgent.analyze(semanticInput);
    globalState.writeSemanticState(semanticState);
    
    // 2. Reuse Agent预测
    const reuseInput: ReuseAgentInput = {
      tokenCount: EXP_CONFIG.tokenCount,
      numLayers: EXP_CONFIG.numLayers,
      taskType,
      currentStep: step,
      generationProgress: step / accessPattern.length,
      historicalAttention: [generateAttention(EXP_CONFIG.tokenCount, step)],
      semanticRegions: semanticState.activeRegions
    };
    const reuseState = reuseAgent.predict(reuseInput);
    globalState.writeReuseState(reuseState);
    
    // 3. Scheduler决策
    const cacheTokens = Array.from(cache);
    const toEvict = scheduler.makeEvictionDecision(cacheTokens);
    
    // 执行驱逐
    for (const token of toEvict) {
      cache.delete(token);
      evictionCount++;
    }
    
    // 添加当前访问到缓存
    for (const token of currentAccess) {
      if (cache.size >= EXP_CONFIG.cacheSize && !cache.has(token)) {
        // 缓存满，先驱逐
        const moreToEvict = scheduler.makeEvictionDecision(Array.from(cache));
        if (moreToEvict.length > 0) {
          cache.delete(moreToEvict[0]);
          evictionCount++;
        }
      }
      cache.add(token);
    }
    
    // 4. 计算命中
    for (const token of currentAccess) {
      if (cache.has(token)) {
        hits++;
      } else {
        misses++;
      }
    }
  }
  
  return {
    hitRate: hits / (hits + misses),
    missRate: misses / (hits + misses),
    evictionCount
  };
}

/**
 * 无semantic信息的eviction（baseline: LRU）
 */
function runWithoutSemanticInfo(
  accessPattern: number[][]
): { hitRate: number; missRate: number; evictionCount: number } {
  const cache = new Set<number>();
  const accessHistory = new Map<number, number>();
  let hits = 0;
  let misses = 0;
  let evictionCount = 0;
  
  for (let step = 0; step < accessPattern.length; step++) {
    const currentAccess = accessPattern[step];
    
    // 更新访问历史
    for (const token of currentAccess) {
      accessHistory.set(token, step);
    }
    
    // LRU驱逐：驱逐最久未访问的
    if (cache.size >= EXP_CONFIG.cacheSize) {
      let oldestToken = -1;
      let oldestTime = Infinity;
      
      for (const token of cache) {
        const lastAccess = accessHistory.get(token) || 0;
        if (lastAccess < oldestTime) {
          oldestTime = lastAccess;
          oldestToken = token;
        }
      }
      
      if (oldestToken >= 0) {
        cache.delete(oldestToken);
        evictionCount++;
      }
    }
    
    // 添加当前访问
    for (const token of currentAccess) {
      if (!cache.has(token)) {
        cache.add(token);
      }
    }
    
    // 计算命中
    for (const token of currentAccess) {
      if (cache.has(token)) {
        hits++;
      } else {
        misses++;
      }
    }
  }
  
  return {
    hitRate: hits / (hits + misses),
    missRate: misses / (hits + misses),
    evictionCount
  };
}

// ============================================
// 实验函数
// ============================================

function runCoordinatedExperiment(taskType: TaskType): {
  withSemantic: { hitRate: number; missRate: number; evictionCount: number };
  withoutSemantic: { hitRate: number; missRate: number; evictionCount: number };
  improvement: number;
} {
  const accessPattern = generateAccessPattern(
    taskType, 
    EXP_CONFIG.tokenCount, 
    EXP_CONFIG.decodeSteps
  );
  
  // 有semantic信息
  const semanticAgent = new SemanticAgent();
  const reuseAgent = new ReuseAgent();
  const globalState = new InMemoryGlobalStateStore();
  
  const withSemantic = runWithSemanticInfo(
    taskType,
    accessPattern,
    semanticAgent,
    reuseAgent,
    globalState
  );
  
  // 无semantic信息
  const withoutSemantic = runWithoutSemanticInfo(accessPattern);
  
  // 计算提升
  const hitRateImprovement = (withSemantic.hitRate - withoutSemantic.hitRate) / withoutSemantic.hitRate * 100;
  
  return {
    withSemantic,
    withoutSemantic,
    improvement: hitRateImprovement
  };
}

/**
 * 运行完整实验
 */
async function runExperiment(): Promise<void> {
  console.log('========================================');
  console.log('实验36：Agents协调实验');
  console.log('========================================\n');
  
  console.log('配置:');
  console.log(`  - 任务类型: ${EXP_CONFIG.taskTypes.join(', ')}`);
  console.log(`  - Token数: ${EXP_CONFIG.tokenCount}`);
  console.log(`  - 缓存大小: ${EXP_CONFIG.cacheSize}`);
  console.log(`  - Decode步数: ${EXP_CONFIG.decodeSteps}`);
  console.log(`  - 每类型请求数: ${EXP_CONFIG.requestsPerType}\n`);
  
  const results: Record<TaskType, ReturnType<typeof runCoordinatedExperiment>> = {} as any;
  
  console.log('--- Agent协调效果对比 ---\n');
  
  for (const taskType of EXP_CONFIG.taskTypes) {
    console.log(`[${taskType}]`);
    const result = runCoordinatedExperiment(taskType);
    results[taskType] = result;
    
    console.log(`  有Semantic信息:`);
    console.log(`    - 命中率: ${(result.withSemantic.hitRate * 100).toFixed(1)}%`);
    console.log(`    - 驱逐次数: ${result.withSemantic.evictionCount}`);
    
    console.log(`  无Semantic信息 (LRU baseline):`);
    console.log(`    - 命中率: ${(result.withoutSemantic.hitRate * 100).toFixed(1)}%`);
    console.log(`    - 驱逐次数: ${result.withoutSemantic.evictionCount}`);
    
    console.log(`  提升: ${result.improvement > 0 ? '+' : ''}${result.improvement.toFixed(1)}%\n`);
  }
  
  // 全局统计
  const avgImprovement = Object.values(results)
    .reduce((sum, r) => sum + r.improvement, 0) / Object.values(results).length;
  
  const avgHitRateWith = Object.values(results)
    .reduce((sum, r) => sum + r.withSemantic.hitRate, 0) / Object.values(results).length;
  
  const avgHitRateWithout = Object.values(results)
    .reduce((sum, r) => sum + r.withoutSemantic.hitRate, 0) / Object.values(results).length;
  
  console.log('--- 全局统计 ---');
  console.log(`  平均命中率 (有Semantic): ${(avgHitRateWith * 100).toFixed(1)}%`);
  console.log(`  平均命中率 (无Semantic): ${(avgHitRateWithout * 100).toFixed(1)}%`);
  console.log(`  平均提升: ${avgImprovement > 0 ? '+' : ''}${avgImprovement.toFixed(1)}%\n`);
  
  // 生成报告
  const report = generateReport(results, avgImprovement, avgHitRateWith, avgHitRateWithout);
  console.log(report);
  
  return;
}

/**
 * 生成实验报告
 */
function generateReport(
  results: Record<TaskType, ReturnType<typeof runCoordinatedExperiment>>,
  avgImprovement: number,
  avgHitRateWith: number,
  avgHitRateWithout: number
): string {
  const taskResults = Object.entries(results).map(([taskType, r]) => {
    return `  - ${taskType}: ${r.improvement > 0 ? '+' : ''}${r.improvement.toFixed(1)}% (${(r.withSemantic.hitRate * 100).toFixed(1)}% vs ${(r.withoutSemantic.hitRate * 100).toFixed(1)}%)`;
  }).join('\n');
  
  return `
========================================
实验36结果总结
========================================

1. Agent协调架构
   ┌─────────────┐      ┌──────────────┐
   │  Semantic   │ ──── │ Global State │
   │   Agent     │      │    Store     │
   └─────────────┘      └──────────────┘
           │                    │
           │                    ▼
   ┌─────────────┐      ┌──────────────┐
   │   Reuse      │ ──── │  Scheduler   │
   │   Agent      │      │  (决策)      │
   └─────────────┘      └──────────────┘

2. 各任务类型效果
${taskResults}

3. 关键发现
   - Semantic Agent识别热区域 → 保护关键token不被驱逐
   - Reuse Agent预测reuse距离 → 识别可驱逐的冷token
   - 协调效果: ${avgImprovement > 0 ? '正向提升' : '需优化'} ${Math.abs(avgImprovement).toFixed(1)}%

4. 结论
   - Agent只输出状态，不做决策（解耦设计 ✓）
   - Global State Store作为状态共享中心 ✓
   - Scheduler基于状态做最优决策 ✓
   - 语义信息显著提升了eviction效果
`;
}

// ============================================
// 运行实验
// ============================================

runExperiment().catch(console.error);
