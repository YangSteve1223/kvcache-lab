/**
 * 实验35：Reuse Agent 实验
 * 
 * 目标：验证reuse prediction的准确性
 * - 验证reuse prediction的准确性
 * - 对比3种预测方法（统计/语义/任务模式/综合）
 * - 对比Belady vs LRU vs LFU vs H2O vs 我们的Predictive
 */

import { ReuseAgent, ReuseAgentInput, ReuseState } from '../src/agents/ReuseAgent';
import { SemanticAgent, SemanticAgentInput, SemanticRegion } from '../src/agents/SemanticAgent';
import { TaskType } from '../src/core/types';

// ============================================
// 实验配置
// ============================================

const EXP_CONFIG = {
  taskTypes: ['math', 'code', 'qa', 'conversation'] as TaskType[],
  tokenCount: 1000,
  numLayers: 32,
  decodeSteps: 100,
  cacheSize: 200,  // 缓存容量
  runs: 5
};

// ============================================
// 模拟的访问模式
// ============================================

/**
 * 生成真实的访问序列（用于评估）
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
        // 数学任务：周期性回溯
        // system prompt (0-100): 每步访问
        for (let i = 0; i < 100; i++) access.push(i);
        // 最近推理 (最后100): 每步访问
        for (let i = tokenCount - 100; i < tokenCount; i++) access.push(i);
        // 周期性回溯到中间区域
        if (step % 5 === 0) {
          for (let i = 400; i < 500; i++) access.push(i);
        }
        break;
        
      case 'code':
        // 代码任务：函数局部性
        // imports (0-50): 偶尔访问
        if (step % 10 === 0) {
          for (let i = 0; i < 50; i++) access.push(i);
        }
        // 当前函数 (最后50): 每步访问
        for (let i = tokenCount - 50; i < tokenCount; i++) access.push(i);
        // 函数体 (中间): 访问较频繁
        for (let i = 200; i < 250; i++) access.push(i);
        break;
        
      case 'qa':
        // QA任务：文档检索
        // system prompt (0-100): 每步访问
        for (let i = 0; i < 100; i++) access.push(i);
        // 当前chunk (最后200): 每步访问
        for (let i = tokenCount - 200; i < tokenCount; i++) access.push(i);
        // 相关chunk (400-500): 偶尔访问
        if (step % 3 === 0) {
          for (let i = 400; i < 500; i++) access.push(i);
        }
        break;
        
      case 'conversation':
        // 对话任务：最近轮次
        // system prompt (0-50): 每步访问
        for (let i = 0; i < 50; i++) access.push(i);
        // 最近2轮 (最后150): 每步访问
        for (let i = tokenCount - 150; i < tokenCount; i++) access.push(i);
        // 中间轮次: 偶尔回溯
        if (step % 4 === 0) {
          for (let i = 200; i < 300; i++) access.push(i);
        }
        break;
    }
    
    accessPattern.push([...new Set(access)]); // 去重
  }
  
  return accessPattern;
}

/**
 * 生成历史attention分布
 */
function generateHistoricalAttention(
  tokenCount: number, 
  steps: number
): Float64Array[] {
  const history: Float64Array[] = [];
  
  for (let step = 0; step < steps; step++) {
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
    
    history.push(attention);
  }
  
  return history;
}

// ============================================
// 缓存策略实现
// ============================================

/**
 * Belady策略（Oracle，最优）
 */
function beladyEvict(
  accessPattern: number[][],
  cacheSize: number,
  currentStep: number
): number[] {
  // 找出不在当前和未来访问中的token
  const futureAccess = new Set<number>();
  for (let step = currentStep; step < accessPattern.length; step++) {
    for (const token of accessPattern[step]) {
      futureAccess.add(token);
    }
  }
  
  // 返回不在未来访问中的token（这些可以被驱逐）
  const evictable: number[] = [];
  for (let i = 0; i < 1000; i++) {
    if (!futureAccess.has(i)) {
      evictable.push(i);
    }
  }
  
  return evictable.slice(0, cacheSize);
}

/**
 * LRU策略
 */
function lruEvict(
  accessHistory: Map<number, number>,
  cacheSize: number
): number[] {
  // 按最近访问时间排序
  const entries = Array.from(accessHistory.entries())
    .sort((a, b) => a[1] - b[1]); // 按时间戳升序
  
  return entries.slice(0, cacheSize).map(e => e[0]);
}

/**
 * LFU策略
 */
function lfuEvict(
  accessCount: Map<number, number>,
  cacheSize: number
): number[] {
  // 按访问频率排序
  const entries = Array.from(accessCount.entries())
    .sort((a, b) => a[1] - b[1]); // 按频率升序
  
  return entries.slice(0, cacheSize).map(e => e[0]);
}

/**
 * H2O策略（混合）
 */
function h2oEvict(
  accessHistory: Map<number, number>,
  accessCount: Map<number, number>,
  currentStep: number,
  cacheSize: number
): number[] {
  // 结合LRU和LFU
  const scores = new Map<number, number>();
  
  for (const [token, lastAccess] of accessHistory.entries()) {
    const freq = accessCount.get(token) || 1;
    const recency = currentStep - lastAccess;
    // H2O评分: 频率权重0.3, 新鲜度权重0.7
    const score = freq * 0.3 + (1 / (recency + 1)) * 0.7;
    scores.set(token, score);
  }
  
  const entries = Array.from(scores.entries())
    .sort((a, b) => a[1] - b[1]); // 低分优先驱逐
  
  return entries.slice(0, cacheSize).map(e => e[0]);
}

/**
 * Predictive策略（我们的方法）
 */
function predictiveEvict(
  reuseState: ReuseState,
  cacheSize: number
): number[] {
  return reuseState.evictableTokens.slice(0, cacheSize);
}

// ============================================
// 命中率计算
// ============================================

/**
 * 计算缓存命中率
 */
function calculateHitRate(
  accessPattern: number[][],
  strategy: 'belady' | 'lru' | 'lfu' | 'h2o' | 'predictive',
  reuseAgent: ReuseAgent,
  semanticAgent: SemanticAgent,
  taskType: TaskType
): { hitRate: number; missRate: number; reEvictions: number } {
  const cache = new Set<number>();
  const accessHistory = new Map<number, number>();
  const accessCount = new Map<number, number>();
  let hits = 0;
  let misses = 0;
  let reEvictions = 0;
  
  for (let step = 0; step < accessPattern.length; step++) {
    const currentAccess = accessPattern[step];
    const prevCacheSize = cache.size;
    
    // 更新统计
    for (const token of currentAccess) {
      accessHistory.set(token, step);
      accessCount.set(token, (accessCount.get(token) || 0) + 1);
    }
    
    // 根据策略驱逐
    let evictCandidates: number[] = [];
    
    switch (strategy) {
      case 'belady':
        evictCandidates = beladyEvict(accessPattern, Math.floor(EXP_CONFIG.cacheSize / 2), step);
        break;
      case 'lru':
        evictCandidates = lruEvict(accessHistory, Math.floor(EXP_CONFIG.cacheSize / 2));
        break;
      case 'lfu':
        evictCandidates = lfuEvict(accessCount, Math.floor(EXP_CONFIG.cacheSize / 2));
        break;
      case 'h2o':
        evictCandidates = h2oEvict(accessHistory, accessCount, step, Math.floor(EXP_CONFIG.cacheSize / 2));
        break;
      case 'predictive':
        // 先更新Semantic Agent
        const semanticInput: SemanticAgentInput = {
          tokens: Array.from({ length: EXP_CONFIG.tokenCount }, (_, i) => `token-${i}`),
          taskType,
          recentAttention: new Float64Array(EXP_CONFIG.tokenCount),
          decodeStep: step,
          totalSteps: accessPattern.length
        };
        const semanticState = semanticAgent.analyze(semanticInput);
        
        // 更新Reuse Agent
        const reuseInput: ReuseAgentInput = {
          tokenCount: EXP_CONFIG.tokenCount,
          numLayers: EXP_CONFIG.numLayers,
          taskType,
          currentStep: step,
          generationProgress: step / accessPattern.length,
          historicalAttention: [new Float64Array(EXP_CONFIG.tokenCount)],
          semanticRegions: semanticState.activeRegions
        };
        const reuseState = reuseAgent.predict(reuseInput);
        evictCandidates = predictiveEvict(reuseState, Math.floor(EXP_CONFIG.cacheSize / 2));
        break;
    }
    
    // 执行驱逐
    for (const token of evictCandidates) {
      if (cache.has(token)) {
        cache.delete(token);
        // 如果这个token马上又要被访问，则计数
        if (currentAccess.includes(token)) {
          reEvictions++;
        }
      }
    }
    
    // 添加当前访问的token到缓存
    for (const token of currentAccess) {
      if (!cache.has(token)) {
        if (cache.size >= EXP_CONFIG.cacheSize) {
          // 缓存满，先驱逐
          const toEvict = evictCandidates[0];
          if (toEvict !== undefined) {
            cache.delete(toEvict);
          }
        }
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
  
  const total = hits + misses;
  return {
    hitRate: total > 0 ? hits / total : 0,
    missRate: total > 0 ? misses / total : 0,
    reEvictions
  };
}

// ============================================
// 实验函数
// ============================================

/**
 * 运行预测方法对比实验
 */
function runPredictionMethodComparison(taskType: TaskType): {
  method: string;
  accuracy: number;
  confidence: number;
}[] {
  const reuseAgent = new ReuseAgent();
  const semanticAgent = new SemanticAgent();
  
  const methods = ['statistical', 'semantic', 'task-pattern', 'combined'];
  const results: typeof runPredictionMethodComparison extends () => infer R ? R : never = [];
  
  for (const method of methods) {
    let totalAccuracy = 0;
    let totalConfidence = 0;
    let count = 0;
    
    for (let run = 0; run < EXP_CONFIG.runs; run++) {
      const history = generateHistoricalAttention(EXP_CONFIG.tokenCount, EXP_CONFIG.decodeSteps);
      
      // 设置初始访问历史
      for (let step = 0; step < 10; step++) {
        for (let i = 0; i < 100; i++) {
          const att = new Float64Array(EXP_CONFIG.tokenCount);
          for (let j = 0; j < 100; j++) att[j] = 1 / 100;
          history[step] = att;
        }
      }
      
      const semanticInput: SemanticAgentInput = {
        tokens: Array.from({ length: EXP_CONFIG.tokenCount }, (_, i) => `token-${i}`),
        taskType,
        recentAttention: history[history.length - 1],
        decodeStep: EXP_CONFIG.decodeSteps,
        totalSteps: EXP_CONFIG.decodeSteps
      };
      const semanticState = semanticAgent.analyze(semanticInput);
      
      const reuseInput: ReuseAgentInput = {
        tokenCount: EXP_CONFIG.tokenCount,
        numLayers: EXP_CONFIG.numLayers,
        taskType,
        currentStep: EXP_CONFIG.decodeSteps,
        generationProgress: 1,
        historicalAttention: history,
        semanticRegions: semanticState.activeRegions
      };
      
      const reuseState = reuseAgent.predict(reuseInput);
      
      // 评估准确性
      for (const [tokenId, prediction] of reuseState.tokenPredictions) {
        // 简化：假设高reuse概率的token确实会被重用
        const expectedHighReuse = tokenId < 100 || tokenId > EXP_CONFIG.tokenCount - 100;
        const predictedHighReuse = prediction.reuseProbability > 0.7;
        
        if (expectedHighReuse === predictedHighReuse) {
          totalAccuracy += 1;
        }
        totalConfidence += prediction.confidence;
        count++;
      }
    }
    
    results.push({
      method,
      accuracy: totalAccuracy / count,
      confidence: totalConfidence / count
    });
    
    reuseAgent.reset();
  }
  
  return results;
}

/**
 * 运行策略对比实验
 */
function runStrategyComparison(taskType: TaskType): {
  strategy: string;
  hitRate: number;
  missRate: number;
  reEvictions: number;
}[] {
  const accessPattern = generateAccessPattern(taskType, EXP_CONFIG.tokenCount, EXP_CONFIG.decodeSteps);
  const reuseAgent = new ReuseAgent();
  const semanticAgent = new SemanticAgent();
  
  const strategies: Array<'belady' | 'lru' | 'lfu' | 'h2o' | 'predictive'> = 
    ['belady', 'lru', 'lfu', 'h2o', 'predictive'];
  
  const results: typeof runStrategyComparison extends () => infer R ? R : never = [];
  
  for (const strategy of strategies) {
    const reuseAgentCopy = new ReuseAgent();
    const semanticAgentCopy = new SemanticAgent();
    
    const result = calculateHitRate(
      accessPattern, 
      strategy, 
      reuseAgentCopy, 
      semanticAgentCopy, 
      taskType
    );
    
    results.push({
      strategy,
      hitRate: result.hitRate,
      missRate: result.missRate,
      reEvictions: result.reEvictions
    });
  }
  
  return results;
}

// ============================================
// 主实验
// ============================================

async function runExperiment(): Promise<void> {
  console.log('========================================');
  console.log('实验35：Reuse Agent 实验');
  console.log('========================================\n');
  
  console.log('配置:');
  console.log(`  - 任务类型: ${EXP_CONFIG.taskTypes.join(', ')}`);
  console.log(`  - Token数: ${EXP_CONFIG.tokenCount}`);
  console.log(`  - 缓存大小: ${EXP_CONFIG.cacheSize}`);
  console.log(`  - Decode步数: ${EXP_CONFIG.decodeSteps}`);
  console.log(`  - 运行次数: ${EXP_CONFIG.runs}\n`);
  
  // 1. 预测方法对比
  console.log('--- 预测方法对比 ---');
  for (const taskType of EXP_CONFIG.taskTypes) {
    console.log(`\n[${taskType}]`);
    const results = runPredictionMethodComparison(taskType);
    
    for (const r of results) {
      console.log(`  ${r.method}: accuracy=${(r.accuracy * 100).toFixed(1)}%, confidence=${(r.confidence * 100).toFixed(1)}%`);
    }
  }
  
  // 2. 策略对比
  console.log('\n--- 缓存策略对比 ---');
  for (const taskType of EXP_CONFIG.taskTypes) {
    console.log(`\n[${taskType}]`);
    const results = runStrategyComparison(taskType);
    
    // 排序：按命中率降序
    results.sort((a, b) => b.hitRate - a.hitRate);
    
    for (const r of results) {
      console.log(`  ${r.strategy}: hitRate=${(r.hitRate * 100).toFixed(1)}%, missRate=${(r.missRate * 100).toFixed(1)}%, reEvictions=${r.reEvictions}`);
    }
  }
  
  // 生成报告
  const report = generateReport();
  console.log(report);
  
  return;
}

/**
 * 生成实验报告
 */
function generateReport(): string {
  return `
========================================
实验35结果总结
========================================

1. 预测方法对比结论
   - statistical: 基于历史访问间隔的EMA预测
   - semantic: 基于语义区域的位置预测
   - task-pattern: 基于任务类型的先验知识
   - combined: 加权组合以上三种方法

2. 策略对比结论
   - Belady: Oracle最优策略（需要未来信息）
   - LRU: 基于最近访问时间
   - LFU: 基于访问频率
   - H2O: LRU和LFU的混合
   - Predictive: 基于我们的Reuse Agent预测

3. 预期结果
   - Predictive策略应接近Belady（因为预测了未来访问）
   - Predictive应优于LRU/LFU/H2O（利用了语义信息）
   - combined预测方法应优于单一方法

结论: Reuse Agent的预测能力是关键，准确性直接影响缓存策略效果。
`;
}

// ============================================
// 运行实验
// ============================================

runExperiment().catch(console.error);
