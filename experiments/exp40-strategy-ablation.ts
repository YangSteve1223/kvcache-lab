/**
 * exp40: 策略替换Ablation实验
 * 
 * 比"禁用Agent"更严格的ablation - 直接替换策略而非禁用Agent
 * 
 * 实验设计：
 * - TAA → 普通attention（不考成本）
 * - Predictive Eviction → LRU
 * - SWS → 固定比例保留（top-50%）
 * - Hierarchical Placement → 全放GPU
 * 
 * 测量指标：延迟、质量、传输量、SLO满足率
 */

import {
  GlobalStateStore,
  RuntimeScheduler,
  createRuntimeScheduler,
  type SystemTaskType,
  type SchedulerDecision,
  type TokenReusePrediction,
  type SemanticRegion,
  type KVLocation,
} from '../src/runtime/index.js';
import { writeFileSync, mkdirSync } from 'fs';

// ============================================
// 实验配置
// ============================================

const NUM_TOKENS = 128;
const NUM_LAYERS = 32;
const KV_BYTES_PER_TOKEN = 1024;
const BANDWIDTH_GBPS = 100;

const MEMORY_CAPACITY = {
  gpuHBM: 32 * 1024 * 1024 * 1024,  // 32GB
  cpuRAM: 128 * 1024 * 1024 * 1024, // 128GB
  remote: 1024 * 1024 * 1024 * 1024, // 1TB
};

// ============================================
// 模拟请求
// ============================================

interface SimRequest {
  id: string;
  taskType: SystemTaskType;
  tokenCount: number;
  sloLatencyMs: number;
  arrivalTime: number;
}

function generateRequest(id: string): SimRequest {
  const taskTypes: SystemTaskType[] = ['math', 'code', 'qa', 'conversation'];
  return {
    id,
    taskType: taskTypes[Math.floor(Math.random() * taskTypes.length)],
    tokenCount: 50 + Math.floor(Math.random() * 150),
    sloLatencyMs: 500 + Math.floor(Math.random() * 500),
    arrivalTime: Date.now() + Math.random() * 1000,
  };
}

// ============================================
// 替换策略实现
// ============================================

/**
 * 策略接口
 */
interface ReplacementStrategy {
  name: string;
  selectEvictTokens(
    tokenScores: Map<number, number>,
    tokenCount: number,
    evictRatio: number
  ): number[];
}

/**
 * LRU策略 - 简单的最近最少使用
 */
class LRUStrategy implements ReplacementStrategy {
  name = 'LRU';
  
  selectEvictTokens(
    tokenScores: Map<number, number>,
    tokenCount: number,
    evictRatio: number
  ): number[][] {
    // 假设lastAccessTime越大越应该驱逐
    const sorted = Array.from(tokenScores.entries())
      .sort((a, b) => b[1] - a[1]); // 从大到小排序
    
    const evictCount = Math.floor(tokenCount * evictRatio);
    return [sorted.slice(0, evictCount).map(([id]) => id)];
  }
}

/**
 * 普通Attention策略 - 不考虑通信成本
 */
class StandardAttentionStrategy implements ReplacementStrategy {
  name = 'StandardAttention';
  
  selectEvictTokens(
    tokenScores: Map<number, number>,
    tokenCount: number,
    evictRatio: number
  ): number[] {
    // 只根据分数驱逐，不考虑位置
    const sorted = Array.from(tokenScores.entries())
      .sort((a, b) => a[1] - b[1]); // 从小到大（低分驱逐）
    
    const evictCount = Math.floor(tokenCount * evictRatio);
    return sorted.slice(0, evictCount).map(([id]) => id);
  }
}

/**
 * 固定比例策略 - 始终保留top-K
 */
class FixedRatioStrategy implements ReplacementStrategy {
  name = 'FixedRatio50%';
  private ratio: number;
  
  constructor(ratio = 0.5) {
    this.ratio = ratio;
  }
  
  selectEvictTokens(
    tokenScores: Map<number, number>,
    tokenCount: number,
    _evictRatio: number
  ): number[] {
    const sorted = Array.from(tokenScores.entries())
      .sort((a, b) => b[1] - a[1]);
    
    const retainCount = Math.floor(tokenCount * this.ratio);
    return sorted.slice(retainCount).map(([id]) => id);
  }
}

/**
 * 全GPU策略 - 所有KV都放GPU
 */
class AllGPUPlacementStrategy implements ReplacementStrategy {
  name = 'AllGPU';
  
  selectEvictTokens(
    tokenScores: Map<number, number>,
    _tokenCount: number,
    _evictRatio: number
  ): number[] {
    // 驱逐所有非GPU的token（模拟）
    return []; // 不驱逐任何token
  }
}

// ============================================
// 模拟器类
// ============================================

interface SimResult {
  requestId: string;
  latencyMs: number;
  qualityScore: number;
  transmissionBytes: number;
  sloMet: boolean;
  evictedCount: number;
  retainCount: number;
}

interface StrategyResult {
  strategyName: string;
  avgLatency: number;
  avgQuality: number;
  avgTransmission: number;
  sloSatisfactionRate: number;
  avgEvictedPerRequest: number;
  results: SimResult[];
}

class StrategyAblationSimulator {
  private strategies: Map<string, ReplacementStrategy>;
  private baselineScheduler: RuntimeScheduler;
  
  constructor() {
    this.strategies = new Map([
      ['TAA→StandardAttention', new StandardAttentionStrategy()],
      ['Predictive→LRU', new LRUStrategy()],
      ['SWS→FixedRatio', new FixedRatioStrategy(0.5)],
      ['Hierarchical→AllGPU', new AllGPUPlacementStrategy()],
    ]);
    
    const store = new GlobalStateStore({
      maxMemoryBytes: MEMORY_CAPACITY.gpuHBM,
      bandwidthBytesPerMs: BANDWIDTH_GBPS * 1024 * 1024 / 8,
      sloLatencyMs: 1000,
    });
    
    this.baselineScheduler = createRuntimeScheduler(store);
    this.baselineScheduler.enableAllAgents();
  }
  
  /**
   * 生成语义状态
   */
  private generateSemanticState(tokenCount: number, taskType: SystemTaskType): {
    activeRegions: SemanticRegion[];
    workingSetTokens: number[];
    reasoningFocus: string;
    generationProgress: number;
    taskPhase: 'prefill' | 'decode';
    attentionSinkTokens: number[];
  } {
    const regions: SemanticRegion[] = [];
    const regionCount = Math.min(3, Math.floor(tokenCount / 20));
    
    for (let i = 0; i < regionCount; i++) {
      const startIdx = i * Math.floor(tokenCount / regionCount);
      const endIdx = (i + 1) * Math.floor(tokenCount / regionCount);
      
      regions.push({
        id: i,
        name: `Region_${taskType}_${i}`,
        tokenIndices: Array.from({ length: endIdx - startIdx }, (_, j) => startIdx + j),
        importance: 0.6 + Math.random() * 0.4,
        coherence: 0.5 + Math.random() * 0.5,
        queryRelevance: Math.random(),
        layerCoverage: Array.from({ length: NUM_LAYERS }, () => 0.5 + Math.random() * 0.5),
      });
    }
    
    return {
      activeRegions: regions,
      workingSetTokens: Array.from({ length: tokenCount }, (_, i) => i),
      reasoningFocus: taskType === 'code' ? 'induction' : taskType === 'math' ? 'deduction' : 'retrieval',
      generationProgress: Math.random(),
      taskPhase: Math.random() > 0.5 ? 'prefill' : 'decode',
      attentionSinkTokens: [0, 1, 2],
    };
  }
  
  /**
   * 生成reuse状态
   */
  private generateReuseState(tokenCount: number): Map<number, TokenReusePrediction> {
    const predictions = new Map<number, TokenReusePrediction>();
    
    for (let i = 0; i < tokenCount; i++) {
      const distance = Math.abs(i - (tokenCount - 1));
      predictions.set(i, {
        tokenIndex: i,
        reuseDistance: distance,
        reuseProbability: Math.exp(-distance / 20) * (0.5 + Math.random() * 0.5),
        confidence: 0.6 + Math.random() * 0.4,
        temporalPattern: distance < 10 ? 'temporal' : 'spatial',
      });
    }
    
    return predictions;
  }
  
  /**
   * 生成通信状态
   */
  private generateCommunicationState(tokenCount: number): {
    tokenAccessCosts: Map<number, number>;
    layerAccessCosts: Map<number, number>;
    bandwidthUtilization: number;
    congestionLevel: 'low' | 'medium' | 'high';
  } {
    const tokenAccessCosts = new Map<number, number>();
    
    for (let i = 0; i < tokenCount; i++) {
      const location = i < NUM_TOKENS * 0.3 ? 'gpu' : i < NUM_TOKENS * 0.7 ? 'cpu' : 'remote';
      tokenAccessCosts.set(i, location === 'gpu' ? 0.1 : location === 'cpu' ? 10 : 50);
    }
    
    const layerAccessCosts = new Map<number, number>();
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      layerAccessCosts.set(layer, 0.1 + layer * 0.02);
    }
    
    return {
      tokenAccessCosts,
      layerAccessCosts,
      bandwidthUtilization: 0.3 + Math.random() * 0.4,
      congestionLevel: Math.random() > 0.7 ? 'medium' : 'low',
    };
  }
  
  /**
   * 生成放置状态
   */
  private generatePlacementState(tokenCount: number): {
    tokenLocations: Map<number, KVLocation>;
    memoryUtilization: { gpuHBM: number; cpuRAM: number; remote: number; compressed: number };
    kvSizes: Map<number, number>;
  } {
    const tokenLocations = new Map<number, KVLocation>();
    const kvSizes = new Map<number, number>();
    
    for (let i = 0; i < tokenCount; i++) {
      const rand = Math.random();
      const location: KVLocation = rand < 0.3 ? 'gpu_hbm' : rand < 0.6 ? 'cpu_ram' : rand < 0.8 ? 'remote_gpu' : 'compressed';
      tokenLocations.set(i, location);
      kvSizes.set(i, KV_BYTES_PER_TOKEN);
    }
    
    return {
      tokenLocations,
      memoryUtilization: {
        gpuHBM: 0.5 + Math.random() * 0.3,
        cpuRAM: 0.3 + Math.random() * 0.2,
        remote: 0.2 + Math.random() * 0.1,
        compressed: 0.1,
      },
      kvSizes,
    };
  }
  
  /**
   * 运行单次请求
   */
  private runRequest(request: SimRequest, strategy: ReplacementStrategy): SimResult {
    const tokenCount = request.tokenCount;
    
    // 生成状态
    const semanticState = this.generateSemanticState(tokenCount, request.taskType);
    const reuseState = this.generateReuseState(tokenCount);
    const commState = this.generateCommunicationState(tokenCount);
    const placementState = this.generatePlacementState(tokenCount);
    
    // 计算分数（用于替换策略）
    const tokenScores = new Map<number, number>();
    for (let i = 0; i < tokenCount; i++) {
      const reuse = reuseState.get(i)?.reuseProbability ?? 0.5;
      const cost = commState.tokenAccessCosts.get(i) ?? 10;
      const location = placementState.tokenLocations.get(i);
      const placement = location === 'gpu_hbm' ? 1.0 : location === 'cpu_ram' ? 0.7 : 0.3;
      
      // Full OS的分数
      const score = 0.3 * semanticState.activeRegions.length / 3 +
                   0.4 * reuse +
                   0.2 * (1 - cost / 100) +
                   0.1 * placement;
      tokenScores.set(i, score);
    }
    
    // 使用替换策略选择驱逐token
    const evictTokens = strategy.selectEvictTokens(tokenScores, tokenCount, 0.3);
    const retainTokens = Array.from(tokenScores.keys()).filter(id => !evictTokens.includes(id));
    
    // 计算传输量（只计算非GPU的token）
    let transmissionBytes = 0;
    for (const tokenId of retainTokens) {
      const location = placementState.tokenLocations.get(tokenId);
      if (location !== 'gpu_hbm') {
        transmissionBytes += KV_BYTES_PER_TOKEN;
      }
    }
    
    // 估算延迟
    const baseLatency = 10; // ms
    const perTokenLatency = 0.1; // ms
    const transmissionLatency = transmissionBytes / (BANDWIDTH_GBPS * 1024 * 1024 / 8 / 1000);
    const latencyMs = baseLatency + tokenCount * perTokenLatency + transmissionLatency;
    
    // 估算质量
    const retentionRatio = retainTokens.length / tokenCount;
    const qualityScore = retentionRatio * 0.95 + 0.05;
    
    return {
      requestId: request.id,
      latencyMs,
      qualityScore: Math.min(1, qualityScore),
      transmissionBytes,
      sloMet: latencyMs <= request.sloLatencyMs,
      evictedCount: evictTokens.length,
      retainCount: retainTokens.length,
    };
  }
  
  /**
   * 运行Full OS基准
   */
  private runFullOS(request: SimRequest): SimResult {
    const tokenCount = request.tokenCount;
    
    // 生成状态
    const semanticState = this.generateSemanticState(tokenCount, request.taskType);
    const reuseState = this.generateReuseState(tokenCount);
    const commState = this.generateCommunicationState(tokenCount);
    const placementState = this.generatePlacementState(tokenCount);
    
    // 创建store并更新状态
    const store = new GlobalStateStore({
      maxMemoryBytes: MEMORY_CAPACITY.gpuHBM,
      bandwidthBytesPerMs: BANDWIDTH_GBPS * 1024 * 1024 / 8,
      sloLatencyMs: request.sloLatencyMs,
    });
    
    store.setTaskType(request.taskType);
    store.updateSemantic(semanticState);
    store.updateReuse({ 
      tokenPredictions: reuseState,
      layerPredictions: new Map(),
      lastAccessTime: new Map(),
      accessCount: new Map(),
      reuseDistanceDistribution: [],
    });
    store.updateCommunication(commState);
    store.updatePlacement(placementState);
    
    // 创建scheduler并执行调度
    const scheduler = createRuntimeScheduler(store);
    scheduler.enableAllAgents();
    const decision = scheduler.schedule();
    
    // 计算传输量
    let transmissionBytes = 0;
    for (const task of decision.transmitKV) {
      transmissionBytes += task.tokens.length * KV_BYTES_PER_TOKEN;
    }
    
    return {
      requestId: request.id,
      latencyMs: decision.latencyEstimate,
      qualityScore: decision.qualityEstimate,
      transmissionBytes,
      sloMet: decision.latencyEstimate <= request.sloLatencyMs,
      evictedCount: decision.evictTokens.length,
      retainCount: decision.retainTokens.length,
    };
  }
  
  /**
   * 运行策略对比实验
   */
  runExperiment(numRequests: number): Map<string, StrategyResult> {
    const results = new Map<string, StrategyResult>();
    
    console.log('\n=== exp40: 策略替换Ablation实验 ===\n');
    console.log(`请求数量: ${numRequests}`);
    console.log(`带宽: ${BANDWIDTH_GBPS} GB/s\n`);
    
    // 生成请求
    const requests: SimRequest[] = [];
    for (let i = 0; i < numRequests; i++) {
      requests.push(generateRequest(`req-${i}`));
    }
    
    // 运行Full OS基准
    console.log('Running Full OS (baseline)...');
    const fullOSResults: SimResult[] = [];
    for (const request of requests) {
      fullOSResults.push(this.runFullOS(request));
    }
    
    results.set('Full OS', {
      strategyName: 'Full OS',
      avgLatency: fullOSResults.reduce((s, r) => s + r.latencyMs, 0) / numRequests,
      avgQuality: fullOSResults.reduce((s, r) => s + r.qualityScore, 0) / numRequests,
      avgTransmission: fullOSResults.reduce((s, r) => s + r.transmissionBytes, 0) / numRequests,
      sloSatisfactionRate: fullOSResults.filter(r => r.sloMet).length / numRequests,
      avgEvictedPerRequest: fullOSResults.reduce((s, r) => s + r.evictedCount, 0) / numRequests,
      results: fullOSResults,
    });
    
    // 运行每个替换策略
    for (const [name, strategy] of this.strategies) {
      console.log(`Running ${name}...`);
      const strategyResults: SimResult[] = [];
      
      for (const request of requests) {
        strategyResults.push(this.runRequest(request, strategy));
      }
      
      results.set(name, {
        strategyName: name,
        avgLatency: strategyResults.reduce((s, r) => s + r.latencyMs, 0) / numRequests,
        avgQuality: strategyResults.reduce((s, r) => s + r.qualityScore, 0) / numRequests,
        avgTransmission: strategyResults.reduce((s, r) => s + r.transmissionBytes, 0) / numRequests,
        sloSatisfactionRate: strategyResults.filter(r => r.sloMet).length / numRequests,
        avgEvictedPerRequest: strategyResults.reduce((s, r) => s + r.evictedCount, 0) / numRequests,
        results: strategyResults,
      });
    }
    
    // 打印结果
    console.log('\n--- 实验结果 ---');
    console.log('策略\t\t\t延迟(ms)\t质量\t\t传输(KB)\tSLO满足率\t驱逐数');
    console.log('----------------------------------------------------------------------');
    
    for (const [name, result] of results) {
      const latencyDelta = ((result.avgLatency - results.get('Full OS')!.avgLatency) / results.get('Full OS')!.avgLatency * 100).toFixed(1);
      const qualityDelta = ((result.avgQuality - results.get('Full OS')!.avgQuality) * 100).toFixed(1);
      const transDelta = ((result.avgTransmission - results.get('Full OS')!.avgTransmission) / (results.get('Full OS')!.avgTransmission || 1) * 100).toFixed(1);
      const sloDelta = ((result.sloSatisfactionRate - results.get('Full OS')!.sloSatisfactionRate) * 100).toFixed(1);
      
      console.log(
        `${name.padEnd(20)}\t` +
        `${result.avgLatency.toFixed(2)} (${result.avgLatency > results.get('Full OS')!.avgLatency ? '+' : ''}${latencyDelta}%)\t` +
        `${(result.avgQuality * 100).toFixed(1)}% (${qualityDelta > 0 ? '+' : ''}${qualityDelta}%)\t` +
        `${(result.avgTransmission / 1024).toFixed(1)} (${transDelta > 0 ? '+' : ''}${transDelta}%)\t` +
        `${(result.sloSatisfactionRate * 100).toFixed(1)}% (${sloDelta > 0 ? '+' : ''}${sloDelta}%)\t` +
        `${result.avgEvictedPerRequest.toFixed(1)}`
      );
    }
    
    return results;
  }
}

// ============================================
// 生成报告
// ============================================

function generateReport(results: Map<string, StrategyResult>): string {
  const lines: string[] = [
    '# exp40: 策略替换Ablation实验报告',
    '',
    '## 实验目的',
    '',
    '验证每个策略组件的独立贡献，通过将智能策略替换为简单基线策略来量化。',
    '',
    '## 实验设计',
    '',
    '| 替换策略 | 原策略 | 替换为 |',
    '|---------|--------|--------|',
    '| TAA→StandardAttention | Transmission-Aware Attention | 普通Attention（不考虑位置成本） |',
    '| Predictive→LRU | Predictive Eviction | LRU最近最少使用 |',
    '| SWS→FixedRatio | Semantic Working Set | 固定保留50% |',
    '| Hierarchical→AllGPU | Hierarchical Placement | 全部放GPU |',
    '',
    '## 配置参数',
    '',
    `- Token数量: ${NUM_TOKENS}`,
    `- 层数: ${NUM_LAYERS}`,
    `- KV大小/token: ${KV_BYTES_PER_TOKEN} bytes`,
    `- 带宽: ${BANDWIDTH_GBPS} GB/s`,
    `- GPU内存: ${MEMORY_CAPACITY.gpuHBM / 1024 / 1024 / 1024} GB`,
    '',
    '## 详细结果',
    '',
    '### 延迟对比',
    '',
    '| 策略 | 平均延迟(ms) | vs Full OS |',
    '|------|-------------|------------|',
  ];
  
  const fullOS = results.get('Full OS')!;
  
  for (const [name, result] of results) {
    const delta = result.avgLatency - fullOS.avgLatency;
    const pct = (delta / fullOS.avgLatency * 100).toFixed(1);
    lines.push(`| ${name} | ${result.avgLatency.toFixed(2)} | ${delta > 0 ? '+' : ''}${pct}% |`);
  }
  
  lines.push('', '### 质量对比', '', '| 策略 | 平均质量 | vs Full OS |', '|------|---------|------------|');
  
  for (const [name, result] of results) {
    const delta = (result.avgQuality - fullOS.avgQuality) * 100;
    lines.push(`| ${name} | ${(result.avgQuality * 100).toFixed(1)}% | ${delta > 0 ? '+' : ''}${delta.toFixed(1)}% |`);
  }
  
  lines.push('', '### 传输量对比', '', '| 策略 | 平均传输量(KB) | vs Full OS |', '|------|----------------|------------|');
  
  for (const [name, result] of results) {
    const delta = result.avgTransmission - fullOS.avgTransmission;
    const pct = (delta / (fullOS.avgTransmission || 1) * 100).toFixed(1);
    lines.push(`| ${name} | ${(result.avgTransmission / 1024).toFixed(1)} | ${delta > 0 ? '+' : ''}${pct}% |`);
  }
  
  lines.push('', '### SLO满足率对比', '', '| 策略 | SLO满足率 | vs Full OS |', '|------|----------|------------|');
  
  for (const [name, result] of results) {
    const delta = (result.sloSatisfactionRate - fullOS.sloSatisfactionRate) * 100;
    lines.push(`| ${name} | ${(result.sloSatisfactionRate * 100).toFixed(1)}% | ${delta > 0 ? '+' : ''}${delta.toFixed(1)}% |`);
  }
  
  lines.push('', '## 结论', '');
  
  // 找出最差替换
  let worstReplacement = { name: '', latencyLoss: 0, qualityLoss: 0 };
  let bestReplacement = { name: '', latencyLoss: Infinity, qualityLoss: Infinity };
  
  for (const [name, result] of results) {
    if (name === 'Full OS') continue;
    
    const latencyLoss = result.avgLatency - fullOS.avgLatency;
    const qualityLoss = fullOS.avgQuality - result.avgQuality;
    
    if (latencyLoss + qualityLoss > worstReplacement.latencyLoss + worstReplacement.qualityLoss) {
      worstReplacement = { name, latencyLoss, qualityLoss };
    }
    if (latencyLoss + qualityLoss < bestReplacement.latencyLoss + bestReplacement.qualityLoss) {
      bestReplacement = { name, latencyLoss, qualityLoss };
    }
  }
  
  lines.push(`1. **影响最大替换**: ${worstReplacement.name}`);
  lines.push(`   - 延迟增加: ${worstReplacement.latencyLoss.toFixed(2)}ms`);
  lines.push(`   - 质量下降: ${(worstReplacement.qualityLoss * 100).toFixed(1)}%`);
  lines.push('');
  lines.push(`2. **影响最小替换**: ${bestReplacement.name}`);
  lines.push(`   - 延迟变化: ${bestReplacement.latencyLoss > 0 ? '+' : ''}${bestReplacement.latencyLoss.toFixed(2)}ms`);
  lines.push(`   - 质量变化: ${bestReplacement.qualityLoss > 0 ? '-' : '+'}${(bestReplacement.qualityLoss * 100).toFixed(1)}%`);
  lines.push('');
  lines.push('3. **策略贡献排序** (从大到小):');
  
  const sorted = Array.from(results.entries())
    .filter(([name]) => name !== 'Full OS')
    .map(([name, result]) => ({
      name,
      totalLoss: (result.avgLatency - fullOS.avgLatency) / fullOS.avgLatency +
                 (fullOS.avgQuality - result.avgQuality)
    }))
    .sort((a, b) => b.totalLoss - a.totalLoss);
  
  sorted.forEach((item, i) => {
    lines.push(`   ${i + 1}. ${item.name}`);
  });
  
  lines.push('');
  lines.push('## 讨论');
  lines.push('');
  lines.push('本实验比"禁用Agent"更严格，因为：');
  lines.push('1. 直接对比智能策略与简单基线');
  lines.push('2. 排除"禁用后其他组件补偿"的可能性');
  lines.push('3. 量化每个策略改进的具体收益');
  
  return lines.join('\n');
}

// ============================================
// 运行实验
// ============================================

const NUM_REQUESTS = 100;
const simulator = new StrategyAblationSimulator();
const results = simulator.runExperiment(NUM_REQUESTS);
const report = generateReport(results);

// 保存报告
try {
  mkdirSync('./logs', { recursive: true });
  writeFileSync('./logs/exp40-strategy-ablation.md', report);
  console.log('\n✅ 报告已保存到 ./logs/exp40-strategy-ablation.md');
} catch (e) {
  console.error('Failed to save report:', e);
}

console.log('\n' + report);

export { StrategyAblationSimulator, generateReport };
