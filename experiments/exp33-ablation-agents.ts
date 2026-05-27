/**
 * exp33 - Ablation Study: 各Agent贡献度分析
 * 
 * 实验目标：
 * 逐个替换/禁用Agent，测量系统性能变化
 * 
 * 实验设计：
 * - Full system: 所有Agent都启用
 * - w/o Semantic: 禁用Semantic Agent
 * - w/o Reuse: 禁用Reuse Agent
 * - w/o Communication: 禁用Communication Agent
 * - w/o Placement: 禁用Placement Agent
 * 
 * 每种配置跑100个请求，测量：
 * - 质量评分
 * - 延迟
 * - 内存使用
 * - SLO满足率
 */

import {
  GlobalStateStore,
  RuntimeScheduler,
  SystemTaskType,
  SchedulerDecision,
  TokenReusePrediction,
  SemanticRegion
} from '../src/runtime/index.js';

// ============================================
// 配置
// ============================================

interface AblationConfig {
  name: string;
  enabledAgents: Set<'semantic' | 'reuse' | 'communication' | 'placement'>;
}

const ABLATION_CONFIGS: AblationConfig[] = [
  { name: 'Full system', enabledAgents: new Set(['semantic', 'reuse', 'communication', 'placement']) },
  { name: 'w/o Semantic', enabledAgents: new Set(['reuse', 'communication', 'placement']) },
  { name: 'w/o Reuse', enabledAgents: new Set(['semantic', 'communication', 'placement']) },
  { name: 'w/o Communication', enabledAgents: new Set(['semantic', 'reuse', 'placement']) },
  { name: 'w/o Placement', enabledAgents: new Set(['semantic', 'reuse', 'communication']) }
];

const REQUESTS_PER_CONFIG = 100;

// ============================================
// 模拟数据生成器
// ============================================

/**
 * 生成模拟请求
 */
interface SimulatedRequest {
  id: string;
  taskType: SystemTaskType;
  tokenCount: number;
  sloLatencyMs: number;
  arrivalTime: number;
}

/**
 * 生成批量请求
 */
function generateRequests(count: number): SimulatedRequest[] {
  const taskTypes: SystemTaskType[] = ['math', 'code', 'qa', 'conversation'];
  const requests: SimulatedRequest[] = [];
  
  for (let i = 0; i < count; i++) {
    requests.push({
      id: `req_${i}`,
      taskType: taskTypes[Math.floor(Math.random() * taskTypes.length)],
      tokenCount: 50 + Math.floor(Math.random() * 150),
      sloLatencyMs: 500 + Math.floor(Math.random() * 500),
      arrivalTime: Date.now() + i * 10
    });
  }
  
  return requests;
}

/**
 * 生成模拟语义状态
 */
function generateSemanticState(
  taskType: SystemTaskType,
  tokenCount: number
): {
  semantic: {
    activeRegions: SemanticRegion[];
    workingSetTokens: number[];
    reasoningFocus: string;
    generationProgress: number;
    taskPhase: 'prefill' | 'decode';
    attentionSinkTokens: number[];
  };
} {
  const regions: SemanticRegion[] = [];
  const regionCount = Math.min(3, Math.floor(tokenCount / 10));
  
  for (let i = 0; i < regionCount; i++) {
    const startIdx = i * Math.floor(tokenCount / regionCount);
    const endIdx = (i + 1) * Math.floor(tokenCount / regionCount);
    
    regions.push({
      id: i,
      name: `Region_${i}`,
      tokenIndices: Array.from({ length: endIdx - startIdx }, (_, j) => startIdx + j),
      importance: 0.7 + Math.random() * 0.3,
      coherence: 0.6 + Math.random() * 0.4,
      queryRelevance: Math.random(),
      layerCoverage: Array.from({ length: 32 }, () => Math.random() * 0.5 + 0.5)
    });
  }
  
  return {
    semantic: {
      activeRegions: regions,
      workingSetTokens: Array.from({ length: tokenCount }, (_, i) => i),
      reasoningFocus: taskType === 'code' ? 'induction' : 'retrieval',
      generationProgress: Math.random(),
      taskPhase: Math.random() > 0.5 ? 'prefill' : 'decode',
      attentionSinkTokens: [0, 1, 2]
    }
  };
}

/**
 * 生成模拟重用状态
 */
function generateReuseState(tokenCount: number): {
  reuse: {
    tokenPredictions: Map<number, TokenReusePrediction>;
    layerPredictions: Map<number, unknown>;
    lastAccessTime: Map<number, number>;
    accessCount: Map<number, number>;
    reuseDistanceDistribution: number[];
  };
} {
  const tokenPredictions = new Map<number, TokenReusePrediction>();
  
  for (let i = 0; i < tokenCount; i++) {
    const distance = Math.abs(i - (tokenCount - 1));
    const reuseProb = Math.exp(-distance / 20) * (0.5 + Math.random() * 0.5);
    
    tokenPredictions.set(i, {
      tokenIndex: i,
      reuseDistance: distance,
      reuseProbability: reuseProb,
      confidence: 0.6 + Math.random() * 0.4,
      temporalPattern: distance < 10 ? 'temporal' : 'spatial'
    });
  }
  
  const layerPredictions = new Map<number, {
    layerIndex: number;
    avgReuseDistance: number;
    hotTokens: number[];
    accessFrequency: number;
    importanceScore: number;
  }>();
  
  for (let layer = 0; layer < 32; layer++) {
    layerPredictions.set(layer, {
      layerIndex: layer,
      avgReuseDistance: 5 + layer * 0.5,
      hotTokens: Array.from({ length: 10 }, (_, i) => tokenCount - 1 - i).filter(i => i >= 0),
      accessFrequency: Math.exp(-layer / 10),
      importanceScore: 1 - layer / 64
    });
  }
  
  return {
    reuse: {
      tokenPredictions,
      layerPredictions,
      lastAccessTime: new Map(Array.from({ length: tokenCount }, (_, i) => [i, Date.now() - i * 10])),
      accessCount: new Map(Array.from({ length: tokenCount }, (_, i) => [i, Math.floor(Math.random() * 10) + 1])),
      reuseDistanceDistribution: Array.from({ length: 20 }, () => Math.random())
    }
  };
}

/**
 * 生成模拟通信状态
 */
function generateCommunicationState(tokenCount: number): {
  communication: {
    tokenAccessCosts: Map<number, number>;
    layerAccessCosts: Map<number, number>;
    bandwidthUtilization: number;
    congestionLevel: 'low' | 'medium' | 'high';
    estimatedTransferLatency: number;
    availableBandwidthBytesPerMs: number;
    totalBandwidthBytesPerMs: number;
    pendingTransfers: number;
  };
} {
  const tokenAccessCosts = new Map<number, number>();
  
  for (let i = 0; i < tokenCount; i++) {
    const location = i < 50 ? 'gpu_hbm' : i < 80 ? 'cpu_ram' : 'compressed';
    let cost: number;
    switch (location) {
      case 'gpu_hbm': cost = 0.1; break;
      case 'cpu_ram': cost = 10; break;
      default: cost = 20;
    }
    tokenAccessCosts.set(i, cost + Math.random() * 10);
  }
  
  const layerAccessCosts = new Map<number, number>();
  for (let layer = 0; layer < 32; layer++) {
    layerAccessCosts.set(layer, 0.1 + layer * 0.05);
  }
  
  return {
    communication: {
      tokenAccessCosts,
      layerAccessCosts,
      bandwidthUtilization: 0.5 + Math.random() * 0.3,
      congestionLevel: Math.random() > 0.7 ? 'medium' : 'low',
      estimatedTransferLatency: 50 + Math.random() * 100,
      availableBandwidthBytesPerMs: 80 + Math.random() * 20,
      totalBandwidthBytesPerMs: 100,
      pendingTransfers: Math.floor(Math.random() * 10)
    }
  };
}

/**
 * 生成模拟放置状态
 */
function generatePlacementState(tokenCount: number): {
  placement: {
    tokenLocations: Map<number, 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted'>;
    layerLocations: Map<number, 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted'>;
    memoryUtilization: { gpuHBM: number; cpuRAM: number; remote: number; compressed: number };
    migrationQueue: unknown[];
    kvSizes: Map<number, number>;
    lastMigrationTime: number;
  };
} {
  const tokenLocations = new Map<number, 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted'>();
  
  for (let i = 0; i < tokenCount; i++) {
    if (i < 50) {
      tokenLocations.set(i, 'gpu_hbm');
    } else if (i < 80) {
      tokenLocations.set(i, 'cpu_ram');
    } else {
      tokenLocations.set(i, Math.random() > 0.5 ? 'remote_gpu' : 'compressed');
    }
  }
  
  const layerLocations = new Map<number, 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted'>();
  for (let layer = 0; layer < 32; layer++) {
    layerLocations.set(layer, layer < 16 ? 'gpu_hbm' : 'cpu_ram');
  }
  
  return {
    placement: {
      tokenLocations,
      layerLocations,
      memoryUtilization: {
        gpuHBM: 0.7 + Math.random() * 0.2,
        cpuRAM: 0.4 + Math.random() * 0.2,
        remote: 0.2 + Math.random() * 0.2,
        compressed: 0.1 + Math.random() * 0.1
      },
      migrationQueue: [],
      kvSizes: new Map(Array.from({ length: tokenCount }, (_, i) => [i, 1024 + Math.floor(Math.random() * 512)])),
      lastMigrationTime: Date.now() - 1000
    }
  };
}

// ============================================
// 运行单配置实验
// ============================================

interface ConfigResult {
  configName: string;
  enabledAgents: string[];
  results: {
    requestId: string;
    quality: number;
    latency: number;
    memory: number;
    sloMet: boolean;
    decision: SchedulerDecision | null;
  }[];
  summary: {
    avgQuality: number;
    avgLatency: number;
    avgMemory: number;
    sloSatisfactionRate: number;
    avgEvictCount: number;
    avgTransmitCount: number;
  };
}

/**
 * 运行单个配置实验
 */
function runConfigExperiment(config: AblationConfig): ConfigResult {
  const results: ConfigResult['results'] = [];
  const requests = generateRequests(REQUESTS_PER_CONFIG);
  
  // 创建Scheduler并设置启用的Agent
  const store = new GlobalStateStore({
    maxMemoryBytes: 16 * 1024 * 1024 * 1024,
    bandwidthBytesPerMs: 100,
    sloLatencyMs: 1000
  });
  
  const scheduler = new RuntimeScheduler(store);
  
  // 设置启用的Agent
  scheduler.enableAllAgents();
  if (!config.enabledAgents.has('semantic')) scheduler.disableAgent('semantic');
  if (!config.enabledAgents.has('reuse')) scheduler.disableAgent('reuse');
  if (!config.enabledAgents.has('communication')) scheduler.disableAgent('communication');
  if (!config.enabledAgents.has('placement')) scheduler.disableAgent('placement');
  
  for (const request of requests) {
    // 更新状态
    const semanticData = generateSemanticState(request.taskType, request.tokenCount);
    const reuseData = generateReuseState(request.tokenCount);
    const commData = generateCommunicationState(request.tokenCount);
    const placementData = generatePlacementState(request.tokenCount);
    
    store.setTaskType(request.taskType);
    store.updateSemantic(semanticData.semantic);
    store.updateReuse(reuseData.reuse);
    store.updateCommunication(commData.communication);
    store.updatePlacement(placementData.placement);
    
    // 执行调度
    const decision = scheduler.schedule();
    
    // 记录结果
    const sloMet = decision.latencyEstimate <= request.sloLatencyMs;
    
    results.push({
      requestId: request.id,
      quality: decision.qualityEstimate,
      latency: decision.latencyEstimate,
      memory: decision.memoryEstimate,
      sloMet,
      decision
    });
  }
  
  // 计算汇总
  const avgQuality = results.reduce((sum, r) => sum + r.quality, 0) / results.length;
  const avgLatency = results.reduce((sum, r) => sum + r.latency, 0) / results.length;
  const avgMemory = results.reduce((sum, r) => sum + r.memory, 0) / results.length;
  const sloSatisfactionRate = results.filter(r => r.sloMet).length / results.length;
  const avgEvictCount = results.reduce((sum, r) => sum + (r.decision?.evictTokens.length ?? 0), 0) / results.length;
  const avgTransmitCount = results.reduce((sum, r) => 
    sum + (r.decision?.transmitKV.reduce((s, t) => s + t.tokens.length, 0) ?? 0), 0) / results.length;
  
  return {
    configName: config.name,
    enabledAgents: Array.from(config.enabledAgents),
    results,
    summary: {
      avgQuality,
      avgLatency,
      avgMemory,
      sloSatisfactionRate,
      avgEvictCount,
      avgTransmitCount
    }
  };
}

// ============================================
// 运行完整实验
// ============================================

/**
 * 运行完整Ablation实验
 */
function runAblationExperiment(): ConfigResult[] {
  console.log('='.repeat(60));
  console.log('Ablation Study: Agent Contribution Analysis');
  console.log('='.repeat(60));
  console.log();
  console.log(`Requests per config: ${REQUESTS_PER_CONFIG}`);
  console.log(`Total configs: ${ABLATION_CONFIGS.length}`);
  console.log();
  
  const allResults: ConfigResult[] = [];
  
  for (const config of ABLATION_CONFIGS) {
    console.log(`\n🔄 Running: ${config.name}`);
    console.log(`   Enabled agents: ${Array.from(config.enabledAgents).join(', ') || 'none'}`);
    
    const result = runConfigExperiment(config);
    allResults.push(result);
    
    console.log(`   Avg Quality: ${(result.summary.avgQuality * 100).toFixed(1)}%`);
    console.log(`   Avg Latency: ${result.summary.avgLatency.toFixed(2)}ms`);
    console.log(`   SLO Satisfaction: ${(result.summary.sloSatisfactionRate * 100).toFixed(1)}%`);
    console.log(`   Avg Evict: ${result.summary.avgEvictCount.toFixed(1)}`);
    console.log(`   Avg Transmit: ${result.summary.avgTransmitCount.toFixed(1)}`);
  }
  
  return allResults;
}

/**
 * 生成Ablation报告
 */
function generateAblationReport(results: ConfigResult[]): string {
  const report: string[] = [
    '# exp33 - Ablation Study: Agent Contribution Analysis',
    '',
    '## 实验概述',
    '',
    '**实验目标**: 分析各Agent对系统整体性能的贡献',
    '',
    '**实验设计**:',
    '- 对每个配置运行100个模拟请求',
    '- 测量质量、延迟、内存使用和SLO满足率',
    '- 对比禁用各Agent后的性能变化',
    '',
    '## 配置列表',
    '',
    '| 配置 | 启用的Agent |',
    '|------|------------|'
  ];
  
  for (const r of results) {
    report.push(`| ${r.configName} | ${r.enabledAgents.join(', ') || 'none'} |`);
  }
  
  // 汇总表格
  report.push('');
  report.push('## 性能对比');
  report.push('');
  report.push('| 配置 | Quality | Latency (ms) | Memory (MB) | SLO Rate | Evict | Transmit |');
  report.push('|------|---------|--------------|-------------|----------|-------|----------|');
  
  const fullSystem = results.find(r => r.configName === 'Full system')!;
  
  for (const r of results) {
    const qualityDelta = ((r.summary.avgQuality - fullSystem.summary.avgQuality) * 100).toFixed(1);
    const latencyDelta = (r.summary.avgLatency - fullSystem.summary.avgLatency).toFixed(2);
    const sloDelta = ((r.summary.sloSatisfactionRate - fullSystem.summary.sloSatisfactionRate) * 100).toFixed(1);
    
    report.push(
      `| ${r.configName} | ${(r.summary.avgQuality * 100).toFixed(1)}% (${qualityDelta > 0 ? '+' : ''}${qualityDelta}%) | ` +
      `${r.summary.avgLatency.toFixed(2)} (${latencyDelta > 0 ? '+' : ''}${latencyDelta}) | ` +
      `${(r.summary.avgMemory / 1024 / 1024).toFixed(1)} | ` +
      `${(r.summary.sloSatisfactionRate * 100).toFixed(1)}% (${sloDelta > 0 ? '+' : ''}${sloDelta}%) | ` +
      `${r.summary.avgEvictCount.toFixed(1)} | ` +
      `${r.summary.avgTransmitCount.toFixed(1)} |`
    );
  }
  
  // 贡献度分析
  report.push('');
  report.push('## Agent贡献度分析');
  report.push('');
  report.push('基于 Full system vs 各 w/o 配置的对比：');
  report.push('');
  
  for (const r of results) {
    if (r.configName === 'Full system') continue;
    
    const agent = r.configName.replace('w/o ', '');
    const qualityLoss = (fullSystem.summary.avgQuality - r.summary.avgQuality) * 100;
    const sloLoss = (fullSystem.summary.sloSatisfactionRate - r.summary.sloSatisfactionRate) * 100;
    const latencyIncrease = r.summary.avgLatency - fullSystem.summary.avgLatency;
    
    report.push(`### 禁用 ${agent} 的影响`);
    report.push('');
    report.push(`- 质量下降: ${qualityLoss.toFixed(1)}%`);
    report.push(`- SLO满足率下降: ${sloLoss.toFixed(1)}%`);
    report.push(`- 延迟增加: ${latencyIncrease.toFixed(2)}ms`);
    report.push('');
  }
  
  // 排名
  report.push('## 综合排名');
  report.push('');
  
  const ranked = [...results].sort((a, b) => {
    // 综合评分：质量*0.4 + SLO满足率*0.4 - 延迟*0.1 - 驱逐数*0.1
    const scoreA = a.summary.avgQuality * 0.4 + a.summary.sloSatisfactionRate * 0.4 
                 - a.summary.avgLatency * 0.001 - a.summary.avgEvictCount * 0.01;
    const scoreB = b.summary.avgQuality * 0.4 + b.summary.sloSatisfactionRate * 0.4 
                 - b.summary.avgLatency * 0.001 - b.summary.avgEvictCount * 0.01;
    return scoreB - scoreA;
  });
  
  report.push('| 排名 | 配置 | 综合评分 |');
  report.push('|------|------|----------|');
  
  ranked.forEach((r, i) => {
    const score = (r.summary.avgQuality * 0.4 + r.summary.sloSatisfactionRate * 0.4 
                 - r.summary.avgLatency * 0.001 - r.summary.avgEvictCount * 0.01).toFixed(3);
    report.push(`| ${i + 1} | ${r.configName} | ${score} |`);
  });
  
  // 结论
  report.push('');
  report.push('## 结论');
  report.push('');
  
  const mostImportant = ranked[0];
  const leastImportant = ranked[ranked.length - 1];
  
  report.push(`1. **最重要组件**: ${mostImportant.configName}（综合评分最高）`);
  report.push(`2. **最不重要组件**: ${leastImportant.configName}（综合评分最低）`);
  report.push('');
  
  // 计算各Agent的平均贡献
  const agentContributions: Record<string, { quality: number; slo: number; latency: number }> = {};
  
  for (const r of results) {
    if (r.configName === 'Full system') continue;
    
    const agent = r.configName.replace('w/o ', '');
    agentContributions[agent] = {
      quality: fullSystem.summary.avgQuality - r.summary.avgQuality,
      slo: fullSystem.summary.sloSatisfactionRate - r.summary.sloSatisfactionRate,
      latency: r.summary.avgLatency - fullSystem.summary.avgLatency
    };
  }
  
  report.push('3. **各Agent平均贡献度**:');
  report.push('');
  
  const sortedAgents = Object.entries(agentContributions)
    .sort((a, b) => (b[1].quality + b[1].slo) - (a[1].quality + a[1].slo));
  
  for (const [agent, contrib] of sortedAgents) {
    report.push(`   - ${agent}: 质量+${(contrib.quality * 100).toFixed(1)}%, SLO+${(contrib.slo * 100).toFixed(1)}%, 延迟-${contrib.latency.toFixed(2)}ms`);
  }
  
  report.push('');
  report.push('## 建议');
  report.push('');
  report.push('1. 优先保证关键Agent的正常运行');
  report.push('2. 对于资源受限场景，可考虑禁用贡献度较低的Agent');
  report.push('3. 各Agent协同工作才能发挥最大效果');
  
  return report.join('\n');
}

// ============================================
// 运行实验
// ============================================

const results = runAblationExperiment();
const report = generateAblationReport(results);

console.log('\n' + '='.repeat(60));
console.log('Ablation Experiment Report');
console.log('='.repeat(60));
console.log(report);

// 保存报告
import { writeFileSync, mkdirSync } from 'fs';

try {
  mkdirSync('./kvcache-lab/logs', { recursive: true });
  writeFileSync('./logs/exp33-ablation-agents.md', report);
  console.log('\n✅ Report saved to ./kvcache-lab/logs/exp33-ablation-agents.md');
} catch (e) {
  console.error('Failed to save report:', e);
}

export { runAblationExperiment, generateAblationReport, ABLATION_CONFIGS };
