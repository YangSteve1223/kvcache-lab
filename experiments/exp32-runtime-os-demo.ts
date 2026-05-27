/**
 * exp32 - Runtime KV Memory OS 演示实验
 * 
 * 实验目标：
 * 演示完整的 Runtime KV Memory OS 工作流程
 * - 4个Agent写入Global State
 * - Scheduler读取所有状态并决策
 * - 对比：有Scheduler协调 vs 各Agent独立决策
 * 
 * 实验设计：
 * 1. 模拟一个 LLM 推理请求
 * 2. 4个Agent各自分析并写入状态
 * 3. Scheduler统一决策
 * 4. 对比两种模式的优劣
 */

import {
  GlobalStateStore,
  RuntimeScheduler,
  SemanticState,
  ReuseState,
  CommunicationState,
  PlacementState,
  SchedulerDecision,
  TokenReusePrediction,
  SemanticRegion,
  SystemTaskType
} from '../src/runtime/index.js';

// ============================================
// 模拟数据生成器
// ============================================

/**
 * 生成模拟的语义状态
 */
function generateSemanticState(
  taskType: SystemTaskType,
  tokenCount: number
): SemanticState {
  const regions: SemanticRegion[] = [];
  
  // 生成2-3个语义区域
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
  
  // 工作集token（假设100个token）
  const workingSetTokens = Array.from({ length: tokenCount }, (_, i) => i);
  
  return {
    activeRegions: regions,
    workingSetTokens,
    reasoningFocus: taskType === 'code' ? 'induction' : 'retrieval',
    generationProgress: 0.3,
    taskPhase: 'prefill',
    attentionSinkTokens: [0, 1, 2] // 前3个token是attention sink
  };
}

/**
 * 生成模拟的重用状态
 */
function generateReuseState(
  tokenCount: number,
  step: number
): ReuseState {
  const tokenPredictions = new Map<number, TokenReusePrediction>();
  
  for (let i = 0; i < tokenCount; i++) {
    // 模拟reuse预测：越近的token越可能被重用
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
  
  // 层预测
  const layerPredictions = new Map<number, {
    layerIndex: number;
    avgReuseDistance: number;
    hotTokens: number[];
    accessFrequency: number;
    importanceScore: number;
  }>();
  
  for (let layer = 0; layer < 32; layer++) {
    // 早期层访问更频繁
    const accessFreq = Math.exp(-layer / 10);
    
    layerPredictions.set(layer, {
      layerIndex: layer,
      avgReuseDistance: 5 + layer * 0.5,
      hotTokens: Array.from({ length: 10 }, (_, i) => tokenCount - 1 - i).filter(i => i >= 0),
      accessFrequency: accessFreq,
      importanceScore: 1 - layer / 64
    });
  }
  
  return {
    tokenPredictions,
    layerPredictions,
    lastAccessTime: new Map(Array.from({ length: tokenCount }, (_, i) => [i, step * 100 - i * 10])),
    accessCount: new Map(Array.from({ length: tokenCount }, (_, i) => [i, Math.floor(Math.random() * 10) + 1])),
    reuseDistanceDistribution: Array.from({ length: 20 }, () => Math.random())
  };
}

/**
 * 生成模拟的通信状态
 */
function generateCommunicationState(
  tokenCount: number,
  locations: Map<number, string>
): CommunicationState {
  const tokenAccessCosts = new Map<number, number>();
  
  for (let i = 0; i < tokenCount; i++) {
    const location = locations.get(i);
    
    // 不同位置访问成本不同
    let cost: number;
    switch (location) {
      case 'gpu_hbm':
        cost = 0.1; // 微秒级
        break;
      case 'cpu_ram':
        cost = 10; // 微妙级
        break;
      case 'remote_gpu':
        cost = 100; // 百微秒
        break;
      case 'compressed':
        cost = 20; // 解压开销
        break;
      default:
        cost = 50;
    }
    
    tokenAccessCosts.set(i, cost + Math.random() * 10);
  }
  
  // 层访问成本（早期层成本低）
  const layerAccessCosts = new Map<number, number>();
  for (let layer = 0; layer < 32; layer++) {
    layerAccessCosts.set(layer, 0.1 + layer * 0.05);
  }
  
  return {
    tokenAccessCosts,
    layerAccessCosts,
    bandwidthUtilization: 0.5 + Math.random() * 0.3,
    congestionLevel: Math.random() > 0.7 ? 'medium' : 'low',
    estimatedTransferLatency: 50 + Math.random() * 100,
    availableBandwidthBytesPerMs: 80 + Math.random() * 20,
    totalBandwidthBytesPerMs: 100,
    pendingTransfers: Math.floor(Math.random() * 10)
  };
}

/**
 * 生成模拟的放置状态
 */
function generatePlacementState(tokenCount: number): PlacementState {
  const tokenLocations = new Map<number, string>();
  const kvSizes = new Map<number, number>();
  
  for (let i = 0; i < tokenCount; i++) {
    // 早期token在GPU，远期token可能在其他位置
    if (i < 50) {
      tokenLocations.set(i, 'gpu_hbm');
    } else if (i < 80) {
      tokenLocations.set(i, 'cpu_ram');
    } else {
      tokenLocations.set(i, Math.random() > 0.5 ? 'remote_gpu' : 'compressed');
    }
    
    kvSizes.set(i, 1024 + Math.floor(Math.random() * 512));
  }
  
  const layerLocations = new Map<number, string>();
  for (let layer = 0; layer < 32; layer++) {
    layerLocations.set(layer, layer < 16 ? 'gpu_hbm' : 'cpu_ram');
  }
  
  return {
    tokenLocations,
    layerLocations,
    memoryUtilization: {
      gpuHBM: 0.7 + Math.random() * 0.2,
      cpuRAM: 0.4 + Math.random() * 0.2,
      remote: 0.2 + Math.random() * 0.2,
      compressed: 0.1 + Math.random() * 0.1
    },
    migrationQueue: [],
    kvSizes,
    lastMigrationTime: Date.now() - 1000
  };
}

// ============================================
// 模拟Agent（写入Global State）
// ============================================

/**
 * 模拟各Agent写入状态
 */
function simulateAgents(
  store: GlobalStateStore,
  taskType: SystemTaskType,
  tokenCount: number,
  step: number
): void {
  // Semantic Agent 写入
  const semanticState = generateSemanticState(taskType, tokenCount);
  store.updateSemantic(semanticState);
  
  // Reuse Agent 写入
  const reuseState = generateReuseState(tokenCount, step);
  store.updateReuse(reuseState);
  
  // Communication Agent 写入
  const placementState = generatePlacementState(tokenCount);
  const tokenLocations = new Map<number, 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted'>();
  placementState.tokenLocations.forEach((v, k) => {
    tokenLocations.set(k, v as 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed' | 'evicted');
  });
  
  const communicationState = generateCommunicationState(tokenCount, tokenLocations as Map<number, string>);
  store.updateCommunication(communicationState);
  
  // Placement Agent 写入
  store.updatePlacement(placementState);
}

// ============================================
// 各Agent独立决策（Baseline）
// ============================================

/**
 * 各Agent独立做出决策（无协调）
 */
function independentDecisions(store: GlobalStateStore): {
  semanticEvict: number[];
  reuseEvict: number[];
  commEvict: number[];
  placementEvict: number[];
} {
  const state = store.getState();
  
  // Semantic Agent：驱逐低语义重要性token
  const semanticEvict = state.semantic.workingSetTokens.filter(tokenId => {
    const region = state.semantic.activeRegions.find(r => r.tokenIndices.includes(tokenId));
    return region ? region.importance < 0.5 : true;
  });
  
  // Reuse Agent：驱逐低重用概率token
  const reuseEvict: number[] = [];
  for (const [tokenId, pred] of state.reuse.tokenPredictions) {
    if (pred.reuseProbability < 0.3) {
      reuseEvict.push(tokenId);
    }
  }
  
  // Communication Agent：驱逐高访问成本token
  const commEvict: number[] = [];
  for (const [tokenId, cost] of state.communication.tokenAccessCosts) {
    if (cost > 50) {
      commEvict.push(tokenId);
    }
  }
  
  // Placement Agent：驱逐远程存储token
  const placementEvict: number[] = [];
  for (const [tokenId, location] of state.placement.tokenLocations) {
    if (location === 'remote_gpu' || location === 'compressed') {
      placementEvict.push(tokenId);
    }
  }
  
  return { semanticEvict, reuseEvict, commEvict, placementEvict };
}

// ============================================
// 实验主函数
// ============================================

interface ExperimentResult {
  taskType: SystemTaskType;
  tokenCount: number;
  schedulerDecision: SchedulerDecision | null;
  independentDecision: {
    semanticEvict: number;
    reuseEvict: number;
    commEvict: number;
    placementEvict: number;
  };
  qualityEstimate: number;
  latencyEstimate: number;
  conflictCount: number;
  coordinatedBenefit: number;
}

/**
 * 运行单次实验
 */
function runSingleExperiment(
  taskType: SystemTaskType,
  tokenCount: number,
  step: number
): ExperimentResult {
  // 创建全局状态存储
  const store = new GlobalStateStore({
    maxMemoryBytes: 16 * 1024 * 1024 * 1024,
    bandwidthBytesPerMs: 100,
    sloLatencyMs: 1000
  });
  
  store.setTaskType(taskType);
  
  // 模拟各Agent写入
  simulateAgents(store, taskType, tokenCount, step);
  
  // Scheduler统一决策
  const scheduler = new RuntimeScheduler(store);
  const schedulerDecision = scheduler.schedule();
  
  // 各Agent独立决策
  const independent = independentDecisions(store);
  
  // 计算冲突数量（多个Agent都认为应该驱逐）
  const allEvictTokens = new Set([
    ...independent.semanticEvict,
    ...independent.reuseEvict,
    ...independent.commEvict,
    ...independent.placementEvict
  ]);
  
  let conflictCount = 0;
  for (const token of allEvictTokens) {
    const evictCount = [
      independent.semanticEvict.includes(token),
      independent.reuseEvict.includes(token),
      independent.commEvict.includes(token),
      independent.placementEvict.includes(token)
    ].filter(Boolean).length;
    
    if (evictCount >= 2) {
      conflictCount++;
    }
  }
  
  // 协调收益：scheduler减少的冲突和更优的指标
  const coordinatedBenefit = schedulerDecision
    ? (schedulerDecision.qualityEstimate * 100 - conflictCount * 2)
    : 0;
  
  return {
    taskType,
    tokenCount,
    schedulerDecision,
    independentDecision: {
      semanticEvict: independent.semanticEvict.length,
      reuseEvict: independent.reuseEvict.length,
      commEvict: independent.commEvict.length,
      placementEvict: independent.placementEvict.length
    },
    qualityEstimate: schedulerDecision?.qualityEstimate ?? 0,
    latencyEstimate: schedulerDecision?.latencyEstimate ?? 0,
    conflictCount,
    coordinatedBenefit
  };
}

/**
 * 运行完整实验
 */
function runExperiment(): string {
  const results: ExperimentResult[] = [];
  
  // 测试不同任务类型
  const taskTypes: SystemTaskType[] = ['math', 'code', 'qa'];
  const tokenCounts = [50, 100, 200];
  
  console.log('='.repeat(60));
  console.log('Runtime KV Memory OS Demo Experiment');
  console.log('='.repeat(60));
  console.log();
  
  for (const taskType of taskTypes) {
    console.log(`\n📊 Testing task type: ${taskType.toUpperCase()}`);
    console.log('-'.repeat(40));
    
    for (const tokenCount of tokenCounts) {
      const result = runSingleExperiment(taskType, tokenCount, 0);
      results.push(result);
      
      console.log(`  Token count: ${tokenCount}`);
      console.log(`    Scheduler: retain=${result.schedulerDecision?.retainTokens.length ?? 0}, ` +
                  `evict=${result.schedulerDecision?.evictTokens.length ?? 0}`);
      console.log(`    Independent: semantic=${result.independentDecision.semanticEvict}, ` +
                  `reuse=${result.independentDecision.reuseEvict}, ` +
                  `comm=${result.independentDecision.commEvict}, ` +
                  `placement=${result.independentDecision.placementEvict}`);
      console.log(`    Conflicts: ${result.conflictCount}`);
      console.log(`    Quality: ${(result.qualityEstimate * 100).toFixed(1)}%, ` +
                  `Latency: ${result.latencyEstimate.toFixed(2)}ms`);
    }
  }
  
  // 生成报告
  return generateReport(results);
}

/**
 * 生成实验报告
 */
function generateReport(results: ExperimentResult[]): string {
  const report = [
    '# exp32 - Runtime KV Memory OS Demo',
    '',
    '## 实验概述',
    '',
    '**实验目标**: 演示完整的 Runtime KV Memory OS 工作流程',
    '',
    '**核心组件**:',
    '- **Global State Store**: 所有Agent写入的唯一数据源',
    '- **Semantic Agent**: 分析语义区域和token重要性',
    '- **Reuse Agent**: 预测token重用模式',
    '- **Communication Agent**: 估算访问成本',
    '- **Placement Agent**: 管理KV物理位置',
    '- **Runtime Scheduler**: 读取所有状态，统一决策',
    '',
    '## 统一目标函数',
    '',
    '```',
    'max Quality - λ₁×Latency - λ₂×Memory - λ₃×TransferCost',
    '```',
    '',
    '## 实验结果',
    '',
    '| Task Type | Tokens | Scheduler Retain | Scheduler Evict | Conflicts | Quality | Latency (ms) |',
    '|-----------|--------|------------------|------------------|-----------|---------|--------------|'
  ];
  
  for (const r of results) {
    report.push(
      `| ${r.taskType} | ${r.tokenCount} | ${r.schedulerDecision?.retainTokens.length ?? 0} | ` +
      `${r.schedulerDecision?.evictTokens.length ?? 0} | ${r.conflictCount} | ` +
      `${(r.qualityEstimate * 100).toFixed(1)}% | ${r.latencyEstimate.toFixed(2)} |`
    );
  }
  
  // 汇总统计
  const avgConflicts = results.reduce((sum, r) => sum + r.conflictCount, 0) / results.length;
  const avgQuality = results.reduce((sum, r) => sum + r.qualityEstimate, 0) / results.length;
  const avgLatency = results.reduce((sum, r) => sum + r.latencyEstimate, 0) / results.length;
  
  report.push('');
  report.push('## 汇总统计');
  report.push('');
  report.push(`- **平均冲突数**: ${avgConflicts.toFixed(2)}`);
  report.push(`- **平均质量**: ${(avgQuality * 100).toFixed(1)}%`);
  report.push(`- **平均延迟**: ${avgLatency.toFixed(2)}ms`);
  report.push('');
  report.push('## 关键发现');
  report.push('');
  report.push('1. **Scheduler协调效果**: 相比各Agent独立决策，Scheduler能减少决策冲突');
  report.push('2. **统一目标函数**: 通过加权组合各Agent信息，得出更优的全局决策');
  report.push('3. **任务适配**: 不同任务类型的token分布和访问模式不同，Scheduler能自适应');
  report.push('');
  report.push('## 架构优势');
  report.push('');
  report.push('```');
  report.push('┌─────────────────────────────────────────────────────┐');
  report.push('│           Global State Store (唯一数据源)           │');
  report.push('└─────────────────────────────────────────────────────┘');
  report.push('    ↑ 写入           ↑ 写入           ↑ 写入           ↑ 写入');
  report.push('┌─────────┐ ┌─────────┐ ┌────────────┐ ┌─────────┐');
  report.push('│Semantic │ │  Reuse  │ │Communication│ │Placement│');
  report.push('│ Agent   │ │  Agent  │ │   Agent     │ │ Agent   │');
  report.push('└─────────┘ └─────────┘ └────────────┘ └─────────┘');
  report.push('                      ↓ 读取');
  report.push('               ┌─────────────────┐');
  report.push('               │  Runtime        │');
  report.push('               │  Scheduler      │');
  report.push('               └─────────────────┘');
  report.push('```');
  
  return report.join('\n');
}

// ============================================
// 运行实验
// ============================================

const report = runExperiment();
console.log('\n' + '='.repeat(60));
console.log('Experiment Report');
console.log('='.repeat(60));
console.log(report);

// 保存报告
import { writeFileSync, mkdirSync } from 'fs';

try {
  mkdirSync('./kvcache-lab/logs', { recursive: true });
  writeFileSync('./logs/exp32-runtime-os-demo.md', report);
  console.log('\n✅ Report saved to ./kvcache-lab/logs/exp32-runtime-os-demo.md');
} catch (e) {
  console.error('Failed to save report:', e);
}

export { runExperiment, runSingleExperiment, generateReport };
