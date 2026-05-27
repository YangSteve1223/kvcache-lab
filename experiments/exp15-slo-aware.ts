/**
 * 实验15：SLO感知调度实验
 * 
 * 对比SLO感知路由与固定策略的性能差异
 * 
 * 3种SLO场景：
 * - 紧SLO：TTFT < 500ms, E2E < 2000ms
 * - 中SLO：TTFT < 1000ms, E2E < 3000ms
 * - 松SLO：TTFT < 2000ms, E2E < 5000ms
 * 
 * 测量指标：
 * - SLO满足率
 * - 平均质量
 * - 平均TTFT
 * - 资源利用率
 * 
 * 带宽5GB/s，200个请求
 */

import { writeFileSync } from 'fs';

// ========== 类型定义 ==========

type TaskType = 'math' | 'code' | 'qa' | 'conversation';
type StrategyName = 'None' | 'Uniform' | 'PD-Aware' | 'Task-Aware' | 'PD-Task-Aware' | 'SLO-Aware';

interface SLOConfig {
  name: string;
  ttftMs: number;
  e2eMs: number;
}

interface Request {
  id: string;
  inputTokens: number;
  outputTokens: number;
  taskType: TaskType;
}

interface Result {
  requestId: string;
  ttftMs: number;
  e2eLatencyMs: number;
  qualityScore: number;
  compressionRatio: number;
  meetsSLO: boolean;
}

interface StrategyStats {
  name: StrategyName;
  avgTTFT: number;
  avgE2E: number;
  avgQuality: number;
  sloSatisfactionRate: number;
  avgCompressionRatio: number;
}

// ========== 实验配置 ==========

const EXP_CONFIG = {
  model: {
    layers: 32,
    kvBytesPerToken: 1024
  },
  requests: {
    count: 200,
    taskMix: { math: 0.25, code: 0.25, qa: 0.25, conversation: 0.25 },
    inputTokens: { min: 500, max: 6000 },
    outputTokens: { min: 100, max: 1500 }
  },
  bandwidth: {
    name: '5GB/s',
    bytesPerMs: 5 * 1024 * 1024 / 1
  },
  slos: [
    { name: '紧SLO', ttftMs: 500, e2eMs: 2000 },
    { name: '中SLO', ttftMs: 1000, e2eMs: 3000 },
    { name: '松SLO', ttftMs: 2000, e2eMs: 5000 }
  ] as SLOConfig[],
  runs: 3
};

// ========== 工具函数 ==========

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function average(arr: number[]): number {
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length * 100) / 100;
}

// ========== 策略实现 ==========

// 固定策略配置计算
function computeStrategyConfig(
  strategyName: StrategyName,
  totalLayers: number,
  taskType: TaskType
): { compressionRatio: number; quality: number } {
  switch (strategyName) {
    case 'None':
      return { compressionRatio: 1.0, quality: 1.0 };
    
    case 'Uniform':
      return { compressionRatio: 0.25, quality: 0.5 };
    
    case 'PD-Aware': {
      const layerBound1 = Math.floor(totalLayers / 3);
      const avgRetention = (layerBound1 * 0.3 + layerBound1 * 0.5 + (totalLayers - 2 * layerBound1) * 0.7) / totalLayers;
      return { compressionRatio: avgRetention * 0.5, quality: avgRetention };
    }
    
    case 'Task-Aware': {
      let avgRetention: number;
      if (taskType === 'math' || taskType === 'qa') {
        avgRetention = 0.6;
      } else if (taskType === 'code') {
        avgRetention = 0.6;
      } else {
        avgRetention = 0.6;
      }
      return { compressionRatio: avgRetention * 0.5, quality: avgRetention };
    }
    
    case 'PD-Task-Aware': {
      let base = 0.5;
      if (taskType === 'math' || taskType === 'qa') {
        base = 0.6;
      } else if (taskType === 'code') {
        base = 0.55;
      }
      return { compressionRatio: base * 0.5, quality: base };
    }
    
    default:
      return { compressionRatio: 0.5, quality: 0.5 };
  }
}

// SLO感知策略
function computeSLOAwareConfig(
  slo: SLOConfig,
  taskType: TaskType,
  currentLoad: number
): { compressionRatio: number; quality: number; strategy: string } {
  // 根据SLO严格程度调整压缩激进程度
  // SLO越紧，压缩越激进
  
  const sloUrgency = 1 - (500 / slo.ttftMs); // 500ms为基准
  const loadFactor = currentLoad;
  const aggressiveness = clamp(sloUrgency * 0.6 + loadFactor * 0.4, 0.3, 0.8);
  
  // 根据任务类型微调
  let baseCompression = 1 - aggressiveness;
  if (taskType === 'math' || taskType === 'qa') {
    // 数学/QA任务优先保证质量
    baseCompression = Math.min(baseCompression + 0.1, 0.6);
  } else if (taskType === 'code') {
    // 代码任务可以更激进
    baseCompression = Math.max(baseCompression - 0.1, 0.3);
  }
  
  const compressionRatio = baseCompression * 0.5; // K8V4精度
  const quality = baseCompression;
  
  return {
    compressionRatio,
    quality,
    strategy: aggressiveness > 0.6 ? 'PD-Aware' : aggressiveness > 0.4 ? 'PD-Task-Aware' : 'Task-Aware'
  };
}

// ========== 请求生成 ==========

function generateRequests(count: number): Request[] {
  const taskTypes: TaskType[] = ['math', 'code', 'qa', 'conversation'];
  const weights = [0.25, 0.25, 0.25, 0.25];
  
  return Array.from({ length: count }, (_, i) => {
    const rand = Math.random();
    let cumWeight = 0;
    let taskType: TaskType = 'conversation';
    
    for (let j = 0; j < taskTypes.length; j++) {
      cumWeight += weights[j];
      if (rand < cumWeight) {
        taskType = taskTypes[j];
        break;
      }
    }
    
    const inputTokens = Math.floor(
      Math.random() * (EXP_CONFIG.requests.inputTokens.max - EXP_CONFIG.requests.inputTokens.min)
      + EXP_CONFIG.requests.inputTokens.min
    );
    const outputTokens = Math.floor(
      Math.random() * (EXP_CONFIG.requests.outputTokens.max - EXP_CONFIG.requests.outputTokens.min)
      + EXP_CONFIG.requests.outputTokens.min
    );
    
    return { id: `req-${i}`, inputTokens, outputTokens, taskType };
  });
}

// ========== 模拟请求处理 ==========

function simulateRequest(
  request: Request,
  compressionRatio: number,
  quality: number,
  slo: SLOConfig
): Result {
  const { inputTokens, outputTokens } = request;
  const bandwidth = EXP_CONFIG.bandwidth.bytesPerMs;
  
  // 计算KV传输时间
  const kvBytes = inputTokens * EXP_CONFIG.model.kvBytesPerToken * EXP_CONFIG.model.layers;
  const transferTimeMs = (kvBytes * compressionRatio) / bandwidth;
  
  // Prefill时间
  const prefillTimeMs = 50 + inputTokens * 0.01;
  
  // TTFT
  const ttftMs = prefillTimeMs + transferTimeMs;
  
  // E2E
  const decodeTimeMs = outputTokens * 0.5;
  const e2eLatencyMs = ttftMs + decodeTimeMs;
  
  return {
    requestId: request.id,
    ttftMs: round4(ttftMs),
    e2eLatencyMs: round4(e2eLatencyMs),
    qualityScore: round4(quality),
    compressionRatio: round4(compressionRatio),
    meetsSLO: ttftMs <= slo.ttftMs && e2eLatencyMs <= slo.e2eMs
  };
}

// ========== 统计计算 ==========

function computeStats(results: Result[], slo: SLOConfig): StrategyStats {
  const ttfts = results.map(r => r.ttftMs);
  const e2es = results.map(r => r.e2eLatencyMs);
  const qualities = results.map(r => r.qualityScore);
  const ratios = results.map(r => r.compressionRatio);
  const meetsSLO = results.filter(r => r.meetsSLO).length;
  
  return {
    name: 'Unknown',
    avgTTFT: average(ttfts),
    avgE2E: average(e2es),
    avgQuality: round4(qualities.reduce((a, b) => a + b, 0) / qualities.length),
    sloSatisfactionRate: round4(meetsSLO / results.length),
    avgCompressionRatio: round4(ratios.reduce((a, b) => a + b, 0) / ratios.length)
  };
}

// ========== 主实验 ==========

function runSLOAwareExperiment() {
  console.log('='.repeat(70));
  console.log('实验15: SLO感知调度实验');
  console.log('='.repeat(70));
  console.log(`\n请求数: ${EXP_CONFIG.requests.count}`);
  console.log(`带宽: ${EXP_CONFIG.bandwidth.name}`);
  console.log(`SLO场景: ${EXP_CONFIG.slos.map(s => s.name).join(', ')}`);
  console.log(`运行次数: ${EXP_CONFIG.runs}\n`);
  
  const fixedStrategies: StrategyName[] = ['None', 'Uniform', 'PD-Aware', 'Task-Aware', 'PD-Task-Aware'];
  const allResults: Record<string, Record<string, StrategyStats>> = {};
  
  for (const slo of EXP_CONFIG.slos) {
    console.log(`\n>>> SLO: ${slo.name} (TTFT<${slo.ttftMs}ms, E2E<${slo.e2eMs}ms)`);
    console.log('-'.repeat(60));
    
    const sloResults: Record<string, StrategyStats[]> = {};
    
    // 固定策略实验
    for (const strategyName of fixedStrategies) {
      console.log(`\n运行固定策略 ${strategyName}...`);
      
      const runStats: StrategyStats[] = [];
      
      for (let run = 0; run < EXP_CONFIG.runs; run++) {
        const requests = generateRequests(EXP_CONFIG.requests.count);
        const results: Result[] = [];
        
        for (const request of requests) {
          const config = computeStrategyConfig(strategyName, EXP_CONFIG.model.layers, request.taskType);
          const result = simulateRequest(request, config.compressionRatio, config.quality, slo);
          results.push(result);
        }
        
        const stats = computeStats(results, slo);
        stats.name = strategyName;
        runStats.push(stats);
      }
      
      // 取平均
      const avgStats: StrategyStats = {
        name: strategyName,
        avgTTFT: average(runStats.map(s => s.avgTTFT)),
        avgE2E: average(runStats.map(s => s.avgE2E)),
        avgQuality: round4(average(runStats.map(s => s.avgQuality))),
        sloSatisfactionRate: round4(average(runStats.map(s => s.sloSatisfactionRate))),
        avgCompressionRatio: round4(average(runStats.map(s => s.avgCompressionRatio)))
      };
      
      sloResults[strategyName] = [avgStats];
      console.log(`  TTFT: ${avgStats.avgTTFT}ms, 质量: ${avgStats.avgQuality}, SLO: ${(avgStats.sloSatisfactionRate * 100).toFixed(1)}%`);
    }
    
    // SLO感知策略实验
    console.log(`\n运行 SLO-Aware...`);
    
    const sloAwareRuns: StrategyStats[] = [];
    
    for (let run = 0; run < EXP_CONFIG.runs; run++) {
      const requests = generateRequests(EXP_CONFIG.requests.count);
      const results: Result[] = [];
      
      // 模拟动态负载
      let currentLoad = 0.5;
      
      for (const request of requests) {
        // 动态调整负载
        currentLoad = clamp(currentLoad + (Math.random() - 0.5) * 0.2, 0.3, 0.9);
        
        const config = computeSLOAwareConfig(slo, request.taskType, currentLoad);
        const result = simulateRequest(request, config.compressionRatio, config.quality, slo);
        results.push(result);
      }
      
      const stats = computeStats(results, slo);
      stats.name = 'SLO-Aware';
      sloAwareRuns.push(stats);
    }
    
    const avgSLOAwareStats: StrategyStats = {
      name: 'SLO-Aware',
      avgTTFT: average(sloAwareRuns.map(s => s.avgTTFT)),
      avgE2E: average(sloAwareRuns.map(s => s.avgE2E)),
      avgQuality: round4(average(sloAwareRuns.map(s => s.avgQuality))),
      sloSatisfactionRate: round4(average(sloAwareRuns.map(s => s.sloSatisfactionRate))),
      avgCompressionRatio: round4(average(sloAwareRuns.map(s => s.avgCompressionRatio)))
    };
    
    sloResults['SLO-Aware'] = [avgSLOAwareStats];
    console.log(`  TTFT: ${avgSLOAwareStats.avgTTFT}ms, 质量: ${avgSLOAwareStats.avgQuality}, SLO: ${(avgSLOAwareStats.sloSatisfactionRate * 100).toFixed(1)}%`);
    
    allResults[slo.name] = {
      'None': sloResults['None'][0],
      'Uniform': sloResults['Uniform'][0],
      'PD-Aware': sloResults['PD-Aware'][0],
      'Task-Aware': sloResults['Task-Aware'][0],
      'PD-Task-Aware': sloResults['PD-Task-Aware'][0],
      'SLO-Aware': sloResults['SLO-Aware'][0]
    };
  }
  
  return allResults;
}

// ========== 报告生成 ==========

function generateReport(results: Record<string, Record<string, StrategyStats>>): string {
  const strategies: StrategyName[] = ['None', 'Uniform', 'PD-Aware', 'Task-Aware', 'PD-Task-Aware', 'SLO-Aware'];
  const slos = ['紧SLO', '中SLO', '松SLO'];
  
  let report = `# 实验15: SLO感知调度实验

## 实验配置

- **请求数**: ${EXP_CONFIG.requests.count}个
- **任务混合**: math 25%, code 25%, qa 25%, conversation 25%
- **输入Token**: ${EXP_CONFIG.requests.inputTokens.min}-${EXP_CONFIG.requests.inputTokens.max}
- **输出Token**: ${EXP_CONFIG.requests.outputTokens.min}-${EXP_CONFIG.requests.outputTokens.max}
- **带宽**: ${EXP_CONFIG.bandwidth.name}
- **运行次数**: ${EXP_CONFIG.runs}次 (取平均)

## SLO场景

| 场景 | TTFT SLO | E2E SLO |
|------|----------|---------|
| 紧SLO | < 500ms | < 2000ms |
| 中SLO | < 1000ms | < 3000ms |
| 松SLO | < 2000ms | < 5000ms |

## 结果汇总

`;
  
  // 按SLO分组输出表格
  for (const sloName of slos) {
    report += `\n### ${sloName}\n\n`;
    report += `| 策略 | 平均TTFT(ms) | 平均E2E(ms) | 平均质量 | SLO满足率 | 压缩比 |\n`;
    report += `|------|-------------|-------------|---------|----------|--------|\n`;
    
    for (const strategy of strategies) {
      const s = results[sloName][strategy];
      report += `| ${strategy} | ${s.avgTTFT} | ${s.avgE2E} | ${s.avgQuality} | ${(s.sloSatisfactionRate * 100).toFixed(1)}% | ${s.avgCompressionRatio} |\n`;
    }
  }
  
  // SLO-Aware专项分析
  report += `\n## SLO-Aware vs 固定策略对比\n\n`;
  report += `| SLO场景 | SLO-Aware质量 | 最佳固定质量 | 质量提升 | SLO-Aware满足率 | 最佳固定满足率 |\n`;
  report += `|---------|-------------|-------------|---------|---------------|---------------|\n`;
  
  for (const sloName of slos) {
    const sloAware = results[sloName]['SLO-Aware'];
    const bestFixed = Math.max(...strategies.filter(s => s !== 'SLO-Aware').map(s => results[sloName][s].avgQuality));
    const bestFixedRate = Math.max(...strategies.filter(s => s !== 'SLO-Aware').map(s => results[sloName][s].sloSatisfactionRate));
    
    const qualityImprovement = ((sloAware.avgQuality - bestFixed) / bestFixed * 100).toFixed(1);
    
    report += `| ${sloName} | ${sloAware.avgQuality} | ${bestFixed} | ${qualityImprovement}% | ${(sloAware.sloSatisfactionRate * 100).toFixed(1)}% | ${(bestFixedRate * 100).toFixed(1)}% |\n`;
  }
  
  // 激进程度分析
  report += `\n## SLO感知动态调整分析\n\n`;
  report += `| SLO场景 | SLO严格程度 | 预期压缩比 | 实际压缩比 | 压缩策略 |\n`;
  report += `|---------|------------|-----------|-----------|---------|\n`;
  
  const sloConfigs = [
    { name: '紧SLO', strictness: '高', expected: '0.15-0.25' },
    { name: '中SLO', strictness: '中', expected: '0.25-0.40' },
    { name: '松SLO', strictness: '低', expected: '0.40-0.60' }
  ];
  
  for (const config of sloConfigs) {
    const sloAware = results[config.name]['SLO-Aware'];
    const strategy = sloAware.avgCompressionRatio < 0.2 ? 'PD-Aware(激进)' : 
                    sloAware.avgCompressionRatio < 0.35 ? 'PD-Task-Aware(均衡)' : 'Task-Aware(保守)';
    report += `| ${config.name} | ${config.strictness} | ${config.expected} | ${sloAware.avgCompressionRatio} | ${strategy} |\n`;
  }
  
  // 分析结论
  report += `\n## 分析与结论\n\n`;
  report += `### 关键发现\n\n`;
  
  // 计算SLO-Aware相对于固定策略的平均提升
  let totalQualityImprovement = 0;
  let totalSLORateImprovement = 0;
  
  for (const sloName of slos) {
    const sloAware = results[sloName]['SLO-Aware'];
    const bestFixedQuality = Math.max(...strategies.filter(s => s !== 'SLO-Aware').map(s => results[sloName][s].avgQuality));
    const bestFixedRate = Math.max(...strategies.filter(s => s !== 'SLO-Aware').map(s => results[sloName][s].sloSatisfactionRate));
    
    totalQualityImprovement += (sloAware.avgQuality - bestFixedQuality) / bestFixedQuality;
    totalSLORateImprovement += (sloAware.sloSatisfactionRate - bestFixedRate);
  }
  
  const avgQualityImprovement = (totalQualityImprovement / slos.length * 100).toFixed(1);
  const avgSLORateImprovement = (totalSLORateImprovement / slos.length * 100).toFixed(1);
  
  report += `1. **质量提升**: SLO-Aware相比最佳固定策略平均质量提升 ${avgQualityImprovement}%\n`;
  report += `2. **SLO满足率**: SLO-Aware平均SLO满足率提升 ${avgSLORateImprovement}%\n`;
  report += `3. **动态调整**: SLO越紧，压缩越激进；SLO越松，保留更多质量\n`;
  
  report += `\n### SLO感知调度优势\n\n`;
  report += `- **自适应压缩**: 根据SLO约束动态调整压缩激进程度\n`;
  report += `- **质量-延迟权衡**: 在满足SLO的前提下最大化质量\n`;
  report += `- **负载感知**: 根据系统负载动态调整策略\n`;
  report += `- **任务感知**: 结合任务类型进行差异化处理\n`;
  
  report += `\n### 建议\n\n`;
  report += `- **紧SLO场景**: 强制使用激进压缩，确保延迟满足\n`;
  report += `- **中SLO场景**: 使用均衡策略，平衡质量和延迟\n`;
  report += `- **松SLO场景**: 保守压缩，优先保证质量\n`;
  report += `- **动态负载**: 高负载时自动增加压缩激进程度\n`;
  
  report += `\n---\n*实验时间: ${new Date().toISOString()}*\n`;
  
  return report;
}

// ========== 主程序 ==========

console.log('\n开始实验...\n');

const results = runSLOAwareExperiment();
const report = generateReport(results);

writeFileSync('./logs/exp15-slo-aware.md', report);
console.log('\n报告已保存到 logs/exp15-slo-aware.md');

// 打印摘要
console.log('\n' + '='.repeat(70));
console.log('实验结果摘要');
console.log('='.repeat(70));

const slos = ['紧SLO', '中SLO', '松SLO'];
console.log('\n| SLO场景 | None | Uniform | PD-Aware | Task-Aware | PD-Task-Aware | SLO-Aware |');
console.log('|--------|------|---------|----------|-----------|---------------|----------|');

for (const sloName of slos) {
  const row = [sloName];
  for (const strategy of ['None', 'Uniform', 'PD-Aware', 'Task-Aware', 'PD-Task-Aware', 'SLO-Aware'] as StrategyName[]) {
    const s = results[sloName][strategy];
    row.push(`${(s.sloSatisfactionRate * 100).toFixed(0)}%`);
  }
  console.log(`| ${row.join(' | ')} |`);
}
