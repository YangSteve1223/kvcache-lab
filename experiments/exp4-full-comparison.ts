/**
 * 实验4：4策略完整对比（仿真）
 * 
 * 对比 None, Uniform, PD-Aware, Task-Aware 四种压缩策略
 */

import { PDSimulator, DEFAULT_CONFIG } from '../src/core/PDSimulator.ts';
import { CompressionOrchestrator } from '../src/compression/CompressionOrchestrator.ts';
import { NoneCompression } from '../src/compression/strategies/NoneCompression.ts';
import { UniformCompression } from '../src/compression/strategies/UniformCompression.ts';
import { PDAwareCompression } from '../src/compression/strategies/PDAwareCompression.ts';
import { TaskAwareCompression } from '../src/compression/strategies/TaskAwareCompression.ts';
import { ServingRequest, TaskType, CompressionConfig, LayerCompressionConfig } from '../src/core/types.ts';
import { writeFileSync } from 'fs';

/**
 * 实验配置
 */
const EXP_CONFIG = {
  model: {
    layers: 32,
    hidden: 4096,
    heads: 32
  },
  requests: {
    count: 200,
    taskMix: {
      math: 0.3,
      code: 0.3,
      qa: 0.3,
      conversation: 0.1
    },
    inputTokens: { min: 500, max: 8000 },
    outputTokens: { min: 100, max: 2000 }
  },
  network: {
    bandwidthBytesPerMs: 10 * 1024 * 1024 * 1024 / 1000 // 10GB/s
  },
  gpu: {
    memoryBytes: 80 * 1024 * 1024 * 1024 // 80GB
  },
  batch: {
    maxSize: 32
  },
  runs: 3 // 运行3次取平均
};

/**
 * 生成随机请求序列
 */
function generateRequests(count: number): ServingRequest[] {
  const taskTypes: TaskType[] = ['math', 'code', 'qa', 'conversation'];
  const weights = [0.3, 0.3, 0.3, 0.1];
  
  return Array.from({ length: count }, (_, i) => {
    // 根据权重随机选择任务类型
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
    
    return {
      id: `req-${i}`,
      inputTokens,
      outputTokens,
      taskType,
      prefixHash: Math.random() < 0.2 ? `prefix-${Math.floor(Math.random() * 20)}` : undefined,
      arrivalTimeMs: Date.now() + i * 100
    };
  });
}

/**
 * 创建压缩配置
 */
function createCompressionConfig(strategy: 'none' | 'uniform' | 'pd-aware' | 'task-aware', taskType: TaskType): CompressionConfig {
  const layers = EXP_CONFIG.model.layers;
  const pLayers: LayerCompressionConfig[] = [];
  const dLayers: LayerCompressionConfig[] = [];
  
  for (let i = 0; i < layers; i++) {
    let pRetention: number;
    let pKeyPrecision: number;
    let pValuePrecision: number;
    
    switch (strategy) {
      case 'none':
        pRetention = 1.0;
        pKeyPrecision = 16;
        pValuePrecision = 16;
        break;
      case 'uniform':
        pRetention = 0.5;
        pKeyPrecision = 16;
        pValuePrecision = 8;
        break;
      case 'pd-aware':
        // PD感知：低层激进压缩，高层保守
        const bound1 = Math.floor(layers / 3);
        const bound2 = Math.floor((2 * layers) / 3);
        if (i < bound1) {
          pRetention = 0.3;
        } else if (i < bound2) {
          pRetention = 0.5;
        } else {
          pRetention = 0.7;
        }
        pKeyPrecision = 8;
        pValuePrecision = 4;
        break;
      case 'task-aware':
        // 任务感知：根据任务类型调整
        if (taskType === 'math' || taskType === 'qa') {
          // 数学和QA：中高层重要
          const bound1 = Math.floor(layers / 3);
          if (i < bound1) {
            pRetention = 0.3;
          } else {
            pRetention = 0.75;
          }
        } else if (taskType === 'code') {
          // 代码：低层重要
          const bound1 = Math.floor(layers / 3);
          if (i < bound1) {
            pRetention = 0.8;
          } else {
            pRetention = 0.4;
          }
        } else {
          // 对话：均匀
          pRetention = 0.55;
        }
        pKeyPrecision = 8;
        pValuePrecision = 4;
        break;
    }
    
    pLayers.push({
      layerIndex: i,
      totalLayers: layers,
      retentionRatio: pRetention,
      keyPrecision: pKeyPrecision,
      valuePrecision: pValuePrecision
    });
    
    // D端配置：全部中等保真
    dLayers.push({
      layerIndex: i,
      totalLayers: layers,
      retentionRatio: 0.6,
      keyPrecision: 16,
      valuePrecision: 8
    });
  }
  
  return {
    strategy,
    pLayers,
    dLayers,
    taskType: strategy === 'task-aware' ? taskType : undefined
  };
}

/**
 * 运行单次实验
 */
function runSingleExperiment(strategy: 'none' | 'uniform' | 'pd-aware' | 'task-aware') {
  const simulator = new PDSimulator({
    bandwidthBytesPerMs: EXP_CONFIG.network.bandwidthBytesPerMs,
    gpuMemoryBytes: EXP_CONFIG.gpu.memoryBytes,
    kvBytesPerToken: 1024
  });
  
  const requests = generateRequests(EXP_CONFIG.requests.count);
  
  // 按任务类型分组应用压缩配置
  const results: any[] = [];
  for (const request of requests) {
    const compression = createCompressionConfig(strategy, request.taskType);
    const result = simulator.simulateRequest(request, compression, 1);
    results.push(result);
  }
  
  const stats = simulator.computeStats();
  
  return stats;
}

/**
 * 运行多次实验取平均
 */
function runMultiExperiment(strategy: 'none' | 'uniform' | 'pd-aware' | 'task-aware', runs: number) {
  console.log(`\n运行 ${strategy} 策略实验 (${runs}次)...`);
  
  const allStats: any[] = [];
  for (let i = 0; i < runs; i++) {
    const stats = runSingleExperiment(strategy);
    allStats.push(stats);
  }
  
  // 计算平均值
  const avgStats = {
    totalRequests: EXP_CONFIG.requests.count,
    avgTTFT: average(allStats.map(s => s.avgTTFT)),
    avgTPOT: average(allStats.map(s => s.avgTPOT)),
    avgE2E: average(allStats.map(s => s.avgE2E)),
    p50TTFT: average(allStats.map(s => s.p50TTFT)),
    p95TTFT: average(allStats.map(s => s.p95TTFT)),
    p99TTFT: average(allStats.map(s => s.p99TTFT)),
    avgCompressionRatio: average(allStats.map(s => s.avgCompressionRatio)),
    avgQualityScore: average(allStats.map(s => s.avgQualityScore)),
    cacheHitRate: average(allStats.map(s => s.cacheHitRate)),
    throughputTokensPerSec: average(allStats.map(s => s.throughputTokensPerSec)),
    perTaskStats: {
      math: averageTaskStats(allStats, 'math'),
      code: averageTaskStats(allStats, 'code'),
      qa: averageTaskStats(allStats, 'qa'),
      conversation: averageTaskStats(allStats, 'conversation')
    }
  };
  
  return avgStats;
}

function average(arr: number[]): number {
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length * 10000) / 10000;
}

function averageTaskStats(allStats: any[], taskType: TaskType) {
  return {
    count: allStats[0].perTaskStats[taskType].count,
    avgTTFT: average(allStats.map(s => s.perTaskStats[taskType].avgTTFT)),
    avgE2E: average(allStats.map(s => s.perTaskStats[taskType].avgE2E)),
    avgQuality: average(allStats.map(s => s.perTaskStats[taskType].avgQuality))
  };
}

/**
 * 生成实验报告
 */
function generateReport(results: Record<string, any>): string {
  const strategies = ['none', 'uniform', 'pd-aware', 'task-aware'];
  
  let report = `# 实验4：4策略完整对比（仿真）

## 实验配置

- **模型**: ${EXP_CONFIG.model.layers}层, ${EXP_CONFIG.model.hidden} hidden, ${EXP_CONFIG.model.heads} heads
- **请求数**: ${EXP_CONFIG.requests.count}个
- **任务混合**: math 30%, code 30%, qa 30%, conversation 10%
- **输入Token**: ${EXP_CONFIG.requests.inputTokens.min}-${EXP_CONFIG.requests.inputTokens.max}
- **输出Token**: ${EXP_CONFIG.requests.outputTokens.min}-${EXP_CONFIG.requests.outputTokens.max}
- **网络带宽**: ${EXP_CONFIG.network.bandwidthBytesPerMs * 1000 / (1024 * 1024 * 1024)} GB/s
- **GPU显存**: ${EXP_CONFIG.gpu.memoryBytes / (1024 * 1024 * 1024)} GB
- **运行次数**: ${EXP_CONFIG.runs}次 (取平均)

## 总体指标对比

| 策略 | 平均TTFT(ms) | P50 TTFT(ms) | P95 TTFT(ms) | P99 TTFT(ms) | 平均E2E(ms) | 吞吐量(tokens/s) | 压缩比 | 质量评分 | 缓存命中率 |
|------|-------------|-------------|-------------|-------------|------------|-----------------|--------|---------|----------|
`;

  for (const strategy of strategies) {
    const r = results[strategy];
    report += `| ${strategy} | ${r.avgTTFT} | ${r.p50TTFT} | ${r.p95TTFT} | ${r.p99TTFT} | ${r.avgE2E} | ${r.throughputTokensPerSec} | ${r.avgCompressionRatio} | ${r.avgQualityScore} | ${(r.cacheHitRate * 100).toFixed(1)}% |\n`;
  }

  report += `
## 按任务类型分组统计

### TTFT (ms)

| 任务类型 | None | Uniform | PD-Aware | Task-Aware |
|---------|------|---------|----------|------------|
`;

  const taskTypes: TaskType[] = ['math', 'code', 'qa', 'conversation'];
  for (const taskType of taskTypes) {
    report += `| ${taskType} | ${results.none.perTaskStats[taskType].avgTTFT} | ${results.uniform.perTaskStats[taskType].avgTTFT} | ${results['pd-aware'].perTaskStats[taskType].avgTTFT} | ${results['task-aware'].perTaskStats[taskType].avgTTFT} |\n`;
  }

  report += `
### E2E延迟 (ms)

| 任务类型 | None | Uniform | PD-Aware | Task-Aware |
|---------|------|---------|----------|------------|
`;

  for (const taskType of taskTypes) {
    report += `| ${taskType} | ${results.none.perTaskStats[taskType].avgE2E} | ${results.uniform.perTaskStats[taskType].avgE2E} | ${results['pd-aware'].perTaskStats[taskType].avgE2E} | ${results['task-aware'].perTaskStats[taskType].avgE2E} |\n`;
  }

  report += `
### 质量评分

| 任务类型 | None | Uniform | PD-Aware | Task-Aware |
|---------|------|---------|----------|------------|
`;

  for (const taskType of taskTypes) {
    report += `| ${taskType} | ${results.none.perTaskStats[taskType].avgQuality} | ${results.uniform.perTaskStats[taskType].avgQuality} | ${results['pd-aware'].perTaskStats[taskType].avgQuality} | ${results['task-aware'].perTaskStats[taskType].avgQuality} |\n`;
  }

  // 分析和结论
  const bestTTFT = strategies.reduce((a, b) => results[a].avgTTFT < results[b].avgTTFT ? a : b);
  const bestQuality = strategies.reduce((a, b) => results[a].avgQualityScore > results[b].avgQualityScore ? a : b);
  const bestThroughput = strategies.reduce((a, b) => results[a].throughputTokensPerSec > results[b].throughputTokensPerSec ? a : b);

  report += `
## 分析与结论

### 关键发现

1. **TTFT性能**: ${bestTTFT} 策略的平均TTFT最低（${results[bestTTFT].avgTTFT}ms）
2. **质量评分**: ${bestQuality} 策略的质量评分最高（${results[bestQuality].avgQualityScore}）
3. **吞吐量**: ${bestThroughput} 策略的吞吐量最高（${results[bestThroughput].throughputTokensPerSec} tokens/s）

### 策略对比

- **None (无压缩)**: 质量最高(1.0)，但无带宽节省。适合带宽充足场景。
- **Uniform (均匀压缩)**: 统一50%保留率，质量约0.5。简单但不够智能。
- **PD-Aware**: 低层激进压缩(0.3)，高层保守(0.7)。在质量和带宽间取得平衡。
- **Task-Aware**: 根据任务类型动态调整。数学/QA任务保持高层，代码任务保持低层。

### 建议

- 带宽受限场景：推荐 PD-Aware 或 Task-Aware
- 质量优先场景：推荐 None 或 PD-Aware
- 计算资源受限：推荐 Task-Aware（任务自适应）

---
*实验时间: ${new Date().toISOString()}*
`;

  return report;
}

// ========== 主程序 ==========

console.log('='.repeat(60));
console.log('实验4：4策略完整对比（仿真）');
console.log('='.repeat(60));

const strategies: Array<'none' | 'uniform' | 'pd-aware' | 'task-aware'> = ['none', 'uniform', 'pd-aware', 'task-aware'];
const results: Record<string, any> = {};

for (const strategy of strategies) {
  results[strategy] = runMultiExperiment(strategy, EXP_CONFIG.runs);
}

// 生成报告
const report = generateReport(results);

// 保存报告
writeFileSync('./logs/exp4-full-comparison.md', report);
console.log('\n报告已保存到 logs/exp4-full-comparison.md');

// 打印摘要
console.log('\n' + '='.repeat(60));
console.log('实验结果摘要');
console.log('='.repeat(60));

console.log('\n| 策略 | TTFT | E2E | 压缩比 | 质量 |');
console.log('|------|------|-----|--------|------|');
for (const strategy of strategies) {
  const r = results[strategy];
  console.log(`| ${strategy.padEnd(10)} | ${r.avgTTFT.toFixed(2)} | ${r.avgE2E.toFixed(2)} | ${r.avgCompressionRatio.toFixed(2)} | ${r.avgQualityScore.toFixed(4)} |`);
}
