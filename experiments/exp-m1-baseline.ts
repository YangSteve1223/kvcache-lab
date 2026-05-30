/**
 * M1 综合基准实验
 * 
 * 目标：验证 CompressionOrchestrator → CompressionAdapter → PDSimulator 全链路
 * 覆盖：5种策略 × 4种任务 × 3种带宽 × 3种序列长度 = 180组实验
 * 
 * 输出：CSV格式对比表 + JSON完整结果
 */

import { PDSimulator } from '../src/core/PDSimulator.ts';
import { CompressionOrchestrator } from '../src/compression/CompressionOrchestrator.ts';
import { adaptCompressionOutput } from '../src/compression/CompressionAdapter.ts';
import { NoneCompression } from '../src/compression/strategies/NoneCompression.ts';
import { UniformCompression } from '../src/compression/strategies/UniformCompression.ts';
import { PDAwareCompression } from '../src/compression/strategies/PDAwareCompression.ts';
import { TaskAwareCompression } from '../src/compression/strategies/TaskAwareCompression.ts';
import { PDTaskAwareCompression } from '../src/compression/strategies/PDTaskAwareCompression.ts';
import { QualityConstrainedCompression } from '../src/compression/strategies/QualityConstrainedCompression.ts';
import {
  ServingRequest,
  TaskType,
  CompressionParams,
  SimulatorConfig,
} from '../src/core/types.ts';
import { writeFileSync, mkdirSync } from 'fs';

// ============================================================
// 实验配置
// ============================================================

const STRATEGIES = [
  { name: 'None', cls: NoneCompression },
  { name: 'Uniform', cls: UniformCompression },
  { name: 'PD-Aware', cls: PDAwareCompression },
  { name: 'Task-Aware', cls: TaskAwareCompression },
  { name: 'PD-Task-Aware', cls: PDTaskAwareCompression },
  { name: 'QCBM', cls: QualityConstrainedCompression },
] as const;

const TASK_TYPES: TaskType[] = ['math', 'code', 'qa', 'conversation'];

const BANDWIDTH_SCENARIOS = [
  { name: 'Low-1Gbps',  bytesPerMs: 1e9 / 8 / 1000 },    // 1 Gbps (cross-rack Ethernet)
  { name: 'Mid-10Gbps', bytesPerMs: 10e9 / 8 / 1000 },   // 10 Gbps (typical DC)
  { name: 'High-100Gbps', bytesPerMs: 100e9 / 8 / 1000 }, // 100 Gbps (NVLink/intra-rack)
];

const SEQ_LENGTHS = [2048, 4096, 8192, 16384, 32768];

const MODEL_CONFIG = {
  totalLayers: 32,
  hiddenSize: 4096,
  numHeads: 32,
  headDim: 128,
  // kvBytesPerToken = 2(K+V) × headDim × 2bytes(FP16) = 512 bytes/token/layer
  // PDSimulator.computeKVSize = tokens × kvBytesPerToken × totalLayers
  kvBytesPerToken: 2 * 128 * 2, // 512 bytes per token per layer
};

const GPU_MEMORY = 80 * 1024 * 1024 * 1024; // 80GB (A100-80GB)

// ============================================================
// 工具函数
// ============================================================

function generateRequest(taskType: TaskType, seqLen: number, id: number): ServingRequest {
  return {
    id: `req-${id}`,
    inputTokens: seqLen,
    outputTokens: Math.floor(seqLen * 0.1),  // output ≈ 10% of input
    taskType,
    arrivalTimeMs: id * 100,  // 100ms间隔
    sloLatencyMs: 2000,
  };
}

function createSimulator(bandwidthBytesPerMs: number): PDSimulator {
  const config: Partial<SimulatorConfig> = {
    prefillBaseMs: 50,
    prefillMsPerToken: 0.1,
    decodeBaseMs: 10,
    decodeMsPerToken: 1.5,
    kvBytesPerToken: MODEL_CONFIG.kvBytesPerToken,
    bandwidthBytesPerMs,
    gpuMemoryBytes: GPU_MEMORY,
    cpuMemoryBytes: 200 * 1024 * 1024 * 1024,
  };
  return new PDSimulator(config);
}

// ============================================================
// 主实验逻辑
// ============================================================

interface ExperimentResult {
  strategy: string;
  taskType: string;
  bandwidth: string;
  seqLen: number;
  ttftMs: number;
  tpotMs: number;
  e2eLatencyMs: number;
  kvTransferTimeMs: number;
  compressionRatio: number;
  qualityScore: number;
  bandwidthSavingPct: number;
  ttftReductionPct: number;  // vs None baseline
}

function runExperiment(): ExperimentResult[] {
  const results: ExperimentResult[] = [];
  
  // 注册所有策略
  const orchestrator = new CompressionOrchestrator();
  for (const { name, cls } of STRATEGIES) {
    orchestrator.registerStrategy(new cls());
  }
  console.log(`Registered strategies: ${orchestrator.getRegisteredStrategies().join(', ')}`);

  let total = TASK_TYPES.length * BANDWIDTH_SCENARIOS.length * SEQ_LENGTHS.length * STRATEGIES.length;
  let done = 0;

  for (const taskType of TASK_TYPES) {
    for (const bw of BANDWIDTH_SCENARIOS) {
      for (const seqLen of SEQ_LENGTHS) {
        // None策略的baseline结果
        let baselineTTFT = 0;

        for (const { name } of STRATEGIES) {
          done++;
          if (done % 20 === 0) {
            console.log(`  Progress: ${done}/${total} (${(done/total*100).toFixed(1)}%)`);
          }

          // 构建压缩参数
          const params: CompressionParams = {
            totalLayers: MODEL_CONFIG.totalLayers,
            totalTokens: seqLen,
            bandwidthBytesPerMs: bw.bytesPerMs,
            gpuMemoryBytes: GPU_MEMORY,
            currentMemoryUsage: GPU_MEMORY * 0.7,  // 70% utilization
            taskType,
            sloLatencyMs: 2000,
          };

          // 获取压缩配置
          const strategy = orchestrator.getRegisteredStrategies().includes(name + 'Compression')
            ? orchestrator.selectStrategy(params)  // auto select 不一定选当前策略
            : null;

          // 直接用对应策略计算配置
          const selectedStrategy = (() => {
            const strategyMap: Record<string, any> = {
              'None': new NoneCompression(),
              'Uniform': new UniformCompression(),
              'PD-Aware': new PDAwareCompression(),
              'Task-Aware': new TaskAwareCompression(),
              'PD-Task-Aware': new PDTaskAwareCompression(),
              'QCBM': new QualityConstrainedCompression(),
            };
            return strategyMap[name];
          })();

          const compressionOutput = selectedStrategy.computeConfig(params);
          const compressionConfig = adaptCompressionOutput(compressionOutput, taskType);

          // 运行仿真
          const sim = createSimulator(bw.bytesPerMs);
          const request = generateRequest(taskType, seqLen, 1);
          const result = sim.simulateRequest(request, compressionConfig);

          // 计算带宽节省 (基于compression ratio which now includes precision)
          const bandwidthSavingPct = (1 - result.compressionRatio) * 100;

          // 记录None baseline
          if (name === 'None') {
            baselineTTFT = result.ttftMs;
          }

          results.push({
            strategy: name,
            taskType,
            bandwidth: bw.name,
            seqLen,
            ttftMs: result.ttftMs,
            tpotMs: result.tpotMs,
            e2eLatencyMs: result.e2eLatencyMs,
            kvTransferTimeMs: result.kvTransferTimeMs,
            compressionRatio: result.compressionRatio,
            qualityScore: result.qualityScore,
            bandwidthSavingPct: Math.round(bandwidthSavingPct * 100) / 100,
            ttftReductionPct: baselineTTFT > 0
              ? Math.round((1 - result.ttftMs / baselineTTFT) * 10000) / 100
              : 0,
          });
        }
      }
    }
  }

  return results;
}

// ============================================================
// 输出格式化
// ============================================================

function toCSV(results: ExperimentResult[]): string {
  const header = 'strategy,taskType,bandwidth,seqLen,ttftMs,tpotMs,e2eLatencyMs,kvTransferTimeMs,compressionRatio,qualityScore,bandwidthSavingPct,ttftReductionPct';
  const rows = results.map(r =>
    `${r.strategy},${r.taskType},${r.bandwidth},${r.seqLen},${r.ttftMs.toFixed(2)},${r.tpotMs.toFixed(2)},${r.e2eLatencyMs.toFixed(2)},${r.kvTransferTimeMs.toFixed(2)},${r.compressionRatio.toFixed(4)},${r.qualityScore.toFixed(4)},${r.bandwidthSavingPct.toFixed(1)},${r.ttftReductionPct.toFixed(1)}`
  );
  return header + '\n' + rows.join('\n');
}

function printSummaryTable(results: ExperimentResult[]): void {
  console.log('\n' + '='.repeat(100));
  console.log('M1 Baseline Summary (averaged across tasks, seq=4096, Low-1Gbps)');
  console.log('='.repeat(100));
  
  const filtered = results.filter(r => r.seqLen === 4096 && r.bandwidth === 'Low-1Gbps');
  
  // Group by strategy
  const byStrategy: Record<string, ExperimentResult[]> = {};
  for (const r of filtered) {
    if (!byStrategy[r.strategy]) byStrategy[r.strategy] = [];
    byStrategy[r.strategy].push(r);
  }
  
  console.log(
    'Strategy'.padEnd(18) +
    'Avg TTFT(ms)'.padEnd(14) +
    'TTFT Reduction'.padEnd(16) +
    'BW Saving'.padEnd(12) +
    'Quality'.padEnd(10) +
    'Compression'.padEnd(12)
  );
  console.log('-'.repeat(82));
  
  for (const [strategy, rs] of Object.entries(byStrategy)) {
    const avgTTFT = rs.reduce((s, r) => s + r.ttftMs, 0) / rs.length;
    const avgReduction = rs.reduce((s, r) => s + r.ttftReductionPct, 0) / rs.length;
    const avgBWSaving = rs.reduce((s, r) => s + r.bandwidthSavingPct, 0) / rs.length;
    const avgQuality = rs.reduce((s, r) => s + r.qualityScore, 0) / rs.length;
    const avgComp = rs.reduce((s, r) => s + r.compressionRatio, 0) / rs.length;
    
    console.log(
      strategy.padEnd(18) +
      avgTTFT.toFixed(1).padEnd(14) +
      (avgReduction.toFixed(1) + '%').padEnd(16) +
      (avgBWSaving.toFixed(1) + '%').padEnd(12) +
      avgQuality.toFixed(3).padEnd(10) +
      avgComp.toFixed(3).padEnd(12)
    );
  }
  
  // Per-task breakdown for best strategy
  console.log('\n' + '='.repeat(100));
  console.log('PD-Task-Aware Per-Task Breakdown (Low-1Gbps, seq=4096)');
  console.log('='.repeat(100));
  
  const pdTaskResults = results.filter(
    r => r.strategy === 'PD-Task-Aware' && r.seqLen === 4096 && r.bandwidth === 'Low-1Gbps'
  );
  
  console.log(
    'Task'.padEnd(14) +
    'TTFT(ms)'.padEnd(12) +
    'TTFT Reduction'.padEnd(16) +
    'BW Saving'.padEnd(12) +
    'Quality'.padEnd(10)
  );
  console.log('-'.repeat(64));
  
  for (const r of pdTaskResults) {
    console.log(
      r.taskType.padEnd(14) +
      r.ttftMs.toFixed(1).padEnd(12) +
      (r.ttftReductionPct.toFixed(1) + '%').padEnd(16) +
      (r.bandwidthSavingPct.toFixed(1) + '%').padEnd(12) +
      r.qualityScore.toFixed(3).padEnd(10)
    );
  }

  // Bandwidth sensitivity
  console.log('\n' + '='.repeat(100));
  console.log('Bandwidth Sensitivity (PD-Task-Aware, seq=4096, avg across tasks)');
  console.log('='.repeat(100));
  
  for (const bw of BANDWIDTH_SCENARIOS) {
    const bwResults = results.filter(
      r => r.strategy === 'PD-Task-Aware' && r.seqLen === 4096 && r.bandwidth === bw.name
    );
    if (bwResults.length === 0) continue;
    
    const avgTTFT = bwResults.reduce((s, r) => s + r.ttftMs, 0) / bwResults.length;
    const avgQuality = bwResults.reduce((s, r) => s + r.qualityScore, 0) / bwResults.length;
    const avgBWSaving = bwResults.reduce((s, r) => s + r.bandwidthSavingPct, 0) / bwResults.length;
    
    console.log(
      `${bw.name.padEnd(14)} TTFT=${avgTTFT.toFixed(1)}ms  Quality=${avgQuality.toFixed(3)}  BW Saving=${avgBWSaving.toFixed(1)}%`
    );
  }

  // Seq length scaling
  console.log('\n' + '='.repeat(100));
  console.log('Sequence Length Scaling (PD-Task-Aware, Low-1Gbps, avg across tasks)');
  console.log('='.repeat(100));
  
  for (const seqLen of SEQ_LENGTHS) {
    const seqResults = results.filter(
      r => r.strategy === 'PD-Task-Aware' && r.seqLen === seqLen && r.bandwidth === 'Low-1Gbps'
    );
    if (seqResults.length === 0) continue;
    
    const avgTTFT = seqResults.reduce((s, r) => s + r.ttftMs, 0) / seqResults.length;
    const avgQuality = seqResults.reduce((s, r) => s + r.qualityScore, 0) / seqResults.length;
    
    console.log(
      `seq=${seqLen.toString().padEnd(6)} TTFT=${avgTTFT.toFixed(1)}ms  Quality=${avgQuality.toFixed(3)}`
    );
  }
}

// ============================================================
// 主函数
// ============================================================

async function main() {
  console.log('kvcache-lab M1 Comprehensive Baseline Benchmark');
  console.log('================================================');
  console.log(`Config: ${STRATEGIES.length} strategies × ${TASK_TYPES.length} tasks × ${BANDWIDTH_SCENARIOS.length} bandwidth × ${SEQ_LENGTHS.length} seq_lens`);
  console.log(`KV bytes/token/layer: ${MODEL_CONFIG.kvBytesPerToken} (2 × ${MODEL_CONFIG.headDim} dim × 2 bytes); total across ${MODEL_CONFIG.totalLayers} layers = ${MODEL_CONFIG.kvBytesPerToken * MODEL_CONFIG.totalLayers} bytes`);
  console.log('');
  
  const results = runExperiment();
  
  // 输出摘要表
  printSummaryTable(results);
  
  // 保存JSON
  const outDir = './experiment_logs';
  mkdirSync(outDir, { recursive: true });
  
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const jsonPath = `${outDir}/m1-baseline-${timestamp}.json`;
  writeFileSync(jsonPath, JSON.stringify(results, null, 2));
  console.log(`\n✅ JSON results saved: ${jsonPath}`);
  
  // 保存CSV
  const csvPath = `${outDir}/m1-baseline-${timestamp}.csv`;
  writeFileSync(csvPath, toCSV(results));
  console.log(`✅ CSV results saved: ${csvPath}`);
  
  // 保存摘要
  const summary = {
    timestamp: new Date().toISOString(),
    config: {
      strategies: STRATEGIES.map(s => s.name),
      taskTypes: TASK_TYPES,
      bandwidthScenarios: BANDWIDTH_SCENARIOS.map(b => b.name),
      seqLengths: SEQ_LENGTHS,
      modelConfig: MODEL_CONFIG,
    },
    totalExperiments: results.length,
    keyFindings: extractKeyFindings(results),
  };
  
  const summaryPath = `${outDir}/m1-baseline-summary-${timestamp}.json`;
  writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
  console.log(`✅ Summary saved: ${summaryPath}`);
}

function extractKeyFindings(results: ExperimentResult[]): string[] {
  const findings: string[] = [];
  
  // 1. Best strategy by TTFT reduction
  const byStrategy: Record<string, ExperimentResult[]> = {};
  for (const r of results) {
    if (!byStrategy[r.strategy]) byStrategy[r.strategy] = [];
    byStrategy[r.strategy].push(r);
  }
  
  for (const [strategy, rs] of Object.entries(byStrategy)) {
    if (strategy === 'None') continue;
    const avgReduction = rs.reduce((s, r) => s + r.ttftReductionPct, 0) / rs.length;
    const avgQuality = rs.reduce((s, r) => s + r.qualityScore, 0) / rs.length;
    const avgBWSaving = rs.reduce((s, r) => s + r.bandwidthSavingPct, 0) / rs.length;
    findings.push(`${strategy}: avg TTFT reduction=${avgReduction.toFixed(1)}%, quality=${avgQuality.toFixed(3)}, BW saving=${avgBWSaving.toFixed(1)}%`);
  }
  
  // 2. Low bandwidth benefit
  const lowBW = results.filter(r => r.bandwidth === 'Low-1Gbps' && r.strategy !== 'None');
  if (lowBW.length > 0) {
    const avgReduction = lowBW.reduce((s, r) => s + r.ttftReductionPct, 0) / lowBW.length;
    findings.push(`Low bandwidth (1Gbps): avg TTFT reduction=${avgReduction.toFixed(1)}%`);
  }
  
  // 3. Quality preservation
  const pdTask = results.filter(r => r.strategy === 'PD-Task-Aware');
  if (pdTask.length > 0) {
    const minQuality = Math.min(...pdTask.map(r => r.qualityScore));
    findings.push(`PD-Task-Aware min quality=${minQuality.toFixed(3)} across all scenarios`);
  }
  
  return findings;
}

main().catch(console.error);
