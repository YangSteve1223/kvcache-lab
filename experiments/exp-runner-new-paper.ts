/**
 * 新论文实验运行器 - PD-Aware + Task-Aware KV Compression
 * 
 * 实验设计：
 * Exp1: 策略对比 (5策略 × 4任务 × 4预算)
 * Exp2: 带宽敏感性 (6级带宽 × 5策略)
 * Exp3: 任务类型影响 (4任务 × 3策略 × 4预算)
 * Exp4: 层预算消融 (Uniform vs PD-only vs Task-only vs Combined)
 * Exp5: Baseline对比 (vs PDTrim, KVServe)
 * Exp6: 长上下文缩放 (1K→32K)
 */

import * as fs from 'fs';

import {
  CompressionParams,
  CompressionOutput,
  TaskType,
  ICompressionStrategy
} from '../src/core/types.js';

import { CompressionOrchestrator } from '../src/compression/CompressionOrchestrator.js';
import { NoneCompression } from '../src/compression/strategies/NoneCompression.js';
import { UniformCompression } from '../src/compression/strategies/UniformCompression.js';
import { PDAwareCompression } from '../src/compression/strategies/PDAwareCompression.js';
import { TaskAwareCompression } from '../src/compression/strategies/TaskAwareCompression.js';
import { PDTaskAwareCompression } from '../src/compression/strategies/PDTaskAwareCompression.js';
import { QualityModel } from '../src/core/QualityModel.js';
import { PDTrimBaseline } from '../src/baselines/PDTrimBaseline.js';
import { KVServeBaseline } from '../src/baselines/KVServeBaseline.js';

// ============================================
// 常量配置
// ============================================

const TOTAL_LAYERS = 32;
const NUM_HEADS = 32;
const HEAD_DIM = 128;
// KV bytes per token: 2(K+V) × heads × headDim × layers × 2bytes(FP16)
const KV_BYTES_PER_TOKEN = 2 * NUM_HEADS * HEAD_DIM * TOTAL_LAYERS * 2; // 524,288 ≈ 512KB

const BANDWIDTH_LEVELS = [
  { name: '0.1Gbps', bps: 0.1e9 / 8 },
  { name: '0.5Gbps', bps: 0.5e9 / 8 },
  { name: '1Gbps',   bps: 1.0e9 / 8 },
  { name: '5Gbps',   bps: 5.0e9 / 8 },
  { name: '10Gbps',  bps: 10.0e9 / 8 },
  { name: '25Gbps',  bps: 25.0e9 / 8 },
];

const TASK_TYPES: TaskType[] = ['math', 'code', 'qa', 'conversation'];
const BUDGET_LEVELS = [0.3, 0.5, 0.7, 1.0];
const SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768];

const GPU_MEMORY = 48 * 1024 ** 3; // 48GB

// 策略实例映射
const STRATEGIES: Record<string, ICompressionStrategy> = {
  'NoneCompression': new NoneCompression(),
  'UniformCompression': new UniformCompression(),
  'PDAwareCompression': new PDAwareCompression(),
  'TaskAwareCompression': new TaskAwareCompression(),
  'PDTaskAwareCompression': new PDTaskAwareCompression(),
};

// ============================================
// 类型定义
// ============================================

interface ExpResult {
  strategy: string;
  taskType: string;
  bandwidth: string;
  seqLen: number;
  budgetRatio: number;
  avgCompressionRatio: number;
  estimatedBandwidthSaving: number;
  estimatedQuality: number;
  kvTransferTimeMs: number;
  fullTransferTimeMs: number;
  transferTimeReductionPct: number;
  pAvgRetention: number;
  dAvgRetention: number;
  pdDifferentiation: number;
}

// ============================================
// 工具函数
// ============================================

function mkParams(seqLen: number, bwBps: number, taskType: TaskType): CompressionParams {
  return {
    totalLayers: TOTAL_LAYERS,
    totalTokens: seqLen,
    bandwidthBytesPerMs: bwBps / 1000,
    gpuMemoryBytes: GPU_MEMORY,
    currentMemoryUsage: GPU_MEMORY * 0.7,
    taskType,
    sloLatencyMs: 1000,
    prefixHitRate: 0.1,
  };
}

function runStrategy(name: string, params: CompressionParams): ExpResult {
  const strategy = STRATEGIES[name];
  if (!strategy) throw new Error(`Unknown strategy: ${name}`);

  const config = strategy.computeConfig(params);
  const quality = strategy.estimateQualityImpact(config, params.taskType);

  const kvTotal = KV_BYTES_PER_TOKEN * params.totalTokens;
  const fullMs = kvTotal / params.bandwidthBytesPerMs;
  const compBytes = kvTotal * config.avgCompressionRatio;
  const compMs = compBytes / params.bandwidthBytesPerMs;

  const pAvg = avg(config.pLayerRetention);
  const dAvg = avg(config.dLayerRetention);

  return {
    strategy: name,
    taskType: params.taskType,
    bandwidth: '',
    seqLen: params.totalTokens,
    budgetRatio: 0,
    avgCompressionRatio: round4(config.avgCompressionRatio),
    estimatedBandwidthSaving: round4(config.estimatedBandwidthSaving),
    estimatedQuality: round4(quality),
    kvTransferTimeMs: round2(compMs),
    fullTransferTimeMs: round2(fullMs),
    transferTimeReductionPct: round2((1 - compMs / fullMs) * 100),
    pAvgRetention: round4(pAvg),
    dAvgRetention: round4(dAvg),
    pdDifferentiation: round4(Math.abs(pAvg - dAvg)),
  };
}

function avg(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}
function round2(n: number): number { return Math.round(n * 100) / 100; }
function round4(n: number): number { return Math.round(n * 10000) / 10000; }

function groupBy<T>(arr: T[], key: keyof T): Record<string, T[]> {
  return arr.reduce((acc, item) => {
    const k = String(item[key]);
    (acc[k] = acc[k] || []).push(item);
    return acc;
  }, {} as Record<string, T[]>);
}

// ============================================
// Exp1: 策略全对比 (5策略 × 4任务 × 4预算)
// ============================================

function exp1(): ExpResult[] {
  console.log('\n========== Exp1: Strategy Comparison ==========');
  const results: ExpResult[] = [];
  const bw = BANDWIDTH_LEVELS[2]; // 1Gbps
  const seqLen = 4096;

  for (const task of TASK_TYPES) {
    for (const budget of BUDGET_LEVELS) {
      const params = mkParams(seqLen, bw.bps, task);
      for (const sName of Object.keys(STRATEGIES)) {
        try {
          const r = runStrategy(sName, params);
          r.bandwidth = bw.name;
          r.budgetRatio = budget;
          results.push(r);
        } catch (e) {
          console.warn(`  FAIL: ${sName}/${task}/b${budget}: ${e}`);
        }
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp2: 带宽敏感性 (6级带宽 × 5策略)
// ============================================

function exp2(): ExpResult[] {
  console.log('\n========== Exp2: Bandwidth Sensitivity ==========');
  const results: ExpResult[] = [];
  const seqLen = 8192;
  const task: TaskType = 'conversation';

  for (const bw of BANDWIDTH_LEVELS) {
    const params = mkParams(seqLen, bw.bps, task);
    for (const sName of Object.keys(STRATEGIES)) {
      try {
        const r = runStrategy(sName, params);
        r.bandwidth = bw.name;
        results.push(r);
      } catch (e) {
        console.warn(`  FAIL: ${sName}/${bw.name}: ${e}`);
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp3: 任务类型影响 (4任务 × 3核心策略 × 4预算)
// ============================================

function exp3(): ExpResult[] {
  console.log('\n========== Exp3: Task Type Impact ==========');
  const results: ExpResult[] = [];
  const coreStrategies = ['PDAwareCompression', 'TaskAwareCompression', 'PDTaskAwareCompression'];
  const bw = BANDWIDTH_LEVELS[2];

  for (const task of TASK_TYPES) {
    for (const budget of BUDGET_LEVELS) {
      const params = mkParams(4096, bw.bps, task);
      for (const sName of coreStrategies) {
        try {
          const r = runStrategy(sName, params);
          r.bandwidth = bw.name;
          r.budgetRatio = budget;
          results.push(r);
        } catch (e) {
          console.warn(`  FAIL: ${sName}/${task}: ${e}`);
        }
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp4: 消融 (Uniform vs PD-only vs Task-only vs Combined)
// ============================================

function exp4(): ExpResult[] {
  console.log('\n========== Exp4: Ablation Study ==========');
  const results: ExpResult[] = [];
  const labels: Record<string, string> = {
    'UniformCompression': 'Uniform (baseline)',
    'PDAwareCompression': 'PD-Aware only',
    'TaskAwareCompression': 'Task-Aware only',
    'PDTaskAwareCompression': 'PD+Task Combined',
  };
  const bw = BANDWIDTH_LEVELS[1]; // 0.5Gbps (low BW → PD-aware matters)
  const seqLen = 8192;

  for (const task of TASK_TYPES) {
    const params = mkParams(seqLen, bw.bps, task);
    for (const [sName, label] of Object.entries(labels)) {
      try {
        const r = runStrategy(sName, params);
        r.strategy = label;
        r.bandwidth = bw.name;
        results.push(r);
      } catch (e) {
        console.warn(`  FAIL: ${label}/${task}: ${e}`);
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp5: Baseline对比 (vs PDTrim, KVServe)
// ============================================

interface BaselineResult {
  taskType: string;
  bandwidth: string;
  seqLen: number;
  fullTransferMs: number;
  pdtrim: { avgCompressionRatio: number; transferTimeMs: number; transferReductionPct: number };
  kvserve: { avgCompressionRatio: number; transferTimeMs: number; transferReductionPct: number };
  ours: { avgCompressionRatio: number; transferTimeMs: number; transferReductionPct: number; estimatedQuality: number };
}

function exp5(): BaselineResult[] {
  console.log('\n========== Exp5: Baseline Comparison ==========');
  const results: BaselineResult[] = [];
  const bw = BANDWIDTH_LEVELS[2];
  const seqLen = 4096;

  const pdtrim = new PDTrimBaseline();
  const kvserve = new KVServeBaseline();

  for (const task of TASK_TYPES) {
    const params = mkParams(seqLen, bw.bps, task);

    const pdtrimCfg = pdtrim.computeConfig(params);
    const kvserveCfg = kvserve.computeConfig(params);
    const ours = runStrategy('PDTaskAwareCompression', params);

    const kvTotal = KV_BYTES_PER_TOKEN * seqLen;
    const fullMs = kvTotal / params.bandwidthBytesPerMs;

    results.push({
      taskType: task,
      bandwidth: bw.name,
      seqLen,
      fullTransferMs: round2(fullMs),
      pdtrim: {
        avgCompressionRatio: round4(pdtrimCfg.avgCompressionRatio),
        transferTimeMs: round2(kvTotal * pdtrimCfg.avgCompressionRatio / params.bandwidthBytesPerMs),
        transferReductionPct: round2((1 - pdtrimCfg.avgCompressionRatio) * 100),
      },
      kvserve: {
        avgCompressionRatio: round4(kvserveCfg.avgCompressionRatio),
        transferTimeMs: round2(kvTotal * kvserveCfg.avgCompressionRatio / params.bandwidthBytesPerMs),
        transferReductionPct: round2((1 - kvserveCfg.avgCompressionRatio) * 100),
      },
      ours: {
        avgCompressionRatio: ours.avgCompressionRatio,
        transferTimeMs: ours.kvTransferTimeMs,
        transferReductionPct: ours.transferTimeReductionPct,
        estimatedQuality: ours.estimatedQuality,
      },
    });
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp6: 长上下文缩放 (1K→32K)
// ============================================

function exp6(): ExpResult[] {
  console.log('\n========== Exp6: Long Context Scaling ==========');
  const results: ExpResult[] = [];
  const strategies = ['UniformCompression', 'PDAwareCompression', 'PDTaskAwareCompression'];
  const bw = BANDWIDTH_LEVELS[2];
  const task: TaskType = 'conversation';

  for (const seqLen of SEQ_LENS) {
    const params = mkParams(seqLen, bw.bps, task);
    for (const sName of strategies) {
      try {
        const r = runStrategy(sName, params);
        r.bandwidth = bw.name;
        results.push(r);
      } catch (e) {
        console.warn(`  FAIL: ${sName}/${seqLen}: ${e}`);
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// 汇总打印
// ============================================

function printSummary(all: Record<string, any>) {
  console.log('\n===================================================');
  console.log('  EXPERIMENT SUMMARY');
  console.log('===================================================');

  // Exp1
  const exp1 = all.exp1 as ExpResult[];
  if (exp1?.length) {
    console.log('\n--- Exp1: Strategy Comparison (4K tokens, 1Gbps) ---');
    const byStrat = groupBy(exp1, 'strategy');
    for (const [name, items] of Object.entries(byStrat)) {
      const q = avg(items.map(i => i.estimatedQuality));
      const r = avg(items.map(i => i.transferTimeReductionPct));
      const c = avg(items.map(i => i.avgCompressionRatio));
      console.log(`  ${name.padEnd(25)} Quality=${q.toFixed(4)}  TransferReduction=${r.toFixed(1)}%  CompressRatio=${c.toFixed(4)}`);
    }
  }

  // Exp2
  const exp2 = all.exp2 as ExpResult[];
  if (exp2?.length) {
    console.log('\n--- Exp2: Bandwidth Sensitivity (8K tokens, conversation) ---');
    const byBw = groupBy(exp2, 'bandwidth');
    for (const [bw, items] of Object.entries(byBw)) {
      const pdTask = items.find(i => i.strategy === 'PDTaskAwareCompression');
      if (pdTask) {
        console.log(`  ${bw.padEnd(8)} Transfer=${pdTask.kvTransferTimeMs.toFixed(1)}ms (full=${pdTask.fullTransferTimeMs.toFixed(1)}ms)  Reduction=${pdTask.transferTimeReductionPct.toFixed(1)}%  Quality=${pdTask.estimatedQuality.toFixed(4)}`);
      }
    }
  }

  // Exp4
  const exp4 = all.exp4 as ExpResult[];
  if (exp4?.length) {
    console.log('\n--- Exp4: Ablation Study (8K tokens, 0.5Gbps, conversation) ---');
    const byStrat = groupBy(exp4, 'strategy');
    for (const [name, items] of Object.entries(byStrat)) {
      const q = avg(items.map(i => i.estimatedQuality));
      const r = avg(items.map(i => i.transferTimeReductionPct));
      const pd = avg(items.map(i => i.pdDifferentiation));
      console.log(`  ${name.padEnd(25)} Quality=${q.toFixed(4)}  TransferReduction=${r.toFixed(1)}%  P/D Diff=${pd.toFixed(4)}`);
    }
  }

  // Exp5
  const exp5 = all.exp5 as BaselineResult[];
  if (exp5?.length) {
    console.log('\n--- Exp5: Baseline Comparison (4K tokens, 1Gbps) ---');
    console.log(`  ${'Task'.padEnd(14)} ${'PDTrim'.padEnd(18)} ${'KVServe'.padEnd(18)} ${'Ours(PD+Task)'.padEnd(22)} Full(ms)`);
    for (const r of exp5) {
      const pt = `${r.pdtrim.transferReductionPct.toFixed(1)}%`;
      const ks = `${r.kvserve.transferReductionPct.toFixed(1)}%`;
      const our = `${r.ours.transferReductionPct.toFixed(1)}%/Q=${r.ours.estimatedQuality.toFixed(3)}`;
      console.log(`  ${r.taskType.padEnd(14)} ${pt.padEnd(18)} ${ks.padEnd(18)} ${our.padEnd(22)} ${r.fullTransferMs.toFixed(1)}`);
    }
  }

  // Exp6
  const exp6 = all.exp6 as ExpResult[];
  if (exp6?.length) {
    console.log('\n--- Exp6: Long Context Scaling (1Gbps, conversation) ---');
    const bySeq = groupBy(exp6, 'seqLen');
    for (const [seq, items] of Object.entries(bySeq).sort((a, b) => +a[0] - +b[0])) {
      const pdTask = items.find(i => i.strategy === 'PDTaskAwareCompression');
      if (pdTask) {
        console.log(`  ${(seq + ' tok').padEnd(10)} Transfer=${pdTask.kvTransferTimeMs.toFixed(1)}ms  Full=${pdTask.fullTransferTimeMs.toFixed(1)}ms  Reduction=${pdTask.transferTimeReductionPct.toFixed(1)}%  Quality=${pdTask.estimatedQuality.toFixed(4)}`);
      }
    }
  }
}

// ============================================
// Main
// ============================================

function main() {
  console.log('===================================================');
  console.log('  PD-Aware + Task-Aware KV Compression Experiments');
  console.log('  Date: ' + new Date().toISOString().split('T')[0]);
  console.log('===================================================');

  const all: Record<string, any> = {};

  all.exp1 = exp1();
  all.exp2 = exp2();
  all.exp3 = exp3();
  all.exp4 = exp4();
  all.exp5 = exp5();
  all.exp6 = exp6();

  // Save JSON
  const outPath = './experiment_logs/new-paper-experiments.json';
  fs.writeFileSync(outPath, JSON.stringify(all, null, 2));
  console.log(`\n✅ Results saved to ${outPath}`);

  printSummary(all);
}

main();
