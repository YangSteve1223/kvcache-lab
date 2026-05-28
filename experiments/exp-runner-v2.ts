/**
 * 新论文实验运行器 v2 - 使用校准质量模型
 * 
 * 改进:
 * - 使用CalibratedQualityModel (基于真实GPU实验数据)
 * - 支持4种模型locality profile (Qwen-7B/14B, Mistral-7B, Gemma-9B)
 * - 更准确的质量→PPL映射
 * - 7组实验完整覆盖
 */

import * as fs from 'fs';

import {
  CompressionParams,
  CompressionOutput,
  TaskType,
  ICompressionStrategy,
  clamp,
  round4
} from '../src/core/types.js';

import { NoneCompression } from '../src/compression/strategies/NoneCompression.js';
import { UniformCompression } from '../src/compression/strategies/UniformCompression.js';
import { PDAwareCompression } from '../src/compression/strategies/PDAwareCompression.js';
import { TaskAwareCompression } from '../src/compression/strategies/TaskAwareCompression.js';
import { PDTaskAwareCompression } from '../src/compression/strategies/PDTaskAwareCompression.js';
import { PDTrimBaseline } from '../src/baselines/PDTrimBaseline.js';
import { KVServeBaseline } from '../src/baselines/KVServeBaseline.js';
import {
  computeCalibratedQuality,
  LOCALITY_PROFILES,
  LocalityProfile,
  TASK_SENSITIVITY,
  TaskSensitivityProfile,
  estimatePPLChange,
  pplChangeToQuality,
} from '../src/core/CalibratedQualityModel.js';

// ============================================
// 常量
// ============================================

const TOTAL_LAYERS = 32;
const NUM_HEADS = 32;
const HEAD_DIM = 128;
const KV_BYTES_PER_TOKEN = 2 * NUM_HEADS * HEAD_DIM * TOTAL_LAYERS * 2; // ≈512KB/token

const BW_LEVELS = [
  { name: '0.1Gbps', bps: 0.1e9 / 8 },
  { name: '0.5Gbps', bps: 0.5e9 / 8 },
  { name: '1Gbps',   bps: 1.0e9 / 8 },
  { name: '5Gbps',   bps: 5.0e9 / 8 },
  { name: '10Gbps',  bps: 10.0e9 / 8 },
  { name: '25Gbps',  bps: 25.0e9 / 8 },
];

const TASKS: TaskType[] = ['math', 'code', 'qa', 'conversation'];
const MODELS = ['qwen-7b', 'qwen-14b', 'mistral-7b', 'gemma-9b'];
const BUDGETS = [0.3, 0.5, 0.7, 1.0];
const SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768];
const GPU_MEM = 48 * 1024 ** 3;

const STRATEGIES: Record<string, ICompressionStrategy> = {
  'NoneCompression': new NoneCompression(),
  'UniformCompression': new UniformCompression(),
  'PDAwareCompression': new PDAwareCompression(),
  'TaskAwareCompression': new TaskAwareCompression(),
  'PDTaskAwareCompression': new PDTaskAwareCompression(),
};

// ============================================
// 类型
// ============================================

interface ExpResult {
  strategy: string;
  taskType: string;
  model: string;
  bandwidth: string;
  seqLen: number;
  budgetRatio: number;
  // 压缩
  avgCompressionRatio: number;
  estimatedBandwidthSaving: number;
  // 校准质量
  quality: number;
  pplChangePct: number;
  // 传输
  kvTransferTimeMs: number;
  fullTransferTimeMs: number;
  transferReductionPct: number;
  // P/D差异化
  pAvgRetention: number;
  dAvgRetention: number;
  pdDifferentiation: number;
  // Locality
  gini: number;
  pattern: string;
}

// ============================================
// 工具
// ============================================

const r2 = (n: number) => Math.round(n * 100) / 100;
const r4 = (n: number) => Math.round(n * 10000) / 10000;

function mkParams(seqLen: number, bwBps: number, task: TaskType): CompressionParams {
  return {
    totalLayers: TOTAL_LAYERS,
    totalTokens: seqLen,
    bandwidthBytesPerMs: bwBps / 1000,
    gpuMemoryBytes: GPU_MEM,
    currentMemoryUsage: GPU_MEM * 0.7,
    taskType: task,
    sloLatencyMs: 1000,
    prefixHitRate: 0.1,
  };
}

function runStrategy(
  name: string,
  params: CompressionParams,
  model: string,
): ExpResult {
  const strategy = STRATEGIES[name];
  if (!strategy) throw new Error(`Unknown: ${name}`);
  const config = strategy.computeConfig(params);

  // 校准质量
  const calibrated = computeCalibratedQuality(config, model, params.taskType as TaskType, true);

  // 传输时间
  const kvTotal = KV_BYTES_PER_TOKEN * params.totalTokens;
  const fullMs = kvTotal / params.bandwidthBytesPerMs;
  const compMs = kvTotal * config.avgCompressionRatio / params.bandwidthBytesPerMs;

  const locality = LOCALITY_PROFILES[model];

  return {
    strategy: name,
    taskType: params.taskType,
    model,
    bandwidth: '',
    seqLen: params.totalTokens,
    budgetRatio: 0,
    avgCompressionRatio: r4(config.avgCompressionRatio),
    estimatedBandwidthSaving: r4(config.estimatedBandwidthSaving),
    quality: calibrated.quality,
    pplChangePct: calibrated.pplChangePct,
    kvTransferTimeMs: r2(compMs),
    fullTransferTimeMs: r2(fullMs),
    transferReductionPct: r2((1 - compMs / fullMs) * 100),
    pAvgRetention: calibrated.pAvgRetention,
    dAvgRetention: calibrated.dAvgRetention,
    pdDifferentiation: calibrated.pdDifferentiation,
    gini: locality.gini,
    pattern: locality.pattern,
  };
}

function groupBy<T>(arr: T[], key: keyof T): Record<string, T[]> {
  return arr.reduce((acc, item) => {
    const k = String(item[key]);
    (acc[k] = acc[k] || []).push(item);
    return acc;
  }, {} as Record<string, T[]>);
}

function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

// ============================================
// Exp1: 跨模型策略对比 (4模型 × 5策略 × 4任务)
// ============================================

function exp1(): ExpResult[] {
  console.log('\n========== Exp1: Cross-Model Strategy Comparison ==========');
  const results: ExpResult[] = [];
  const bw = BW_LEVELS[2]; // 1Gbps

  for (const model of MODELS) {
    for (const task of TASKS) {
      const params = mkParams(4096, bw.bps, task);
      for (const sName of Object.keys(STRATEGIES)) {
        try {
          const r = runStrategy(sName, params, model);
          r.bandwidth = bw.name;
          results.push(r);
        } catch (e) { console.warn(`FAIL: ${sName}/${model}/${task}: ${e}`); }
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp2: 带宽敏感性 (6带宽 × 5策略 × 4模型)
// ============================================

function exp2(): ExpResult[] {
  console.log('\n========== Exp2: Bandwidth Sensitivity ==========');
  const results: ExpResult[] = [];
  const task: TaskType = 'conversation';

  for (const model of MODELS) {
    for (const bw of BW_LEVELS) {
      const params = mkParams(8192, bw.bps, task);
      for (const sName of Object.keys(STRATEGIES)) {
        try {
          const r = runStrategy(sName, params, model);
          r.bandwidth = bw.name;
          results.push(r);
        } catch (e) { console.warn(`FAIL: ${sName}/${model}/${bw.name}: ${e}`); }
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp3: 任务类型 × 模型交互 (4模型 × 4任务 × 3核心策略)
// ============================================

function exp3(): ExpResult[] {
  console.log('\n========== Exp3: Task × Model Interaction ==========');
  const results: ExpResult[] = [];
  const core = ['PDAwareCompression', 'TaskAwareCompression', 'PDTaskAwareCompression'];
  const bw = BW_LEVELS[2];

  for (const model of MODELS) {
    for (const task of TASKS) {
      const params = mkParams(4096, bw.bps, task);
      for (const sName of core) {
        try {
          const r = runStrategy(sName, params, model);
          r.bandwidth = bw.name;
          results.push(r);
        } catch (e) { console.warn(`FAIL: ${sName}/${model}/${task}: ${e}`); }
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp4: 消融 (4模型 × 4策略变体)
// ============================================

function exp4(): ExpResult[] {
  console.log('\n========== Exp4: Ablation Study ==========');
  const results: ExpResult[] = [];
  const labels: Record<string, string> = {
    'UniformCompression': 'Uniform',
    'PDAwareCompression': 'PD-Aware',
    'TaskAwareCompression': 'Task-Aware',
    'PDTaskAwareCompression': 'PD+Task',
  };
  const bw = BW_LEVELS[1]; // 0.5Gbps

  for (const model of MODELS) {
    for (const task of TASKS) {
      const params = mkParams(8192, bw.bps, task);
      for (const [sName, label] of Object.entries(labels)) {
        try {
          const r = runStrategy(sName, params, model);
          r.strategy = label;
          r.bandwidth = bw.name;
          results.push(r);
        } catch (e) { console.warn(`FAIL: ${label}/${model}: ${e}`); }
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
  model: string;
  bandwidth: string;
  seqLen: number;
  fullTransferMs: number;
  pdtrim: { compressionRatio: number; transferMs: number; reductionPct: number; quality: number };
  kvserve: { compressionRatio: number; transferMs: number; reductionPct: number; quality: number };
  ours: { compressionRatio: number; transferMs: number; reductionPct: number; quality: number; pplChangePct: number };
}

function exp5(): BaselineResult[] {
  console.log('\n========== Exp5: Baseline Comparison ==========');
  const results: BaselineResult[] = [];
  const bw = BW_LEVELS[2];
  const seqLen = 4096;

  const pdtrim = new PDTrimBaseline();
  const kvserve = new KVServeBaseline();

  for (const model of MODELS) {
    for (const task of TASKS) {
      const params = mkParams(seqLen, bw.bps, task);

      const ptCfg = pdtrim.computeConfig(params);
      const ksCfg = kvserve.computeConfig(params);
      const ours = runStrategy('PDTaskAwareCompression', params, model);

      const kvTotal = KV_BYTES_PER_TOKEN * seqLen;
      const fullMs = kvTotal / params.bandwidthBytesPerMs;

      // 对PDTrim和KVServe也用校准质量模型
      const ptQuality = computeCalibratedQuality(ptCfg, model, task, true);
      const ksQuality = computeCalibratedQuality(ksCfg, model, task, true);

      results.push({
        taskType: task,
        model,
        bandwidth: bw.name,
        seqLen,
        fullTransferMs: r2(fullMs),
        pdtrim: {
          compressionRatio: r4(ptCfg.avgCompressionRatio),
          transferMs: r2(kvTotal * ptCfg.avgCompressionRatio / params.bandwidthBytesPerMs),
          reductionPct: r2((1 - ptCfg.avgCompressionRatio) * 100),
          quality: ptQuality.quality,
        },
        kvserve: {
          compressionRatio: r4(ksCfg.avgCompressionRatio),
          transferMs: r2(kvTotal * ksCfg.avgCompressionRatio / params.bandwidthBytesPerMs),
          reductionPct: r2((1 - ksCfg.avgCompressionRatio) * 100),
          quality: ksQuality.quality,
        },
        ours: {
          compressionRatio: ours.avgCompressionRatio,
          transferMs: ours.kvTransferTimeMs,
          reductionPct: ours.transferReductionPct,
          quality: ours.quality,
          pplChangePct: ours.pplChangePct,
        },
      });
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp6: 长上下文缩放 (4模型 × 6序列长度 × 3策略)
// ============================================

function exp6(): ExpResult[] {
  console.log('\n========== Exp6: Long Context Scaling ==========');
  const results: ExpResult[] = [];
  const strategies = ['UniformCompression', 'PDAwareCompression', 'PDTaskAwareCompression'];
  const bw = BW_LEVELS[2];
  const task: TaskType = 'conversation';

  for (const model of MODELS) {
    for (const seqLen of SEQ_LENS) {
      const params = mkParams(seqLen, bw.bps, task);
      for (const sName of strategies) {
        try {
          const r = runStrategy(sName, params, model);
          r.bandwidth = bw.name;
          results.push(r);
        } catch (e) { console.warn(`FAIL: ${sName}/${model}/${seqLen}: ${e}`); }
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// Exp7: Sink Token敏感性 (仅hybrid/sink-dominant模型)
// ============================================

function exp7(): any[] {
  console.log('\n========== Exp7: Sink Token Sensitivity ==========');
  const results: any[] = [];
  const models = ['mistral-7b', 'gemma-9b'];
  const bw = BW_LEVELS[2];
  const task: TaskType = 'conversation';

  for (const model of models) {
    const locality = LOCALITY_PROFILES[model];
    const params = mkParams(4096, bw.bps, task);

    for (const budget of [0.3, 0.5, 0.7]) {
      for (const sinkCount of [0, 4, 8, 16]) {
        const config = STRATEGIES['PDAwareCompression'].computeConfig(params);
        const calibrated = computeCalibratedQuality(config, model, task, sinkCount > 0);

        // 精确PPL估算 (使用拟合曲线)
        const retention = budget;
        const pplChange = estimatePPLChange(retention, locality, TASK_SENSITIVITY[task], sinkCount > 0);

        results.push({
          model,
          budget,
          sinkCount,
          pattern: locality.pattern,
          gini: locality.gini,
          estimatedPPLChange: r4(pplChange),
          quality: pplChangeToQuality(pplChange),
          sinkPreserved: sinkCount > 0,
        });
      }
    }
  }
  console.log(`  → ${results.length} results`);
  return results;
}

// ============================================
// 汇总
// ============================================

function printSummary(all: Record<string, any>) {
  console.log('\n' + '='.repeat(70));
  console.log('  CALIBRATED EXPERIMENT SUMMARY');
  console.log('='.repeat(70));

  // Exp1
  const exp1 = all.exp1 as ExpResult[];
  if (exp1?.length) {
    console.log('\n--- Exp1: Cross-Model Strategy Comparison ---');
    for (const model of MODELS) {
      const modelResults = exp1.filter(r => r.model === model);
      if (!modelResults.length) continue;
      const locality = LOCALITY_PROFILES[model];
      console.log(`\n  📊 ${model} (Gini=${locality.gini}, ${locality.pattern})`);
      const byStrat = groupBy(modelResults, 'strategy');
      for (const [name, items] of Object.entries(byStrat)) {
        const q = avg(items.map(i => i.quality));
        const p = avg(items.map(i => i.pplChangePct));
        const r = avg(items.map(i => i.transferReductionPct));
        console.log(`    ${name.padEnd(26)} Q=${q.toFixed(3)} PPL=${p >= 0 ? '+' : ''}${p.toFixed(2)}% TransRed=${r.toFixed(1)}%`);
      }
    }
  }

  // Exp4
  const exp4 = all.exp4 as ExpResult[];
  if (exp4?.length) {
    console.log('\n--- Exp4: Ablation (8K tokens, 0.5Gbps) ---');
    for (const model of MODELS) {
      const modelResults = exp4.filter(r => r.model === model);
      if (!modelResults.length) continue;
      console.log(`\n  📊 ${model}`);
      const byStrat = groupBy(modelResults, 'strategy');
      for (const [name, items] of Object.entries(byStrat)) {
        const q = avg(items.map(i => i.quality));
        const p = avg(items.map(i => i.pplChangePct));
        const pd = avg(items.map(i => i.pdDifferentiation));
        console.log(`    ${name.padEnd(16)} Q=${q.toFixed(3)} PPL=${p >= 0 ? '+' : ''}${p.toFixed(2)}% P/D_diff=${pd.toFixed(3)}`);
      }
    }
  }

  // Exp5
  const exp5 = all.exp5 as BaselineResult[];
  if (exp5?.length) {
    console.log('\n--- Exp5: Baseline Comparison ---');
    for (const model of MODELS) {
      const modelResults = exp5.filter(r => r.model === model);
      if (!modelResults.length) continue;
      console.log(`\n  📊 ${model}`);
      for (const r of modelResults) {
        console.log(
          `    ${r.taskType.padEnd(14)} ` +
          `PDTrim: ${r.pdtrim.reductionPct.toFixed(1)}%/Q=${r.pdtrim.quality.toFixed(3)} | ` +
          `KVServe: ${r.kvserve.reductionPct.toFixed(1)}%/Q=${r.kvserve.quality.toFixed(3)} | ` +
          `Ours: ${r.ours.reductionPct.toFixed(1)}%/Q=${r.ours.quality.toFixed(3)}`
        );
      }
    }
  }

  // Exp7
  const exp7 = all.exp7 as any[];
  if (exp7?.length) {
    console.log('\n--- Exp7: Sink Token Sensitivity ---');
    for (const model of ['mistral-7b', 'gemma-9b']) {
      const modelResults = exp7.filter((r: any) => r.model === model);
      if (!modelResults.length) continue;
      console.log(`\n  📊 ${model} (${modelResults[0].pattern})`);
      for (const budget of [0.3, 0.5, 0.7]) {
        const budgetResults = modelResults.filter((r: any) => r.budget === budget);
        const line = budgetResults.map((r: any) =>
          `sink=${r.sinkCount}: PPL${r.estimatedPPLChange >= 0 ? '+' : ''}${r.estimatedPPLChange.toFixed(2)}%`
        ).join(' | ');
        console.log(`    budget=${budget}: ${line}`);
      }
    }
  }

  // Exp6
  const exp6 = all.exp6 as ExpResult[];
  if (exp6?.length) {
    console.log('\n--- Exp6: Long Context Scaling (1Gbps, conversation, PD+Task) ---');
    for (const model of MODELS) {
      const modelResults = exp6.filter(r => r.model === model && r.strategy === 'PDTaskAwareCompression');
      if (!modelResults.length) continue;
      console.log(`\n  📊 ${model}`);
      for (const r of modelResults) {
        console.log(
          `    ${(r.seqLen + ' tok').padEnd(10)} ` +
          `Transfer=${r.kvTransferTimeMs.toFixed(1)}ms Full=${r.fullTransferTimeMs.toFixed(1)}ms ` +
          `Reduction=${r.transferReductionPct.toFixed(1)}% Q=${r.quality.toFixed(3)}`
        );
      }
    }
  }
}

// ============================================
// Main
// ============================================

function main() {
  console.log('╔═══════════════════════════════════════════════════════════════╗');
  console.log('║  PD-Aware + Task-Aware KV Compression Experiments (v2)       ║');
  console.log('║  Calibrated Quality Model — Date: ' + new Date().toISOString().split('T')[0] + '               ║');
  console.log('╚═══════════════════════════════════════════════════════════════╝');

  const all: Record<string, any> = {};
  all.exp1 = exp1();
  all.exp2 = exp2();
  all.exp3 = exp3();
  all.exp4 = exp4();
  all.exp5 = exp5();
  all.exp6 = exp6();
  all.exp7 = exp7();

  // Save
  const outPath = './experiment_logs/new-paper-experiments-v2.json';
  fs.writeFileSync(outPath, JSON.stringify(all, null, 2));
  console.log(`\n✅ Results saved to ${outPath}`);

  printSummary(all);

  // 关键发现摘要
  console.log('\n' + '='.repeat(70));
  console.log('  KEY FINDINGS');
  console.log('='.repeat(70));
  
  // 找出最佳策略
  const allResults = all.exp1 as ExpResult[];
  const byStrat = groupBy(allResults, 'strategy');
  const stratAvg: Record<string, { quality: number; reduction: number }> = {};
  for (const [name, items] of Object.entries(byStrat)) {
    stratAvg[name] = {
      quality: avg(items.map(i => i.quality)),
      reduction: avg(items.map(i => i.transferReductionPct)),
    };
  }
  
  // 找出quality × reduction综合最优
  let bestStrategy = '';
  let bestScore = -1;
  for (const [name, s] of Object.entries(stratAvg)) {
    // 综合得分 = quality * 0.6 + (reduction/100) * 0.4
    const score = s.quality * 0.6 + (s.reduction / 100) * 0.4;
    if (score > bestScore) {
      bestScore = score;
      bestStrategy = name;
    }
  }
  
  console.log(`\n  🏆 Best overall strategy: ${bestStrategy} (score=${bestScore.toFixed(4)})`);
  
  // PD-Aware vs Uniform
  const pdAware = stratAvg['PDAwareCompression'];
  const uniform = stratAvg['UniformCompression'];
  if (pdAware && uniform) {
    console.log(`  📈 PD-Aware vs Uniform: Quality +${((pdAware.quality - uniform.quality) * 100).toFixed(1)}pp, Reduction +${(pdAware.reduction - uniform.reduction).toFixed(1)}pp`);
  }
  
  // PD+Task vs PD-only
  const pdTask = stratAvg['PDTaskAwareCompression'];
  const pdOnly = stratAvg['PDAwareCompression'];
  if (pdTask && pdOnly) {
    console.log(`  📈 PD+Task vs PD-only: Quality ${pdTask.quality > pdOnly.quality ? '+' : ''}${((pdTask.quality - pdOnly.quality) * 100).toFixed(1)}pp, Reduction ${pdTask.reduction > pdOnly.reduction ? '+' : ''}${(pdTask.reduction - pdOnly.reduction).toFixed(1)}pp`);
  }
}

main();
