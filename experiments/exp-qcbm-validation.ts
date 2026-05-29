/**
 * 实验: Quality-Constrained Bandwidth Minimization (QCBM) 验证
 * 
 * 目标：
 * 1. 验证QCBM在满足质量约束的前提下，带宽是否低于PD-Task-Aware
 * 2. 对比6种策略在4种模型×4种任务下的综合表现
 * 3. 生成质量-带宽Pareto前沿
 * 4. 分析QCBM的层分配合理性
 */

import { CompressionOrchestrator } from '../src/compression/CompressionOrchestrator.js';
import { NoneCompression } from '../src/compression/strategies/NoneCompression.js';
import { UniformCompression } from '../src/compression/strategies/UniformCompression.js';
import { PDAwareCompression } from '../src/compression/strategies/PDAwareCompression.js';
import { TaskAwareCompression } from '../src/compression/strategies/TaskAwareCompression.js';
import { PDTaskAwareCompression } from '../src/compression/strategies/PDTaskAwareCompression.js';
import { QualityConstrainedCompression } from '../src/compression/strategies/QualityConstrainedCompression.js';
import { CompressionParams, CompressionOutput, round4 } from '../src/core/types.js';
import { computeCalibratedQuality } from '../src/core/CalibratedQualityModel.js';

// ============================================
// 实验配置
// ============================================

const MODELS = ['qwen-7b', 'qwen-14b', 'mistral-7b', 'gemma-9b'] as const;
const TASKS = ['math', 'code', 'qa', 'conversation'] as const;
const BANDWIDTHS = [
  { name: '1Gbps', value: 125 },    // 1Gbps = 125MB/s = 125 bytes/ms
  { name: '5Gbps', value: 625 },
  { name: '10Gbps', value: 1250 },
  { name: '50Gbps', value: 6250 },
];
const QUALITY_TARGETS = [0.90, 0.95, 0.99, 1.00];

const TOTAL_LAYERS = 28; // Qwen2.5-7B has 28 layers

// ============================================
// 工具函数
// ============================================

function createOrchestrator(): CompressionOrchestrator {
  const oc = new CompressionOrchestrator();
  oc.registerStrategy(new NoneCompression());
  oc.registerStrategy(new UniformCompression());
  oc.registerStrategy(new PDAwareCompression());
  oc.registerStrategy(new TaskAwareCompression());
  oc.registerStrategy(new PDTaskAwareCompression());
  return oc;
}

function evalQuality(config: CompressionOutput, model: string, task: string): {
  quality: number; pplChange: number; pAvg: number; dAvg: number; pdDiff: number;
} {
  // 使用修正后的物理模型：effectiveRetention = pAvg
  const result = computeCalibratedQuality(config, model, task as any, true, 'transmission');
  return {
    quality: round4(result.quality),
    pplChange: round4(result.pplChangePct),
    pAvg: round4(result.pAvgRetention),
    dAvg: round4(result.dAvgRetention),
    pdDiff: round4(result.pdDifferentiation),
  };
}

// ============================================
// 实验1: 6策略全对比（4模型×4任务×4带宽）
// ============================================

function experiment1(): void {
  console.log('\n' + '='.repeat(80));
  console.log('实验1: 6策略全对比');
  console.log('='.repeat(80));
  
  const oc = createOrchestrator();
  
  // 记录各策略的综合得分
  const strategyScores: Record<string, { quality: number[]; bandwidthSaving: number[]; count: number }> = {};
  
  for (const model of MODELS) {
    for (const task of TASKS) {
      for (const bw of BANDWIDTHS) {
        const params: CompressionParams = {
          totalLayers: TOTAL_LAYERS,
          totalTokens: 4096,
          bandwidthBytesPerMs: bw.value,
          gpuMemoryBytes: 40 * 1024 * 1024 * 1024,
          currentMemoryUsage: 20 * 1024 * 1024 * 1024,
          taskType: task,
          sloLatencyMs: 500,
        };
        
        // 评估5种已有策略
        for (const [name, strategy] of oc['strategies'] as Map<string, any>) {
          try {
            const config = strategy.computeConfig(params);
            const q = evalQuality(config, model, task);
            
            if (!strategyScores[name]) {
              strategyScores[name] = { quality: [], bandwidthSaving: [], count: 0 };
            }
            strategyScores[name].quality.push(q.quality);
            strategyScores[name].bandwidthSaving.push(config.estimatedBandwidthSaving);
            strategyScores[name].count++;
          } catch (e) {
            // skip failed
          }
        }
        
        // 评估QCBM在不同quality target下
        for (const qt of QUALITY_TARGETS) {
          const qcbm = new QualityConstrainedCompression({ qualityTarget: qt, modelName: model });
          try {
            const config = qcbm.computeConfig(params);
            const q = evalQuality(config, model, task);
            const name = `QCBM(qt=${qt})`;
            
            if (!strategyScores[name]) {
              strategyScores[name] = { quality: [], bandwidthSaving: [], count: 0 };
            }
            strategyScores[name].quality.push(q.quality);
            strategyScores[name].bandwidthSaving.push(config.estimatedBandwidthSaving);
            strategyScores[name].count++;
          } catch (e) {
            // skip
          }
        }
      }
    }
  }
  
  // 输出汇总
  console.log('\n📊 策略综合对比（4模型×4任务×4带宽 = 64组）');
  console.log('-'.repeat(90));
  console.log('策略'.padEnd(30) + '平均质量'.padEnd(12) + '平均带宽节省'.padEnd(14) + '综合得分'.padEnd(12));
  console.log('-'.repeat(90));
  
  const results: Array<{ name: string; avgQuality: number; avgSaving: number; score: number }> = [];
  
  for (const [name, data] of Object.entries(strategyScores)) {
    const avgQ = data.quality.reduce((a, b) => a + b, 0) / data.quality.length;
    const avgS = data.bandwidthSaving.reduce((a, b) => a + b, 0) / data.bandwidthSaving.length;
    // 综合得分: Quality×0.6 + BandwidthSaving×0.4
    const score = avgQ * 0.6 + avgS * 0.4;
    results.push({ name, avgQuality: round4(avgQ), avgSaving: round4(avgS), score: round4(score) });
  }
  
  results.sort((a, b) => b.score - a.score);
  for (const r of results) {
    const marker = r.name.includes('QCBM') ? ' ⭐' : '';
    console.log(
      r.name.padEnd(30) + 
      r.avgQuality.toFixed(4).padEnd(12) + 
      (r.avgSaving * 100).toFixed(1) + '%'.padEnd(14) + 
      r.score.toFixed(4).padEnd(12) + marker
    );
  }
}

// ============================================
// 实验2: QCBM vs PD-Task-Aware 逐场景对比
// ============================================

function experiment2(): void {
  console.log('\n' + '='.repeat(80));
  console.log('实验2: QCBM vs PD-Task-Aware 逐场景对比');
  console.log('='.repeat(80));
  
  const pdta = new PDTaskAwareCompression();
  
  // 选关键场景
  const scenarios = [
    { model: 'gemma-9b', task: 'qa', bw: 125, desc: 'hybrid模型+QA+低带宽' },
    { model: 'gemma-9b', task: 'math', bw: 125, desc: 'hybrid模型+数学+低带宽' },
    { model: 'qwen-7b', task: 'code', bw: 125, desc: 'local模型+代码+低带宽' },
    { model: 'mistral-7b', task: 'qa', bw: 125, desc: 'sink模型+QA+低带宽' },
    { model: 'gemma-9b', task: 'conversation', bw: 625, desc: 'hybrid模型+对话+中带宽' },
    { model: 'qwen-14b', task: 'math', bw: 1250, desc: 'local14B+数学+高带宽' },
  ];
  
  for (const s of scenarios) {
    console.log(`\n📌 ${s.desc} (${s.model}, ${s.task}, ${s.bw}B/ms)`);
    console.log('-'.repeat(70));
    
    const params: CompressionParams = {
      totalLayers: TOTAL_LAYERS,
      totalTokens: 4096,
      bandwidthBytesPerMs: s.bw,
      gpuMemoryBytes: 40 * 1024 * 1024 * 1024,
      currentMemoryUsage: 20 * 1024 * 1024 * 1024,
      taskType: s.task,
      sloLatencyMs: 500,
    };
    
    // PD-Task-Aware
    const pdtaConfig = pdta.computeConfig(params);
    const pdtaQ = evalQuality(pdtaConfig, s.model, s.task);
    
    // QCBM (quality target = 0.95)
    const qcbm = new QualityConstrainedCompression({ qualityTarget: 0.95, modelName: s.model });
    const qcbmConfig = qcbm.computeConfig(params);
    const qcbmQ = evalQuality(qcbmConfig, s.model, s.task);
    
    console.log('策略              | 质量   | PPL变化% | P端均保留 | D端均保留 | 带宽节省%');
    console.log('-'.repeat(70));
    console.log(
      'PD-Task-Aware     '.padEnd(18) + '|' +
      pdtaQ.quality.toFixed(4).padStart(8) + ' |' +
      (pdtaQ.pplChange >= 0 ? '+' : '') + pdtaQ.pplChange.toFixed(2) + '%'.padStart(7) + ' |' +
      pdtaQ.pAvg.toFixed(3).padStart(10) + ' |' +
      pdtaQ.dAvg.toFixed(3).padStart(10) + ' |' +
      (pdtaConfig.estimatedBandwidthSaving * 100).toFixed(1)
    );
    console.log(
      'QCBM(qt=0.95)     '.padEnd(18) + '|' +
      qcbmQ.quality.toFixed(4).padStart(8) + ' |' +
      (qcbmQ.pplChange >= 0 ? '+' : '') + qcbmQ.pplChange.toFixed(2) + '%'.padStart(7) + ' |' +
      qcbmQ.pAvg.toFixed(3).padStart(10) + ' |' +
      qcbmQ.dAvg.toFixed(3).padStart(10) + ' |' +
      (qcbmConfig.estimatedBandwidthSaving * 100).toFixed(1)
    );
    
    // 计算QCBM改善
    const bwImprovement = (qcbmConfig.estimatedBandwidthSaving - pdtaConfig.estimatedBandwidthSaving) * 100;
    const qualityDiff = qcbmQ.quality - pdtaQ.quality;
    console.log(`→ QCBM改善: 带宽节省${bwImprovement >= 0 ? '+' : ''}${bwImprovement.toFixed(1)}pp, 质量${qualityDiff >= 0 ? '+' : ''}${(qualityDiff * 100).toFixed(1)}pp`);
  }
}

// ============================================
// 实验3: 质量-带宽Pareto前沿
// ============================================

function experiment3(): void {
  console.log('\n' + '='.repeat(80));
  console.log('实验3: 质量-带宽Pareto前沿（QCBM不同quality target）');
  console.log('='.repeat(80));
  
  for (const model of MODELS) {
    console.log(`\n📊 ${model}`);
    console.log('-'.repeat(60));
    
    const params: CompressionParams = {
      totalLayers: TOTAL_LAYERS,
      totalTokens: 4096,
      bandwidthBytesPerMs: 125, // 1Gbps
      gpuMemoryBytes: 40 * 1024 * 1024 * 1024,
      currentMemoryUsage: 20 * 1024 * 1024 * 1024,
      taskType: 'qa',
      sloLatencyMs: 500,
    };
    
    const qcbm = new QualityConstrainedCompression({ modelName: model });
    const curve = qcbm.searchQualityBudgetCurve(params, [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99, 1.00]);
    
    console.log('质量目标 | 实际质量 | P端均保留 | 带宽节省% | 精度');
    console.log('-'.repeat(60));
    for (const point of curve) {
      console.log(
        point.qualityTarget.toFixed(2).padStart(8) + ' |' +
        point.quality.toFixed(4).padStart(9) + ' |' +
        point.avgPRetention.toFixed(4).padStart(10) + ' |' +
        (point.bandwidthSaving * 100).toFixed(1).padStart(10) + ' |' +
        point.precision.padStart(6)
      );
    }
  }
}

// ============================================
// 实验4: QCBM层分配可视化（Gemma-9B, QA任务）
// ============================================

function experiment4(): void {
  console.log('\n' + '='.repeat(80));
  console.log('实验4: QCBM层分配 vs PD-Task-Aware层分配（Gemma-9B, QA, 1Gbps）');
  console.log('='.repeat(80));
  
  const params: CompressionParams = {
    totalLayers: TOTAL_LAYERS,
    totalTokens: 4096,
    bandwidthBytesPerMs: 125,
    gpuMemoryBytes: 40 * 1024 * 1024 * 1024,
    currentMemoryUsage: 20 * 1024 * 1024 * 1024,
    taskType: 'qa',
    sloLatencyMs: 500,
  };
  
  const pdta = new PDTaskAwareCompression();
  const pdtaConfig = pdta.computeConfig(params);
  
  const qcbm = new QualityConstrainedCompression({ qualityTarget: 0.95, modelName: 'gemma-9b' });
  const qcbmConfig = qcbm.computeConfig(params);
  
  console.log('\nLayer | PD-Task P端 | QCBM P端 | PD-Task D端 | QCBM D端 | 差异(P端)');
  console.log('-'.repeat(70));
  
  for (let i = 0; i < TOTAL_LAYERS; i++) {
    const pdP = pdtaConfig.pLayerRetention[i];
    const qcP = qcbmConfig.pLayerRetention[i];
    const pdD = pdtaConfig.dLayerRetention[i];
    const qcD = qcbmConfig.dLayerRetention[i];
    const diff = qcP - pdP;
    
    const segment = i < 9 ? '低层' : i < 19 ? '中层' : '高层';
    console.log(
      `L${String(i).padStart(2, '0')}(${segment}) | ` +
      pdP.toFixed(3).padStart(11) + ' | ' +
      qcP.toFixed(3).padStart(8) + ' | ' +
      pdD.toFixed(3).padStart(11) + ' | ' +
      qcD.toFixed(3).padStart(8) + ' | ' +
      (diff >= 0 ? '+' : '') + diff.toFixed(3)
    );
  }
  
  // 段统计
  const segments = [
    { name: '低层(L00-L09)', start: 0, end: 9 },
    { name: '中层(L10-L18)', start: 9, end: 19 },
    { name: '高层(L19-L27)', start: 19, end: 28 },
  ];
  
  console.log('\n📊 段统计对比');
  console.log('-'.repeat(70));
  console.log('段              | PD-Task P端均 | QCBM P端均 | PD-Task D端均 | QCBM D端均');
  
  for (const seg of segments) {
    const pdPavg = pdtaConfig.pLayerRetention.slice(seg.start, seg.end).reduce((a, b) => a + b, 0) / (seg.end - seg.start);
    const qcPavg = qcbmConfig.pLayerRetention.slice(seg.start, seg.end).reduce((a, b) => a + b, 0) / (seg.end - seg.start);
    const pdDavg = pdtaConfig.dLayerRetention.slice(seg.start, seg.end).reduce((a, b) => a + b, 0) / (seg.end - seg.start);
    const qcDavg = qcbmConfig.dLayerRetention.slice(seg.start, seg.end).reduce((a, b) => a + b, 0) / (seg.end - seg.start);
    
    console.log(
      seg.name.padEnd(16) + '| ' +
      pdPavg.toFixed(3).padStart(14) + ' | ' +
      qcPavg.toFixed(3).padStart(11) + ' | ' +
      pdDavg.toFixed(3).padStart(14) + ' | ' +
      qcDavg.toFixed(3).padStart(11)
    );
  }
}

// ============================================
// 主入口
// ============================================

function main(): void {
  console.log('🔬 kvcache-lab QCBM验证实验');
  console.log('日期: 2026-05-29');
  console.log('='.repeat(80));
  
  experiment1();  // 6策略全对比
  experiment2();  // QCBM vs PD-Task-Aware
  experiment3();  // Pareto前沿
  experiment4();  // 层分配可视化
  
  console.log('\n✅ 实验完成');
}

main();
