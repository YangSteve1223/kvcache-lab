/**
 * QCBM扩展实验：消融分析 + SWS对比
 */
import { QualityConstrainedCompression } from '../src/compression/strategies/QualityConstrainedCompression.js';
import { PDTaskAwareCompression } from '../src/compression/strategies/PDTaskAwareCompression.js';
import { CompressionParams, round4 } from '../src/core/types.js';
import { computeCalibratedQuality } from '../src/core/CalibratedQualityModel.js';

const MODELS = ['qwen-7b', 'qwen-14b', 'mistral-7b', 'gemma-9b'] as const;
const TASKS = ['math', 'code', 'qa', 'conversation'] as const;

// ============ 实验5: QCBM消融——精度vs保留率的独立贡献 ============
function ablation() {
  console.log('\n' + '='.repeat(80));
  console.log('实验5: QCBM消融分析——精度选择 vs 保留率优化的独立贡献');
  console.log('='.repeat(80));

  for (const model of MODELS) {
    console.log(`\n📊 ${model} (QA, 1Gbps)`);
    console.log('-'.repeat(80));

    const params: CompressionParams = {
      totalLayers: 28, totalTokens: 4096, bandwidthBytesPerMs: 125,
      gpuMemoryBytes: 40e9, currentMemoryUsage: 20e9, taskType: 'qa', sloLatencyMs: 500,
    };

    // Baseline: QCBM full (自适应精度 + 优化保留率)
    const qcbmFull = new QualityConstrainedCompression({ qualityTarget: 0.95, modelName: model });
    const fullConfig = qcbmFull.computeConfig(params);
    const fullQ = computeCalibratedQuality(fullConfig, model, 'qa', true, 'transmission');

    // Ablation 1: 固定K16V16精度（只优化保留率）
    const qcbmFP16 = new QualityConstrainedCompression({ qualityTarget: 0.95, modelName: model });
    // 强制用FP16 — 在computeConfig后无法修改，所以我们手动计算

    // Ablation 2: 固定保留率=0.5（只优化精度）
    // 手动构建配置
    const halfRetention = new Array(28).fill(0.5);
    const halfK16 = { ...fullConfig, pLayerRetention: halfRetention, pKeyPrecision: new Array(28).fill(16), pValuePrecision: new Array(28).fill(16) };
    const halfK8V4 = { ...fullConfig, pLayerRetention: halfRetention, pKeyPrecision: new Array(28).fill(8), pValuePrecision: new Array(28).fill(4) };
    const halfK4V4 = { ...fullConfig, pLayerRetention: halfRetention, pKeyPrecision: new Array(28).fill(4), pValuePrecision: new Array(28).fill(4) };

    const halfK16Q = computeCalibratedQuality(halfK16, model, 'qa', true, 'transmission');
    const halfK8V4Q = computeCalibratedQuality(halfK8V4, model, 'qa', true, 'transmission');
    const halfK4V4Q = computeCalibratedQuality(halfK4V4, model, 'qa', true, 'transmission');

    console.log('配置                              | 质量   | PPL变化% | 带宽成本');
    console.log('-'.repeat(80));
    console.log(`QCBM Full (自适应精度+优化保留)    | ${fullQ.quality.toFixed(4)} | ${(fullQ.pplChangePct >= 0 ? '+' : '')}${fullQ.pplChangePct.toFixed(2)}%  | ${(1 - fullConfig.estimatedBandwidthSaving).toFixed(4)}`);
    console.log(`固定50%保留 + K16V16 (FP16)       | ${halfK16Q.quality.toFixed(4)} | ${(halfK16Q.pplChangePct >= 0 ? '+' : '')}${halfK16Q.pplChangePct.toFixed(2)}%  | 0.5000`);
    console.log(`固定50%保留 + K8V4                | ${halfK8V4Q.quality.toFixed(4)} | ${(halfK8V4Q.pplChangePct >= 0 ? '+' : '')}${halfK8V4Q.pplChangePct.toFixed(2)}%  | 0.0625`);
    console.log(`固定50%保留 + K4V4                | ${halfK4V4Q.quality.toFixed(4)} | ${(halfK4V4Q.pplChangePct >= 0 ? '+' : '')}${halfK4V4Q.pplChangePct.toFixed(2)}%  | 0.0313`);
  }
}

// ============ 实验6: QCBM在不同带宽下的策略切换 ============
function bandwidthSweep() {
  console.log('\n' + '='.repeat(80));
  console.log('实验6: QCBM策略随带宽变化——精度选择与保留率自适应');
  console.log('='.repeat(80));

  const bandwidths = [
    { name: '100Mbps', val: 12.5 },
    { name: '500Mbps', val: 62.5 },
    { name: '1Gbps', val: 125 },
    { name: '5Gbps', val: 625 },
    { name: '10Gbps', val: 1250 },
    { name: '25Gbps', val: 3125 },
    { name: '50Gbps', val: 6250 },
    { name: '100Gbps', val: 12500 },
  ];

  for (const model of ['qwen-7b', 'gemma-9b'] as const) {
    console.log(`\n📊 ${model} (QA)`);
    console.log('-'.repeat(70));
    console.log('带宽      | P端均保留 | 精度   | 质量   | PPL变化% | 带宽节省%');
    console.log('-'.repeat(70));

    for (const bw of bandwidths) {
      const params: CompressionParams = {
        totalLayers: 28, totalTokens: 4096, bandwidthBytesPerMs: bw.val,
        gpuMemoryBytes: 40e9, currentMemoryUsage: 20e9, taskType: 'qa', sloLatencyMs: 500,
      };
      const qcbm = new QualityConstrainedCompression({ qualityTarget: 0.95, modelName: model });
      const config = qcbm.computeConfig(params);
      const q = computeCalibratedQuality(config, model, 'qa', true, 'transmission');

      const prec = `${config.pKeyPrecision[0]}K${config.pValuePrecision[0]}V`;
      const pAvg = config.pLayerRetention.reduce((a, b) => a + b, 0) / 28;

      console.log(
        bw.name.padEnd(10) + '| ' +
        pAvg.toFixed(3).padStart(10) + ' | ' +
        prec.padStart(6) + ' | ' +
        q.quality.toFixed(4).padStart(7) + ' | ' +
        (q.pplChangePct >= 0 ? '+' : '') + q.pplChangePct.toFixed(2) + '%'.padStart(7) + ' | ' +
        (config.estimatedBandwidthSaving * 100).toFixed(1)
      );
    }
  }
}

// ============ 实验7: QCBM + SWS联合——先选token再压缩 ============
function swsQcbmJoint() {
  console.log('\n' + '='.repeat(80));
  console.log('实验7: QCBM与SWS的联合效果分析（概念验证）');
  console.log('='.repeat(80));
  console.log('\nSWS选择50% hot tokens → QCBM对这50%做压缩');
  console.log('对比：全量KV传输 vs SWS-only vs QCBM-only vs SWS+QCBM联合\n');

  for (const model of MODELS) {
    console.log(`\n📊 ${model} (QA, 1Gbps, 4K tokens)`);
    console.log('-'.repeat(70));

    // 场景：4K tokens，7B模型
    // 全量KV大小：4000 * 28层 * (128*2 key + 128*2 value) * 2 bytes = ~57MB
    const totalKV_MB = (4000 * 28 * 256 * 2) / (1024 * 1024);

    const scenarios = [
      { name: '全量FP16传输', retention: 1.0, keyBits: 16, valBits: 16 },
      { name: 'SWS-only (50%FP16)', retention: 0.5, keyBits: 16, valBits: 16 },
      { name: 'QCBM-only (100%K4V4)', retention: 1.0, keyBits: 4, valBits: 4 },
      { name: 'SWS+QCBM (50%K4V4)', retention: 0.5, keyBits: 4, valBits: 4 },
      { name: 'SWS+QCBM (30%K4V4)', retention: 0.3, keyBits: 4, valBits: 4 },
    ];

    console.log('方案                   | 传输量(MB) | 节省%  | 说明');
    console.log('-'.repeat(70));

    for (const s of scenarios) {
      const bwFactor = (s.keyBits / 16) * (s.valBits / 16);
      const txMB = totalKV_MB * s.retention * bwFactor;
      const saving = ((1 - txMB / totalKV_MB) * 100).toFixed(1);
      console.log(
        s.name.padEnd(23) + '| ' +
        txMB.toFixed(2).padStart(10) + ' | ' +
        saving.padStart(5) + '% | '
      );
    }
  }
}

ablation();
bandwidthSweep();
swsQcbmJoint();

console.log('\n✅ 扩展实验完成');
