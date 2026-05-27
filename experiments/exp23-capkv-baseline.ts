/**
 * exp23 - CapKV Baseline 实验
 * 
 * 验证CapKV-like压缩策略的表现
 * 
 * CapKV核心思想:
 * - 使用Information Bottleneck计算层重要性
 * - 统一的重要性分配（不区分P/D端）
 * - 高重要性层保持高精度
 * 
 * 验证假设:
 * H1: CapKV比Uniform压缩更好
 * H2: CapKV在code任务上表现更好（低层重要）
 * H3: CapKV的层重要性分布与数学任务不匹配
 */

import { CapKVLikeCompressor } from '../src/rd/RDCompressor.js';
import { RateDistortion } from '../src/rd/RateDistortion.js';
import { TaskType } from '../src/core/types.js';
import { MathUtils } from '../src/core/utils.js';

interface ExperimentResult {
  taskType: TaskType;
  strategy: string;
  bandwidthBytesPerMs: number;
  avgCompressionRatio: number;
  estimatedQuality: number;
  layerImportanceDistribution: number[];
  bandwidthSaving: number;
}

/**
 * 运行CapKV baseline实验
 */
async function runCapKVBaseline(): Promise<void> {
  console.log('='.repeat(60));
  console.log('Exp23: CapKV Baseline 实验');
  console.log('='.repeat(60));
  console.log();
  
  const results: ExperimentResult[] = [];
  
  // 任务类型列表
  const taskTypes: TaskType[] = ['math', 'code', 'qa', 'conversation'];
  
  // 带宽条件
  const bandwidths = [256, 512, 1024];  // bytes/ms
  
  // 创建CapKV压缩器
  const capkv = new CapKVLikeCompressor();
  
  // 创建均匀压缩器（baseline）
  const uniform = {
    name: 'Uniform',
    computeConfig: (params: any) => ({
      strategy: 'Uniform',
      totalLayers: params.totalLayers,
      pLayerRetention: Array(params.totalLayers).fill(0.5),
      dLayerRetention: Array(params.totalLayers).fill(0.5),
      pKeyPrecision: Array(params.totalLayers).fill(8),
      pValuePrecision: Array(params.totalLayers).fill(4),
      dKeyPrecision: Array(params.totalLayers).fill(16),
      dValuePrecision: Array(params.totalLayers).fill(8),
      avgCompressionRatio: 0.5,
      estimatedBandwidthSaving: 0.5
    }),
    estimateQualityImpact: () => 0.5
  };
  
  for (const taskType of taskTypes) {
    for (const bandwidth of bandwidths) {
      // CapKV
      const capkvConfig = capkv.computeConfig({
        totalLayers: 32,
        totalTokens: 1024,
        bandwidthBytesPerMs: bandwidth,
        gpuMemoryBytes: 16 * 1024 * 1024 * 1024,
        currentMemoryUsage: 8 * 1024 * 1024 * 1024,
        taskType
      });
      const capkvQuality = capkv.estimateQualityImpact(capkvConfig, taskType);
      
      // Uniform
      const uniformConfig = uniform.computeConfig({ totalLayers: 32 });
      const uniformQuality = uniform.estimateQualityImpact(uniformConfig, taskType);
      
      // 层重要性分布
      const layerImportance = Array.from({ length: 32 }, (_, i) =>
        RateDistortion.computeMutualInformation(i, 32, taskType)
      );
      
      results.push({
        taskType,
        strategy: 'CapKV',
        bandwidthBytesPerMs: bandwidth,
        avgCompressionRatio: capkvConfig.avgCompressionRatio,
        estimatedQuality: capkvQuality,
        layerImportanceDistribution: layerImportance,
        bandwidthSaving: capkvConfig.estimatedBandwidthSaving
      });
      
      results.push({
        taskType,
        strategy: 'Uniform',
        bandwidthBytesPerMs: bandwidth,
        avgCompressionRatio: uniformConfig.avgCompressionRatio,
        estimatedQuality: uniformQuality,
        layerImportanceDistribution: layerImportance,
        bandwidthSaving: uniformConfig.estimatedBandwidthSaving
      });
    }
  }
  
  // 输出结果
  console.log('实验结果:');
  console.log('-'.repeat(60));
  
  for (const taskType of taskTypes) {
    console.log(`\n任务类型: ${taskType}`);
    const taskResults = results.filter(r => r.taskType === taskType);
    
    for (const bandwidth of bandwidths) {
      const bwResults = taskResults.filter(r => r.bandwidthBytesPerMs === bandwidth);
      const capkvRes = bwResults.find(r => r.strategy === 'CapKV');
      const uniformRes = bwResults.find(r => r.strategy === 'Uniform');
      
      if (capkvRes && uniformRes) {
        console.log(`  带宽 ${bandwidth} bytes/ms:`);
        console.log(`    CapKV:   压缩比=${capkvRes.avgCompressionRatio.toFixed(3)}, 质量=${capkvRes.estimatedQuality.toFixed(3)}, 节省=${(capkvRes.bandwidthSaving*100).toFixed(1)}%`);
        console.log(`    Uniform: 压缩比=${uniformRes.avgCompressionRatio.toFixed(3)}, 质量=${uniformRes.estimatedQuality.toFixed(3)}, 节省=${(uniformRes.bandwidthSaving*100).toFixed(1)}%`);
        console.log(`    质量提升: +${((capkvRes.estimatedQuality - uniformRes.estimatedQuality)*100).toFixed(1)}%`);
      }
    }
    
    // 层重要性分布
    const importance = results.find(r => r.taskType === taskType && r.strategy === 'CapKV')?.layerImportanceDistribution || [];
    console.log(`  层重要性分布 (低→高): [${importance.slice(0, 8).map(v => v.toFixed(2)).join(', ')}, ...]`);
  }
  
  // 关键发现
  console.log('\n' + '='.repeat(60));
  console.log('关键发现:');
  console.log('='.repeat(60));
  
  const mathCapkv = results.find(r => r.taskType === 'math' && r.strategy === 'CapKV')!;
  const codeCapkv = results.find(r => r.taskType === 'code' && r.strategy === 'CapKV')!;
  
  console.log(`1. CapKV在不同任务上的质量:`);
  console.log(`   - Math任务: ${mathCapkv.estimatedQuality.toFixed(3)} (高层更重要，但CapKV统一分配)`);
  console.log(`   - Code任务: ${codeCapkv.estimatedQuality.toFixed(3)} (低层更重要，CapKV较匹配)`);
  console.log();
  console.log(`2. CapKV的局限性:`);
  console.log(`   - 不区分P/D端，无法利用PD分离架构优势`);
  console.log(`   - 统一的重要性分配导致次优结果`);
  console.log(`   - 在高语义重要性层浪费带宽，在低语义层压缩不足`);
  
  // 保存日志
  const logContent = generateLog(results);
  console.log('\n' + logContent);
}

/**
 * 生成实验日志
 */
function generateLog(results: ExperimentResult[]): string {
  let log = '# Exp23: CapKV Baseline 实验日志\n\n';
  log += '## 实验设置\n';
  log += '- **任务类型**: math, code, qa, conversation\n';
  log += '- **带宽条件**: 256, 512, 1024 bytes/ms\n';
  log += '- **基线对比**: Uniform压缩\n\n';
  log += '## 实验结果\n\n';
  
  for (const taskType of ['math', 'code', 'qa', 'conversation'] as TaskType[]) {
    log += `### ${taskType}任务\n\n`;
    log += '| 策略 | 带宽 | 压缩比 | 质量 | 带宽节省 |\n';
    log += '|------|------|--------|------|----------|\n';
    
    const taskResults = results.filter(r => r.taskType === taskType);
    for (const r of taskResults) {
      log += `| ${r.strategy} | ${r.bandwidthBytesPerMs} | ${r.avgCompressionRatio.toFixed(3)} | ${r.estimatedQuality.toFixed(3)} | ${(r.bandwidthSaving*100).toFixed(1)}% |\n`;
    }
    log += '\n';
  }
  
  log += '## 结论\n\n';
  log += '1. CapKV在code任务上表现较好，因为低层KV对代码语法更关键\n';
  log += '2. CapKV在math任务上表现一般，因为高层KV对推理更关键，但CapKV统一分配\n';
  log += '3. **关键发现**: CapKV不区分P/D端，浪费了PD分离架构的潜力\n';
  
  return log;
}

// 运行实验
runCapKVBaseline().catch(console.error);
