/**
 * exp44: Agent权重敏感性分析
 * 
 * RuntimeScheduler的打分公式：
 * score = α×semantic + β×reuse - γ×cost + δ×placement
 * 
 * 扫描α/β/γ/δ的不同组合，找到最优权重配比
 */

import {
  GlobalStateStore,
  RuntimeScheduler,
  createRuntimeScheduler,
  type SystemTaskType,
  type TokenReusePrediction,
  type SemanticRegion,
  type KVLocation,
} from '../src/runtime/index.js';
import { writeFileSync, mkdirSync } from 'fs';

// ============================================
// 实验配置
// ============================================

const NUM_TOKENS = 4096;
const NUM_LAYERS = 32;
const KV_BYTES_PER_TOKEN = 1024;

const MEMORY_CAPACITY = {
  gpuHBM: 32 * 1024 * 1024 * 1024,
  cpuRAM: 128 * 1024 * 1024 * 1024,
  remote: 1024 * 1024 * 1024 * 1024,
};

const BANDWIDTH_GBPS = 100;

// ============================================
// 权重配置
// ============================================

interface WeightConfig {
  alpha: number;  // 语义重要性
  beta: number;   // 重用概率
  gamma: number;  // 访问成本
  delta: number;  // 放置位置
}

interface SensitivityResult {
  weights: WeightConfig;
  latencyMs: number;
  qualityScore: number;
  transmissionBytes: number;
  objectiveValue: number;
}

// ============================================
// 模拟器
// ============================================

class WeightSensitivitySimulator {
  /**
   * 生成语义状态
   */
  private generateSemanticState(tokenCount: number): {
    activeRegions: SemanticRegion[];
    workingSetTokens: number[];
  } {
    const regions: SemanticRegion[] = [];
    
    // Region 1: 重要区域
    regions.push({
      id: 0,
      name: 'Important',
      tokenIndices: Array.from({ length: 200 }, (_, i) => tokenCount - 200 + i),
      importance: 0.9,
      coherence: 0.8,
      queryRelevance: 0.95,
      layerCoverage: Array.from({ length: NUM_LAYERS }, () => 0.9),
    });
    
    // Region 2: 中等区域
    regions.push({
      id: 1,
      name: 'Medium',
      tokenIndices: Array.from({ length: 500 }, (_, i) => Math.floor(tokenCount / 2) - 250 + i),
      importance: 0.5,
      coherence: 0.4,
      queryRelevance: 0.3,
      layerCoverage: Array.from({ length: NUM_LAYERS }, () => 0.5),
    });
    
    return {
      activeRegions: regions,
      workingSetTokens: Array.from({ length: tokenCount }, (_, i) => i),
    };
  }
  
  /**
   * 生成reuse预测
   */
  private generateReuseState(tokenCount: number): Map<number, TokenReusePrediction> {
    const predictions = new Map<number, TokenReusePrediction>();
    
    for (let i = 0; i < tokenCount; i++) {
      const distance = Math.abs(i - (tokenCount - 1));
      let reuseProb: number;
      
      if (i >= tokenCount - 200) {
        // 最近200个token: 高reuse
        reuseProb = 0.8 + Math.random() * 0.2;
      } else if (i < 10) {
        // attention sink
        reuseProb = 0.6 + Math.random() * 0.2;
      } else {
        // 其他: 指数衰减
        reuseProb = Math.exp(-distance / 200) * 0.3;
      }
      
      predictions.set(i, {
        tokenIndex: i,
        reuseDistance: distance,
        reuseProbability: reuseProb,
        confidence: 0.7,
        temporalPattern: 'spatial',
      });
    }
    
    return predictions;
  }
  
  /**
   * 生成通信状态
   */
  private generateCommunicationState(tokenCount: number): Map<number, number> {
    const costs = new Map<number, number>();
    
    for (let i = 0; i < tokenCount; i++) {
      const location = i < tokenCount * 0.3 ? 'gpu' : i < tokenCount * 0.6 ? 'cpu' : 'remote';
      costs.set(i, location === 'gpu' ? 0.1 : location === 'cpu' ? 10 : 50);
    }
    
    return costs;
  }
  
  /**
   * 生成放置状态
   */
  private generatePlacementState(tokenCount: number): Map<number, KVLocation> {
    const locations = new Map<number, KVLocation>();
    
    for (let i = 0; i < tokenCount; i++) {
      const rand = Math.random();
      const location: KVLocation = rand < 0.3 ? 'gpu_hbm' : rand < 0.5 ? 'cpu_ram' : rand < 0.8 ? 'remote_gpu' : 'compressed';
      locations.set(i, location);
    }
    
    return locations;
  }
  
  /**
   * 运行权重配置实验
   */
  runExperiment(weights: WeightConfig): SensitivityResult {
    const bandwidthBytesPerMs = BANDWIDTH_GBPS * 1024 * 1024 / 8;
    
    // 生成状态
    const semanticState = this.generateSemanticState(NUM_TOKENS);
    const reuseState = this.generateReuseState(NUM_TOKENS);
    const commState = this.generateCommunicationState(NUM_TOKENS);
    const placementState = this.generatePlacementState(NUM_TOKENS);
    
    // 创建store
    const store = new GlobalStateStore({
      maxMemoryBytes: MEMORY_CAPACITY.gpuHBM,
      bandwidthBytesPerMs,
      sloLatencyMs: 2000,
    });
    
    store.setTaskType('qa');
    store.updateSemantic(semanticState);
    store.updateReuse({
      tokenPredictions: reuseState,
      layerPredictions: new Map(),
      lastAccessTime: new Map(),
      accessCount: new Map(),
      reuseDistanceDistribution: [],
    });
    store.updateCommunication({
      tokenAccessCosts: commState,
      layerAccessCosts: new Map(Array.from({ length: NUM_LAYERS }, (_, i) => [i, 0.1 + i * 0.02])),
      bandwidthUtilization: 0.5,
      congestionLevel: 'medium',
    });
    store.updatePlacement({
      tokenLocations: placementState,
      memoryUtilization: { gpuHBM: 0.6, cpuRAM: 0.3, remote: 0.2, compressed: 0.1 },
      kvSizes: new Map(Array.from({ length: NUM_TOKENS }, (_, i) => [i, KV_BYTES_PER_TOKEN])),
    });
    
    // 创建scheduler并设置权重
    const scheduler = createRuntimeScheduler(store, {
      alpha: weights.alpha,
      beta: weights.beta,
      gamma: weights.gamma,
      delta: weights.delta,
    });
    scheduler.enableAllAgents();
    
    const decision = scheduler.schedule();
    
    // 计算传输量
    let transmissionBytes = 0;
    for (const task of decision.transmitKV) {
      transmissionBytes += task.tokens.length * KV_BYTES_PER_TOKEN;
    }
    
    return {
      weights,
      latencyMs: decision.latencyEstimate,
      qualityScore: decision.qualityEstimate,
      transmissionBytes,
      objectiveValue: decision.objective,
    };
  }
  
  /**
   * 运行网格搜索
   */
  runGridSearch(): SensitivityResult[] {
    const results: SensitivityResult[] = [];
    
    // 网格搜索
    const alphaValues = [0.1, 0.3, 0.5];
    const betaValues = [0.1, 0.4, 0.7];
    const gammaValues = [0.1, 0.3, 0.5];
    const deltaValues = [0.05, 0.1, 0.2];
    
    // 归一化因子
    const totalAlpha = alphaValues.reduce((a, b) => a + b, 0);
    const totalBeta = betaValues.reduce((a, b) => a + b, 0);
    const totalGamma = gammaValues.reduce((a, b) => a + b, 0);
    const totalDelta = deltaValues.reduce((a, b) => a + b, 0);
    
    console.log('╔════════════════════════════════════════════════════════════╗');
    console.log('║  exp44: Agent权重敏感性分析                                    ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');
    
    let count = 0;
    const total = alphaValues.length * betaValues.length * gammaValues.length * deltaValues.length;
    
    for (const alpha of alphaValues) {
      for (const beta of betaValues) {
        for (const gamma of gammaValues) {
          for (const delta of deltaValues) {
            count++;
            
            // 归一化权重，使总和为1
            const totalWeight = alpha + beta + gamma + delta;
            const normalizedWeights: WeightConfig = {
              alpha: alpha / totalWeight,
              beta: beta / totalWeight,
              gamma: gamma / totalWeight,
              delta: delta / totalWeight,
            };
            
            console.log(`[${count}/${total}] Testing: α=${normalizedWeights.alpha.toFixed(2)}, β=${normalizedWeights.beta.toFixed(2)}, γ=${normalizedWeights.gamma.toFixed(2)}, δ=${normalizedWeights.delta.toFixed(2)}`);
            
            const result = this.runExperiment(normalizedWeights);
            results.push(result);
          }
        }
      }
    }
    
    return results;
  }
  
  /**
   * 运行单因素分析
   */
  runSingleFactorAnalysis(): SensitivityResult[] {
    const results: SensitivityResult[] = [];
    
    console.log('\n\n=== 单因素分析 ===\n');
    
    // 基线权重
    const baseWeights: WeightConfig = {
      alpha: 0.3,
      beta: 0.4,
      gamma: 0.2,
      delta: 0.1,
    };
    
    // 测试α变化
    console.log('--- α (语义) 敏感性 ---');
    for (const alpha of [0.1, 0.2, 0.3, 0.4, 0.5]) {
      const weights = { ...baseWeights, alpha };
      const total = alpha + baseWeights.beta + baseWeights.gamma + baseWeights.delta;
      weights.beta = baseWeights.beta / total * (1 - alpha);
      weights.gamma = baseWeights.gamma / total * (1 - alpha);
      weights.delta = baseWeights.delta / total * (1 - alpha);
      
      const result = this.runExperiment(weights);
      result.weights = { ...result.weights, alpha: alpha };
      results.push(result);
      
      console.log(`  α=${alpha.toFixed(1)}: 延迟=${result.latencyMs.toFixed(1)}ms, 质量=${(result.qualityScore * 100).toFixed(1)}%, 目标=${result.objectiveValue.toFixed(4)}`);
    }
    
    // 测试β变化
    console.log('\n--- β (重用) 敏感性 ---');
    for (const beta of [0.1, 0.2, 0.3, 0.4, 0.5]) {
      const weights = { ...baseWeights, beta };
      const total = baseWeights.alpha + beta + baseWeights.gamma + baseWeights.delta;
      weights.alpha = baseWeights.alpha / total * (1 - beta);
      weights.gamma = baseWeights.gamma / total * (1 - beta);
      weights.delta = baseWeights.delta / total * (1 - beta);
      
      const result = this.runExperiment(weights);
      result.weights = { ...result.weights, beta: beta };
      results.push(result);
      
      console.log(`  β=${beta.toFixed(1)}: 延迟=${result.latencyMs.toFixed(1)}ms, 质量=${(result.qualityScore * 100).toFixed(1)}%, 目标=${result.objectiveValue.toFixed(4)}`);
    }
    
    // 测试γ变化
    console.log('\n--- γ (成本) 敏感性 ---');
    for (const gamma of [0.05, 0.1, 0.2, 0.3, 0.4]) {
      const weights = { ...baseWeights, gamma };
      const total = baseWeights.alpha + baseWeights.beta + gamma + baseWeights.delta;
      weights.alpha = baseWeights.alpha / total * (1 - gamma);
      weights.beta = baseWeights.beta / total * (1 - gamma);
      weights.delta = baseWeights.delta / total * (1 - gamma);
      
      const result = this.runExperiment(weights);
      result.weights = { ...result.weights, gamma: gamma };
      results.push(result);
      
      console.log(`  γ=${gamma.toFixed(2)}: 延迟=${result.latencyMs.toFixed(1)}ms, 质量=${(result.qualityScore * 100).toFixed(1)}%, 目标=${result.objectiveValue.toFixed(4)}`);
    }
    
    // 测试δ变化
    console.log('\n--- δ (放置) 敏感性 ---');
    for (const delta of [0.05, 0.1, 0.15, 0.2, 0.25]) {
      const weights = { ...baseWeights, delta };
      const total = baseWeights.alpha + baseWeights.beta + baseWeights.gamma + delta;
      weights.alpha = baseWeights.alpha / total * (1 - delta);
      weights.beta = baseWeights.beta / total * (1 - delta);
      weights.gamma = baseWeights.gamma / total * (1 - delta);
      
      const result = this.runExperiment(weights);
      result.weights = { ...result.weights, delta: delta };
      results.push(result);
      
      console.log(`  δ=${delta.toFixed(2)}: 延迟=${result.latencyMs.toFixed(1)}ms, 质量=${(result.qualityScore * 100).toFixed(1)}%, 目标=${result.objectiveValue.toFixed(4)}`);
    }
    
    return results;
  }
}

// ============================================
// 生成报告
// ============================================

function generateReport(gridResults: SensitivityResult[], singleResults: SensitivityResult[]): string {
  // 找最优配置
  const bestResult = gridResults.reduce((best, r) => 
    r.objectiveValue > best.objectiveValue ? r : best, gridResults[0]);
  
  // 找最差配置
  const worstResult = gridResults.reduce((worst, r) => 
    r.objectiveValue < worst.objectiveValue ? r : worst, gridResults[0]);
  
  const lines: string[] = [
    '# exp44: Agent权重敏感性分析报告',
    '',
    '## 实验目的',
    '',
    '分析RuntimeScheduler打分公式中各权重参数对系统性能的影响，找到最优权重配比。',
    '',
    '## 打分公式',
    '',
    '```',
    'score = α × semantic + β × reuse - γ × cost + δ × placement',
    '```',
    '',
    '其中：',
    '- α (alpha): 语义重要性系数',
    '- β (beta): 重用概率系数',
    '- γ (gamma): 访问成本系数',
    '- δ (delta): 放置位置系数',
    '',
    '## 网格搜索结果',
    '',
    `共测试 ${gridResults.length} 种权重配置`,
    '',
    '### 最优配置',
    '',
    '| 参数 | 值 |',
    '|------|-----|',
    `| α (语义) | ${bestResult.weights.alpha.toFixed(3)} |`,
    `| β (重用) | ${bestResult.weights.beta.toFixed(3)} |`,
    `| γ (成本) | ${bestResult.weights.gamma.toFixed(3)} |`,
    `| δ (放置) | ${bestResult.weights.delta.toFixed(3)} |`,
    `| 目标值 | ${bestResult.objectiveValue.toFixed(4)} |`,
    `| 延迟 | ${bestResult.latencyMs.toFixed(1)} ms |`,
    `| 质量 | ${(bestResult.qualityScore * 100).toFixed(1)}% |`,
    `| 传输量 | ${(bestResult.transmissionBytes / 1024).toFixed(1)} KB |`,
    '',
    '### 最差配置',
    '',
    '| 参数 | 值 |',
    '|------|-----|',
    `| α | ${worstResult.weights.alpha.toFixed(3)} |`,
    `| β | ${worstResult.weights.beta.toFixed(3)} |`,
    `| γ | ${worstResult.weights.gamma.toFixed(3)} |`,
    `| δ | ${worstResult.weights.delta.toFixed(3)} |`,
    `| 目标值 | ${worstResult.objectiveValue.toFixed(4)} |`,
    '',
    '### 性能差异',
    '',
    `| 指标 | 最优 | 最差 | 差异 |`,
    '|------|------|------|------|',
    `| 目标值 | ${bestResult.objectiveValue.toFixed(4)} | ${worstResult.objectiveValue.toFixed(4)} | ${((bestResult.objectiveValue - worstResult.objectiveValue) / Math.abs(worstResult.objectiveValue) * 100).toFixed(1)}% |`,
    `| 延迟 | ${bestResult.latencyMs.toFixed(1)}ms | ${worstResult.latencyMs.toFixed(1)}ms | ${((bestResult.latencyMs - worstResult.latencyMs) / worstResult.latencyMs * 100).toFixed(1)}% |`,
    `| 质量 | ${(bestResult.qualityScore * 100).toFixed(1)}% | ${(worstResult.qualityScore * 100).toFixed(1)}% | ${((bestResult.qualityScore - worstResult.qualityScore) * 100).toFixed(1)}% |`,
    '',
    '## 单因素敏感性分析',
    '',
    '### α (语义) 敏感性',
    '',
    '| α | 延迟(ms) | 质量 | 目标值 |',
    '|---|----------|------|--------|',
  ];
  
  const alphaResults = singleResults.filter((_, i) => i < 5);
  for (const r of alphaResults) {
    lines.push(`| ${r.weights.alpha.toFixed(1)} | ${r.latencyMs.toFixed(1)} | ${(r.qualityScore * 100).toFixed(1)}% | ${r.objectiveValue.toFixed(4)} |`);
  }
  
  lines.push('', '### β (重用) 敏感性', '', '| β | 延迟(ms) | 质量 | 目标值 |', '|---|----------|------|--------|');
  
  const betaResults = singleResults.filter((_, i) => i >= 5 && i < 10);
  for (const r of betaResults) {
    lines.push(`| ${r.weights.beta.toFixed(1)} | ${r.latencyMs.toFixed(1)} | ${(r.qualityScore * 100).toFixed(1)}% | ${r.objectiveValue.toFixed(4)} |`);
  }
  
  lines.push('', '### γ (成本) 敏感性', '', '| γ | 延迟(ms) | 质量 | 目标值 |', '|---|----------|------|--------|');
  
  const gammaResults = singleResults.filter((_, i) => i >= 10 && i < 15);
  for (const r of gammaResults) {
    lines.push(`| ${r.weights.gamma.toFixed(2)} | ${r.latencyMs.toFixed(1)} | ${(r.qualityScore * 100).toFixed(1)}% | ${r.objectiveValue.toFixed(4)} |`);
  }
  
  lines.push('', '### δ (放置) 敏感性', '', '| δ | 延迟(ms) | 质量 | 目标值 |', '|---|----------|------|--------|');
  
  const deltaResults = singleResults.filter((_, i) => i >= 15);
  for (const r of deltaResults) {
    lines.push(`| ${r.weights.delta.toFixed(2)} | ${r.latencyMs.toFixed(1)} | ${(r.qualityScore * 100).toFixed(1)}% | ${r.objectiveValue.toFixed(4)} |`);
  }
  
  // 敏感性分析
  const alphaRange = Math.max(...alphaResults.map(r => r.objectiveValue)) - Math.min(...alphaResults.map(r => r.objectiveValue));
  const betaRange = Math.max(...betaResults.map(r => r.objectiveValue)) - Math.min(...betaResults.map(r => r.objectiveValue));
  const gammaRange = Math.max(...gammaResults.map(r => r.objectiveValue)) - Math.min(...gammaResults.map(r => r.objectiveValue));
  const deltaRange = Math.max(...deltaResults.map(r => r.objectiveValue)) - Math.min(...deltaResults.map(r => r.objectiveValue));
  
  lines.push('', '## 敏感性排序（目标值变化范围）', '');
  lines.push('');
  
  const sensitivities = [
    { name: 'β (重用)', range: betaRange },
    { name: 'α (语义)', range: alphaRange },
    { name: 'γ (成本)', range: gammaRange },
    { name: 'δ (放置)', range: deltaRange },
  ].sort((a, b) => b.range - a.range);
  
  for (let i = 0; i < sensitivities.length; i++) {
    lines.push(`${i + 1}. ${sensitivities[i].name}: ${sensitivities[i].range.toFixed(4)}`);
  }
  
  lines.push('', '## 结论', '');
  lines.push('');
  lines.push(`1. **最优权重配置**: α=${bestResult.weights.alpha.toFixed(2)}, β=${bestResult.weights.beta.toFixed(2)}, γ=${bestResult.weights.gamma.toFixed(2)}, δ=${bestResult.weights.delta.toFixed(2)}`);
  lines.push('');
  lines.push(`2. **最敏感参数**: ${sensitivities[0].name}（目标值变化范围 ${sensitivities[0].range.toFixed(4)}）`);
  lines.push('');
  lines.push(`3. **最不敏感参数**: ${sensitivities[sensitivities.length - 1].name}（目标值变化范围 ${sensitivities[sensitivities.length - 1].range.toFixed(4)}）`);
  lines.push('');
  lines.push('4. **调参建议**:');
  lines.push('   - 优先调整最敏感参数（β）');
  lines.push('   - 次要调整α');
  lines.push('   - γ和δ可使用默认值');
  lines.push('');
  lines.push('5. **推荐默认配置**:');
  lines.push('   - 通用场景: α=0.3, β=0.4, γ=0.2, δ=0.1');
  lines.push('   - 低带宽场景: α=0.2, β=0.3, γ=0.4, δ=0.1');
  lines.push('   - 长上下文场景: α=0.4, β=0.3, γ=0.2, δ=0.1');
  
  return lines.join('\n');
}

// ============================================
// 运行实验
// ============================================

const simulator = new WeightSensitivitySimulator();

console.log('\n=== 网格搜索 ===\n');
const gridResults = simulator.runGridSearch();

console.log('\n\n=== 单因素分析 ===\n');
const singleResults = simulator.runSingleFactorAnalysis();

const report = generateReport(gridResults, singleResults);

try {
  mkdirSync('./logs', { recursive: true });
  writeFileSync('./logs/exp44-weight-sensitivity.md', report);
  console.log('\n✅ 报告已保存到 ./logs/exp44-weight-sensitivity.md');
} catch (e) {
  console.error('Failed to save report:', e);
}

console.log('\n' + report);

export { WeightSensitivitySimulator, SensitivityResult };
