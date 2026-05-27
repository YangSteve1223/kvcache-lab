/**
 * 压缩策略测试套件
 * 使用 tsx 直接运行，无外部测试框架依赖
 */

import {
  CompressionOrchestrator,
  NoneCompression,
  UniformCompression,
  PDAwareCompression,
  TaskAwareCompression,
  clamp,
  ensureRetentionRange
} from '../src/compression/index.js';

// 测试计数器
let passed = 0;
let failed = 0;

// 断言辅助函数
function assert(condition: boolean, message: string) {
  if (condition) {
    passed++;
    console.log(`  ✓ ${message}`);
  } else {
    failed++;
    console.log(`  ✗ ${message}`);
    throw new Error(`Assertion failed: ${message}`);
  }
}

// 测试参数工厂函数
function createDefaultParams() {
  return {
    totalLayers: 32,
    totalTokens: 1000,
    bandwidthBytesPerMs: 100,
    gpuMemoryBytes: 16 * 1024 * 1024 * 1024, // 16GB
    currentMemoryUsage: 12 * 1024 * 1024 * 1024, // 12GB (75%)
    taskType: 'math',
    sloLatencyMs: 500,
    prefixHitRate: 0.5,
  };
}

// ==================== NoneCompression 测试 ====================
console.log('\n【测试】NoneCompression 策略');

console.log('  - 应返回全1.0的保留率');
{
  const strategy = new NoneCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  config.pLayerRetention.forEach((retention) => {
    assert(retention === 1.0, `P端保留率为 ${retention}`);
  });
  config.dLayerRetention.forEach((retention) => {
    assert(retention === 1.0, `D端保留率为 ${retention}`);
  });
}

console.log('  - 应使用16bit精度');
{
  const strategy = new NoneCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  config.pKeyPrecision.forEach((precision) => {
    assert(precision === 16, `Key精度为 ${precision}`);
  });
}

console.log('  - 应返回压缩比1.0，带宽节省0');
{
  const strategy = new NoneCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  assert(config.avgCompressionRatio === 1.0, `压缩比为 ${config.avgCompressionRatio}`);
  assert(config.estimatedBandwidthSaving === 0, `带宽节省为 ${config.estimatedBandwidthSaving}`);
}

console.log('  - 质量影响应为1.0');
{
  const strategy = new NoneCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  const quality = strategy.estimateQualityImpact(config, 'math');
  assert(quality === 1.0, `质量影响为 ${quality}`);
}

// ==================== UniformCompression 测试 ====================
console.log('\n【测试】UniformCompression 策略');

console.log('  - 低显存压力时应返回1.0保留率');
{
  const strategy = new UniformCompression();
  const params = createDefaultParams();
  params.currentMemoryUsage = params.gpuMemoryBytes * 0.7;
  const config = strategy.computeConfig(params);
  config.pLayerRetention.forEach((retention) => {
    assert(retention === 1.0, `P端保留率为 ${retention}`);
  });
}

console.log('  - 高显存压力时应动态调整保留率');
{
  const strategy = new UniformCompression();
  const params = createDefaultParams();
  params.currentMemoryUsage = params.gpuMemoryBytes * 0.95;
  const config = strategy.computeConfig(params);
  assert(config.pLayerRetention[0] < 1.0, '保留率应小于1.0');
  assert(config.pLayerRetention[0] >= 0.3, `保留率应为>=0.3，实际为${config.pLayerRetention[0]}`);
}

console.log('  - 所有层应使用统一保留率');
{
  const strategy = new UniformCompression();
  const params = createDefaultParams();
  params.currentMemoryUsage = params.gpuMemoryBytes * 0.9;
  const config = strategy.computeConfig(params);
  const firstRetention = config.pLayerRetention[0];
  config.pLayerRetention.forEach((retention) => {
    assert(retention === firstRetention, `各层保留率一致`);
  });
}

console.log('  - Key和Value应使用相同精度');
{
  const strategy = new UniformCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  assert(config.pKeyPrecision[0] === config.pValuePrecision[0], 'P端Key和Value精度相同');
}

// ==================== PDAwareCompression 测试 ====================
console.log('\n【测试】PDAwareCompression 策略');

console.log('  - P端保留率应满足: 低层 < 中层 < 高层');
{
  const strategy = new PDAwareCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  
  const totalLayers = params.totalLayers;
  const layerBound1 = Math.floor(totalLayers / 3);
  const layerBound2 = Math.floor((2 * totalLayers) / 3);
  
  const lowerAvg = config.pLayerRetention.slice(0, layerBound1).reduce((a, b) => a + b, 0) / layerBound1;
  const middleAvg = config.pLayerRetention.slice(layerBound1, layerBound2).reduce((a, b) => a + b, 0) / (layerBound2 - layerBound1);
  const upperAvg = config.pLayerRetention.slice(layerBound2).reduce((a, b) => a + b, 0) / (totalLayers - layerBound2);
  
  assert(lowerAvg < middleAvg, `低层(${lowerAvg}) < 中层(${middleAvg})`);
  assert(middleAvg < upperAvg, `中层(${middleAvg}) < 高层(${upperAvg})`);
}

console.log('  - 带宽越低，P端应越激进');
{
  const strategy = new PDAwareCompression();
  const params = createDefaultParams();
  
  params.bandwidthBytesPerMs = 200;
  const highBandwidthConfig = strategy.computeConfig(params);
  
  params.bandwidthBytesPerMs = 20;
  const lowBandwidthConfig = strategy.computeConfig(params);
  
  const highBandwidthAvg = highBandwidthConfig.pLayerRetention.reduce((a, b) => a + b, 0) / params.totalLayers;
  const lowBandwidthAvg = lowBandwidthConfig.pLayerRetention.reduce((a, b) => a + b, 0) / params.totalLayers;
  
  assert(lowBandwidthAvg < highBandwidthAvg, `低带宽(${lowBandwidthAvg}) < 高带宽(${highBandwidthAvg})`);
}

console.log('  - D端保留率不应低于0.6');
{
  const strategy = new PDAwareCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  
  config.dLayerRetention.forEach((retention) => {
    assert(retention >= 0.6, `D端保留率 ${retention} >= 0.6`);
  });
}

console.log('  - P端应使用K8V4精度配置');
{
  const strategy = new PDAwareCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  assert(config.pKeyPrecision[0] === 8, `P端Key精度为8`);
  assert(config.pValuePrecision[0] === 4, `P端Value精度为4`);
}

console.log('  - D端应使用K16V8精度配置');
{
  const strategy = new PDAwareCompression();
  const params = createDefaultParams();
  const config = strategy.computeConfig(params);
  assert(config.dKeyPrecision[0] === 16, `D端Key精度为16`);
  assert(config.dValuePrecision[0] === 8, `D端Value精度为8`);
}

// ==================== TaskAwareCompression 测试 ====================
console.log('\n【测试】TaskAwareCompression 策略');

console.log('  - math任务中高层保留率应更高');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  params.taskType = 'math';
  const config = strategy.computeConfig(params);
  
  const totalLayers = params.totalLayers;
  const layerBound2 = Math.floor((2 * totalLayers) / 3);
  
  const upperAvg = config.pLayerRetention.slice(layerBound2).reduce((a, b) => a + b, 0) / (totalLayers - layerBound2);
  const lowerAvg = config.pLayerRetention.slice(0, Math.floor(totalLayers / 3)).reduce((a, b) => a + b, 0) / Math.floor(totalLayers / 3);
  
  assert(upperAvg > lowerAvg, `math任务高层(${upperAvg}) > 低层(${lowerAvg})`);
}

console.log('  - code任务低层保留率应更高');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  params.taskType = 'code';
  const config = strategy.computeConfig(params);
  
  const totalLayers = params.totalLayers;
  const layerBound1 = Math.floor(totalLayers / 3);
  const layerBound2 = Math.floor((2 * totalLayers) / 3);
  
  const lowerAvg = config.pLayerRetention.slice(0, layerBound1).reduce((a, b) => a + b, 0) / layerBound1;
  const upperAvg = config.pLayerRetention.slice(layerBound2).reduce((a, b) => a + b, 0) / (totalLayers - layerBound2);
  
  assert(lowerAvg > upperAvg, `code任务低层(${lowerAvg}) > 高层(${upperAvg})`);
}

console.log('  - qa任务高层保留率应更高');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  params.taskType = 'qa';
  const config = strategy.computeConfig(params);
  
  const totalLayers = params.totalLayers;
  const layerBound2 = Math.floor((2 * totalLayers) / 3);
  
  const upperAvg = config.pLayerRetention.slice(layerBound2).reduce((a, b) => a + b, 0) / (totalLayers - layerBound2);
  const lowerAvg = config.pLayerRetention.slice(0, Math.floor(totalLayers / 3)).reduce((a, b) => a + b, 0) / Math.floor(totalLayers / 3);
  
  assert(upperAvg > lowerAvg, `qa任务高层(${upperAvg}) > 低层(${lowerAvg})`);
}

console.log('  - conversation任务各层保留率应较均匀');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  params.taskType = 'conversation';
  const config = strategy.computeConfig(params);
  
  const totalLayers = params.totalLayers;
  const layerBound1 = Math.floor(totalLayers / 3);
  const layerBound2 = Math.floor((2 * totalLayers) / 3);
  
  const lowerAvg = config.pLayerRetention.slice(0, layerBound1).reduce((a, b) => a + b, 0) / layerBound1;
  const middleAvg = config.pLayerRetention.slice(layerBound1, layerBound2).reduce((a, b) => a + b, 0) / (layerBound2 - layerBound1);
  const upperAvg = config.pLayerRetention.slice(layerBound2).reduce((a, b) => a + b, 0) / (totalLayers - layerBound2);
  
  const mean = (lowerAvg + middleAvg + upperAvg) / 3;
  const variance = (Math.pow(lowerAvg - mean, 2) + Math.pow(middleAvg - mean, 2) + Math.pow(upperAvg - mean, 2)) / 3;
  
  assert(variance < 0.03, `conversation任务各层方差(${variance}) < 0.03`);
}

// ==================== CompressionOrchestrator 测试 ====================
console.log('\n【测试】CompressionOrchestrator 编排器');

console.log('  - 应能注册和选择策略');
{
  const orchestrator = new CompressionOrchestrator();
  const strategy = new NoneCompression();
  orchestrator.registerStrategy(strategy);
  const selected = orchestrator.selectStrategy(createDefaultParams());
  assert(selected.name === 'NoneCompression', `选中策略: ${selected.name}`);
}

console.log('  - 应根据参数自动选择最优策略');
{
  const orchestrator = new CompressionOrchestrator();
  orchestrator.registerStrategy(new NoneCompression());
  orchestrator.registerStrategy(new UniformCompression());
  orchestrator.registerStrategy(new PDAwareCompression());
  orchestrator.registerStrategy(new TaskAwareCompression());
  
  // 无压缩需求时应选择None
  const params1 = createDefaultParams();
  params1.currentMemoryUsage = params1.gpuMemoryBytes * 0.5;
  params1.bandwidthBytesPerMs = 200;
  const selected1 = orchestrator.selectStrategy(params1);
  assert(selected1.type === 'none', `无压缩需求选中: ${selected1.type}`);
  
  // 有压缩需求时应选择TaskAware或PDAware
  const params2 = createDefaultParams();
  params2.taskType = 'code';
  const selected2 = orchestrator.selectStrategy(params2);
  assert(['pd-aware', 'task-aware'].includes(selected2.type), `压缩需求选中: ${selected2.type}`);
}

console.log('  - 应能对比所有策略');
{
  const orchestrator = new CompressionOrchestrator();
  orchestrator.registerStrategy(new NoneCompression());
  orchestrator.registerStrategy(new UniformCompression());
  orchestrator.registerStrategy(new PDAwareCompression());
  orchestrator.registerStrategy(new TaskAwareCompression());
  
  const comparisons = orchestrator.compareStrategies(createDefaultParams());
  assert(comparisons.size === 4, `对比了${comparisons.size}个策略`);
  assert(comparisons.has('NoneCompression'), '包含NoneCompression');
  assert(comparisons.has('UniformCompression'), '包含UniformCompression');
  assert(comparisons.has('PDAwareCompression'), '包含PDAwareCompression');
  assert(comparisons.has('TaskAwareCompression'), '包含TaskAwareCompression');
}

console.log('  - computeOptimalConfig应返回选中策略的配置');
{
  const orchestrator = new CompressionOrchestrator();
  orchestrator.registerStrategy(new TaskAwareCompression());
  orchestrator.registerStrategy(new PDAwareCompression());
  
  const params = createDefaultParams();
  const config = orchestrator.computeOptimalConfig(params);
  assert(config.totalLayers === params.totalLayers, `总层数: ${config.totalLayers}`);
  assert(config.pLayerRetention.length === params.totalLayers, `保留率数组长度: ${config.pLayerRetention.length}`);
}

// ==================== 边界条件测试 ====================
console.log('\n【测试】边界条件');

console.log('  - 保留率应始终在[0.1, 1.0]范围内');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  
  // 极端低带宽
  params.bandwidthBytesPerMs = 1;
  const config1 = strategy.computeConfig(params);
  config1.pLayerRetention.forEach((r) => {
    assert(r >= 0.1 && r <= 1.0, `保留率${r}在有效范围内`);
  });
  
  // 极端高显存
  params.bandwidthBytesPerMs = 100;
  params.currentMemoryUsage = params.gpuMemoryBytes * 0.99;
  const config2 = strategy.computeConfig(params);
  config2.pLayerRetention.forEach((r) => {
    assert(r >= 0.1 && r <= 1.0, `保留率${r}在有效范围内`);
  });
}

console.log('  - 应处理小层数模型');
{
  const strategy = new PDAwareCompression();
  const params = createDefaultParams();
  params.totalLayers = 3;
  const config = strategy.computeConfig(params);
  assert(config.pLayerRetention.length === 3, `保留率数组长度: ${config.pLayerRetention.length}`);
  assert(config.dLayerRetention.length === 3, `D端保留率数组长度: ${config.dLayerRetention.length}`);
}

console.log('  - 应处理大层数模型');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  params.totalLayers = 128;
  const config = strategy.computeConfig(params);
  assert(config.pLayerRetention.length === 128, `保留率数组长度: ${config.pLayerRetention.length}`);
}

console.log('  - 应处理未知任务类型');
{
  const strategy = new TaskAwareCompression();
  const params = createDefaultParams();
  params.taskType = 'unknown-task';
  const config = strategy.computeConfig(params);
  assert(config.pLayerRetention.length === params.totalLayers, `支持未知任务类型`);
}

// ==================== 工具函数测试 ====================
console.log('\n【测试】工具函数');

console.log('  - clamp应正确限制数值范围');
{
  assert(clamp(5, 0, 10) === 5, `clamp(5, 0, 10) = 5`);
  assert(clamp(-5, 0, 10) === 0, `clamp(-5, 0, 10) = 0`);
  assert(clamp(15, 0, 10) === 10, `clamp(15, 0, 10) = 10`);
}

console.log('  - ensureRetentionRange应确保保留率在有效范围内');
{
  assert(ensureRetentionRange(0.05) === 0.1, `ensureRetentionRange(0.05) = 0.1`);
  assert(ensureRetentionRange(1.5) === 1.0, `ensureRetentionRange(1.5) = 1.0`);
  assert(ensureRetentionRange(0.5) === 0.5, `ensureRetentionRange(0.5) = 0.5`);
}

// ==================== 测试结果汇总 ====================
console.log('\n' + '='.repeat(50));
console.log(`测试结果: ${passed} 通过, ${failed} 失败`);
console.log('='.repeat(50));

if (failed > 0) {
  console.error(`\n有 ${failed} 个测试失败!`);
  process.exit(1);
} else {
  console.log('\n✓ 所有测试通过!');
  process.exit(0);
}
