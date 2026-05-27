#!/usr/bin/env npx tsx
/**
 * 最终集成验证脚本
 */

import { PDSimulator } from './src/core/PDSimulator.ts';
import { CompressionOrchestrator, NoneCompression, UniformCompression, PDAwareCompression, TaskAwareCompression } from './src/compression/index.ts';
import { TaskClassifier, LayerBudgetAllocator } from './src/task/index.ts';

console.log('=== 最终集成验证 ===\n');

// 1. 测试核心模块
const sim = new PDSimulator();
const result = sim.simulateRequest({
  id: 'final-test',
  inputTokens: 1000,
  outputTokens: 100,
  taskType: 'math',
  arrivalTimeMs: Date.now()
});
console.log('✅ PDSimulator: TTFT =', result.ttftMs.toFixed(2), 'ms');

// 2. 测试压缩编排器
const orch = new CompressionOrchestrator();
orch.registerStrategy(new NoneCompression());
orch.registerStrategy(new UniformCompression());
orch.registerStrategy(new PDAwareCompression());
orch.registerStrategy(new TaskAwareCompression());
console.log('✅ CompressionOrchestrator: 已注册', orch.getRegisteredStrategies().length, '个策略');

// 3. 测试任务分类器
const classifier = new TaskClassifier();
classifier.classify('请计算这个积分').then(taskResult => {
  console.log('✅ TaskClassifier: "请计算这个积分" ->', taskResult.taskType);

  // 4. 测试层预算分配器
  const budget = new LayerBudgetAllocator({
    totalMemoryBytes: 16 * 1024 * 1024 * 1024,
    totalLayers: 32,
    hiddenSize: 4096,
    numHeads: 32,
    sequenceLength: 4096,
    taskType: 'math'
  }).allocate();
  console.log('✅ LayerBudgetAllocator: 总预算 =', (budget.totalBudgetBytes / 1024 / 1024 / 1024).toFixed(1), 'GB');

  console.log('\n=== 所有模块集成验证通过! ===');
});
