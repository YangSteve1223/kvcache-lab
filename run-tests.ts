#!/usr/bin/env npx tsx
/**
 * 简单的测试运行器
 * 使用 tsx 运行测试文件
 */

import { PDSimulator } from './src/core/PDSimulator.ts';
import { TaskClassifier } from './src/task/TaskClassifier.ts';

let passed = 0;
let failed = 0;

function test(name: string, fn: () => void | Promise<void>) {
  try {
    const result = fn();
    if (result instanceof Promise) {
      result.then(() => {
        console.log(`✅ ${name}`);
        passed++;
      }).catch((err) => {
        console.log(`❌ ${name}: ${err.message}`);
        failed++;
      });
    } else {
      console.log(`✅ ${name}`);
      passed++;
    }
  } catch (err: any) {
    console.log(`❌ ${name}: ${err.message}`);
    failed++;
  }
}

function assert(condition: boolean, message?: string) {
  if (!condition) {
    throw new Error(message || 'Assertion failed');
  }
}

// ========== PDSimulator 测试 ==========

console.log('\n========== PDSimulator Tests ==========\n');

test('应该使用默认配置初始化', () => {
  const simulator = new PDSimulator();
  const config = simulator.getConfig();
  assert(config.prefillBaseMs === 50);
});

test('应该允许自定义配置', () => {
  const simulator = new PDSimulator({ prefillBaseMs: 100 });
  const config = simulator.getConfig();
  assert(config.prefillBaseMs === 100);
});

test('应该模拟单个请求并返回结果', () => {
  const simulator = new PDSimulator();
  const result = simulator.simulateRequest({
    id: 'test-req',
    inputTokens: 500,
    outputTokens: 50,
    taskType: 'math',
    arrivalTimeMs: Date.now()
  });
  assert(result.ttftMs > 0);
  assert(result.e2eLatencyMs > result.ttftMs);
});

test('应该正确处理缓存命中', () => {
  const simulator = new PDSimulator();
  const prefixHash = 'test-prefix';
  
  simulator.simulateRequest({
    id: 'req-1',
    inputTokens: 100,
    outputTokens: 10,
    taskType: 'math',
    prefixHash,
    arrivalTimeMs: Date.now()
  });
  
  const result2 = simulator.simulateRequest({
    id: 'req-2',
    inputTokens: 100,
    outputTokens: 10,
    taskType: 'math',
    prefixHash,
    arrivalTimeMs: Date.now()
  });
  
  assert(result2.cacheHit === true);
});

test('应该正确处理不同任务类型', () => {
  const simulator = new PDSimulator();
  const taskTypes = ['math', 'code', 'qa', 'conversation'] as const;
  
  for (const taskType of taskTypes) {
    const result = simulator.simulateRequest({
      id: `req-${taskType}`,
      inputTokens: 100,
      outputTokens: 10,
      taskType,
      arrivalTimeMs: Date.now()
    });
    assert(result.taskType === taskType);
  }
});

test('应该模拟批量请求', () => {
  const simulator = new PDSimulator();
  const requests = Array.from({ length: 10 }, (_, i) => ({
    id: `batch-req-${i}`,
    inputTokens: 500,
    outputTokens: 50,
    taskType: 'math' as const,
    arrivalTimeMs: Date.now() + i * 100
  }));
  
  const stats = simulator.simulateBatch(requests);
  assert(stats.totalRequests === 10);
  assert(stats.avgTTFT > 0);
});

test('应该正确计算TTFT百分位数', () => {
  const simulator = new PDSimulator();
  const requests = Array.from({ length: 50 }, (_, i) => ({
    id: `req-${i}`,
    inputTokens: 200 + i * 10,
    outputTokens: 50,
    taskType: 'math' as const,
    arrivalTimeMs: Date.now()
  }));
  
  const stats = simulator.simulateBatch(requests);
  assert(stats.p50TTFT > 0);
  assert(stats.p95TTFT >= stats.p50TTFT);
  assert(stats.p99TTFT >= stats.p95TTFT);
});

test('应该正确按任务类型分组统计', () => {
  const simulator = new PDSimulator();
  const requests = [
    { id: 'm1', taskType: 'math' as const },
    { id: 'm2', taskType: 'math' as const },
    { id: 'c1', taskType: 'code' as const },
    { id: 'q1', taskType: 'qa' as const }
  ].map(r => ({
    ...r,
    inputTokens: 100,
    outputTokens: 10,
    arrivalTimeMs: Date.now()
  }));
  
  const stats = simulator.simulateBatch(requests);
  assert(stats.perTaskStats.math.count === 2);
  assert(stats.perTaskStats.code.count === 1);
  assert(stats.perTaskStats.qa.count === 1);
});

test('应该处理无压缩配置', () => {
  const simulator = new PDSimulator();
  const result = simulator.simulateRequest({
    id: 'no-compress',
    inputTokens: 100,
    outputTokens: 10,
    taskType: 'math',
    arrivalTimeMs: Date.now()
  }, null);
  
  assert(result.compressionRatio === 1.0);
  assert(result.qualityScore === 1.0);
});

test('应该正确处理空请求列表', () => {
  const simulator = new PDSimulator();
  const stats = simulator.simulateBatch([]);
  assert(stats.totalRequests === 0);
  assert(stats.avgTTFT === 0);
});

test('应该正确重置模拟器状态', () => {
  const simulator = new PDSimulator();
  simulator.simulateRequest({
    id: 'req',
    inputTokens: 100,
    outputTokens: 10,
    taskType: 'math',
    arrivalTimeMs: Date.now()
  });
  assert(simulator.getResults().length === 1);
  
  simulator.reset();
  assert(simulator.getResults().length === 0);
});

test('应该正确计算吞吐量', () => {
  const simulator = new PDSimulator();
  const requests = Array.from({ length: 20 }, (_, i) => ({
    id: `req-${i}`,
    inputTokens: 500,
    outputTokens: 50,
    taskType: 'math' as const,
    arrivalTimeMs: Date.now()
  }));
  
  const stats = simulator.simulateBatch(requests);
  assert(stats.throughputTokensPerSec > 0);
});

test('应该处理超大token数量', () => {
  const simulator = new PDSimulator();
  const result = simulator.simulateRequest({
    id: 'large',
    inputTokens: 100000,
    outputTokens: 10000,
    taskType: 'math',
    arrivalTimeMs: Date.now()
  });
  
  assert(result.ttftMs > 0);
  assert(result.e2eLatencyMs > result.ttftMs);
});

test('应该获取缓存管理器', () => {
  const simulator = new PDSimulator();
  const cacheManager = simulator.getCacheManager();
  assert(cacheManager !== null);
  assert(typeof cacheManager.lookup === 'function');
  assert(typeof cacheManager.store === 'function');
});

// ========== TaskClassifier 测试 ==========

console.log('\n========== TaskClassifier Tests ==========\n');

test('应该正确分类数学任务', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('请帮我计算一下这个积分');
  assert(result.taskType === 'math');
  assert(result.confidence > 0);
});

test('应该正确分类代码任务', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('帮我写一个函数实现排序');
  assert(result.taskType === 'code');
});

test('应该正确分类QA任务', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('什么是机器学习？');
  assert(result.taskType === 'qa');
});

test('应该正确分类对话任务', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('你好，今天天气怎么样？');
  assert(result.taskType === 'conversation');
});

test('classifyTask便捷函数应该正确工作', async () => {
  const result = await import('./src/task/TaskClassifier.ts').then(m => m.classifyTask('计算这个积分'));
  assert(result.taskType === 'math');
});

test('批量分类应该正确工作', async () => {
  const classifier = new TaskClassifier();
  const results = await classifier.classifyBatch(['计算积分', '写代码']);
  assert(results.length === 2);
  assert(results[0].taskType === 'math');
  assert(results[1].taskType === 'code');
});

test('分类结果应该包含必要字段', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('测试');
  assert(typeof result.taskType === 'string');
  assert(typeof result.confidence === 'number');
  assert(typeof result.method === 'string');
  assert(typeof result.latencyMs === 'number');
  assert(['rule', 'api'].includes(result.method));
});

test('置信度应该在0-1之间', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('测试');
  assert(result.confidence >= 0 && result.confidence <= 1);
});

test('应该正确处理空字符串', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('');
  // 空字符串应该被分类为conversation
  assert(result.taskType === 'conversation');
});

test('应该正确处理Unicode字符', async () => {
  const classifier = new TaskClassifier();
  const result = await classifier.classify('请解释什么是人工智能 αβγδ');
  assert(result.taskType === 'qa');
});

test('应该支持设置API超时时间', () => {
  const classifier = new TaskClassifier();
  classifier.setAPITimeout(5000);
  // 不抛异常即通过
});

test('应该正确处理大小写', async () => {
  const classifier = new TaskClassifier();
  const lower = await classifier.classify('什么是机器学习');
  const upper = await classifier.classify('什么是机器学习'.toUpperCase());
  assert(lower.taskType === upper.taskType);
});

// 等待异步测试完成
setTimeout(() => {
  console.log('\n========== Test Summary ==========');
  console.log(`✅ Passed: ${passed}`);
  console.log(`❌ Failed: ${failed}`);
  console.log(`Total: ${passed + failed}`);
  
  if (failed > 0) {
    process.exit(1);
  }
}, 100);
