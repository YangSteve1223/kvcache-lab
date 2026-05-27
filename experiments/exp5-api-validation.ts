/**
 * 实验5：DeepSeek API验证
 * 
 * 验证任务分类器的准确性
 */

import { TaskClassifier, classifyTask } from '../src/task/TaskClassifier.ts';
import { writeFileSync } from 'fs';

/**
 * 测试样本
 */
const SAMPLES = [
  // Math (8个)
  { text: '请帮我计算一下这个积分: ∫x²dx', expected: 'math' },
  { text: '推导一下勾股定理的证明过程', expected: 'math' },
  { text: '求方程 2x + 5 = 15 的解', expected: 'math' },
  { text: '计算矩阵 [[1,2],[3,4]] 的行列式', expected: 'math' },
  { text: '证明这个数列收敛', expected: 'math' },
  { text: '求函数 f(x) = x² + 2x + 1 的极值', expected: 'math' },
  { text: '计算概率: 投掷两个骰子点数之和为7的概率', expected: 'math' },
  { text: '求解这个微分方程 dy/dx = y', expected: 'math' },
  
  // Code (8个)
  { text: '用Python写一个快速排序算法', expected: 'code' },
  { text: '帮我debug这段代码', expected: 'code' },
  { text: '如何用JavaScript实现一个Promise', expected: 'code' },
  { text: '写一个函数来反转链表', expected: 'code' },
  { text: '解释一下闭包的概念和用法', expected: 'code' },
  { text: '这段代码为什么会报错？', expected: 'code' },
  { text: '用TypeScript定义一个泛型接口', expected: 'code' },
  { text: '实现一个LRU缓存类', expected: 'code' },
  
  // QA (8个)
  { text: '什么是机器学习？', expected: 'qa' },
  { text: '请解释一下什么是深度学习', expected: 'qa' },
  { text: '为什么天空是蓝色的？', expected: 'qa' },
  { text: '如何学习编程？有什么建议吗？', expected: 'qa' },
  { text: '请总结一下这篇文章的主要内容', expected: 'qa' },
  { text: '量子计算和经典计算有什么区别？', expected: 'qa' },
  { text: '介绍一下人工智能的发展历史', expected: 'qa' },
  { text: '云计算的优势和劣势是什么？', expected: 'qa' },
  
  // Conversation (6个)
  { text: '你好，今天天气怎么样？', expected: 'conversation' },
  { text: '我们聊聊吧，随便什么都行', expected: 'conversation' },
  { text: '最近怎么样？有什么新鲜事吗？', expected: 'conversation' },
  { text: '谢谢你帮我解答问题！', expected: 'conversation' },
  { text: '我今天心情不太好', expected: 'conversation' },
  { text: '周末有什么计划吗？', expected: 'conversation' }
];

/**
 * 运行规则模式分类
 */
async function testRuleMode(classifier: TaskClassifier) {
  const results: any[] = [];
  
  for (const sample of SAMPLES) {
    const start = Date.now();
    const result = await classifier.classify(sample.text);
    const latency = Date.now() - start;
    
    results.push({
      text: sample.text.substring(0, 30) + (sample.text.length > 30 ? '...' : ''),
      expected: sample.expected,
      predicted: result.taskType,
      confidence: result.confidence,
      latencyMs: latency,
      correct: result.taskType === sample.expected
    });
  }
  
  return results;
}

/**
 * 运行API模式分类
 */
async function testAPIMode(classifier: TaskClassifier) {
  const results: any[] = [];
  
  for (const sample of SAMPLES) {
    const start = Date.now();
    const result = await classifier.classify(sample.text);
    const latency = Date.now() - start;
    
    results.push({
      text: sample.text.substring(0, 30) + (sample.text.length > 30 ? '...' : ''),
      expected: sample.expected,
      predicted: result.taskType,
      confidence: result.confidence,
      latencyMs: latency,
      method: result.method,
      correct: result.taskType === sample.expected
    });
  }
  
  return results;
}

/**
 * 计算统计信息
 */
function computeStats(results: any[]) {
  const correct = results.filter(r => r.correct).length;
  const total = results.length;
  const accuracy = (correct / total * 100).toFixed(1);
  
  // 按类型统计
  const byType: Record<string, { correct: number; total: number; accuracy: string }> = {};
  for (const expected of ['math', 'code', 'qa', 'conversation']) {
    const typeResults = results.filter(r => r.expected === expected);
    const typeCorrect = typeResults.filter(r => r.correct).length;
    byType[expected] = {
      correct: typeCorrect,
      total: typeResults.length,
      accuracy: (typeCorrect / typeResults.length * 100).toFixed(1)
    };
  }
  
  // 平均延迟
  const avgLatency = (results.reduce((sum, r) => sum + r.latencyMs, 0) / results.length).toFixed(2);
  
  return { correct, total, accuracy, byType, avgLatency };
}

/**
 * 生成报告
 */
function generateReport(
  ruleStats: any,
  apiStats: any,
  ruleResults: any[],
  apiResults: any[]
): string {
  let report = `# 实验5：DeepSeek API验证

## 实验概述

- **样本数量**: ${SAMPLES.length}个
- **任务类型**: math (8), code (8), qa (8), conversation (6)
- **测试时间**: ${new Date().toISOString()}

## 测试结果对比

| 模式 | 正确数 | 总数 | 准确率 | 平均延迟 |
|------|--------|------|--------|----------|
| 规则模式 | ${ruleStats.correct} | ${ruleStats.total} | ${ruleStats.accuracy}% | ${ruleStats.avgLatency}ms |
| API模式 | ${apiStats.correct} | ${apiStats.total} | ${apiStats.accuracy}% | ${apiStats.avgLatency}ms |

## 按任务类型准确率

| 任务类型 | 规则模式 | API模式 | 样本数 |
|---------|---------|---------|--------|
`;

  for (const type of ['math', 'code', 'qa', 'conversation']) {
    const r = ruleStats.byType[type];
    const a = apiStats.byType[type];
    report += `| ${type} | ${r.accuracy}% | ${a.accuracy}% | ${r.total} |\n`;
  }

  report += `
## 详细分类结果

### 规则模式

| 文本 | 期望 | 预测 | 正确 | 置信度 |
|------|------|------|------|--------|
`;

  for (const r of ruleResults) {
    const correct = r.correct ? '✅' : '❌';
    report += `| ${r.text} | ${r.expected} | ${r.predicted} | ${correct} | ${(r.confidence * 100).toFixed(0)}% |\n`;
  }

  report += `
### API模式

| 文本 | 期望 | 预测 | 方法 | 正确 | 置信度 |
|------|------|------|------|------|--------|
`;

  for (const r of apiResults) {
    const correct = r.correct ? '✅' : '❌';
    report += `| ${r.text} | ${r.expected} | ${r.predicted} | ${r.method} | ${correct} | ${(r.confidence * 100).toFixed(0)}% |\n`;
  }

  // 分析
  const ruleAccuracy = parseFloat(ruleStats.accuracy);
  const apiAccuracy = parseFloat(apiStats.accuracy);
  
  let conclusion = '';
  if (apiAccuracy > ruleAccuracy + 10) {
    conclusion = 'API模式显著优于规则模式（差距>10%），建议使用API模式';
  } else if (apiAccuracy > ruleAccuracy) {
    conclusion = 'API模式略优于规则模式，但差距不大';
  } else if (ruleAccuracy > 90) {
    conclusion = '规则模式表现优秀（>90%），准确率已经很高';
  } else {
    conclusion = '两种模式准确率相近，可根据延迟和成本选择';
  }

  report += `
## 分析与结论

### 关键发现

1. **规则模式准确率**: ${ruleStats.accuracy}%
2. **API模式准确率**: ${apiStats.accuracy}%
3. **规则模式延迟**: ${ruleStats.avgLatency}ms（无需网络请求）
4. **API模式延迟**: ${apiStats.avgLatency}ms（包含网络开销）

### 结论

${conclusion}

### 建议

- **生产环境**: 推荐使用规则模式（低延迟、无API成本）
- **高精度场景**: 可结合规则+API（规则快速筛选，API处理疑难样本）
- **持续优化**: 根据实际请求分布调整关键词权重

---
*实验时间: ${new Date().toISOString()}*
`;

  return report;
}

// ========== 主程序 ==========

console.log('='.repeat(60));
console.log('实验5：DeepSeek API验证');
console.log('='.repeat(60));

// 测试规则模式
console.log('\n测试规则模式...');
const ruleClassifier = new TaskClassifier({ useAPI: false });
const ruleResults = await testRuleMode(ruleClassifier);
const ruleStats = computeStats(ruleResults);

console.log(`规则模式 - 准确率: ${ruleStats.accuracy}%, 平均延迟: ${ruleStats.avgLatency}ms`);

// 测试API模式
console.log('\n测试API模式...');
const apiClassifier = new TaskClassifier({ useAPI: true });
const apiResults = await testAPIMode(apiClassifier);
const apiStats = computeStats(apiResults);

console.log(`API模式 - 准确率: ${apiStats.accuracy}%, 平均延迟: ${apiStats.avgLatency}ms`);

// 生成报告
const report = generateReport(ruleStats, apiStats, ruleResults, apiResults);

// 保存报告
writeFileSync('./logs/exp5-api-validation.md', report);
console.log('\n报告已保存到 logs/exp5-api-validation.md');

// 打印摘要
console.log('\n' + '='.repeat(60));
console.log('实验结果摘要');
console.log('='.repeat(60));

console.log('\n| 模式 | 准确率 | 平均延迟 |');
console.log('|------|--------|----------|');
console.log(`| 规则模式 | ${ruleStats.accuracy}% | ${ruleStats.avgLatency}ms |`);
console.log(`| API模式 | ${apiStats.accuracy}% | ${apiStats.avgLatency}ms |`);

console.log('\n| 任务类型 | 规则模式 | API模式 |');
console.log('|---------|---------|---------|');
for (const type of ['math', 'code', 'qa', 'conversation']) {
  console.log(`| ${type} | ${ruleStats.byType[type].accuracy}% | ${apiStats.byType[type].accuracy}% |`);
}
