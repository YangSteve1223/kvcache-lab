/**
 * 实验8: 增强版TaskClassifier验证
 * 
 * 目标：验证增强后的分类器准确率是否达到90%+
 */

import { classifyByRulesEnhanced, TaskClassifier } from '../src/task/TaskClassifier.ts';
import { TaskType, ClassificationResult } from '../src/core/types.ts';

// 测试样本 - 50个
const TEST_SAMPLES = [
  // ============ Math - 明确 (8个) ============
  { input: "求解方程 x^2 + 3x - 4 = 0", expected: "math" },
  { input: "计算 ∫sin(x)dx 从0到π", expected: "math" },
  { input: "Prove that the sum of angles in a triangle is 180 degrees", expected: "math" },
  { input: "求矩阵 [[1,2],[3,4]] 的特征值", expected: "math" },
  { input: "推导这个泰勒展开式", expected: "math" },
  { input: "计算极限 lim(x→0) sin(x)/x", expected: "math" },
  { input: "求函数 f(x) = x^2 + 2x + 1 的导数", expected: "math" },
  { input: "证明这个定理", expected: "math" },
  
  // ============ Math - 边界 (5个) ============
  { input: "这个算法的时间复杂度是O(n²)，怎么优化？", expected: "math" },  // 数学+代码混合
  { input: "帮我写一个计算斐波那契数列的函数，要求分析其时间复杂度", expected: "code" },  // 代码为主
  { input: "解释一下梯度下降法的工作原理", expected: "math" },  // 数学解释
  { input: "求∫₀^∞ e^(-x²)dx 的值", expected: "math" },
  { input: "用Python计算这个矩阵的逆", expected: "code" },  // 代码为主
  
  // ============ Code - 明确 (8个) ============
  { input: "写一个Python函数实现快速排序", expected: "code" },
  { input: "How to fix this bug: TypeError: undefined is not a function", expected: "code" },
  { input: "实现一个LRU缓存，支持get和put操作", expected: "code" },
  { input: "写一个class处理用户登录逻辑", expected: "code" },
  { input: "帮我debug这段代码", expected: "code" },
  { input: "def quicksort(arr):", expected: "code" },
  { input: "使用async/await重构这个回调函数", expected: "code" },
  { input: "console.log('Hello World')", expected: "code" },
  
  // ============ Code - 边界 (5个) ============
  { input: "分析这段代码的时间复杂度", expected: "code" },  // 代码+数学混合
  { input: "Git commit -m 'fix: resolve issue'", expected: "code" },
  { input: "import numpy as np\nimport pandas as pd", expected: "code" },
  { input: "docker build -t myapp .", expected: "code" },
  { input: "请解释这段代码的作用：function add(a,b){return a+b}", expected: "code" },
  
  // ============ QA - 明确 (8个) ============
  { input: "什么是PD分离推理？", expected: "qa" },
  { input: "Explain the difference between RLHF and DPO", expected: "qa" },
  { input: "为什么KV Cache会占用这么多显存？", expected: "qa" },
  { input: "请总结这篇文章的主要内容", expected: "qa" },
  { input: "什么是大模型的上下文长度？", expected: "qa" },
  { input: "比较一下Python和JavaScript的优缺点", expected: "qa" },
  { input: "如何学习深度学习？", expected: "qa" },
  { input: "Transformer的attention机制是怎么工作的？", expected: "qa" },
  
  // ============ QA - 边界 (5个) ============
  { input: "大模型推理优化有哪些方法？帮我总结一下", expected: "qa" },
  { input: "介绍一下KV Cache技术", expected: "qa" },
  { input: "what is the meaning of life?", expected: "qa" },
  { input: "为什么深度学习需要GPU？", expected: "qa" },
  { input: "请分析一下这个算法的优缺点", expected: "qa" },
  
  // ============ Conversation - 明确 (6个) ============
  { input: "你好，今天天气怎么样？", expected: "conversation" },
  { input: "帮我写一封请假邮件", expected: "conversation" },
  { input: "Can you help me draft an email?", expected: "conversation" },
  { input: "我们聊聊AI的未来吧", expected: "conversation" },
  { input: "Thanks for your help!", expected: "conversation" },
  { input: "再见，有问题再来找你", expected: "conversation" },
  
  // ============ Conversation - 边界 (5个) ============
  { input: "你好", expected: "conversation" },
  { input: "hi", expected: "conversation" },
  { input: "早上好！最近怎么样？", expected: "conversation" },
  { input: "hey, can you help me?", expected: "conversation" },
  { input: "请问能帮我看看这个问题吗？", expected: "conversation" },  // 轻微模糊
];

// 统计结果
interface CategoryStats {
  correct: number;
  total: number;
  accuracy: number;
  samples: { input: string; expected: string; actual: string; correct: boolean }[];
}

function runRuleClassifier(): Map<string, CategoryStats> {
  const stats = new Map<string, CategoryStats>();
  const categories = ['math', 'code', 'qa', 'conversation'];
  
  for (const cat of categories) {
    stats.set(cat, { correct: 0, total: 0, accuracy: 0, samples: [] });
  }
  
  for (const sample of TEST_SAMPLES) {
    const result = classifyByRulesEnhanced(sample.input);
    const actual = result.taskType;
    const correct = actual === sample.expected;
    
    const stat = stats.get(sample.expected)!;
    stat.total++;
    if (correct) stat.correct++;
    stat.samples.push({
      input: sample.input.length > 50 ? sample.input.substring(0, 50) + '...' : sample.input,
      expected: sample.expected,
      actual,
      correct,
    });
  }
  
  // 计算准确率
  for (const [cat, stat] of stats) {
    stat.accuracy = stat.total > 0 ? stat.correct / stat.total : 0;
  }
  
  return stats;
}

async function runAPIClassifier(): Promise<Map<string, CategoryStats>> {
  const apiKey = process.env.DEEEPSEEK_API_KEY;
  if (!apiKey) {
    console.log('⚠️ 未设置 DEEPSEEK_API_KEY，跳过API模式测试');
    return new Map();
  }
  
  const classifier = new TaskClassifier({ useAPI: true });
  const stats = new Map<string, CategoryStats>();
  const categories = ['math', 'code', 'qa', 'conversation'];
  
  for (const cat of categories) {
    stats.set(cat, { correct: 0, total: 0, accuracy: 0, samples: [] });
  }
  
  for (const sample of TEST_SAMPLES) {
    const result = await classifier.classify(sample.input);
    const actual = result.taskType;
    const correct = actual === sample.expected;
    
    const stat = stats.get(sample.expected)!;
    stat.total++;
    if (correct) stat.correct++;
    stat.samples.push({
      input: sample.input.length > 50 ? sample.input.substring(0, 50) + '...' : sample.input,
      expected: sample.expected,
      actual,
      correct,
    });
  }
  
  for (const [cat, stat] of stats) {
    stat.accuracy = stat.total > 0 ? stat.correct / stat.total : 0;
  }
  
  return stats;
}

function printStats(name: string, stats: Map<string, CategoryStats>): void {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`📊 ${name}`);
  console.log('='.repeat(60));
  
  let totalCorrect = 0;
  let totalSamples = 0;
  
  for (const [cat, stat] of stats) {
    const icon = stat.accuracy >= 0.9 ? '✅' : stat.accuracy >= 0.7 ? '⚠️' : '❌';
    console.log(`\n${icon} ${cat.toUpperCase()}: ${stat.correct}/${stat.total} (${(stat.accuracy * 100).toFixed(1)}%)`);
    totalCorrect += stat.correct;
    totalSamples += stat.total;
    
    // 显示错误样本
    const errors = stat.samples.filter(s => !s.correct);
    if (errors.length > 0) {
      console.log('   ❌ 错误样本:');
      for (const e of errors.slice(0, 3)) {
        console.log(`      - "${e.input}"`);
        console.log(`        期望: ${e.expected}, 实际: ${e.actual}`);
      }
      if (errors.length > 3) {
        console.log(`      ... 还有 ${errors.length - 3} 个`);
      }
    }
  }
  
  const overallAccuracy = totalSamples > 0 ? totalCorrect / totalSamples : 0;
  console.log(`\n📈 总体准确率: ${totalCorrect}/${totalSamples} (${(overallAccuracy * 100).toFixed(1)}%)`);
}

// 生成Markdown报告
function generateMarkdownReport(
  ruleStats: Map<string, CategoryStats>,
  apiStats: Map<string, CategoryStats> | null,
  apiLatencies: number[]
): string {
  let report = `# 实验8: TaskClassifier 增强验证报告

## 实验配置
- 测试样本: ${TEST_SAMPLES.length}个
- 样本分布: 每类12-13个（明确+边界case）

## 测试结果

### 1. 规则模式

`;

  // 规则模式结果
  let ruleTotalCorrect = 0;
  let ruleTotalSamples = 0;
  for (const [cat, stat] of ruleStats) {
    ruleTotalCorrect += stat.correct;
    ruleTotalSamples += stat.total;
    report += `| ${cat} | ${stat.correct}/${stat.total} | ${(stat.accuracy * 100).toFixed(1)}% |\n`;
  }
  const ruleAccuracy = ruleTotalCorrect / ruleTotalSamples;
  report += `| **总体** | **${ruleTotalCorrect}/${ruleTotalSamples}** | **${(ruleAccuracy * 100).toFixed(1)}%** |\n\n`;
  
  report += `**结论**: ${ruleAccuracy >= 0.9 ? '✅ 达标 (≥90%)' : '❌ 未达标'}\n\n`;
  
  report += `### 2. API模式\n\n`;
  
  if (apiStats) {
    let apiTotalCorrect = 0;
    let apiTotalSamples = 0;
    for (const [cat, stat] of apiStats) {
      apiTotalCorrect += stat.correct;
      apiTotalSamples += stat.total;
      report += `| ${cat} | ${stat.correct}/${stat.total} | ${(stat.accuracy * 100).toFixed(1)}% |\n`;
    }
    const apiAccuracy = apiTotalCorrect / apiTotalSamples;
    report += `| **总体** | **${apiTotalCorrect}/${apiTotalSamples}** | **${(apiAccuracy * 100).toFixed(1)}%** |\n\n`;
    
    const avgLatency = apiLatencies.length > 0 
      ? (apiLatencies.reduce((a, b) => a + b, 0) / apiLatencies.length).toFixed(2)
      : 'N/A';
    report += `**平均延迟**: ${avgLatency}ms\n`;
    report += `**结论**: ${apiAccuracy >= 0.85 ? '✅ 达标 (≥85%)' : '❌ 未达标'}\n\n`;
  } else {
    report += `*未运行（未配置API Key）*\n\n`;
  }
  
  report += `## 错误分析\n\n`;
  report += `### 规则模式错误样本\n\n`;
  report += `| 输入 | 期望 | 实际 | 分析 |\n`;
  report += `|------|------|------|------|\n`;
  
  for (const [cat, stat] of ruleStats) {
    for (const s of stat.samples.filter(x => !x.correct)) {
      report += `| "${s.input}" | ${s.expected} | ${s.actual} | 需优化 |\n`;
    }
  }
  
  report += `\n## 优化建议\n\n`;
  report += `1. 继续扩充关键词库，特别是针对边界case\n`;
  report += `2. 考虑增加n-gram特征检测\n`;
  report += `3. 可以引入机器学习模型进一步提升准确率\n`;
  
  return report;
}

// 主函数
async function main() {
  console.log('🚀 开始实验8: TaskClassifier 增强验证\n');
  console.log(`📝 测试样本数: ${TEST_SAMPLES.length}`);
  
  // 1. 运行规则模式
  console.log('\n⏳ 运行规则模式分类...');
  const ruleStats = runRuleClassifier();
  printStats('规则模式 (Rule-based)', ruleStats);
  
  // 2. 运行API模式
  console.log('\n⏳ 运行API模式分类...');
  const apiLatencies: number[] = [];
  const apiStats = await runAPIClassifier();
  if (apiStats.size > 0) {
    printStats('API模式 (DeepSeek)', apiStats);
    
    // 收集延迟数据
    for (const sample of TEST_SAMPLES) {
      const classifier = new TaskClassifier({ useAPI: true });
      const result = await classifier.classify(sample.input);
      apiLatencies.push(result.latencyMs);
    }
  }
  
  // 3. 生成报告
  const report = generateMarkdownReport(ruleStats, apiStats, apiLatencies);
  
  // 保存报告
  const fs = await import('fs');
  const logsDir = './logs';
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
  }
  
  const reportPath = `${logsDir}/exp8-classifier-enhanced.md`;
  fs.writeFileSync(reportPath, report, 'utf-8');
  console.log(`\n📄 报告已保存: ${reportPath}`);
  
  // 4. 最终结论
  const ruleAccuracy = Array.from(ruleStats.values())
    .reduce((sum, s) => sum + s.correct, 0) / TEST_SAMPLES.length;
  
  console.log('\n' + '='.repeat(60));
  console.log('🏁 最终结论');
  console.log('='.repeat(60));
  console.log(`规则模式准确率: ${(ruleAccuracy * 100).toFixed(1)}% ${ruleAccuracy >= 0.9 ? '✅ 达标' : '❌ 未达标'}`);
  console.log(`目标: ≥90%`);
}

main().catch(console.error);
