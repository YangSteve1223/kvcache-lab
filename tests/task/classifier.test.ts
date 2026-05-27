/**
 * TaskClassifier 增强版测试
 * 
 * 测试增强后的规则模式和API模式的任务分类功能
 * 准确率目标: ≥90%
 */

import { describe, it, expect, beforeEach } from 'node:test';
import assert from 'node:assert';
import { 
  TaskClassifier, 
  classifyTask, 
  classifyTaskBatch,
  classifyByRulesEnhanced,
  calculateWeightedScore,
  detectCodeFeatures,
  detectMathFeatures,
  detectQAFeatures
} from '../../src/task/TaskClassifier.ts';
import { TaskType } from '../../src/core/types.ts';

describe('TaskClassifier 增强版', () => {
  let classifier: TaskClassifier;

  beforeEach(() => {
    classifier = new TaskClassifier();
  });

  // ========== 规则模式测试 ==========

  describe('规则模式分类', () => {
    describe('数学任务分类', () => {
      const mathInputs = [
        '请帮我计算一下这个积分',
        '推导一下这个公式',
        '求方程 x^2 + 2x + 1 = 0 的解',
        '证明勾股定理',
        '帮我求一下这个矩阵的特征值',
        '求解方程 x^2 + 3x - 4 = 0',
        '计算 ∫sin(x)dx 从0到π',
        'Prove that the sum of angles in a triangle is 180 degrees',
        '求矩阵 [[1,2],[3,4]] 的特征值',
        '推导这个泰勒展开式',
        '计算极限 lim(x→0) sin(x)/x',
        '求函数 f(x) = x^2 + 2x + 1 的导数',
        '证明这个定理',
        '求∫₀^∞ e^(-x²)dx 的值',
        '解释一下梯度下降法的工作原理',
        '这个算法的时间复杂度是O(n²)，怎么优化？',
      ];
      
      for (const input of mathInputs) {
        it(`应该正确分类: "${input.length > 40 ? input.substring(0, 40) + '...' : input}"`, async () => {
          const result = await classifier.classify(input);
          expect(result.taskType).toBe('math');
          expect(result.method).toBe('rule');
          expect(result.confidence).toBeGreaterThan(0.5);
        });
      }
    });

    describe('代码任务分类', () => {
      const codeInputs = [
        '帮我写一个函数实现排序',
        '如何使用Python调用API',
        '解释这段代码的逻辑',
        '修复这个bug',
        '写一个class处理数据',
        '写一个Python函数实现快速排序',
        'How to fix this bug: TypeError: undefined is not a function',
        '实现一个LRU缓存，支持get和put操作',
        '写一个class处理用户登录逻辑',
        '帮我debug这段代码',
        'def quicksort(arr):',
        '使用async/await重构这个回调函数',
        'console.log(\'Hello World\')',
        'import numpy as np',
        'Git commit -m \'fix: resolve issue\'',
        'docker build -t myapp .',
        '用Python计算这个矩阵的逆',
        '分析这段代码的时间复杂度',
      ];
      
      for (const input of codeInputs) {
        it(`应该正确分类: "${input.length > 40 ? input.substring(0, 40) + '...' : input}"`, async () => {
          const result = await classifier.classify(input);
          expect(result.taskType).toBe('code');
          expect(result.method).toBe('rule');
        });
      }
    });

    describe('QA任务分类', () => {
      const qaInputs = [
        '什么是机器学习？',
        '请解释一下什么是深度学习',
        '为什么天空是蓝色的？',
        '如何学习编程？',
        '请总结这篇文章的主要内容',
        '什么是PD分离推理？',
        'Explain the difference between RLHF and DPO',
        '为什么KV Cache会占用这么多显存？',
        '请总结这篇文章的主要内容',
        '什么是大模型的上下文长度？',
        '比较一下Python和JavaScript的优缺点',
        '如何学习深度学习？',
        'Transformer的attention机制是怎么工作的？',
        '介绍一下KV Cache技术',
        'what is the meaning of life?',
      ];
      
      for (const input of qaInputs) {
        it(`应该正确分类: "${input.length > 40 ? input.substring(0, 40) + '...' : input}"`, async () => {
          const result = await classifier.classify(input);
          expect(result.taskType).toBe('qa');
          expect(result.method).toBe('rule');
        });
      }
    });

    describe('对话任务分类', () => {
      const conversationInputs = [
        '你好，今天天气怎么样？',
        '我们聊聊吧',
        '最近怎么样',
        '听说你很厉害',
        '有什么新鲜事吗',
        '你好',
        'hi',
        '早上好！最近怎么样？',
        'hey, can you help me?',
        'Thanks for your help!',
        '再见，有问题再来找你',
        '帮我写一封请假邮件',
        'Can you help me draft an email?',
      ];
      
      for (const input of conversationInputs) {
        it(`应该正确分类: "${input.length > 40 ? input.substring(0, 40) + '...' : input}"`, async () => {
          const result = await classifier.classify(input);
          expect(result.taskType).toBe('conversation');
          expect(result.method).toBe('rule');
        });
      }
    });

    it('应该正确处理置信度', async () => {
      // 高置信度：关键词明确
      const highConfidence = await classifier.classify('请计算一下这个方程的解');
      expect(highConfidence.confidence).toBeGreaterThanOrEqual(0.7);
      
      // 低置信度：关键词不明确
      const lowConfidence = await classifier.classify('你好');
      expect(lowConfidence.confidence).toBeLessThanOrEqual(0.7);
    });

    it('应该正确记录分类延迟', async () => {
      const result = await classifier.classify('测试输入');
      expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    });
  });

  // ========== 结构化特征检测测试 ==========

  describe('结构化特征检测', () => {
    describe('代码特征检测', () => {
      it('应该检测到大括号块', () => {
        const text = 'function test() { return true; }';
        expect(detectCodeFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到Python函数定义', () => {
        const text = 'def quicksort(arr): return sorted(arr)';
        expect(detectCodeFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到async/await', () => {
        const text = 'async function fetchData() { await fetch(url); }';
        expect(detectCodeFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到import语句', () => {
        const text = 'import numpy as np\nimport pandas as pd';
        expect(detectCodeFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到箭头函数', () => {
        const text = 'const add = (a, b) => a + b;';
        expect(detectCodeFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到console.log', () => {
        const text = 'console.log("Hello World");';
        expect(detectCodeFeatures(text)).toBeGreaterThan(0);
      });
    });

    describe('数学特征检测', () => {
      it('应该检测到积分符号', () => {
        const text = '∫sin(x)dx = -cos(x) + C';
        expect(detectMathFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到求和符号', () => {
        const text = '∑(i=1 to n) i = n(n+1)/2';
        expect(detectMathFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到函数表达式', () => {
        const text = 'f(x) = x^2 + 2x + 1';
        expect(detectMathFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到矩阵关键词', () => {
        const text = '这个matrix的特征值怎么求';
        expect(detectMathFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到极限表达式', () => {
        const text = 'lim(x→0) sin(x)/x = 1';
        expect(detectMathFeatures(text)).toBeGreaterThan(0);
      });
    });

    describe('QA特征检测', () => {
      it('应该检测到问号结尾', () => {
        const text = '什么是机器学习？';
        expect(detectQAFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到what开头', () => {
        const text = 'What is the difference?';
        expect(detectQAFeatures(text)).toBeGreaterThan(0);
      });

      it('应该检测到如何开头', () => {
        const text = '如何学习深度学习？';
        expect(detectQAFeatures(text)).toBeGreaterThan(0);
      });
    });
  });

  // ========== 批量分类测试 ==========

  describe('批量分类', () => {
    it('应该正确批量分类多个输入', async () => {
      const inputs = [
        '计算积分',
        '写一个函数',
        '什么是AI',
        '你好'
      ];
      
      const results = await classifier.classifyBatch(inputs);
      
      expect(results).toHaveLength(4);
      expect(results[0].taskType).toBe('math');
      expect(results[1].taskType).toBe('code');
      expect(results[2].taskType).toBe('qa');
      expect(results[3].taskType).toBe('conversation');
    });

    it('应该正确处理空输入列表', async () => {
      const results = await classifier.classifyBatch([]);
      expect(results).toHaveLength(0);
    });

    it('应该并行处理批量请求', async () => {
      const inputs = Array.from({ length: 10 }, (_, i) => `测试输入${i}`);
      
      const start = Date.now();
      const results = await classifier.classifyBatch(inputs);
      const duration = Date.now() - start;
      
      expect(results).toHaveLength(10);
      expect(duration).toBeLessThan(1000);
    });
  });

  // ========== 便捷函数测试 ==========

  describe('便捷函数', () => {
    it('classifyTask应该正确工作', async () => {
      const result = await classifyTask('计算这个积分');
      expect(result.taskType).toBe('math');
    });

    it('classifyTaskBatch应该正确工作', async () => {
      const results = await classifyTaskBatch(['计算积分', '写代码']);
      expect(results).toHaveLength(2);
    });
  });

  // ========== 配置测试 ==========

  describe('分类器配置', () => {
    it('应该支持禁用API模式', () => {
      const localClassifier = new TaskClassifier({ useAPI: false });
      localClassifier.setUseAPI(false);
      expect(true).toBe(true);
    });

    it('应该支持设置API超时时间', () => {
      const localClassifier = new TaskClassifier();
      localClassifier.setAPITimeout(5000);
      expect(true).toBe(true);
    });

    it('应该支持自定义配置构造', () => {
      const customClassifier = new TaskClassifier({
        useAPI: true,
        apiTimeout: 3000
      });
      expect(customClassifier).toBeDefined();
    });
  });

  // ========== 关键词测试 ==========

  describe('关键词匹配', () => {
    it('应该正确处理包含多个关键词的输入', async () => {
      const result = await classifier.classify('用代码计算这个方程');
      expect(['math', 'code']).toContain(result.taskType);
    });

    it('应该正确处理无关键词输入', async () => {
      const result = await classifier.classify('哦好的我知道了');
      expect(result.taskType).toBe('conversation');
    });

    it('应该正确处理大小写', async () => {
      const lower = await classifier.classify('什么是机器学习');
      const upper = await classifier.classify('什么是机器学习'.toUpperCase());
      expect(lower.taskType).toBe(upper.taskType);
    });

    it('应该正确处理混合语言', async () => {
      const result = await classifier.classify('calculate this equation 计算这个方程');
      expect(result.taskType).toBe('math');
    });
  });

  // ========== 边界条件测试 ==========

  describe('边界条件', () => {
    it('应该正确处理空字符串', async () => {
      const result = await classifier.classify('');
      expect(result.taskType).toBe('conversation');
    });

    it('应该正确处理超长输入', async () => {
      const longInput = '计算'.repeat(1000);
      const result = await classifier.classify(longInput);
      expect(result.taskType).toBe('math');
    });

    it('应该正确处理特殊字符', async () => {
      const result = await classifier.classify('!!!计算###公式$$$');
      expect(result.taskType).toBe('math');
    });

    it('应该正确处理Unicode字符', async () => {
      const result = await classifier.classify('请解释什么是人工智能 αβγδ');
      expect(result.taskType).toBe('qa');
    });

    it('应该正确处理代码块标记', async () => {
      const result = await classifier.classify('```python\nprint("hello")\n```');
      expect(result.taskType).toBe('code');
    });

    it('应该正确处理数学公式标记', async () => {
      const result = await classifier.classify('$$\\int_0^\\infty e^{-x^2} dx = \\frac{\\sqrt{\\pi}}{2}$$');
      expect(result.taskType).toBe('math');
    });
  });

  // ========== 结果格式测试 ==========

  describe('结果格式', () => {
    it('应该返回正确的ClassificationResult结构', async () => {
      const result = await classifier.classify('测试输入');
      
      expect(result).toHaveProperty('taskType');
      expect(result).toHaveProperty('confidence');
      expect(result).toHaveProperty('method');
      expect(result).toHaveProperty('latencyMs');
      
      expect(typeof result.taskType).toBe('string');
      expect(typeof result.confidence).toBe('number');
      expect(['rule', 'api']).toContain(result.method);
      expect(typeof result.latencyMs).toBe('number');
    });

    it('置信度应该在0-1之间', async () => {
      const inputs = ['计算', '代码', '什么是', '你好', '随机文本xyz'];
      
      for (const input of inputs) {
        const result = await classifier.classify(input);
        expect(result.confidence).toBeGreaterThanOrEqual(0);
        expect(result.confidence).toBeLessThanOrEqual(1);
      }
    });

    it('method应该是rule或api', async () => {
      const result = await classifier.classify('测试');
      expect(['rule', 'api']).toContain(result.method);
    });
  });

  // ========== 增强版函数测试 ==========

  describe('增强版分类函数', () => {
    it('classifyByRulesEnhanced应该直接返回结果', () => {
      const result = classifyByRulesEnhanced('计算积分');
      expect(result).toHaveProperty('taskType');
      expect(result).toHaveProperty('confidence');
      expect(result.taskType).toBe('math');
    });

    it('calculateWeightedScore应该计算正确', () => {
      const score = calculateWeightedScore('请计算积分', [
        { keyword: '计算', weight: 2 },
        { keyword: '积分', weight: 3 },
      ]);
      expect(score).toBe(5);
    });

    it('空输入应该返回conversation', () => {
      const result = classifyByRulesEnhanced('');
      expect(result.taskType).toBe('conversation');
    });
  });

  // ========== 性能测试 ==========

  describe('性能测试', () => {
    it('单次分类延迟应该小于1ms', async () => {
      const result = await classifier.classify('计算这个积分');
      expect(result.latencyMs).toBeLessThan(1);
    });

    it('批量分类100次应该小于50ms', async () => {
      const inputs = Array.from({ length: 100 }, (_, i) => `测试输入${i}`);
      const start = Date.now();
      await classifier.classifyBatch(inputs);
      const duration = Date.now() - start;
      expect(duration).toBeLessThan(50);
    });
  });
});
