/**
 * TaskClassifier 任务分类器测试
 * 
 * 测试规则模式和API模式的任务分类功能
 */

import { describe, it, expect, beforeEach } from 'node:test';
import assert from 'node:assert';
import { TaskClassifier, classifyTask, classifyTaskBatch } from '../../src/task/TaskClassifier.ts';
import { TaskType } from '../../src/core/types.ts';

describe('TaskClassifier', () => {
  let classifier: TaskClassifier;

  beforeEach(() => {
    classifier = new TaskClassifier();
  });

  // ========== 规则模式测试 ==========

  describe('规则模式分类', () => {
    it('应该正确分类数学任务', async () => {
      const mathInputs = [
        '请帮我计算一下这个积分',
        '推导一下这个公式',
        '求方程 x^2 + 2x + 1 = 0 的解',
        '证明勾股定理',
        '帮我求一下这个矩阵的特征值'
      ];
      
      for (const input of mathInputs) {
        const result = await classifier.classify(input);
        expect(result.taskType).toBe('math');
        expect(result.method).toBe('rule');
        expect(result.confidence).toBeGreaterThan(0.5);
      }
    });

    it('应该正确分类代码任务', async () => {
      const codeInputs = [
        '帮我写一个函数实现排序',
        '如何使用Python调用API',
        '解释这段代码的逻辑',
        '修复这个bug',
        '写一个class处理数据'
      ];
      
      for (const input of codeInputs) {
        const result = await classifier.classify(input);
        expect(result.taskType).toBe('code');
        expect(result.method).toBe('rule');
      }
    });

    it('应该正确分类QA任务', async () => {
      const qaInputs = [
        '什么是机器学习？',
        '请解释一下什么是深度学习',
        '为什么天空是蓝色的？',
        '如何学习编程？',
        '请总结这篇文章的主要内容'
      ];
      
      for (const input of qaInputs) {
        const result = await classifier.classify(input);
        expect(result.taskType).toBe('qa');
        expect(result.method).toBe('rule');
      }
    });

    it('应该正确分类对话任务', async () => {
      const conversationInputs = [
        '你好，今天天气怎么样？',
        '我们聊聊吧',
        '最近怎么样',
        '听说你很厉害',
        '有什么新鲜事吗'
      ];
      
      for (const input of conversationInputs) {
        const result = await classifier.classify(input);
        expect(result.taskType).toBe('conversation');
        expect(result.method).toBe('rule');
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
      // 并行处理应该比串行快很多
      expect(duration).toBeLessThan(1000); // 10个简单请求应该很快
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
      expect(true).toBe(true); // 不抛异常即通过
    });

    it('应该支持设置API超时时间', () => {
      const localClassifier = new TaskClassifier();
      localClassifier.setAPITimeout(5000);
      expect(true).toBe(true); // 不抛异常即通过
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
      // 同时包含数学和代码关键词，取决于权重
      const result = await classifier.classify('用代码计算这个方程');
      // 应该分类为其中一种
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
      // 空字符串应该被分类为conversation
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
});
