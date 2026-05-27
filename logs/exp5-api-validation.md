# 实验5：DeepSeek API验证

## 实验概述

- **样本数量**: 30个
- **任务类型**: math (8), code (8), qa (8), conversation (6)
- **测试时间**: 2026-05-27T03:19:30.294Z

## 测试结果对比

| 模式 | 正确数 | 总数 | 准确率 | 平均延迟 |
|------|--------|------|--------|----------|
| 规则模式 | 24 | 30 | 80.0% | 0.03ms |
| API模式 | 24 | 30 | 80.0% | 103.87ms |

## 按任务类型准确率

| 任务类型 | 规则模式 | API模式 | 样本数 |
|---------|---------|---------|--------|
| math | 100.0% | 100.0% | 8 |
| code | 62.5% | 62.5% | 8 |
| qa | 62.5% | 62.5% | 8 |
| conversation | 100.0% | 100.0% | 6 |

## 详细分类结果

### 规则模式

| 文本 | 期望 | 预测 | 正确 | 置信度 |
|------|------|------|------|--------|
| 请帮我计算一下这个积分: ∫x²dx | math | math | ✅ | 80% |
| 推导一下勾股定理的证明过程 | math | math | ✅ | 80% |
| 求方程 2x + 5 = 15 的解 | math | math | ✅ | 70% |
| 计算矩阵 [[1,2],[3,4]] 的行列式 | math | math | ✅ | 80% |
| 证明这个数列收敛 | math | math | ✅ | 70% |
| 求函数 f(x) = x² + 2x + 1 的极值 | math | math | ✅ | 60% |
| 计算概率: 投掷两个骰子点数之和为7的概率 | math | math | ✅ | 80% |
| 求解这个微分方程 dy/dx = y | math | math | ✅ | 90% |
| 用Python写一个快速排序算法 | code | code | ✅ | 70% |
| 帮我debug这段代码 | code | code | ✅ | 80% |
| 如何用JavaScript实现一个Promise | code | code | ✅ | 70% |
| 写一个函数来反转链表 | code | math | ❌ | 60% |
| 解释一下闭包的概念和用法 | code | qa | ❌ | 80% |
| 这段代码为什么会报错？ | code | code | ✅ | 60% |
| 用TypeScript定义一个泛型接口 | code | qa | ❌ | 70% |
| 实现一个LRU缓存类 | code | code | ✅ | 70% |
| 什么是机器学习？ | qa | qa | ✅ | 70% |
| 请解释一下什么是深度学习 | qa | qa | ✅ | 80% |
| 为什么天空是蓝色的？ | qa | qa | ✅ | 70% |
| 如何学习编程？有什么建议吗？ | qa | code | ❌ | 60% |
| 请总结一下这篇文章的主要内容 | qa | qa | ✅ | 70% |
| 量子计算和经典计算有什么区别？ | qa | math | ❌ | 60% |
| 介绍一下人工智能的发展历史 | qa | qa | ✅ | 70% |
| 云计算的优势和劣势是什么？ | qa | math | ❌ | 70% |
| 你好，今天天气怎么样？ | conversation | conversation | ✅ | 50% |
| 我们聊聊吧，随便什么都行 | conversation | conversation | ✅ | 50% |
| 最近怎么样？有什么新鲜事吗？ | conversation | conversation | ✅ | 50% |
| 谢谢你帮我解答问题！ | conversation | conversation | ✅ | 50% |
| 我今天心情不太好 | conversation | conversation | ✅ | 50% |
| 周末有什么计划吗？ | conversation | conversation | ✅ | 50% |

### API模式

| 文本 | 期望 | 预测 | 方法 | 正确 | 置信度 |
|------|------|------|------|------|--------|
| 请帮我计算一下这个积分: ∫x²dx | math | math | rule | ✅ | 80% |
| 推导一下勾股定理的证明过程 | math | math | rule | ✅ | 80% |
| 求方程 2x + 5 = 15 的解 | math | math | rule | ✅ | 70% |
| 计算矩阵 [[1,2],[3,4]] 的行列式 | math | math | rule | ✅ | 80% |
| 证明这个数列收敛 | math | math | rule | ✅ | 70% |
| 求函数 f(x) = x² + 2x + 1 的极值 | math | math | rule | ✅ | 60% |
| 计算概率: 投掷两个骰子点数之和为7的概率 | math | math | rule | ✅ | 80% |
| 求解这个微分方程 dy/dx = y | math | math | rule | ✅ | 90% |
| 用Python写一个快速排序算法 | code | code | rule | ✅ | 70% |
| 帮我debug这段代码 | code | code | rule | ✅ | 80% |
| 如何用JavaScript实现一个Promise | code | code | rule | ✅ | 70% |
| 写一个函数来反转链表 | code | math | rule | ❌ | 60% |
| 解释一下闭包的概念和用法 | code | qa | rule | ❌ | 80% |
| 这段代码为什么会报错？ | code | code | rule | ✅ | 60% |
| 用TypeScript定义一个泛型接口 | code | qa | rule | ❌ | 70% |
| 实现一个LRU缓存类 | code | code | rule | ✅ | 70% |
| 什么是机器学习？ | qa | qa | rule | ✅ | 70% |
| 请解释一下什么是深度学习 | qa | qa | rule | ✅ | 80% |
| 为什么天空是蓝色的？ | qa | qa | rule | ✅ | 70% |
| 如何学习编程？有什么建议吗？ | qa | code | rule | ❌ | 60% |
| 请总结一下这篇文章的主要内容 | qa | qa | rule | ✅ | 70% |
| 量子计算和经典计算有什么区别？ | qa | math | rule | ❌ | 60% |
| 介绍一下人工智能的发展历史 | qa | qa | rule | ✅ | 70% |
| 云计算的优势和劣势是什么？ | qa | math | rule | ❌ | 70% |
| 你好，今天天气怎么样？ | conversation | conversation | rule | ✅ | 50% |
| 我们聊聊吧，随便什么都行 | conversation | conversation | rule | ✅ | 50% |
| 最近怎么样？有什么新鲜事吗？ | conversation | conversation | rule | ✅ | 50% |
| 谢谢你帮我解答问题！ | conversation | conversation | rule | ✅ | 50% |
| 我今天心情不太好 | conversation | conversation | rule | ✅ | 50% |
| 周末有什么计划吗？ | conversation | conversation | rule | ✅ | 50% |

## 分析与结论

### 关键发现

1. **规则模式准确率**: 80.0%
2. **API模式准确率**: 80.0%
3. **规则模式延迟**: 0.03ms（无需网络请求）
4. **API模式延迟**: 103.87ms（包含网络开销）

### 结论

两种模式准确率相近，可根据延迟和成本选择

### 建议

- **生产环境**: 推荐使用规则模式（低延迟、无API成本）
- **高精度场景**: 可结合规则+API（规则快速筛选，API处理疑难样本）
- **持续优化**: 根据实际请求分布调整关键词权重

---
*实验时间: 2026-05-27T03:19:30.299Z*
