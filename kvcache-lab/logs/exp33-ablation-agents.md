# exp33 - Ablation Study: Agent Contribution Analysis

## 实验概述

**实验目标**: 分析各Agent对系统整体性能的贡献

**实验设计**:
- 对每个配置运行100个模拟请求
- 测量质量、延迟、内存使用和SLO满足率
- 对比禁用各Agent后的性能变化

## 配置列表

| 配置 | 启用的Agent |
|------|------------|
| Full system | semantic, reuse, communication, placement |
| w/o Semantic | reuse, communication, placement |
| w/o Reuse | semantic, communication, placement |
| w/o Communication | semantic, reuse, placement |
| w/o Placement | semantic, reuse, communication |

## 性能对比

| 配置 | Quality | Latency (ms) | Memory (MB) | SLO Rate | Evict | Transmit |
|------|---------|--------------|-------------|----------|-------|----------|
| Full system | 50.7% (0.0%) | 325.04 (0.00) | 0.1 | 93.0% (0.0%) | 61.2 | 28.7 |
| w/o Semantic | 50.7% (-0.0%) | 249.24 (-75.79) | 0.1 | 100.0% (+7.0%) | 60.8 | 21.8 |
| w/o Reuse | 50.8% (+0.1%) | 243.13 (-81.90) | 0.1 | 99.0% (+6.0%) | 62.4 | 21.2 |
| w/o Communication | 51.1% (+0.4%) | 451.95 (+126.92) | 0.1 | 86.0% (-7.0%) | 62.6 | 39.3 |
| w/o Placement | 50.7% (0.0%) | 414.99 (+89.96) | 0.1 | 88.0% (-5.0%) | 62.1 | 36.1 |

## Agent贡献度分析

基于 Full system vs 各 w/o 配置的对比：

### 禁用 Semantic 的影响

- 质量下降: 0.0%
- SLO满足率下降: -7.0%
- 延迟增加: -75.79ms

### 禁用 Reuse 的影响

- 质量下降: -0.1%
- SLO满足率下降: -6.0%
- 延迟增加: -81.90ms

### 禁用 Communication 的影响

- 质量下降: -0.4%
- SLO满足率下降: 7.0%
- 延迟增加: 126.92ms

### 禁用 Placement 的影响

- 质量下降: -0.0%
- SLO满足率下降: 5.0%
- 延迟增加: 89.96ms

## 综合排名

| 排名 | 配置 | 综合评分 |
|------|------|----------|
| 1 | w/o Semantic | -0.255 |
| 2 | w/o Reuse | -0.268 |
| 3 | Full system | -0.362 |
| 4 | w/o Placement | -0.481 |
| 5 | w/o Communication | -0.530 |

## 结论

1. **最重要组件**: w/o Semantic（综合评分最高）
2. **最不重要组件**: w/o Communication（综合评分最低）

3. **各Agent平均贡献度**:

   - Communication: 质量+-0.4%, SLO+7.0%, 延迟-126.92ms
   - Placement: 质量+-0.0%, SLO+5.0%, 延迟-89.96ms
   - Reuse: 质量+-0.1%, SLO+-6.0%, 延迟--81.90ms
   - Semantic: 质量+0.0%, SLO+-7.0%, 延迟--75.79ms

## 建议

1. 优先保证关键Agent的正常运行
2. 对于资源受限场景，可考虑禁用贡献度较低的Agent
3. 各Agent协同工作才能发挥最大效果