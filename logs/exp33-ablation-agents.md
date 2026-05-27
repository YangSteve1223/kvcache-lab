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
| Full system | 50.7% (0.0%) | 311.24 (0.00) | 0.1 | 97.0% (0.0%) | 61.5 | 27.1 |
| w/o Semantic | 50.7% (-0.1%) | 262.67 (-48.57) | 0.1 | 98.0% (+1.0%) | 63.6 | 23.1 |
| w/o Reuse | 50.8% (+0.1%) | 219.51 (-91.73) | 0.1 | 100.0% (+3.0%) | 59.3 | 19.2 |
| w/o Communication | 51.1% (+0.3%) | 406.62 (+95.38) | 0.1 | 94.0% (-3.0%) | 60.0 | 35.5 |
| w/o Placement | 50.8% (+0.1%) | 351.68 (+40.44) | 0.1 | 95.0% (-2.0%) | 60.1 | 30.6 |

## Agent贡献度分析

基于 Full system vs 各 w/o 配置的对比：

### 禁用 Semantic 的影响

- 质量下降: 0.1%
- SLO满足率下降: -1.0%
- 延迟增加: -48.57ms

### 禁用 Reuse 的影响

- 质量下降: -0.1%
- SLO满足率下降: -3.0%
- 延迟增加: -91.73ms

### 禁用 Communication 的影响

- 质量下降: -0.3%
- SLO满足率下降: 3.0%
- 延迟增加: 95.38ms

### 禁用 Placement 的影响

- 质量下降: -0.1%
- SLO满足率下降: 2.0%
- 延迟增加: 40.44ms

## 综合排名

| 排名 | 配置 | 综合评分 |
|------|------|----------|
| 1 | w/o Reuse | -0.209 |
| 2 | w/o Semantic | -0.304 |
| 3 | Full system | -0.335 |
| 4 | w/o Placement | -0.370 |
| 5 | w/o Communication | -0.426 |

## 结论

1. **最重要组件**: w/o Reuse（综合评分最高）
2. **最不重要组件**: w/o Communication（综合评分最低）

3. **各Agent平均贡献度**:

   - Communication: 质量+-0.3%, SLO+3.0%, 延迟-95.38ms
   - Placement: 质量+-0.1%, SLO+2.0%, 延迟-40.44ms
   - Semantic: 质量+0.1%, SLO+-1.0%, 延迟--48.57ms
   - Reuse: 质量+-0.1%, SLO+-3.0%, 延迟--91.73ms

## 建议

1. 优先保证关键Agent的正常运行
2. 对于资源受限场景，可考虑禁用贡献度较低的Agent
3. 各Agent协同工作才能发挥最大效果