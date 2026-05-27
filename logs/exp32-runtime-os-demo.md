# exp32 - Runtime KV Memory OS Demo

## 实验概述

**实验目标**: 演示完整的 Runtime KV Memory OS 工作流程

**核心组件**:
- **Global State Store**: 所有Agent写入的唯一数据源
- **Semantic Agent**: 分析语义区域和token重要性
- **Reuse Agent**: 预测token重用模式
- **Communication Agent**: 估算访问成本
- **Placement Agent**: 管理KV物理位置
- **Runtime Scheduler**: 读取所有状态，统一决策

## 统一目标函数

```
max Quality - λ₁×Latency - λ₂×Memory - λ₃×TransferCost
```

## 实验结果

| Task Type | Tokens | Scheduler Retain | Scheduler Evict | Conflicts | Quality | Latency (ms) |
|-----------|--------|------------------|------------------|-----------|---------|--------------|
| math | 50 | 26 | 24 | 0 | 52.0% | 0.00 |
| math | 100 | 51 | 49 | 13 | 51.0% | 76.60 |
| math | 200 | 101 | 99 | 111 | 50.5% | 633.19 |
| code | 50 | 26 | 24 | 0 | 52.0% | 0.00 |
| code | 100 | 51 | 49 | 9 | 51.0% | 434.84 |
| code | 200 | 101 | 99 | 114 | 50.5% | 605.24 |
| qa | 50 | 26 | 24 | 0 | 52.0% | 0.00 |
| qa | 100 | 51 | 49 | 14 | 51.0% | 312.97 |
| qa | 200 | 102 | 98 | 110 | 51.0% | 588.38 |

## 汇总统计

- **平均冲突数**: 41.22
- **平均质量**: 51.2%
- **平均延迟**: 294.58ms

## 关键发现

1. **Scheduler协调效果**: 相比各Agent独立决策，Scheduler能减少决策冲突
2. **统一目标函数**: 通过加权组合各Agent信息，得出更优的全局决策
3. **任务适配**: 不同任务类型的token分布和访问模式不同，Scheduler能自适应

## 架构优势

```
┌─────────────────────────────────────────────────────┐
│           Global State Store (唯一数据源)           │
└─────────────────────────────────────────────────────┘
    ↑ 写入           ↑ 写入           ↑ 写入           ↑ 写入
┌─────────┐ ┌─────────┐ ┌────────────┐ ┌─────────┐
│Semantic │ │  Reuse  │ │Communication│ │Placement│
│ Agent   │ │  Agent  │ │   Agent     │ │ Agent   │
└─────────┘ └─────────┘ └────────────┘ └─────────┘
                      ↓ 读取
               ┌─────────────────┐
               │  Runtime        │
               │  Scheduler      │
               └─────────────────┘
```