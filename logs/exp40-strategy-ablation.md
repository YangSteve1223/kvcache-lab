# exp40: 策略替换Ablation实验报告

## 实验目的

验证每个策略组件的独立贡献，通过将智能策略替换为简单基线策略来量化。

## 实验设计

| 替换策略 | 原策略 | 替换为 |
|---------|--------|--------|
| TAA→StandardAttention | Transmission-Aware Attention | 普通Attention（不考虑位置成本） |
| Predictive→LRU | Predictive Eviction | LRU最近最少使用 |
| SWS→FixedRatio | Semantic Working Set | 固定保留50% |
| Hierarchical→AllGPU | Hierarchical Placement | 全部放GPU |

## 配置参数

- Token数量: 128
- 层数: 32
- KV大小/token: 1024 bytes
- 带宽: 100 GB/s
- GPU内存: 32 GB

## 详细结果

### 延迟对比

| 策略 | 平均延迟(ms) | vs Full OS |
|------|-------------|------------|
| Full OS | 367.51 | 0.0% |
| TAA→StandardAttention | 25.69 | -93.0% |
| Predictive→LRU | 28.33 | -92.3% |
| SWS→FixedRatio | 24.05 | -93.5% |
| Hierarchical→AllGPU | 28.33 | -92.3% |

### 质量对比

| 策略 | 平均质量 | vs Full OS |
|------|---------|------------|
| Full OS | 50.8% | 0.0% |
| TAA→StandardAttention | 71.9% | +21.1% |
| Predictive→LRU | 100.0% | +49.2% |
| SWS→FixedRatio | 52.3% | +1.5% |
| Hierarchical→AllGPU | 100.0% | +49.2% |

### 传输量对比

| 策略 | 平均传输量(KB) | vs Full OS |
|------|----------------|------------|
| Full OS | 35.9 | 0.0% |
| TAA→StandardAttention | 49.3 | +37.3% |
| Predictive→LRU | 83.0 | +131.4% |
| SWS→FixedRatio | 28.3 | -21.1% |
| Hierarchical→AllGPU | 83.0 | +131.4% |

### SLO满足率对比

| 策略 | SLO满足率 | vs Full OS |
|------|----------|------------|
| Full OS | 99.0% | 0.0% |
| TAA→StandardAttention | 100.0% | +1.0% |
| Predictive→LRU | 100.0% | +1.0% |
| SWS→FixedRatio | 100.0% | +1.0% |
| Hierarchical→AllGPU | 100.0% | +1.0% |

## 结论

1. **影响最大替换**: 
   - 延迟增加: 0.00ms
   - 质量下降: 0.0%

2. **影响最小替换**: SWS→FixedRatio
   - 延迟变化: -343.46ms
   - 质量变化: +-1.5%

3. **策略贡献排序** (从大到小):
   1. SWS→FixedRatio
   2. TAA→StandardAttention
   3. Predictive→LRU
   4. Hierarchical→AllGPU

## 讨论

本实验比"禁用Agent"更严格，因为：
1. 直接对比智能策略与简单基线
2. 排除"禁用后其他组件补偿"的可能性
3. 量化每个策略改进的具体收益