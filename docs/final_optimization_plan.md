# SpectrumKV 最终方案 v3（修正仿真）

> 更新时间：2026-05-31 v3
> 基于：修正仿真（SpectrumKV正确建模全保留+分精度）+ 12种公式仿真 + 文献调研

---

## 一、核心结论

**SpectrumKV（Quantized Cache Bandwidth Management）是SWS的进化方向，不是替代。**

SWS框架不变（带宽受限PD分离场景KV传输），但传输策略从二元（传/不传）升级为三层精度（FP16/INT8/INT4）。

### 修正仿真结果（v3，正确建模）

| 方法 | b=0.3 NIAH | b=0.5 NIAH | b=0.7 NIAH | b=0.5 PPL代价 |
|------|-----------|-----------|-----------|-------------|
| PDTrim (baseline) | 32.8% | 33.9% | 51.3% | +15.6% |
| SWS Original | 32.3% | 42.3% | 51.3% | +15.2% |
| SWS ValueAware | 28.6% | 42.5% | 57.1% | +14.6% |
| **SpectrumKV Greedy AttnOnly** | **73.2%** | **85.2%** | **87.1%** | **+2.5%** |
| SpectrumKV Balanced AttnOnly | 72.7% | 82.3% | 87.1% | +4.2% |

### 量化提升

- **NIAH提升**: b=0.5时 +43pp（85% vs 42%），b=0.3时 +40pp（73% vs 33%）
- **PPL降低**: b=0.5时 +2.5% vs +15.6%（6倍改善）
- **Pareto前沿**: 完全是SpectrumKV方法，选择式方法被全面dominate

---

## 二、为什么SpectrumKV能赢？

### 2.1 根本原因：不是零和博弈

选择式方法（PDTrim/SWS）：在固定budget下，保留中间token必须丢弃sink/recent → 零和博弈
SpectrumKV：不丢弃任何token，只降低精度 → 非零和，所有token都"在"

### 2.2 b=0.5的完美匹配

b=0.5时，所有token变为INT8：
- 带宽：n × 0.5 = 0.5 × n × 1.0（FP16等效）→ 正好等于budget
- NIAH：100% token保留，INT8精度质量=95%
- PPL：全INT8量化仅+2.5%（基于真实INT8结果校准）

### 2.3 各budget下SpectrumKV的精度分配

| Budget | FP16% | INT8% | INT4% | 说明 |
|--------|-------|-------|-------|------|
| b=0.3 | 0% | 20% | 80% | 少量top token升级INT8 |
| b=0.5 | 0% | 100% | 0% | 全INT8，完美匹配 |
| b=0.7 | 40% | 60% | 0% | 部分升级FP16 |

---

## 三、最终方案：SpectrumKV v2.0

### 3.1 评分公式（改进版）

```python
def sws_score_v2(A_j, j, n, layer_idx, num_layers, locality):
    """
    改进版SWS评分：去掉距离衰减，改用attention-only + layer modulation
    """
    base_score = A_j  # 纯attention权重，无exp(-λ·Δ_j)
    
    # Layer调制：低locality层（global attention）→ 中间token更重要
    loc = locality[layer_idx]
    if loc < 0.3:  # Global-dominant layer
        position = j / n
        diversity_bonus = np.sin(np.pi * position) * 0.3
        base_score *= (1 + diversity_bonus)
    
    return base_score
```

**为什么去掉exp(-λ·Δ_j)？**
- A_j本身已偏向sink+recent（attention天然如此）
- 距离衰减进一步惩罚中间token，造成双重惩罚
- 在SpectrumKV模式下，评分只影响"谁先升级到高精度"，不影响"谁被丢弃"
- 所以评分对最终NIAH影响很小——全保留本身就是最大的优势

### 3.2 三层精度分配（SpectrumKV核心）

```python
def qcbm_assign_tiers(scores, budget, sink_size=4):
    """
    全保留 + 贪心精度分配
    
    策略：所有token从INT4起步，按score从高到低升级
    1. 先升级INT8（cost 0.25/tok，quality gain 0.17/tok → 最优性价比）
    2. 再升级FP16（cost 0.50/tok，quality gain 0.05/tok）
    """
    n = len(scores)
    tiers = np.full(n, 4)  # 全部INT4
    
    remaining = (budget - 0.25) * n  # INT4 baseline之上的剩余预算
    ranked = np.argsort(-scores)
    
    # Phase 1: INT4→INT8 (cost 0.25)
    n_int8 = min(n, int(remaining / 0.25))
    for i in range(n_int8):
        tiers[ranked[i]] = 8
    remaining -= n_int8 * 0.25
    
    # Phase 2: INT8→FP16 (cost 0.50)
    n_fp16 = min(n_int8, max(0, int(remaining / 0.50 + 0.5)))
    for i in range(n_fp16):
        tiers[ranked[i]] = 16
    
    return tiers
```

### 3.3 Per-Layer差异化预算

```python
def layer_budget(layer_idx, num_layers, base_budget):
    """PyramidKV启发：下层需更多budget"""
    beta = 0.02
    return max(0.15, min(0.95, 
        base_budget + (num_layers/2 - layer_idx) * beta))
```

### 3.4 On-Demand Fetch（Page Fault）

```python
def on_demand_fetch(tiers, current_attention, fetch_budget=0.05):
    """Decode时发现重要token在低精度 → 按需升级"""
    # 找attention高但在INT4/INT8的token
    candidates = [(j, current_attention[j]) 
                  for j in range(len(current_attention)) 
                  if tiers[j] < 16 and current_attention[j] > threshold]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    n_fetch = int(len(current_attention) * fetch_budget)
    for j, _ in candidates[:n_fetch]:
        tiers[j] = min(16, tiers[j] + 4)  # 升一级精度
    
    return tiers
```

---

## 四、SWS v1 vs v2 对比

| 维度 | SWS v1 (Original) | SpectrumKV v2 |
|------|-------------------|-------------|
| 评分公式 | A_j · exp(-λ·Δ_j) | A_j · f(layer_locality) |
| 传输策略 | 二元（传/不传） | 三层精度（FP16/INT8/INT4） |
| Token保留率 | budget% | 100% |
| NIAH b=0.5 | 42% | **85%** |
| PPL b=0.5 | +15% | **+2.5%** |
| 可恢复性 | 不可逆丢弃 | 可按需升级精度 |
| 系统模型 | 选择+传输 | 分层传输+按需升级 |

---

## 五、GPU验证计划

### 必须验证的关键假设

1. **INT8 KV cache真实质量**：仿真假设95%质量，实际可能是97-99%
2. **INT4 KV cache真实质量**：仿真假设78%质量，实际可能是70-85%
3. **混合精度kernel可行性**：vLLM/SGLang是否支持per-token不同精度

### 实验脚本

已生成：`gpu-experiments/exp_qcbm_quantization.py`

### 成功标准

- SpectrumKV b=0.5 PPL增加 < 5%（INT8量化基准）
- SpectrumKV b=0.5 NIAH > 80%（vs PDTrim ~40%）
- SpectrumKV b=0.3 NIAH > 60%（vs PDTrim ~30%）

---

## 六、论文叙事调整

### 从"选哪些token"→"用多少带宽传每个token"

旧叙事：SWS评分选出最重要的KV传输
新叙事：SWS在带宽约束下为每个KV分配合适的传输精度

**关键差异化 vs PDTrim**：
- PDTrim：单节点eviction，不可逆，只回答"留哪些"
- SpectrumKV：PD分离带宽受限传输，可恢复，回答"传多精确"

---

## 七、风险与限制

1. ⚠️ **仿真结论需GPU验证**：SpectrumKV的NIAH/PPL提升基于修正仿真，INT4真实质量可能不同
2. ⚠️ **评分函数对SpectrumKV影响很小**：这意味着论文难以在"评分公式"上claim贡献，核心贡献在SpectrumKV框架
3. ⚠️ **混合精度KV cache实现复杂度**：当前推理框架不支持per-token动态精度
4. ⚠️ **INT4对检索的影响**：仿真假设78%质量，实际可能更低，需GPU确认
5. ⚠️ **论文创新性**：混合精度KV cache已有KVQuant/MiniKV等工作，差异化在PD分离场景

---

## 八、与旧版方案的关键差异

| 维度 | v2 (错误仿真) | v3 (修正仿真) |
|------|--------------|--------------|
| SpectrumKV建模 | 选1.3x token再分tier | 全保留+贪心精度分配 |
| b=0.5 SpectrumKV NIAH | 42.6% (错误) | **85.2%** (正确) |
| b=0.3 SpectrumKV NIAH | 12.7% (错误) | **73.2%** (正确) |
| SpectrumKV vs PDTrim | 有时输 | **全面碾压** |
| 核心洞察 | 评分优化+SpectrumKV | SpectrumKV全保留是唯一解 |

**旧仿真的bug**：SpectrumKV还是用"选top-K"模式，只是多选了30%。没有体现"全保留"的核心优势。修正后结果天壤之别。
