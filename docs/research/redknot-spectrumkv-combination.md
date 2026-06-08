# RedKnot × SpectrumKV 组合方案技术分析

> **Head-Adaptive Mixed-Precision KV Cache: Head维度与Token/Precision维度的正交组合**

**日期**: 2026-06-07  
**版本**: v1.0  
**状态**: 技术分析草案

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [正交性证明](#2-正交性证明)
3. [组合方案设计: Head-Adaptive Mixed-Precision KV Cache](#3-组合方案设计-head-adaptive-mixed-precision-kv-cache)
4. [预期收益分析](#4-预期收益分析)
5. [潜在论文贡献点](#5-潜在论文贡献点)
6. [实验设计建议](#6-实验设计建议)
7. [风险与挑战](#7-风险与挑战)
8. [参考文献](#8-参考文献)

---

## 1. 背景与动机

### 1.1 RedKnot: Head-Aware KV Cache管理

**论文**: *RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention* (arXiv:2606.06256, 2026)

**核心洞察**: KV cache在不同attention head间具有高度结构化的效用差异——不同head展现不同的功能角色、有效注意力范围和运行时重要性。

**关键机制**:

| 机制 | 描述 |
|------|------|
| **Head-Class Sparsification** | 将每个 (layer, head) 对离线分类为 global head（12-15%，需要全上下文重新计算）或 local head（85-88%，在滑动窗口内直接复用） |
| **Elastic Sparsity** | 自适应运行时恢复机制，当 edge-mass 信号检测到分类不足时，将个别 local head 提升为 full attention |
| **SegPagedAttention** | 替换 dense layout，采用 per-(layer,head) 分页 KV 存储 + fused varlen attention kernel，物理上只保留每个 head 需要的 tokens |
| **Sparse FFN** | 仅对 attention score 最高的 top-k tokens 执行 FFN，独立于上下文长度 |

**关键数据**:

- Local heads 占比: 84.4%–87.5%（Mistral-7B, Qwen3-32B, Llama-3.3-70B）
- TTFT 加速: 1.6–3.5×
- 并发会话提升: 4.7–7.8×
- Prefill FLOPs 减少: 67–79.5%
- SegPagedAttention decode 加速: 2.5–21.0×（vs dense+mask）
- SegPagedAttention prefill 加速: 6.3–23.3×
- KV transfer volume 减少: 4.3–6.3×

**Head分类模型**:

```
┌─────────────────────────────────────────────┐
│              KV Heads (H kv)                 │
├───────────────┬─────────────────────────────┤
│ Global Heads  │  Local Heads                │
│ (12-15%)      │  (85-88%)                   │
│               │                             │
│ - 需要全上下文  │ - 只需 sink + recent window │
│ - 前缀敏感     │ - 前缀无关                  │
│ - 重新计算     │ - 直接复用缓存 KV           │
└───────────────┴─────────────────────────────┘
```

### 1.2 SpectrumKV: Per-Token Mixed-Precision KV Cache

**核心洞察**: 不同 token 的语义重要性（Semantic Weight Score, SWS）差异显著，应根据 SWS 为每个 token 的 KV 对分配不同精度级别。

**关键机制**:

| 机制 | 描述 |
|------|------|
| **SWS (Semantic Weight Scoring)** | 基于语义重要性为每个 token 计算权重分数，驱动精度分配决策 |
| **Per-Token Mixed Precision** | 高 SWS token 保持高精度（FP16/INT8），低 SWS token 使用低精度（INT4/INT2） |
| **Tiered KV Cache** | Tier-1 VRAM 存储量化 KV（INT8 keys + INT4 values），Tier-2 CPU RAM 保留 FP16 originals 作为 fallback |
| **Adaptive Precision Selector** | 基于 attention mass 分布动态选择哪些 block 需要 FP16 keys，保证 attention 分布失真有界 |
| **Compressed-Domain Execution** | 在量化数据上直接执行 attention，寄存器内反量化，无需中间 FP16 buffer |

**关键数据（基于 Runtime-Certified Bounded-Error Quantized Attention, arXiv:2605.20868）**:

- Tier-1 存储开销: ~56% FP16（含元数据）
- Per-channel INT8 keys: 75% FP16（含 per-block per-channel scales/offsets）
- Per-group INT4 values: 37.5% FP16（d=128, g=16）
- PG-19 perplexity: Δppl = +0.03（near-lossless）
- Page-in latency: ~3μs/block (PCIe 5.0)
- Fallback ladder: 4-rung 精度升级机制

**SWS 精度分配模型**:

```
┌──────────────────────────────────────────────────┐
│                Token Sequence (L)                  │
├──────┬──────┬──────┬──────┬──────┬──────┬────────┤
│ tok₁ │ tok₂ │ tok₃ │ tok₄ │ tok₅ │ tok₆ │  ...   │
│ SWS: │ SWS: │ SWS: │ SWS: │ SWS: │ SWS: │        │
│ HIGH │ LOW  │ MED  │ HIGH │ LOW  │ MED  │        │
├──────┼──────┼──────┼──────┼──────┼──────┼────────┤
│FP16  │INT4  │INT8  │FP16  │INT4  │INT8  │  ...   │
└──────┴──────┴──────┴──────┴──────┴──────┴────────┘
```

### 1.3 为什么要组合？

RedKnot 和 SpectrumKV 分别从两个**完全正交的维度**优化 KV cache：

| 维度 | RedKnot | SpectrumKV |
|------|---------|------------|
| **优化轴** | Head 维度（哪些 head 需要什么范围的 KV） | Token/Precision 维度（哪些 token 的 KV 需要多高精度） |
| **压缩方式** | 空间裁剪（local head 只保留短窗口） | 精度裁剪（低重要性 token 使用低精度） |
| **信息保留** | 保留重要 head 的完整 KV，丢弃不重要 head 的冗余 KV | 保留所有 token 的 KV，但按重要性分配精度 |
| **独立有效性** | ✅ 已验证 | ✅ 已验证 |

两者组合的直觉：**一个 local head 的 sink/recent window 中的 token 不需要全部 FP16，而一个 global head 中的低 SWS token 也不需要 FP16**。组合后在两个轴上同时压缩，收益应当**乘性叠加**。

---

## 2. 正交性证明

### 2.1 形式化定义

**KV Cache 张量**: $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times H_{kv} \times L \times D}$

其中 $B$ = batch size, $H_{kv}$ = KV head 数, $L$ = 序列长度, $D$ = head 维度。

**RedKnot 的压缩算子** $\mathcal{R}$:

$$\mathcal{R}: \mathbb{R}^{B \times H_{kv} \times L \times D} \rightarrow \mathbb{R}^{B \times H_{kv} \times L_h \times D}$$

其中 $L_h$ 是 head $h$ 的有效上下文长度：

$$L_h = \begin{cases} L & \text{if } h \in \mathcal{G} \text{ (global head)} \\ W & \text{if } h \in \mathcal{L} \text{ (local head, } W \ll L\text{)} \end{cases}$$

**SpectrumKV 的压缩算子** $\mathcal{S}$:

$$\mathcal{S}: \mathbb{R}^{B \times H_{kv} \times L \times D} \rightarrow \{(\mathbf{K}_t^{b_t}, \mathbf{V}_t^{b_t})\}_{t=1}^{L}$$

其中 $b_t \in \{2, 4, 8, 16\}$ 是 token $t$ 的精度级别，由 SWS 决定：

$$b_t = f_{\text{SWS}}(\text{SWS}(t))$$

### 2.2 正交性定理

**定理 (Head-Token 正交性)**: RedKnot 的压缩算子 $\mathcal{R}$ 和 SpectrumKV 的压缩算子 $\mathcal{S}$ 在 KV cache 张量上是正交的，即它们分别作用于不相交的索引维度，且组合压缩率等于各自压缩率的乘积。

**证明**:

**(1) 作用维度不相交**

RedKnot 算子 $\mathcal{R}$ 的作用域为**第2维（Head 维度）**和**第3维（序列长度的有效子集）**：

$$\mathcal{R} \text{ 修改的维度集合: } \{H_{kv}, L\} \text{，具体为 } L \rightarrow L_h \text{（per-head 裁剪）}$$

SpectrumKV 算子 $\mathcal{S}$ 的作用域为**第3维（Token 精度）**和**第4维（数据表示精度）**：

$$\mathcal{S} \text{ 修改的维度集合: } \{L, D\} \text{，具体为每个 token 的位宽 } b_t$$

关键区别：$\mathcal{R}$ 改变的是每个 head 的**序列长度**（物理上删除 token），$\mathcal{S}$ 改变的是每个 token 的**表示精度**（每个元素占用的比特数）。两者作用于同一序列轴但以不同方式（空间裁剪 vs. 精度裁剪），且互不依赖对方的决策空间。

**(2) 独立决策性**

RedKnot 的 head 分类决策仅依赖于 (layer, head) 的内在属性（是否是 retrieval head / streaming head），这是一个**模型内在属性**，与具体 token 无关：

$$\text{class}(l, h) = \text{Global} \mid \text{Local} \quad \text{（离线分析确定，输入无关）}$$

SpectrumKV 的 SWS 决策依赖于每个 token 的**语义重要性**，这是一个**token 级别属性**，与 head 分类无关：

$$\text{SWS}(t) = g(\text{attn\_scores}_t, \text{value\_norm}_t, \ldots) \quad \text{（运行时计算，head 无关）}$$

因此：

$$\frac{\partial \text{class}(l,h)}{\partial \text{SWS}(t)} = 0, \quad \frac{\partial \text{SWS}(t)}{\partial \text{class}(l,h)} = 0$$

两个决策变量之间**无耦合**。

**(3) 压缩率乘性叠加**

设 RedKnot 的空间压缩率为 $\rho_R$（保留的 token 比例），SpectrumKV 的精度压缩率为 $\rho_S$（平均位宽 / FP16位宽）。

- RedKnot 压缩后存储: $\rho_R \cdot B \cdot H_{kv} \cdot L \cdot D \cdot 16$ bits
- SpectrumKV 压缩后存储: $1 \cdot B \cdot H_{kv} \cdot L \cdot D \cdot \bar{b}$ bits，其中 $\bar{b} = \rho_S \cdot 16$
- 组合压缩后存储: $\rho_R \cdot B \cdot H_{kv} \cdot L \cdot D \cdot \bar{b}$ bits

组合压缩率:

$$\rho_{\text{combined}} = \frac{\rho_R \cdot \bar{b}}{16} = \rho_R \cdot \rho_S$$

即**组合压缩率 = 空间压缩率 × 精度压缩率**，满足乘性叠加。 $\blacksquare$

### 2.3 正交性的边界条件

正交性在以下条件下严格成立：

1. **Head 分类与 token 精度独立**: SWS 的计算不依赖于 head 分类结果（实际中，不同 head 对同一 token 的注意力分数不同，因此 SWS 可能需要 per-head 计算，此时正交性变为近似正交）
2. **Fallback 不产生交叉依赖**: SpectrumKV 的 fallback ladder（FP16 升级）不应触发 RedKnot 的 elastic sparsity（head 提升），反之亦然
3. **物理存储布局兼容**: SegPagedAttention 的 per-head ragged layout 需要支持 per-token 变精度存储

当 SWS 变为 **per-(head, token)** 时，两个维度产生弱耦合，此时需要引入协调机制（详见 Section 3.4）。

---

## 3. 组合方案设计: Head-Adaptive Mixed-Precision KV Cache

### 3.1 设计理念

将 RedKnot 的 head 维度空间裁剪与 SpectrumKV 的 token 维度精度裁剪组合，形成一个**二维压缩空间**：

```
                     Precision (SpectrumKV)
                    FP16   INT8   INT4   INT2
                 ┌───────┬───────┬───────┬───────┐
     Global      │  G-FP │  G-I8 │  G-I4 │  G-I2 │  ← 全上下文
Head             │ 16    │       │       │       │    但可变精度
(RedKnot)       ├───────┼───────┼───────┼───────┤
     Local       │  L-FP │  L-I8 │  L-I4 │  L-I2 │  ← 短窗口
                 │ 16    │       │       │       │    且可变精度
                 └───────┴───────┴───────┴───────┘
                          
     组合策略: Global head 高 SWS token → FP16/INT8
               Global head 低 SWS token → INT4
               Local head  高 SWS token → INT8
               Local head  低 SWS token → INT4/INT2
```

### 3.2 系统架构

```
┌────────────────────────────────────────────────────────────────────┐
│                   Head-Adaptive Mixed-Precision KV Cache           │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Offline Profiling Phase                         │  │
│  │  ┌─────────────────────┐    ┌──────────────────────────┐    │  │
│  │  │  Head Classifier    │    │  SWS Calibration         │    │  │
│  │  │  (RedKnot)          │    │  (SpectrumKV)            │    │  │
│  │  │  - NIAH-based       │    │  - Attention mass dist.  │    │  │
│  │  │  - class(l,h) ∈    │    │  - Value norm profile    │    │  │
│  │  │    {Global, Local}  │    │  - Per-head SWS params   │    │  │
│  │  └─────────┬───────────┘    └──────────┬───────────────┘    │  │
│  │            │                            │                    │  │
│  │            └────────────┬───────────────┘                    │  │
│  │                         ▼                                    │  │
│  │            ┌────────────────────────┐                        │  │
│  │            │  Combined Policy Table │                        │  │
│  │            │  (l, h) → {           │                        │  │
│  │            │    class: G/L,        │                        │  │
│  │            │    ctx_len: L_h,      │                        │  │
│  │            │    default_prec: b_h, │                        │  │
│  │            │    sws_thresholds: ... │                        │  │
│  │            │  }                    │                        │  │
│  │            └────────────────────────┘                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Runtime Execution Phase                         │  │
│  │                                                              │  │
│  │  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐ │  │
│  │  │  Prefill     │   │  Decode      │   │  Adaptive        │ │  │
│  │  │  - Head-aware│   │  - SWS-based │   │  Precision       │ │  │
│  │  │    KV Reuse  │   │    precision │   │  Selector        │ │  │
│  │  │  - Sparse FFN│   │    selection │   │  - Per-head      │ │  │
│  │  └──────┬──────┘   └──────┬───────┘   │    coverage τ_h  │ │  │
│  │         │                 │            └────────┬─────────┘ │  │
│  │         └─────────┬───────┘                     │           │  │
│  │                   ▼                             ▼           │  │
│  │        ┌──────────────────────────────────────────────┐     │  │
│  │        │   SegPagedAttention + Mixed-Precision Store  │     │  │
│  │        │                                              │     │  │
│  │        │   Per-(l,h) page table:                     │     │  │
│  │        │   ┌────────────────────────────────────┐     │     │  │
│  │        │   │ Global head: full context pages     │     │     │  │
│  │        │   │   token t → precision b_{h,t}       │     │     │  │
│  │        │   │ Local head: sink+recent pages only  │     │     │  │
│  │        │   │   token t → precision b_{h,t}       │     │     │  │
│  │        │   └────────────────────────────────────┘     │     │  │
│  │        │                                              │     │  │
│  │        │   Fused varlen attention kernel:             │     │  │
│  │        │   - Mask-free FlashAttention path            │     │  │
│  │        │   - In-register dequantization               │     │  │
│  │        │   - Per-head ragged lengths                  │     │  │
│  │        └──────────────────────────────────────────────┘     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Tiered Storage                                  │  │
│  │  ┌────────────────────────┐  ┌───────────────────────────┐  │  │
│  │  │  Tier-1: VRAM (hot)    │  │  Tier-2: CPU RAM (cold)   │  │  │
│  │  │  - Global head:        │  │  - FP16 originals         │  │  │
│  │  │    high-SWS: INT8 K    │  │  - For adaptive precision │  │  │
│  │  │    low-SWS:  INT4 K    │  │    promotion & fallback   │  │  │
│  │  │    values:  INT4 V     │  │                           │  │  │
│  │  │  - Local head:         │  │                           │  │  │
│  │  │    all tokens: INT4 K/V│  │                           │  │  │
│  │  │  + Error annotations   │  │                           │  │  │
│  │  └────────────────────────┘  └───────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 3.3 核心算法: Head-Adaptive Mixed-Precision Selection (HAMPS)

**输入**: Query $q$; KV cache blocks $\{b_1, \ldots, b_N\}$; Head class map $\text{class}(l,h)$; Per-head SWS thresholds $\tau_h$

**输出**: Per-(head, block) precision decision

```
Algorithm: HAMPS (Head-Adaptive Mixed-Precision Selection)
──────────────────────────────────────────────────────────
For each layer l, head h:
  1. Determine head class:
     if class(l, h) == Global:
         context_len = L (full context)
         base_precision = INT8 (keys)
     else:  // Local
         context_len = W (sink + recent window)
         base_precision = INT4 (keys)
     
  2. Phase 1: INT8 scoring pass
     For each block b in context_len:
         s_{b,t} = q · k̂_{b,t} / √d    // INT8 dequantized keys
         m_b = max_t s_{b,t}
         p_b = Σ_t exp(s_{b,t} - m_b)   // block mass
     
  3. Compute SWS for each block:
     SWS(b, h) = p_b · ||V_b||_2 / V_max
     
  4. Adaptive precision assignment:
     Sort blocks by SWS(b, h) descending
     
     if class(l, h) == Global:
         // Global heads: top-τ_h coverage gets INT8, rest gets INT4
         K* = min blocks for Σ p_b ≥ τ_h
         For b ∈ top-K* blocks:   key_prec = INT8
         For b ∉ top-K* blocks:   key_prec = INT4
     else:
         // Local heads: all use INT4 (short context, low impact)
         key_prec = INT4 for all blocks
     
  5. Value precision:
     // Per-block value error annotation η_b
     For each block b:
         if η_b > v_tol:
             val_prec = FP16  // page-in from Tier-2
         else:
             val_prec = INT4
     
  6. Phase 2: Fused attend with mixed-precision keys
     Execute attention with per-block key precision
     Accumulate into online softmax state
     
  7. Error certification:
     E_key^{(h)} = 2·V_max · e^{2Δ} · α_T · (e^{2Δ} - 1)
     E_val^{(h)} = Σ_b ρ_b^{(h)} · η_b
     E_total^{(h)} = E_key^{(h)} + E_val^{(h)}
     
     if E_total^{(h)} > ε_h:  // head-specific tolerance
         Escalate fallback ladder for head h
```

### 3.4 Head-Specific SWS: 从正交到协调

虽然 head 分类和 token 精度在原则上正交，但实际部署中需要处理**弱耦合**：

**问题**: 不同 head 对同一 token 的注意力分数不同，因此 SWS 应当是 per-(head, token) 的，而非全局的。

**解决方案: Head-Specific SWS (HSWS)**

$$\text{HSWS}(h, t) = \alpha_h \cdot p_t^{(h)} + \beta_h \cdot \frac{\|V_t^{(h)}\|_2}{V_{\max}^{(h)}}$$

其中：
- $p_t^{(h)}$ = head $h$ 对 token $t$ 的注意力概率
- $\alpha_h, \beta_h$ = per-head 权重参数（离线校准）
- Global heads: $\alpha_h$ 较大（更关注注意力分布质量）
- Local heads: $\beta_h$ 较大（更关注 value 重建质量）

**协调规则**:

| Head Class | Key Precision | Value Precision | SWS 决策频率 |
|------------|--------------|-----------------|-------------|
| Global | 高 SWS → INT8, 低 SWS → INT4 | FP16 fallback for high η_b | 每个 decode step |
| Local | 统一 INT4（短上下文无需精细精度） | 统一 INT4（窗口内 token 重要性差异小） | 可降频（每 K step 一次） |

**关键优化**: Local head 的 SWS 计算可**降频**——因为 local head 只关注短窗口（~320 tokens），且其 attention 模式稳定，SWS 的变化幅度小，不需要每个 decode step 重新计算。这显著降低了组合方案的运行时开销。

### 3.5 SegPagedAttention 扩展: 变精度分页

当前 SegPagedAttention 为每个 head 分配固定精度的 page。组合方案需要扩展为**变精度 page**：

**方案A: Per-Page Precision Tag**

```
Page Header:
┌──────────┬──────────┬──────────┬───────────┐
│ head_id  │ ctx_len  │ prec_tag │ data_ptr  │
│ (8 bit)  │ (16 bit) │ (2 bit)  │ (64 bit)  │
└──────────┴──────────┴──────────┴───────────┘

prec_tag ∈ {FP16, INT8, INT4, INT2}
Data size per token = prec_tag → bytes
```

**方案B: Precision Zone（推荐）**

将每个 head 的 KV cache 划分为 precision zone：

```
Global Head KV Layout:
┌─────────────────────────────────────────────────────┐
│ Zone 1: High-SWS tokens (top-K*)                     │
│         Key: INT8, Value: INT4                       │
│         [physically contiguous, aligned]              │
├─────────────────────────────────────────────────────┤
│ Zone 2: Low-SWS tokens (tail)                        │
│         Key: INT4, Value: INT4                       │
│         [physically contiguous, aligned]              │
└─────────────────────────────────────────────────────┘

Local Head KV Layout:
┌─────────────────────────────┐
│ Single Zone: All tokens      │
│ Key: INT4, Value: INT4       │
│ (short window, no zone split)│
└─────────────────────────────┘
```

**方案B优势**:
- 区域内精度统一，kernel 实现简单
- 无需 per-token 精度标记
- 与 FlashAttention varlen 兼容（每个 zone 作为一个 varlen segment）
- Zone 边界可由 adaptive precision selector 的 $K^*$ 参数确定

### 3.6 Fallback Ladder 扩展

原 SpectrumKV 的 4-rung fallback 扩展为 5-rung，增加 head-level 降级：

| Rung | 触发条件 | 动作 |
|------|---------|------|
| 0 | Normal operation | Per-zone mixed precision (default) |
| 1 | $E_{\text{total}}^{(h)} > \epsilon_h$ | Expand $K^*$ (更多 block 升级到 INT8/FP16 keys) |
| 2 | Value error $\eta_b > v_{\text{tol}}$ | Upgrade value precision to FP16 for affected blocks |
| 3 | Ranking inconsistency detected | Per-head fallback to dense attention (all FP16 KV for head $h$) |
| 4 | **Elastic sparsity trigger** | **Per-head promotion: Local → Global** (RedKnot's elastic sparsity) |
| 5 | Unrecoverable error | All-head $O_{\text{dense}}$ (full FP16 recomputation) |

**Rung 4 是新增的协调机制**：当 SpectrumKV 的 fallback 发现某个 local head 在低精度下持续触发错误，说明该 head 可能被错误分类或输入分布发生了偏移。此时，RedKnot 的 elastic sparsity 机制介入，将该 head 从 local 提升为 global，给予完整上下文 + 更高精度的 KV。

---

## 4. 预期收益分析

### 4.1 存储收益

以 Llama-3.3-70B（64层, $H_{kv}=8$, $D=128$, FP16）为例：

**Baseline (Dense FP16)**:
- 每 token 每 head: $2 \times 128 \times 2 = 512$ bytes (K+V)
- 每 token: $512 \times 8 = 4096$ bytes
- 128K context: $4096 \times 128K = 512$ MB per layer; $512 \times 64 = 32$ GB total

**RedKnot Only** (85% local heads, window=320):
- Global heads (15%): $512 \times 128K = 96$ MB/layer
- Local heads (85%): $512 \times 320 \times 6.8 = 1.1$ MB/layer
- Total per layer: ~97.1 MB; 全模型: ~6.2 GB
- 压缩率: 32/6.2 = **5.2×**

**SpectrumKV Only** (平均精度 4-bit effective):
- 每 token 每 head: $(128/8 \times 4 + 128/2 \times 4) / 8 \approx 64$ bytes（含元数据约 96 bytes）
- 平均每 token: $96 \times 8 = 768$ bytes
- 压缩率: 4096/768 = **5.3×**

**组合方案**:
- Global heads (15%): $96 \times 128K$ bytes（混合精度，平均~4-bit effective）
- Local heads (85%): $48 \times 320$ bytes（全 INT4，更激进修减）
- 压缩率: ~**25–30×**（空间裁剪 × 精度裁剪 乘性叠加）

| 方案 | 128K Context 存储占用 | 压缩率 |
|------|----------------------|--------|
| Dense FP16 | ~32 GB | 1× |
| RedKnot Only | ~6.2 GB | 5.2× |
| SpectrumKV Only | ~6.0 GB | 5.3× |
| **组合方案** | **~1.1–1.3 GB** | **25–30×** |

### 4.2 带宽收益

Decode 阶段每个 step 的 KV cache 读取量：

| 方案 | 每 head 每 step 读取 | 总读取/step |
|------|---------------------|-----------|
| Dense FP16 | $2 \times 128 \times 128K \times 2 = 64$ MB | 512 MB |
| RedKnot | ~1.0 MB (local) + 64 MB (global) | ~68 MB |
| SpectrumKV | ~24 MB (4-bit avg) | ~192 MB |
| **组合** | ~0.5 MB (local, INT4) + ~8 MB (global, mixed) | **~12 MB** |

### 4.3 TTFT 收益

Prefill 阶段，组合方案在三个轴上同时减少计算：

1. **Head 轴** (RedKnot): Local heads 跳过全上下文 attention → 减少 67–79% attention FLOPs
2. **FFN 轴** (RedKnot): Sparse FFN 只计算重要 tokens → 减少 40–55% FFN FLOPs
3. **精度轴** (SpectrumKV): 量化 attention 执行 → 进一步减少内存带宽开销

预估组合 TTFT 加速: **4–8×**（vs dense），vs RedKnot alone 的 1.6–3.5× 有显著提升。

### 4.4 并发容量收益

| 方案 | 128K Context 并发会话数/GPU | 提升倍数 |
|------|---------------------------|---------|
| Dense (80GB HBM) | ~3–4 | 1× |
| RedKnot | ~19–31 | 4.7–7.8× |
| **组合** | **~60–90** | **15–23×** |

### 4.5 收益叠加的理论保证

由正交性定理（Section 2.2），组合压缩率满足乘性叠加：

$$\rho_{\text{combined}} = \rho_R \times \rho_S$$

但端到端质量保证需要额外分析。组合方案的输出误差为：

$$\|O_{\text{combined}} - O_{\text{dense}}\| \leq E_{\text{head-truncation}} + E_{\text{quantization}} + E_{\text{interaction}}$$

其中 $E_{\text{interaction}}$ 是两个压缩机制的交互误差。由于 RedKnot 的 local head 截断是确定性的（基于稳定的 head 分类），而 SpectrumKV 的精度降级是有界近似的，交互误差应当可由 Cascade Bound 控制：

$$E_{\text{interaction}} \leq E_{\text{head-truncation}} \cdot (e^{2\Delta} - 1)$$

即 head 截断引入的误差会被量化误差的 softmax distortion 因子放大，但该放大因子有界且可由 fallback ladder 控制。

---

## 5. 潜在论文贡献点

### 5.1 核心贡献

1. **Head-Token 正交性理论框架**
   - 首次形式化证明 KV cache 压缩中 head 维度与 token/precision 维度的正交性
   - 建立乘性压缩率的理论保证
   - 推导 Cascade Error Bound（交互误差有界性）

2. **HAMPS 算法 (Head-Adaptive Mixed-Precision Selection)**
   - 首个在 head 维度和 precision 维度同时做自适应决策的 KV cache 管理算法
   - Head-Specific SWS (HSWS): 将语义重要性评分从 per-token 扩展到 per-(head, token)
   - 精度分配的 head-aware 协调规则

3. **Precision-Zone SegPagedAttention**
   - 扩展 SegPagedAttention 支持变精度分页
   - Precision Zone 设计：同一 head 内按 SWS 划分精度区域
   - 与 FlashAttention varlen 的兼容性设计

4. **5-Rung Coordinated Fallback Ladder**
   - 首个跨 head 分类和精度选择两个维度的协调 fallback 机制
   - Elastic sparsity 与 adaptive precision 的交互升级策略

### 5.2 系统贡献

5. **端到端 Head-Aware Mixed-Precision Serving Engine**
   - 完整的系统实现：从离线 profiling 到在线 serving
   - Prefill-Decode disaggregation 下的 mixed-head mixed-precision KV transfer
   - 与现有框架 (vLLM/SGLang) 的集成路径

6. **Per-Head, Per-Step Certified Error Bounds**
   - 扩展 SpectrumKV 的 certification 框架到 head-aware 场景
   - 每个 head 独立的误差保证，local head 可用更宽松的容忍度

### 5.3 实证贡献

7. **乘性收益的实验验证**
   - 在 3+ 模型规模、6+ 数据集上验证 25–30× 压缩率的可达性
   - 证明组合方案在质量-效率 Pareto 前沿上严格优于任一单方法

8. **Head-Quantization Interaction 分析**
   - 系统分析 global vs local head 对量化误差的敏感度差异
   - 揭示 local head 可以承受更激进量化（精度容忍度更高）的结构性原因

---

## 6. 实验设计建议

### 6.1 实验矩阵

**模型**:
| 模型 | 参数量 | $H_{kv}$ | GQA | 上下文长度 |
|------|--------|---------|-----|-----------|
| Mistral-7B-Instruct | 7B | 8 | GQA-4 | 32K |
| Qwen3-32B | 32B | 8 | GQA-4 | 128K |
| Llama-3.3-70B | 70B | 8 | GQA-8 | 128K |

**数据集**:
| 类别 | 数据集 | 目的 |
|------|--------|------|
| 长上下文 QA | TriviaQA, MultiFieldQA, HotpotQA | 评估长上下文理解 |
| 检索 | Needle-in-a-Haystack (NIAH) | 评估精确检索能力 |
| 结构化推理 | RULER (7 subtasks) | 评估多跳推理 |
| 语言建模 | PG-19 | 评估 perplexity |
| 推理 | GSM8K, MATH-500 | 评估 CoT 推理 |

**上下文长度**: 8K, 16K, 32K, 64K, 128K

### 6.2 Baselines

| 方法 | 维度 | 描述 |
|------|------|------|
| Dense FP16 | — | 全精度无压缩基线 |
| RedKnot | Head | 仅 head 维度压缩 |
| SpectrumKV (INT8K/INT4V) | Precision | 仅精度维度压缩 |
| SpectrumKV (adaptive) | Precision | 自适应精度选择 |
| KIVI | Precision | 非对称 KV 量化 |
| DuoAttention | Head | Head 分类 + streaming |
| **HAMPS (Ours)** | **Head + Precision** | **组合方案** |

### 6.3 核心实验

**实验1: 质量保持 vs 压缩率**

目标: 验证组合方案在质量-压缩 Pareto 前沿上的优势

- 横轴: 平均 KV cache 压缩率 (1× – 40×)
- 纵轴: 任务准确率 / Perplexity
- 每个方法绘制 Pareto 曲线
- **假设**: HAMPS 在所有压缩率下严格 Pareto-dominate 单一方法

**实验2: 乘性压缩率验证**

目标: 验证 $\rho_{\text{combined}} \approx \rho_R \times \rho_S$

- 测量不同 head-class 比例和精度分配下的实际压缩率
- 与理论乘性预测对比
- 分析偏离乘性的原因（元数据开销、对齐填充等）

**实验3: Head 类型对量化敏感度差异**

目标: 揭示 global vs local head 对量化误差的结构性差异

- 固定 RedKnot 的 head 分类
- 对 global heads 和 local heads 分别施加不同精度级别
- 测量 per-head output error $\|O_h^{\text{quant}} - O_h^{\text{FP16}}\|$
- **假设**: Local heads 的精度容忍度显著高于 global heads

**实验4: Fallback 触发率分析**

目标: 评估协调 fallback ladder 的运行时行为

- 统计各 rung 的触发频率
- 分析 rung 4 (elastic sparsity) 的触发模式
- 测量 fallback 对延迟的影响

**实验5: 端到端系统指标**

| 指标 | 定义 | 目标 |
|------|------|------|
| TTFT | Time to First Token | > 4× vs dense |
| Decode latency | 每 token 延迟 | < 1.3× vs dense |
| Throughput | req/s under burst load | > 2× vs RedKnot alone |
| Concurrency | 并发会话数/GPU | > 15× vs dense |
| KV transfer | PD disaggregation 传输量 | > 10× vs dense |

**实验6: Ablation Studies**

| Ablation | 移除的组件 | 验证的假设 |
|----------|-----------|-----------|
| No head-aware precision | 统一精度（不分 global/local） | Head-aware precision 分配的贡献 |
| No SWS | 固定精度（不用 SWS） | SWS 驱动的精度选择贡献 |
| No elastic sparsity | 移除 Rung 4 | Elastic sparsity 的质量保护作用 |
| No fallback | 移除所有 fallback | Fallback 的必要性 |
| Global-local ratio | 改变 global head 比例 | 最优 head 分类阈值 |
| SWS update frequency | 改变 local head 的 SWS 计算频率 | SWS 降频的精度-效率 trade-off |

### 6.4 Head-Quantization Interaction 深度分析

**关键实验: Per-Head Perplexity Sensitivity**

1. 选择模型中所有 KV heads
2. 对每个 head $h$ 独立施加 K4V4 量化，其余 heads 保持 FP16
3. 测量整体 perplexity 变化 $\Delta\text{ppl}^{(h)}$
4. 对比 global vs local heads 的 $\Delta\text{ppl}^{(h)}$ 分布
5. **预期发现**: Local heads 对 INT4 量化的 perplexity 影响远小于 global heads（因为 local heads 的 attention 模式简单，量化误差对输出分布的扰动更小）

**关键实验: Per-Head NIAH Sensitivity**

1. 对 global 和 local heads 分别量化到不同精度
2. 测量 NIAH 准确率
3. 构造 2D heatmap: (head type × precision) → NIAH accuracy
4. **预期发现**: 
   - Global heads: INT8 → near-lossless, INT4 → 显著退化
   - Local heads: INT4 → near-lossless, INT2 → 可接受的退化

### 6.5 统计方法

- NIAH: McNemar's test (paired trial), 报告 p-value
- RULER: 报告 95% CI (20 slices)
- Perplexity: 报告 95% CI
- 每个实验至少 3 seeds，报告 mean ± std

---

## 7. 风险与挑战

### 7.1 技术风险

| 风险 | 严重性 | 缓解策略 |
|------|--------|---------|
| **交互误差不可控** | 高 | Cascade Bound 理论分析 + 5-rung fallback |
| **Per-head SWS 计算开销** | 中 | Local head SWS 降频 + 缓存 |
| **变精度 page 管理复杂度** | 中 | Precision Zone 设计简化 kernel |
| **Fallback 级联导致延迟毛刺** | 中 | Fallback 频率监控 + 预留预算 |
| **Head 分类不准** | 低 | Elastic sparsity 自适应修正 |

### 7.2 工程风险

| 风险 | 描述 | 缓解策略 |
|------|------|---------|
| **Kernel 实现复杂度** | 混合精度 + varlen + per-head 的融合 kernel | 分阶段实现：先统一精度 varlen，再加变精度 |
| **内存碎片** | 变精度 page 导致内存碎片 | 预分配 zone 区域，避免运行时动态分配 |
| **与 vLLM/SGLang 集成** | 现有框架假设均匀 KV layout | 逐步集成：先插件式，再框架级改动 |

### 7.3 评估风险

| 风险 | 描述 | 缓解策略 |
|------|------|---------|
| **乘性叠加不成立** | 实际压缩率低于理论乘性预测 | 详细分析偏离原因，报告实际 vs 理论 |
| **质量退化不可接受** | 极端压缩下准确率崩塌 | 提供可调的 quality-efficiency knob |
| **Fallback 频率过高** | 大量 head 触发 fallback，延迟退化 | 调优 fallback 阈值，分析触发模式 |

---

## 8. 参考文献

1. **RedKnot**: Liu, Y., Luo, Z., Jin, H., Wang, Z., He, R., Wang, B., Chen, G., & Hu, J. (2026). *RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention*. arXiv:2606.06256.

2. **Runtime-Certified Quantized Attention**: Calver, D. (2026). *Runtime-Certified Bounded-Error Quantized Attention: Per-Head, Per-Step Error Bounds for Compressed KV Caches with Dense FP16 Fallback*. arXiv:2605.20868.

3. **Quantize What Counts (spectral-kv)**: Hariri, M., Luo, A., Chen, W., Zhong, S., Zhang, T., Wang, Q., Hu, X., Han, X., & Chaudhary, V. (2025). *Quantize What Counts: Bit Allocation Insights Informed by Spectral Gaps in Keys and Values*. arXiv:2502.15075.

4. **DuoAttention**: Xiao, G. et al. (2025). *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*. 

5. **StreamingLLM**: Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M. (2024). *Efficient Streaming Language Models with Attention Sinks*. ICLR.

6. **H₂O**: Zhang, Z. et al. (2023). *H₂O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*. NeurIPS.

7. **KIVI**: Liu, Z. et al. (2024). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. ICML.

8. **KVQuant**: Hooper, C. et al. (2024). *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization*. NeurIPS.

9. **DynaKV**: arXiv:2603.04411 (2026). *One Size Does Not Fit All: Token-Wise Adaptive Compression for KV Cache*.

10. **Don't Waste Bits!**: Boroujeni, S.P.H. et al. (2026). *Adaptive KV-Cache Quantization for Lightweight On-Device LLMs*. arXiv:2604.04722.

11. **KVTuner**: Li, X. et al. (2025). *Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization*. ICML 2025.

12. **KVmix**: arXiv:2506.08018 (2025). *Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache*.

13. **HeadKV**: arXiv:2410.19258 (2024). *Not All Heads Matter: A Head-Level KV Cache Compression Method with Integrated Retrieval and Reasoning*.

14. **vLLM / PagedAttention**: Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP.

15. **SGLang**: Zheng, L. et al. (2024). *SGLang: Efficient Execution of Structured Language Model Programs*.

---

## 附录 A: 符号表

| 符号 | 含义 |
|------|------|
| $B$ | Batch size |
| $H_{kv}$ | KV head 数量 |
| $L$ | 序列长度 |
| $D$ | Head 维度 |
| $\mathcal{G}$ | Global head 集合 |
| $\mathcal{L}$ | Local head 集合 |
| $L_h$ | Head $h$ 的有效上下文长度 |
| $W$ | Local head 的滑动窗口大小 |
| $\text{SWS}(t)$ | Token $t$ 的 Semantic Weight Score |
| $\text{HSWS}(h, t)$ | Head-specific SWS |
| $b_t$ | Token $t$ 的精度级别 (bits) |
| $\tau_h$ | Head $h$ 的 attention mass 覆盖阈值 |
| $\Delta$ | 量化误差界（per-channel） |
| $\eta_b$ | Block $b$ 的 value 重建误差 |
| $\rho_R$ | RedKnot 的空间压缩率 |
| $\rho_S$ | SpectrumKV 的精度压缩率 |
| $E_{\text{key}}^{(h)}$ | Head $h$ 的 key 量化误差界 |
| $E_{\text{val}}^{(h)}$ | Head $h$ 的 value 量化误差界 |
| $K^*$ | Adaptive precision selector 选择的 top-block 数量 |

## 附录 B: 收益计算详细推导

### B.1 组合存储计算

以 Llama-3.3-70B 为例（64层, $H_{kv}=8$, $D=128$）:

**Dense FP16 基线**:
$$S_{\text{dense}} = 64 \times 8 \times 128K \times 128 \times 2 \times 2 = 32 \text{ GB}$$

**组合方案**:
- Global heads (12.5% = 1 head per layer):
  - High-SWS zone (top ~30%): INT8 K + INT4 V = $(128+64) \times 0.3 \times 128K = 9.2$ MB/layer
  - Low-SWS zone (tail ~70%): INT4 K + INT4 V = $(64+64) \times 0.7 \times 128K = 11.5$ MB/layer
  - Per-layer global: ~20.7 MB

- Local heads (87.5% = 7 heads per layer):
  - All INT4 K+V: $(64+64) \times 320 \times 7 = 0.57$ MB/layer

- Total per layer: ~21.3 MB
- Total: $21.3 \times 64 = 1.36$ GB

**压缩率**: $32 / 1.36 \approx 23.5\times$

### B.2 带宽计算

Decode 阶段每 step 读取量:

- Global head: ~21 MB (全上下文，混合精度)
- 7 Local heads: $128 \times 320 \times 7 / 2 = 0.29$ MB (INT4)
- Total: ~21.3 MB per layer; ~21 MB for all layers (考虑 only 1 layer active at a time)

**vs Dense**: $512 / 21.3 \approx 24\times$ bandwidth reduction
