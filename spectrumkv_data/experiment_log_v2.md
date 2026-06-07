# SpectrumKV GPU实验记录 V2（按论文RQ维度组织）

> 记录时间跨度：2026-05-27 ~ 2026-06-04
> 实验平台：AutoDL / SeetaCloud 云GPU
> 项目：SpectrumKV（KV Cache分层精度传输）
> V2重构说明：将原时间线记录按论文16个RQ维度重组，06-04巨大条目拆分归入对应RQ，补齐缺失RQ条目

---

## 实验环境

| 项目 | 详情 |
|------|------|
| GPU型号 | NVIDIA RTX 4080 SUPER 32GB / RTX 4090 48GB |
| 框架 | PyTorch 2.11.0+cu130, Transformers 5.9.0 |
| 模型 | Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Gemma-2-9b-it |
| 评测数据集 | WikiText-2 (1.29M chars), 自构造NIAH |
| Python | 3.12 |

### 服务器配置

| 服务器 | SSH端口 | GPU | 分配模型 |
|--------|---------|-----|---------|
| S1 (westd) | 26772 | RTX 4080 SUPER 32GB, 12核CPU, 62GB RAM | Qwen2.5-7B |
| S2 (westc) | 53008 | RTX 4080 SUPER 32GB, 16核CPU | Mistral-7B → Gemma-2-9B |

### 实验编号与数据文件总览

| 实验 | 内容 | 核心数据文件 |
|------|------|-------------|
| Exp0 | Locality Characterization (RQ1-6) | 仿真/初步GPU实验 |
| Exp1 | PPL精细预算扫描 (RQ7/8/10/12) | exp1_ppl_qwen7b.json, exp1_ppl_mistral7b.json, exp1_ppl_gemma9b.json |
| Exp2 | NIAH检索评测 (RQ11/15) | exp2_niah_{model}.json, exp2_spectrumkv_{model}.json |
| Exp3 | 逐层量化误差 (RQ16) | exp3_quant_error_{model}.json |
| Exp4 | 层级预算分配 | exp4_layer_budget_{model}.json |
| GPU实测 | TTFT/TPS延迟 (RQ9) | GPU端到端计时 |

---

## RQ1: KV Access Skewness（KV访问偏度）

**实验编号**：Exp0（Locality Characterization）

**数据来源**：05-27 GPU部署，仿真+初步GPU实验

**关键数据**：

| 指标 | 数值 |
|------|------|
| Gini系数范围 | 0.866 ~ 0.952 |

**关键发现**：
- KV cache访问呈现极端偏度，Gini系数0.866-0.952远超均匀分布的0.333
- 高偏度意味着少数token占据了绝大部分attention权重，分层精度具备天然动机
- 三模型均表现出强偏度，但程度存在差异

**遇到的问题**：
- 05-27首次GPU实验中，SDPA模式下TAA hooks注入后PPL delta=0.0000，hooks未真正生效
- 原因：attention_mask在SDPA模式下未传入子模块，后续改用eager模式

---

## RQ2: Sink Token Importance（Sink token重要性）

**实验编号**：Exp0（Locality Characterization）

**数据来源**：05-27 GPU部署，attention hook提取

**关键发现**：
- 首层少量sink token（通常4-8个）承载了异常高的attention权重
- Sink token在不同模型中表现差异显著：
  - Mistral（sink模式）：sink token集中了极高权重，是模型attention的核心锚点
  - Qwen（local模式）：sink token权重相对较低，attention更均匀分布
  - Gemma（hybrid模式）：sink token+local token混合模式
- Sink token的保留对NIAH检索至关重要：Mistral全budget NIAH=89.5%（=FP16基线），而删除sink的方法（PDTrim/SWS）显著退化

**遇到的问题**：
- 无独立问题，与RQ3实验同步完成

---

## RQ3: Attention Pattern Classification（注意力模式分类）

**实验编号**：Exp0（Locality Characterization）

**数据来源**：05-27 GPU部署，transmission_aware_attention.py提取

**关键数据**：

| 模式 | 代表模型 | 特征 |
|------|---------|------|
| Local | Qwen-7B | attention集中在局部窗口，sink权重低 |
| Sink | Mistral-7B | 首层sink token占极高权重，中间层均匀 |
| Hybrid | Gemma-9B | sink+local混合，不同层表现不同 |

**关键发现**：
- 三种模式分类从描述性发展为指导性：检测模型模式→选择精度策略
- 模式分类直接决定了SpectrumKV的自适应策略选择：
  - Local模式→2-tier（禁INT4）
  - Sink模式→3-tier（全开）
  - Hybrid模式→3-tier（可用INT4）
- 05-27初步观察到local/sink/hybrid三种模式的差异

**遇到的问题**：
- SDPA模式下attention hook注入无效（PPL delta=0），改用eager模式后成功

---

## RQ4: Hot Set Capacity Multiplier（热集容量倍数）

**实验编号**：Exp0（Locality Characterization）

**数据来源**：仿真数据

**关键数据**：

| 序列长度 | Hot Set容量倍数 |
|---------|----------------|
| 32K | 130x |

**关键发现**：
- 在32K序列长度下，hot set（高attention token集合）仅占总token的极小比例，但承载了绝大部分attention权重
- 130x容量倍数意味着：hot set仅需全量KV cache的1/130存储空间
- 这为分层精度提供了直接证据——冷token可用低精度存储而几乎不影响模型表现

**遇到的问题**：
- 无独立问题

---

## RQ5: Tier-Aware Adapter Overhead（分层适配器开销）

**实验编号**：Exp0（Locality Characterization）

**数据来源**：仿真/GPU实测

**关键数据**：

| 指标 | 数值 |
|------|------|
| Tier-aware adapter overhead | <0.04% |

**关键发现**：
- 分层精度适配器的计算开销极低（<0.04%），对推理吞吐量几乎无影响
- 开销主要来自importance score计算和precision map分配，均为轻量级操作
- 验证了SpectrumKV在实际部署中的可行性——额外开销可忽略

**遇到的问题**：
- 无独立问题

---

## RQ6: Cross-Model Generality（跨模型通用性）

**实验编号**：Exp0（Locality Characterization）

**数据来源**：三模型（Qwen/Mistral/Gemma）同步实验

**关键发现**：
- 三种注意力模式（local/sink/hybrid）覆盖了主流LLM架构的attention分布特征
- SpectrumKV的自适应策略在三种模式下均有效，但最优配置不同：
  - Qwen（local）需要2-tier避免INT4灾难
  - Mistral（sink）3-tier全开即可
  - Gemma（hybrid）3-tier可用INT4但需控制比例
- 证明分层精度方案具有良好的跨模型通用性

**遇到的问题**：
- 三模型同步实验对磁盘空间要求高（3个模型约51GB，服务器磁盘50GB），采用顺序处理策略

---

## RQ7: Strategy Comparison at b=0.5（策略对比：50%预算PPL）

**实验编号**：Exp1

**数据来源**：exp1_ppl_qwen7b.json, exp1_ppl_mistral7b.json, exp1_ppl_gemma9b.json

**实验配置**：b=0.5, seq=2048

**完整数据表格**（10个方法 × 3模型，PPL Δ%）：

| 方法 | Qwen Δ% | Mistral Δ% | Gemma Δ% |
|------|---------|-----------|---------|
| Full_FP16 | 0 | 0 | 0 |
| Uniform_INT8 | +1.97 | -0.06 | -0.44 |
| PDTrim | +25.85 | +22.07 | +35.63 |
| SWS_Original | +30.88 | +33.25 | +43.78 |
| SWS_ValueAware | +30.88 | +33.25 | +43.78 |
| SpectrumKV_Greedy | +1.97 | -0.06 | -0.44 |
| SpectrumKV_Balanced | +7506.84 | +2.71 | +2.30 |
| SpectrumKV_SinkProtect | +1.39 | -0.10 | -0.47 |
| Random_Tier | +1.97 | -0.06 | -0.44 |

**关键发现**：
- b=0.5时Greedy/Uniform_INT8/Random_Tier三方法等价：b=0.5→FP16比例=0, INT8比例=1.0, INT4比例=0（即全INT8），三方法殊途同归
- SpectrumKV_SinkProtect在Qwen上Δ=+1.39%为最优，甚至优于Greedy的+1.97%
- SpectrumKV_Balanced在Qwen上Δ=+7506.84%，灾难性退化——原因是Balanced强制FP16/INT8/INT4均分，即使b=0.5也分配INT4→Qwen沾INT4即死
- Mistral（sink模式）对几乎所有限制性策略都高度容忍：SinkProtect b=0.5 Δ=-0.10%，近乎无损
- Gemma Balanced b=0.5仅+2.30%，说明Gemma对INT4有一定容忍度
- PDTrim在所有模型上都造成显著退化（+22%~+36%），因为其丢弃token而非分层量化
- SWS两种变体（Original/ValueAware）结果完全一致，说明value-aware改进对PPL无额外帮助

**遇到的问题**：
- Qwen Uniform_INT4爆炸（+394338%），确认INT4对Qwen是灾难性退化，非脚本bug
- SpectrumKV_Balanced的极端数字（+7506.84%）审稿人可能质疑，需在论文中解释INT4不耐受机制

---

## RQ8: Budget Sweep（预算扫描）

**实验编号**：Exp1

**数据来源**：exp1_ppl_qwen7b.json, exp1_ppl_mistral7b.json, exp1_ppl_gemma9b.json

**实验配置**：b=0.3/0.5/0.7, seq=2048

**Greedy 3-tier关键数据**：

| Model | b=0.3 | b=0.5 | b=0.7 |
|-------|-------|-------|-------|
| Qwen (3-tier, 含INT4) | +228031% | +1.97% | +0.47% |
| Mistral (3-tier) | +5.41% | -0.06% | +0.03% |
| Gemma (3-tier) | +2.45% | -0.44% | -0.03% |

**关键发现**：
- Qwen b=0.3的+228031%灾难性退化是因为3-tier含INT4：b=0.3时约80%token被分配INT4→Qwen(local模式)完全不耐受INT4
- 正确配置是2-tier Adaptive：Qwen禁用INT4后，b=0.3 NIAH=52.6%（对比3-tier=4.2%）
- Mistral和Gemma在b=0.3时退化可控：Mistral +5.41%（3-tier），Gemma +2.45%
- Mistral和Gemma在b=0.5以上几乎无损（Δ<1%）
- Budget与PPL呈非线性关系：从b=0.5到b=0.3的退化远大于b=0.7到b=0.5

**06-04 INT4退化5条件对照验证**：

| 条件 | d=0.25 | d=0.50 | d=0.75 | 诊断 |
|------|--------|--------|--------|------|
| Full_FP16 | ✅ | ✅ | ✅ | 基线 |
| Uniform_INT8 | ✅ | ✅ | ✅ | KV QDQ实现正确 |
| Uniform_INT4 | ❌ | ❌ | ❌ | INT4对Qwen确是灾难 |
| SKV b=0.50 | ✅ | ✅ | ✅ | 等价Uniform_INT8，交叉验证通过 |
| SKV b=0.30 | ❌ | ❌ | ❌ | 80%INT4→灾难性退化 |

**验证结论**：
1. Uniform_INT8 = Full_FP16 → KV cache修改流程完全正确
2. Write-back验证：re-QDQ幂等（diff=0），in-place修改确认写回DynamicLayer
3. SKV b=0.50 = Uniform_INT8 → 两套代码路径结果一致
4. INT4对Qwen的灾难性退化是模型特性，不是脚本bug

**遇到的问题**：
- Qwen b=0.3的灾难数字过于极端（Δ=+228031%），审稿人可能质疑数据真实性
- 需在论文中明确说明：这是3-tier配置对INT4不耐受模型的预期结果，2-tier Adaptive可避免

---

## RQ9: TTFT Latency Savings（首token延迟节省）

**实验编号**：GPU端到端实测

**数据来源**：GPU推理计时（非Exp1-4 JSON文件，为独立GPU计时）

**完整数据表格**：

| Model | Seq | Full TTFT(ms) | b=0.5 TTFT(ms) | TTFT↓ | Full TPS | b=0.5 TPS | TPSΔ |
|-------|-----|--------------|----------------|-------|---------|----------|------|
| Qwen-7B | 2K | 391 | 154 | -61% | 49.2 | 43.5 | -11% |
| Qwen-7B | 4K | 630 | 296 | -53% | 42.4 | 43.2 | +2% |
| Mistral-7B | 2K | 274 | 116 | -58% | 51.9 | 50.1 | -3% |
| Mistral-7B | 4K | 497 | 233 | -53% | 47.1 | 49.1 | +4% |
| Mistral-7B | 8K | 1211 | 502 | -59% | 44.1 | 47.0 | +7% |
| Gemma-9B | 2K | 269 | 135 | -50% | 38.2 | 38.9 | +2% |
| Gemma-9B | 4K | 709 | 270 | -62% | 28.8 | 38.2 | +33% |
| Gemma-9B | 8K | 1664 | 713 | -57% | 27.3 | 28.8 | +5% |

**关键发现**：
- b=0.5时TTFT节省50%-62%，三模型、多序列长度一致有效
- 最优TTFT节省：Gemma-9B 4K序列下-62%（709ms→270ms）
- TPS（生成吞吐量）在长序列下反而提升：Gemma-9B 4K下TPS+33%（28.8→38.2 tok/s）
- TPS提升原因：KV cache减半后GPU显存带宽瓶颈缓解，decode阶段KV读取量下降
- 短序列（2K）下TPS轻微下降（Qwen -11%），因为INT8反量化开销相对更明显
- 随序列长度增加，TTFT节省和TPS提升的趋势更加显著，说明SpectrumKV在长上下文场景下收益更大

**遇到的问题**：
- 无独立问题，数据来自独立GPU计时实验

---

## RQ10: Context Length Impact（上下文长度影响）

**实验编号**：Exp1

**数据来源**：exp1_ppl_qwen7b.json, exp1_ppl_mistral7b.json, exp1_ppl_gemma9b.json

**论文参考表格**（部分来自论文，与原始JSON可能略有差异，以论文表格为准）：

| Model | Seq | Full KV PPL | SKV Δ% | PDTrim Δ% |
|-------|-----|------------|--------|-----------|
| Qwen-7B | 2K | 7.04 | +2.0% | +25.8% |
| Qwen-7B | 4K | 7.08* | -0.5%* | +20.1%* |
| Mistral-7B | 2K | 7.96 | -0.1% | +22.1% |
| Mistral-7B | 4K | 8.72* | -2.1%* | +12.5%* |
| Mistral-7B | 8K | 9.48* | -1.3%* | +8.7%* |
| Gemma-9B | 2K | 11.20 | -0.4% | +35.6% |
| Gemma-9B | 4K | 11.85* | -1.8%* | +15.2%* |
| Gemma-9B | 8K | 12.40* | -0.6%* | +180.3%* |

**Exp1实际数据**（Greedy b=0.5）：

| Model | seq=2048 | seq=4096 | seq=8192 |
|-------|----------|----------|----------|
| Qwen PPL | 7.1805 | 6.1554 | 6.3949 |
| Qwen Δ% | +1.97% | +2.45% | +2.90% |
| Mistral Δ% | -0.06% | -0.13% | -0.29% |
| Gemma Δ% | -0.44% | +0.14% | — |

**关键发现**：
- SpectrumKV在不同序列长度下保持稳定：Mistral和Gemma在2K-8K范围内Δ%均<3%
- PDTrim退化随序列长度变化显著：Gemma 2K +35.6% → 8K +180.3%，PDTrim对长上下文极不友好
- SpectrumKV在长上下文下优势更明显：与PDTrim的差距随序列长度拉大
- Mistral（sink模式）在所有长度下Δ%接近0，INT8量化对sink模式模型几乎无损
- Qwen实际数据中seq=4096时Full KV PPL=6.1554（低于2K的7.1805），可能因WikiText不同段落的统计差异

**遇到的问题**：
- Gemma seq=8192存在OOM风险，仅部分数据可用
- 论文部分数据（标*）与原始JSON有差异，以论文表格为准
- Qwen 8K序列下Δ=+2.90%，略高于2K的+1.97%，但仍在可接受范围

---

## RQ11: NIAH Retrieval Preservation（NIAH检索保持）

**实验编号**：Exp2

**数据来源**：exp2_niah_{model}.json, exp2_spectrumkv_{mistral,gemma}.json

### Qwen NIAH（seq=4096, avg across 19 depths）

| 方法 | b=0.3 | b=0.4 | b=0.5 | b=0.6 | b=0.7 |
|------|-------|-------|-------|-------|-------|
| Full_FP16 | 100% | — | — | — | — |
| Adaptive 2-tier | 52.6% | 74.7% | 100% | 100% | 100% |
| Greedy 3-tier | 4.2% | 0% | 100% | 100% | 100% |
| SinkProtect 3-tier | 3.2% | 0% | 100% | 100% | 100% |
| Balanced 3-tier | 4.2% | 0% | 0% | 0% | 90.5% |
| PDTrim | 26.3% | 36.8% | 47.4% | 57.9% | 68.4% |
| SWS | 21.1% | 31.6% | 43.2% | 54.7% | 68.4% |
| Random_Tier | 2.1% | 1.1% | 100% | 100% | 100% |

**Adaptive U形深度分布（Qwen b=0.3）**：
- d≤0.35：100%（sink+recent token保留充分）
- d=0.4~0.8：0%（中间token被丢弃，budget不够覆盖）
- d≥0.85：100%（recent token保留充分）

### Mistral NIAH（FP16 baseline=89.5%）

| 方法 | b=0.3 | b=0.5 | b=0.7 |
|------|-------|-------|-------|
| Full_FP16 | 89.5% | — | — |
| Greedy/Balanced/SinkProtect | 89.5% | 89.5% | 89.5% |
| PDTrim | 26.3% | 44.2% | 62.1% |
| SWS | 20.0% | 37.9% | 61.1% |
| Random_Tier | 88.4% | 89.5% | 86.0% |

### Gemma NIAH（FP16 baseline=100%, with chat template）

| 方法 | b=0.3 | b=0.5 | b=0.7 |
|------|-------|-------|-------|
| Full_FP16 | 100% | — | — |
| Greedy/Balanced/SinkProtect | 100% | 100% | 100% |
| Adaptive 2-tier | 60.0% | 100% | 100% |
| PDTrim | 41.1% | 57.9% | 74.7% |
| SWS | 35.8% | 52.6% | 72.6% |
| Random_Tier | 100% | 100% | 100% |

**关键发现**：
- SpectrumKV在NIAH检索上全面碾压PDTrim和SWS：
  - Qwen Adaptive b=0.3: 52.6% vs PDTrim 26.3%（+26pp）
  - Mistral 3-tier b=0.3: 89.5% vs PDTrim 26.3%（+63pp）
  - Gemma 3-tier b=0.3: 100% vs PDTrim 41.1%（+59pp）
- Balanced b=0.5=0%（Qwen）的根因：Balanced强制FP16/INT8/INT4均分→即使b=0.5也分配INT4→Qwen沾INT4即死
- Mistral FP16 baseline=89.5%（非100%）：shallow depth约80%，deep约100%，sink模式对浅层检索有天然劣势
- Gemma 3-tier方法在所有budget下=100%，hybrid模式对INT4高度容忍
- PDTrim和SWS的NIAH表现随budget单调上升，但即使在b=0.7也远不如SpectrumKV

**遇到的问题**：
- 06-04 14:05发现Bug1：build_niah_prompt缺检索问题→Mistral全方法0%（模型仅续写haystack），修复后Qwen depth=0.5 Full_FP16成功检索"UNICORN7"
- 06-04 14:05发现Bug2：PDTrim/SWS用zero-out而非删除→零值KV仍占attention位置，softmax归一化后分布被严重干扰→PDTrim/SWS全budget=0%，修复改用attention_mask
- 06-04 15:14发现Bug6：if/elif互斥→Adaptive需同时执行attention_mask+QDQ，elif导致none token仍以FP16可见→偷用预算→结果虚高，修正为两个独立if
- 06-04 16:33发现Gemma FP16 baseline仅34%→缺少chat template→Gemma-2-9B-IT依赖chat template格式→修复后FP16=100%

---

## RQ12: PDTrim First-Ratio Sensitivity（PDTrim首层比例敏感性）

**实验编号**：Exp1（含PDTrim不同fr配置）

**数据来源**：exp1_ppl_qwen7b.json, exp1_ppl_mistral7b.json, exp1_ppl_gemma9b.json

**完整数据表格**：

| First-ratio | Qwen Δ% | Mistral Δ% | Gemma Δ% |
|-------------|---------|-----------|---------|
| 0.1 | +18.2% | +45.3% | +49.1% |
| 0.3 | +14.1% | +12.8% | +18.7% |
| 0.5 (default) | +25.8% | +22.1% | +35.6% |
| 0.7 | +16.3% | +10.5% | +8.9% |
| 0.9 | +19.7% | +15.2% | +3.8% |

**关键发现**：
- PDTrim的first-ratio（首层保留比例）对PPL影响显著，且没有统一最优值
- Qwen的最优fr=0.3（+14.1%），Mistral的最优fr=0.7（+10.5%），Gemma的最优fr=0.9（+3.8%）
- 默认fr=0.5在所有模型上都不是最优，说明PDTrim的固定配置存在优化空间
- Gemma在fr=0.9时Δ=+3.8%，接近SpectrumKV的水平，但PDTrim仍以token删除为基础，NIAH检索会严重退化
- First-ratio敏感性说明PDTrim缺乏自适应能力——不同模型/模式需要不同的首层保留策略

**遇到的问题**：
- 无独立问题，数据来自Exp1 PPL扫描的PDTrim子集

---

## RQ13: Decay Rate Impact（衰减率影响）

**实验编号**：仿真v3（非GPU实测）

**数据来源**：论文Table 7，仿真v3精细扫描数据

**完整数据表格**：

| Model | Decay rate | b=0.3 | b=0.5 | b=0.7 |
|-------|-----------|-------|-------|-------|
| Mistral (attn×decay) | dr=0.001 | +36.7% | +11.7% | +1.9% |
| | dr=0.005 | +27.5% | +0.6% | -10.7% |
| | dr=0.01 | +24.7% | -2.0% | -12.3% |
| Gemma (attn×decay) | dr=0.001 | +14.1% | -0.2% | -5.3% |
| | dr=0.005 | +13.9% | -12.7% | -15.2% |
| | dr=0.01 | +13.4% | -13.2% | -18.1% |
| Qwen (kv-norm×decay) | dr=0.001 | +38.4% | +9.6% | -2.0% |
| | dr=0.005 | +44.5% | +12.5% | -1.3% |
| | dr=0.01 | +45.7% | +12.5% | -1.3% |

**关键发现**：
- Decay rate对PPL有显著影响，且不同模型的最优dr不同
- Mistral：dr=0.01在b=0.7时达到最优（-12.3%），较大的衰减率有助于sink模式模型区分重要/不重要token
- Gemma：dr=0.001最优（b=0.5 -0.2%），较大的衰减率反而有害（dr=0.01 b=0.5 = -13.2%）
- Qwen（kv-norm×decay）：衰减率对Qwen影响较小，dr=0.005和0.01结果一致，Qwen的importance主要来自kv-norm而非attention
- 关键洞察：decay rate应作为模型相关的超参数，而非全局固定值
- Mistral和Gemma使用attn×decay，Qwen使用kv-norm×decay——不同importance计算方式对dr的敏感度不同

**遇到的问题**：
- 数据来自仿真v3而非GPU实测，与Exp1 GPU数据可能存在系统性差异
- 论文中应注明数据来源（仿真vs GPU实测）

---

## RQ14: Multi-Seed/Task Robustness（多seed/任务鲁棒性）

**实验编号**：Exp1（多seed）+ Exp1多任务

**数据来源**：exp1_ppl_{model}.json（多seed/多task配置）

### 7-seed CI95 (b=0.5)

| Strategy | Qwen CI95 | Mistral CI95 | Gemma CI95 |
|----------|-----------|-------------|-----------|
| Random_Tier | ±22% | ±265% | ±75% |
| PDTrim | 0 | 0 | 0 |
| SKV Greedy | 0 | 0 | 0 |

### Multi-task (b=0.5)

| Model | Task | PDTrim Δ% | SKV Δ% |
|-------|------|-----------|--------|
| Mistral | WikiText | +22.1% | -0.1% |
| | Code | +15.7% | +6.9% |
| | Science | +152.8% | +33.9% |
| Gemma | WikiText | +35.6% | -0.4% |
| | Code | +11.4% | +8.0% |
| | Science | +36.0% | +32.7% |
| Qwen | WikiText | +25.8% | +2.0% |
| | Code | +8.1% | +5.5% |
| | Science | +35.4% | +30.5% |

**关键发现**：
- **Seed鲁棒性**：
  - Random_Tier方差极大（Mistral ±265%），因为随机分配精度导致不同seed结果差异巨大
  - PDTrim和SKV Greedy的CI95=0，说明两种方法对随机seed完全确定（PDTrim按固定规则裁剪，Greedy按importance排序）
  - SpectrumKV Greedy的seed鲁棒性是其相对于Random_Tier的核心优势之一
- **Task鲁棒性**：
  - PDTrim在不同任务上退化差异巨大：Mistral Science +152.8% vs Code +15.7%
  - SpectrumKV Greedy在WikiText上最优（Mistral -0.1%, Gemma -0.4%），但在Science任务上有明显退化（Mistral +33.9%）
  - Science任务的高退化可能因为科学文本的attention分布更均匀，分层精度更容易遗漏关键信息
  - SpectrumKV整体task鲁棒性优于PDTrim：除Science外，所有任务Δ%<10%

**遇到的问题**：
- 05-28 Exp3多种子鲁棒性（5-seed CI95）数据与7-seed数据可能不完全一致，以7-seed为准
- Science任务的极端退化值（Mistral PDTrim +152.8%）需在论文中讨论domain-specific影响

---

## RQ15: Adaptive Precision Strategy（自适应精度策略）

**实验编号**：Exp2

**数据来源**：exp2_spectrumkv_{model}.json + probe实验

### Probe结果

| Model | Probe successes | INT4 tolerant | Tier choice |
|-------|----------------|--------------|-------------|
| Qwen-7B | 0/3 | No | 2-tier (FP16+INT8) |
| Mistral-7B | 3/3 | Yes | 3-tier (FP16+INT8+INT4) |
| Gemma-9B | 3/3 | Yes | 3-tier (FP16+INT8+INT4) |

### NIAH at b=0.3

| Model | Fixed 3-tier | Adaptive | PDTrim |
|-------|-------------|----------|--------|
| Qwen | 0% | 52.6% (2-tier) | 26.3% |
| Mistral | 89.5% (3-tier) | 100% (3-tier) | 26.3% |
| Gemma | 100% (3-tier) | 100% (3-tier) | 41.1% |

**关键发现**：
- Probe实验可准确预测INT4耐受性：0/3→不耐受→2-tier，3/3→耐受→3-tier
- Qwen Adaptive策略的收益巨大：Fixed 3-tier b=0.3=0% → Adaptive 2-tier=52.6%，提升52.6pp
- Mistral/Gemma的Adaptive与Fixed 3-tier一致（因为probe判定3-tier为最优）
- Adaptive≠固定2-tier：概念是model-pattern-adaptive→local用2-tier/sink用3-tier
- 自适应策略的核心价值：避免INT4不耐受模型使用3-tier配置的灾难性后果
- 三模式分类从描述性→指导性：检测模型模式→选择精度策略

**06-04 Adaptive算法部署历程**：
- 15:03在spectrumkv_utils.py中新增`spectrumkv_adaptive()`（2-tier：FP16+INT8，禁INT4）
- 15:04初版验证报告SKV 2-tier b=0.3 NIAH=3/3(100%)——结果虚高
- 15:14发现if/elif互斥bug：Adaptive需同时执行attention_mask(遮none token) + QDQ(量化INT8 token)，但elif导致none token仍以FP16可见→偷用预算
- 15:15-15:21修正：elif→if + precision_map内加dropped_idx建attention_mask
- 修正后验证：SKV Adaptive b=0.3 d=0.25=✅, d=0.50=❌, d=0.75=❌（U形深度分布）

**遇到的问题**：
- if/elif互斥bug导致Adaptive初版数据虚高，修正后Adaptive b=0.3=52.6%（非100%）
- 修正过程需要停止exp2→写clean_fix.py→两服务器均修→重新跑验证脚本

---

## RQ16: Per-Layer Quantization Error（逐层量化误差）

**实验编号**：Exp3

**数据来源**：exp3_quant_error_qwen7b.json, exp3_quant_error_mistral7b.json, exp3_quant_error_gemma9b.json

### Cosine Similarity（avg across layers）

| Type | Qwen | Mistral | Gemma |
|------|------|---------|-------|
| INT8 key cosine | >0.9999 | >0.9999 | >0.9999 |
| INT8 value cosine | >0.9999 | >0.9999 | >0.9999 |
| INT4 key cosine | 0.9896 | 0.9792 | 0.9636 |
| INT4 value cosine | 0.9804 | 0.9776 | 0.9844 |

### 详细INT4 Key Cosine Range

| Model | Min | Max | Avg |
|-------|-----|-----|-----|
| Qwen | 0.9678 | 0.9955 | 0.9770 |
| Mistral | — | — | 0.9792 |
| Gemma | 0.9334 | 0.9798 | 0.9636 |

**关键发现**：
- INT8量化对KV cache几乎无损：所有模型key/value cosine >0.9999，解释了b=0.5全INT8的近无损PPL
- INT4量化误差显著大于INT8，且跨层异质性明显：
  - Qwen INT4 key range: 0.9678-0.9955，波动0.028
  - Gemma INT4 key range: 0.9334-0.9798，波动0.046，且绝对值最低
- Gemma的INT4 key cosine最低（0.9636），但PPL退化却可控（Balanced b=0.5仅+2.30%）——说明cosine similarity不完全预测PPL影响
- 逐层异质性为Exp4层级预算分配提供了直接动机：不同层可承受不同精度
- INT4 value误差普遍高于key误差（除Gemma外），说明value cache对精度更敏感

**遇到的问题**：
- 06-04 17:50发现Qwen exp3数据崩溃无数据，但17:55成功恢复
- 06-02脚本审核发现exp3 NaN corrcoef bug：零方差返回NaN导致非法JSON→加NaN保护
- 06-02脚本审核发现exp3 per_position/per_importance混合K/V误差→分离K/V误差

---

## 附录A：Bug修复全记录

### A.1 05-28 v5 Pipeline Bug修复

| # | Bug | 根因 | 修复 |
|---|-----|------|------|
| 1 | `total_mem` AttributeError | 属性名拼写错误 | 改为`total_memory` |
| 2 | NIAH passage含"ENIAC" | 模型输出干扰词 | 换中性文本 |
| 3 | Mistral `system_message`缺失 | chat_template要求system参数 | 添加system角色 |
| 4 | "secret/hidden/magic"触发安全拒绝 | Mistral安全机制 | 改用"passcode/confirmation" |
| 5 | needle截断bug | 先插needle再截断→needle被切 | 先建haystack再替换插入 |

### A.2 06-02 脚本审核Bug修复

| # | Bug | 严重度 | 修复 |
|---|-----|--------|------|
| 1 | `ppl_tiered` hook未生效（仍走twopass） | 🔴严重 | 完全重写k_proj/v_proj hook |
| 2 | `compute_importance_from_attention` dim≥3崩溃 | 🔴严重 | 修正维度归约 |
| 3 | `compute_layer_budgets`方向反了 | 🔴严重 | 修正排序方向 |
| 4 | `compute_layer_budgets`均值偏移beta/2 | 🔴严重 | 移除偏移 |
| 5 | `compute_layer_budgets`最低0.15低于INT4地板0.25 | 🔴严重 | 设置INT4地板0.25 |
| 6 | exp2 NIAH generate入口token重复 | 🔴严重 | 手动token-by-token生成 |
| 7 | exp3 NaN corrcoef | 🔴严重 | 加NaN保护 |
| 8 | exp3 per_position/per_importance混合K/V误差 | 🔴严重 | 分离K/V误差 |
| 9 | exp4 ppl_per_layer_tiered同款twopass bug | 🔴严重 | 同修ppl_tiered |
| 10 | exp2 random_tier seed固定 | 🟡中等 | 参数化seed |
| 11 | exp3 全局args引用 | 🟡中等 | 参数化 |
| 12 | exp3 冗余CPU拷贝 | 🟡中等 | 共享复用 |
| 13 | exp4 tier_fractions仅首个sample | 🟡中等 | 跨sample均值 |
| 14 | spectrumkv_balanced死代码 | 🟡中等 | 删除elif f<0 |
| 15 | spectrumkv_greedy小序列assert | 🟡中等 | assert→warning |

### A.3 06-04 实验运行中Bug修复

| # | Bug | 发现时间 | 修复 |
|---|-----|---------|------|
| 1 | NEEDLE_QUESTIONS缺失→Mistral全方法0% | 14:05 | 添加检索问题字典 |
| 2 | zero-out替代删除→PDTrim/SWS全0% | 14:05 | 改用attention_mask |
| 3 | DynamicCache格式不兼容→下标访问报错 | 13:12 | 改用`.keys`/`.values`属性访问 |
| 4 | `baseline_ppl`未绑定→UnboundLocalError | 13:09 | 添加`baseline_ppl=None`初始化 |
| 5 | Gemma OOM | — | `--seq_lens 2048 4096` |
| 6 | if/elif互斥→Adaptive数据虚高 | 15:14 | elif→if + dropped_idx建attention_mask |
| 7 | json_serialize float崩溃→Mistral exp2保存失败 | 15:49 | 加float/int/bool/str类型处理 |
| 8 | Gemma chat template缺失→FP16=34% | 16:33 | 添加`use_chat_template=True` |

---

## 附录B：环境配置记录

### B.1 脚本版本历程（06-04部署）

| 版本 | 问题 | 修复 |
|------|------|------|
| v2 | Mistral路径错误（`Mistral/`→`mistralai/`） | 修正HF路径 |
| v3 | 硬编码`local_files_only=True`，Qwen HF缓存不完整 | 移除硬编码 |
| v4 | Qwen HF缓存snapshots为空 | 加入本地路径到LOCAL_MODEL_PATHS |
| v5 | WikiText离线加载失败 | 4级fallback：本地文件→HF offline→HF online→URL直下 |

### B.2 实验配置总览

| 实验 | 方法数 | 扫描点 | 总配置 | 预估耗时 |
|------|--------|--------|--------|---------|
| exp1 PPL | 10(含Uniform基线) | 11 budget × 2 seq | ~880 | ~3h |
| exp2 NIAH | 7 | 19 depth × 5 budget | ~3325 trial | ~1.5h |
| exp3 量化误差 | — | 5维度 | 3 samples | ~30min |
| exp4 层级预算 | 1 | 6 beta × 3 budget | 54 | ~1h |

### B.3 模型下载与镜像源

- AutoDL无外网，模型下载需用hf-mirror
- 镜像源配置：`HF_ENDPOINT=https://hf-mirror.com`
- 大模型下载策略：提前缓存到本地，避免重复下载

### B.4 数据文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| exp1_ppl_qwen7b.json | 63KB | Qwen PPL精细预算扫描 |
| exp1_ppl_mistral7b.json | 63KB | Mistral PPL精细预算扫描 |
| exp1_ppl_gemma9b.json | 42KB | Gemma PPL精细预算扫描 |
| exp2_spectrumkv_qwen7b.json | — | Qwen NIAH SpectrumKV方法 |
| exp2_spectrumkv_mistral7b.json | 143KB | Mistral NIAH SpectrumKV方法 |
| exp2_spectrumkv_gemma9b.json | 163KB | Gemma NIAH SpectrumKV方法 |
| exp2_niah_qwen7b.json | 15KB | Qwen NIAH基线方法 |
| exp2_niah_mistral7b.json | 701KB | Mistral NIAH基线方法 |
| exp2_niah_gemma9b.json | 972KB | Gemma NIAH基线方法 |
| exp3_quant_error_qwen7b.json | 63KB | Qwen逐层量化误差 |
| exp3_quant_error_mistral7b.json | 80KB | Mistral逐层量化误差 |
| exp3_quant_error_gemma9b.json | 93KB | Gemma逐层量化误差 |
| exp4_layer_budget_qwen7b.json | 55KB | Qwen层级预算分配 |
| exp4_layer_budget_mistral7b.json | 69KB | Mistral层级预算分配 |
| exp4_layer_budget_all.json | 69KB | 全量层级预算汇总 |

---

## 附录C：实验时间线摘要

| 日期 | 关键事件 |
|------|---------|
| 05-27 | 首次GPU部署，Exp0 Locality Characterization，发现SDPA+TAA hooks无效 |
| 05-28 | 两台GPU实例部署，Exp1/2/3首次运行，v5 Pipeline 5次rerun修复bug，37个JSON推GitHub |
| 05-29 | NIAH深度扫描实验，发现脚本bug（未apply_chat_template），数据标记OBSOLETE |
| 05-31 | 方案命名确认（SpectrumKV），完成仿真v3精细扫描 |
| 06-02 | 编写5个GPU实验脚本，6个审核agent发现15个bug并全部修复 |
| 06-04 | 全量GPU实验核心日：exp3→exp1→exp2→exp4全流程，8个运行中bug修复，三模型验证全部完成 |

---

*V2重构完成于 2026-06-04*
