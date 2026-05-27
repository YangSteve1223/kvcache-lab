# kvcache-lab

**Runtime KV Memory Management for PD-Disaggregated LLM Serving**

> We observe that decode-time KV access exhibits strong locality, enabling runtime KV tiering with minimal quality degradation.

## Core Insight

传统方法把KV Cache当作"静态上下文"来压缩或裁剪，我们将其视为**运行时内存**来管理。核心发现：

- **Decode阶段存在极强的KV访问局部性**：Gini系数=0.911，9%的active set覆盖80%的attention mass
- **大部分remote KV对decode不必要**：70% KV放在remote tier仅损失2% attention
- **长上下文下locality更强**：4K上下文50% memory budget下质量无损（ΔPPL=-0.21%）

这不是attention改进或KV压缩，而是**Runtime KV Memory Management System**。

## SWS ≠ Sliding Window Attention

| Sliding Window Attention | SWS (Ours) |
|---|---|
| 改模型attention receptive field | Runtime KV placement/tiering |
| Token不可访问 | Remote KV仍可访问（demoted, not deleted） |
| Architecture change | Runtime memory policy |
| Local-only attention | Hierarchical KV memory (local + remote tier) |

**SWS是KV cache tiering**：将"热"KV保留在local tier（GPU），"冷"KV降级到remote tier（仍可通过PD传输访问）。类比OS的hot/cold pages + working set + tiered memory。

## System Overview

```
┌─────────────────────────────────────────────────────┐
│              PD-Disaggregated Serving               │
│  ┌──────────────┐         ┌──────────────────────┐  │
│  │  Prefill Node │  KV    │    Decode Node        │  │
│  │  (Full KV)    │───────▶│  ┌─────────────────┐  │  │
│  └──────────────┘  Transfer│ │  Local KV (GPU) │  │  │
│                     ▲      │ │  Hot / Working  │  │  │
│                     │      │ └────────┬────────┘  │  │
│                     │      │          │ On-demand  │  │
│                     └──────│──┌───────▼────────┐  │  │
│                            │  │ Remote KV      │  │  │
│                            │  │ Cold / Demoted │  │  │
│                            │  └────────────────┘  │  │
│                            └──────────────────────┘  │
└─────────────────────────────────────────────────────┘

               TAA (Task-Aware Attention)
               score = relevance + α × locality_bias
               → Low-overhead locality-aware guidance
               → Encourages attention to local KV
```

### Architecture Roles

| Module | Role | Description |
|---|---|---|
| **SWS** | Main Contribution | Runtime KV tiering — only maintain active decode-time KV working set locally |
| **Runtime Memory Framing** | Main Contribution | OS analogy: working set, tiered memory, page migration |
| **TAA** | Enabling Mechanism | Low-overhead (0.04%) locality-aware attention bias |
| **Predictive Eviction** | Implementation Policy | LRU/TAA-guided eviction for constrained memory |

## Key Results (Qwen2.5-7B, RTX PRO 6000 Blackwell, bfloat16)

### Locality Characterization

| Metric | Value |
|---|---|
| Gini Coefficient | **0.911** |
| Active Set (80% attention) | **9% of KV** |
| Top 20% KV coverage | **95% attention** |
| Remote KV attention (70% remote) | **≤2%** |
| Active set size | **~45 tokens** (constant across context lengths) |

### Memory-Quality Pareto (Long Context)

| Context | Memory Budget | ΔPPL | KV Saved |
|---|---|---|---|
| 2K | 50% | **-0.2%** (quality intact) | 50% |
| 4K | 50% | **-0.21%** (quality intact) | 50% |
| 4K | 30% | **-0.8%** | 70% |
| 8K | 50% | **+0.08%** (essentially zero) | 50% |

### SWS Memory Savings (32K Context)

| Window Size | KV Memory | Savings |
|---|---|---|
| Full (32K) | 1,776 MB | — |
| ws=2048 | 117 MB | **93.8%** |
| ws=64 | 3.7 MB | **99.8%** |

### TAA Overhead

| Context | Overhead | Relative |
|---|---|---|
| 2K | 55 μs | 0.006% |
| 32K | 3.2 ms | 0.04% |

### Concurrency

SWS supports **8× more concurrent requests** at 32K context length under the same GPU memory budget.

### HF Serving Benchmark (Qwen2.5-7B, bfloat16, RTX PRO 6000 Blackwell)

> vLLM在Blackwell GPU上不兼容（FlashInfer不支持SM 12.x），改用HuggingFace-based serving benchmark。

| Context | Concurrency 1 | Concurrency 4 | Concurrency 8 | Concurrency 16 |
|---------|--------------|--------------|--------------|----------------|
| 1K | 66 TPS | 192 TPS | 327 TPS | 603 TPS |
| 4K | 59 TPS | 141 TPS | 203 TPS | 243 TPS |
| 8K | 50 TPS | 84 TPS | 108 TPS | 128 TPS |
| 16K | 36 TPS | 48 TPS | 55 TPS | 62 TPS |
| 32K | 21 TPS | 24 TPS | 26 TPS | 27 TPS |

> ⚠️ 注意：32K×16 OOM

### Short-Context Behavior (seq=1024)

> 短上下文(1024)下locality弱，这是**seq-length效应**而非系统缺陷，正是我们聚焦长上下文的motivation。

| Workload | Base PPL | 50% Budget ΔPPL |
|----------|----------|-----------------|
| Narrative | 4.72 | +31.1% |
| Narrative (offset=50) | 4.84 | +22.6% |
| Narrative (offset=200) | 4.88 | +35.9% |
| Code | 1.48 | +11.1% |
| QA | 3.78 | +111.6% |

**结论**：短序列locality弱是motivation（长上下文才需要我们的系统），不是系统缺陷。

## Project Structure

```
kvcache-lab/
├── gpu-experiments/          # GPU验证实验
│   ├── scripts/              # 实验脚本
│   │   ├── g1_baseline.py           # G1: 基线测量
│   │   ├── g2_pd_transfer.py        # G2: PD分离传输
│   │   ├── g3_taa_validation.py     # G3: TAA核心验证
│   │   ├── g4_sws.py                # G4: SWS实验
│   │   ├── g5_eviction.py           # G5: Eviction策略
│   │   ├── g6_ablation.py           # G6: Full OS Ablation
│   │   ├── g7g8_concurrency.py      # G7+G8: 并发+质量
│   │   ├── supplement_experiments.py # 补充: Dense α scan + Pareto
│   │   ├── long_context_pareto.py   # 补充: 长上下文Pareto
│   │   └── locality_characterization.py  # Locality统计
│   └── results/              # 实验结果JSON
│       ├── v5f2_all_results.json           # G3 TAA
│       ├── g4_all_results.json             # G4 SWS
│       ├── g5_all_results.json             # G5 Eviction
│       ├── g6_all_results.json             # G6 Ablation
│       ├── g7g8_all_results.json           # G7+G8
│       ├── supplement_all_results.json     # α scan + Pareto
│       ├── locality_all_results.json       # Locality characterization
│       ├── taa_heatmap_results.json        # Per-layer TAA heatmap
│       └── layer_ratio_data.json           # Layer-wise local/remote ratio
├── paper/                    # 论文写作
│   ├── main.tex              # 论文主体
│   ├── references.bib        # 参考文献
│   └── figures/              # 配图脚本
│       ├── fig1_pareto.py
│       ├── fig2_locality_cdf.py
│       ├── fig3_taa_heatmap.py
│       ├── fig4_sws_savings.py
│       ├── fig5_overhead.py
│       ├── fig6_concurrency.py
│       ├── fig7_ablation.py
│       └── run_all_figures.py
├── src/                      # 仿真框架（早期验证）
├── experiments/              # 仿真实验脚本
└── docs/                     # 调研报告
```

## Experiment Groups

### Phase A: Fundamentals ✅
| Group | Experiment | Status |
|---|---|---|
| G1 | Baseline (TTFT/TPOT/PPL) | ✅ |
| G2 | PD Disaggregation Transfer | ✅ |
| G3 | TAA Core Validation | ✅ |

### Phase B: Runtime System ✅
| Group | Experiment | Status |
|---|---|---|
| G4 | SWS Memory Savings | ✅ |
| G5 | Predictive Eviction | ✅ |
| G6 | Full OS Ablation | ✅ |
| G7+G8 | Concurrency + Quality | ✅ |

### Supplement Experiments ✅
| Experiment | Status |
|---|---|
| Dense α scan (15 points) | ✅ |
| Memory-Quality Pareto (19 points) | ✅ |
| Long Context Scaling (256-8192) | ✅ |
| Locality Characterization (5 workloads × 3 lengths) | ✅ |
| TAA Per-layer Heatmap | ✅ |
| Workload-specific Pareto (narrative/code/QA) | ✅ |

### To Do
| Experiment | Priority | Status |
|---|---|---|
| ~~vLLM Serving Benchmark~~ | P0 | ✅ HF Serving Benchmark (vLLM不兼容Blackwell，已用HF替代) |
| Multi-model (Llama3-8B/Mistral-7B) | P1 | — |
| LongBench Task Evaluation | P1 | — |

## Technical Notes

### Blackwell GPU Compatibility
- **FlashInfer不支持SM 12.x (Blackwell)**：vLLM无法在RTX PRO 6000 Blackwell上运行
- **SDPA dtype要求**：attn_mask dtype必须与query dtype一致（bfloat16），不能用float16 mask
- vLLM Serving Benchmark已改用HuggingFace-based serving benchmark

### Qwen2.5-7B + SDPA
- `attention_mask` is `None` by default — inject custom 4D causal mask + TAA bias via **forward hooks**
- 2D causal mask + bias → unsqueeze to 4D (avoid 6D broadcast bug)
- **Never use `eager` attention** with Qwen2.5 → NaN logits

### Model Specs
- 28 layers, 4 KV heads (GQA), head_dim=128
- KV bytes per token: 57,344 bytes (bfloat16)
- SDPA attention, hook injection verified on all 28 layers

## Related Work

| Work | Relation |
|---|---|
| Splitwise (ISCA'24) | PD disaggregation architecture |
| DistServe (OSDI'24) | PD disaggregation scheduling |
| vLLM | Serving system baseline |
| **Mooncake** (Moonshot AI, 2024) | KV-centric disaggregated架构，3-tier KV cache (HBM/DRAM/SSD) |
| **KV Cache in the Wild** (上交+阿里, ATC 2025) | 生产环境KV cache workload刻画 |
| **NVIDIA Dynamo** (GTC 2025) | KV router + KV manager |
| **SGLang RadixAttention** (2024) | Radix tree管理跨请求prefix复用 |
| **StreamingLLM/Attention Sinks** (MIT, ICLR 2024) | 保留初始token实现无限长流式推理 |
| **H2O** (2023) | 动态KV cache eviction (deletion-based, 与tiering不同) |
| **The Sparse Frontier** (2025) | 150+篇sparse attention实证分析 |
| CapKV | KV compression via information bottleneck |
| CacheGen | KV cache compression for transmission |
| Sliding Window Attention | Architecture-level local attention (≠ SWS) |

## Paper Target

**Title**: Runtime KV Memory Management for PD-Disaggregated LLM Serving

**Venue**: MLSys / EuroSys / ATC / SoCC

**Core Narrative**: We observe decode-time KV access exhibits strong locality → enables runtime KV tiering with minimal quality degradation → 50% memory budget with near-lossless quality, 8× more concurrency.

## License

Private Repository — Research content confidential
