# kvcache-lab 完整实验总结

**项目**: Runtime KV Memory Management for PD-Disaggregated LLM Serving  
**时间**: 2026-05-27 ~ 2026-05-28  
**论文目标**: MLSys / ATC / EuroSys

---

## 核心叙事

Decode阶段KV cache访问呈极强locality（Gini>0.85, 4模型验证），使tiered KV memory management可行——50%内存预算下近乎无损。

**证据链**: Locality存在 → Active set仅5-21% → SWS保留window即可 → Sink-aware适配不同模式 → 内存省→并发提升

---

## Phase A: 基础验证（Qwen2.5-7B, PRO 6000 Blackwell 98GB, 05-27）

### A1. Locality Characterization

| 指标 | 数值 | 意义 |
|------|------|------|
| Gini系数 | 0.911 | KV access极强集中度 |
| Top 20% KV → 95% attention | 95% | Zipf-like长尾 |
| Active Set（80%覆盖） | 仅9% | Working set << 总KV |
| Remote KV注意力浪费 | ~2% | 70% remote KV仅获2% attention |

### A2. Long Context Pareto

| 上下文 | 50% budget PPL | 30% budget PPL | 结论 |
|--------|---------------|---------------|------|
| 256 | +28% | — | locality弱 |
| 512 | +29% | — | locality弱 |
| **2048** | **-0.20%** ✅ | ✅ | 质量无损 |
| **4096** | **-0.21%** ✅ | **-0.8%** ✅ | 30%也仅微小退化 |
| **8192** | 更优 | 更优 | 长序列locality更强 |

### A3. Short-Context (seq=1024)

| Workload | 50% Budget ΔPPL | 说明 |
|----------|-----------------|------|
| Narrative | +31.1% | 短序列locality弱 |
| Code | +11.1% | Code稍好 |
| QA | +111.6% | QA高度依赖远程KV |

→ 短序列locality弱是**motivation**（长上下文才需要tiering），不是缺陷

### A4. TAA Validation

- Local attn: 5.82% → TAA(α=0.2): 7.17%（+22%）
- 中间层(14-22)最显著
- Overhead: 32K仅0.04%, 2K仅0.006%
- TAA-guided vs random eviction: 5.19 vs 18.80（3.6x改善）

### A5. HF Serving Benchmark (vLLM不兼容Blackwell)

| Context | C=1 | C=4 | C=8 | C=16 |
|---------|-----|-----|-----|------|
| 1K | 66 TPS | 192 | 327 | 603 |
| 4K | 59 | 141 | 203 | 243 |
| 8K | 50 | 84 | 108 | 128 |
| 16K | 36 | 48 | 55 | 62 |
| 32K | 21 | 24 | 26 | 27 |

> 32K×16 OOM

### A6. GPU实验脚本（8文件5446行）

G1-G8 + 补充实验全部完成，结果在 `gpu-experiments/results/`。

---

## Phase B: 多模型泛化（vGPU-48GB, 05-28）

### B1. 方法论修正

Hook方法（pre-RoPE Q/K）严重低估locality：

| 指标 | Hook(错误) | Eager(正确) | 偏差 |
|------|-----------|-----------|------|
| Mistral Gini | 0.665 | **0.917** | 38% |
| Active Set | 32.5% | 11.9% | — |
| Remote Attn | 68.9% | 78.8% | — |

原因：缺少RoPE位置编码衰减 + 缺少SWA causal mask

修正：`attn_implementation="eager"` + `output_attentions=True`

### B2. 四模型Locality（4/4 PASS）

| 模型 | 参数量 | Gini | Active% | Remote% | 有效窗口 | 模式 |
|------|--------|------|---------|---------|---------|------|
| Qwen2.5-7B | 7B | 0.911 | ~7% | <10% | 全序列 | local-dominant |
| Qwen2.5-14B | 14B | 0.952 | ~5% | <5% | 全序列 | local-dominant |
| Mistral-7B | 7B | 0.917 | 11.9% | 78.8% | 全序列* | sink-dominant |
| Gemma-2-9B | 9B | 0.866 | 20.9% | 60.6% | ~321 | hybrid |

*Mistral SWA window=4096，测试长度内未触发截断

**三种模式**:
1. **local-dominant** (Qwen): attention集中在近处，简单SWS
2. **sink-dominant** (Mistral): 78.8% remote→序列开头(sink)，需保留sink token
3. **hybrid** (Gemma): 交替local/global层，sink是全局层位置锚点

### B3. 逐模型详细数据

**Mistral-7B**: seq_1024 Gini=0.906, seq_2048=0.919, seq_4096=0.926  
**Gemma-2-9B**: seq_1024 Gini=0.854, seq_2048=0.860, seq_4096=0.886  
**Qwen2.5-14B**: 4K Gini=0.952, 8K=0.966

---

## Phase C: Sink-Aware SWS PPL（vGPU-48GB, 05-28）

### C1. Bug修正

之前Mistral PPL=179(+12420%)是脚本bug：缺少position_ids→RoPE位置编码错位→attention崩溃。修正后PPL=1.06。

### C2. Mistral-7B (sink-dominant, Baseline=1.0615)

| Budget | Sink=0 | Sink=4 | Sink=8 | Sink=16 |
|--------|--------|--------|--------|---------|
| 30% | +0.1% | +0.1% | +0.1% | — |
| 50% | ±0.0% | -0.0% | -0.0% | -0.0% |
| 70% | — | -0.0% | -0.0% | — |

→ SWA window=4096覆盖2048序列，sink count影响极小

### C3. Gemma-2-9b-it (hybrid, Baseline=1.0391)

| Budget | Sink=0 | Sink=4 | Sink=8 | Sink=16 |
|--------|--------|--------|--------|---------|
| 30% | +11.84% ⚠️ | +5.64% | +4.42% | **+0.47%** ✅ |
| 50% | +1.79% | +2.54% | +1.88% | **-0.94%** ✅ |
| 70% | +2.73% | +0.38% | -0.38% | **-1.88%** ✅ |

→ **sink token对hybrid模型是必需品**: 30% budget时sink 0→16降11.4pp

### C4. 跨模型对比

| 模型 | 模式 | 30% budget(sink=16) | 50% budget(sink=16) |
|------|------|---------------------|---------------------|
| Mistral-7B | sink-dominant | +0.1% | -0.0% |
| Gemma-2-9B | hybrid | +0.47% | -0.94% |
| Qwen2.5-7B | local-dominant | -0.8%* | -0.2%* |

*Qwen数据来自Phase A不同配置，供参考

---

## 技术教训

| 问题 | 错误 | 正确 | 影响 |
|------|------|------|------|
| Locality测量 | Hook pre-RoPE Q/K | Eager output_attentions | Gini偏差38% |
| SWS PPL | 不传position_ids | 显式传position_ids | PPL 179→1.06 |
| KV cache操作 | 直接操作DynamicCache | token IDs+position_ids | API变更 |

---

## GPU使用记录

| 实例 | GPU | 用途 | 费用 | 状态 |
|------|-----|------|------|------|
| PRO 6000 | RTX PRO 6000 98GB | Phase A + Qwen14B | ~10元 | 已关机 ✅ |
| 双3090 | 2×RTX 3090 | PD benchmark | ~5元 | 已关机 ✅ |
| vGPU-48GB | RTX 4090 48GB | Phase B + Phase C | ~8元 | 运行中 ⚠️ |

---

## 数据资产 (GitHub, private, commit 0b25757)

**核心数据（已push）**:
- `experiments/multimodel_locality/` — 4个JSON (Mistral/Gemma locality + PPL)
- `experiments/scripts/` — 4个Python脚本
- `experiment_logs/` — Phase B+C实验日志
- `gpu-experiments/results/` — Phase A全部结果
- `paper/main.tex` — 论文1008行+11图

**待补push（vGPU commit 1441e07）**:
- `mistral_locality_hook.json` — hook vs eager对比
- `Qwen2.5-14B-Instruct_results.json` — Qwen14B详细
- `all_models_summary.json` — 汇总
- `hf_serving_bench_*.json` — serving benchmark原始
- Phase A raw results

---

## 待完成

| 优先级 | 任务 |
|--------|------|
| P0 | 论文更新（三分法+sink-aware+4/4 PASS） |
| P0 | 关vGPU |
| P1 | 补push vGPU数据 |
| P2 | OSF Preprints → arXiv(需导师endorse) |
