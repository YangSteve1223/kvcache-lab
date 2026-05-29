# kvcache-lab

Runtime KV Memory Management for PD-Disaggregated LLM Serving

##概述

This project investigates **decode-time KV cache locality** in Large Language Models and exploits it for efficient memory management in PD-disaggregated serving systems.

**Core Finding**: During decode phase, KV cache access follows a highly concentrated, long-tail distribution (Gini > 0.85 across 4 model families). This enables tiered KV memory management — compressing cache to 30% budget with near-zero PPL degradation.

## Key Results

### Locality Characterization (4/4 Models PASS)

| Model | Gini | Active Set | Remote Attn | Locality Pattern |
|-------|------|-----------|-------------|-----------------|
| Qwen2.5-7B | 0.911 | ~7% | <10% | local-dominant |
| Qwen2.5-14B | 0.952 | ~5% | <5% | local-dominant |
| Mistral-7B | 0.917 | 11.9% | 78.8% | sink-dominant |
| Gemma-2-9B | 0.866 | 20.9% | 60.6% | hybrid |

### Sink-Aware SWS Eviction — Cross-Model PPL Impact

**Mistral-7B** (sink-dominant, Baseline PPL=1.0615):

| Budget | Sink=0 | Sink=4 | Sink=8 | Sink=16 |
|--------|--------|--------|--------|---------|
| 30% | +0.1% | +0.1% | +0.1% | — |
| 50% | ±0.0% | -0.0% | -0.0% | -0.0% |
| 70% | — | -0.0% | -0.0% | — |

**Gemma-2-9b-it** (hybrid, Baseline PPL=1.0391):

| Budget | Sink=0 | Sink=4 | Sink=8 | Sink=16 |
|--------|--------|--------|--------|---------|
| 30% | +11.84% ⚠️ | +5.64% | +4.42% | **+0.47%** ✅ |
| 50% | +1.79% | +2.54% | +1.88% | **-0.94%** ✅ |
| 70% | +2.73% | +0.38% | -0.38% | **-1.88%** ✅ |

**Key insight**: Sink tokens are essential for hybrid models (Gemma: 11.4pp improvement at 30% budget), less critical for large-SWA models at short sequences (Mistral: <0.1pp variation).

### Three Locality Patterns

1. **local-dominant** (Qwen): Attention concentrated on recent tokens. Simple sliding window works.
2. **sink-dominant** (Mistral): 78.8% remote attention targets first few tokens (attention sink). Requires sink-aware eviction.
3. **hybrid** (Gemma): Alternating local/global attention layers. Moderate remote attention, small effective window (~321).

## Project Structure

## Experiment Logs

- [Multi-Model Locality & Sink-Aware Eviction (2026-05-28)](experiment_logs/multimodel_locality_sink_aware_2026-05-28.md)
- [V2 Calibrated Experiments — M2 Milestone (2026-05-28)](experiment_logs/new-paper-experiments-v2.json)

## Technical Stack

- **Simulation**: TypeScript (Node.js 22, ESM)
- **GPU Experiments**: Python (PyTorch, HuggingFace Transformers)
- **Models Tested**: Qwen2.5-7B/14B, Mistral-7B-v0.3, Gemma-2-9b-it
- **Paper**: LaTeX (Overleaf-compatible)

## Methodology Notes

### Why Eager Mode (Not Hooks) for Locality Characterization

Forward hooks capture pre-RoPE Q/K tensors, missing:
1. Position-dependent decay from RoPE
2. Sliding window causal mask (SWA in Mistral/Gemma)

This severely underestimates locality (e.g., Mistral Gini: hook=0.665 vs eager=0.917).

### Why Position IDs Matter for SWS PPL

When concatenating sink + window tokens, omitting `position_ids` causes RoPE to assign wrong relative positions → attention completely breaks. The earlier Mistral PPL=179 was entirely a bug from missing position_ids, NOT a fundamental limitation.

## Preprint

**Title**: Runtime KV Memory Management for PD-Disaggregated LLM Serving

**状态**: Empirical characterization + prototype preprint. Does not claim a production-quality end-to-end PD serving runtime.

**Core framing**: PD-disaggregated serving 的 KV 管理本质上不是单机 eviction，而是**带宽约束下的 hot-set placement**。SWS 是 sink-aware、attention-weighted 的选择性传输策略（placement policy），而非不可逆压缩。

**Publishing**: OSF Project (时间戳 + DOI via Registration) → 后续视情况提交 arXiv / 顶会

**Latest commit**: `19bae92`

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact model identifiers, revisions, package versions, and hardware details.

See [TABLE_MAPPING.md](TABLE_MAPPING.md) for the mapping from each paper table/figure to the corresponding JSON result files and scripts.

See [CHANGELOG.md](CHANGELOG.md) for known issues, bugged runs, and obsolete data marked per experiment.

## License

- **代码**: MIT License (see [LICENSE](LICENSE))
- **Experiment Data (JSON)**: CC BY 4.0
- **Paper (LaTeX/PDF)**: CC BY 4.0
