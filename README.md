# kvcache-lab

Runtime KV Memory Management for PD-Disaggregated LLM Serving

## Overview

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

```
kvcache-lab/
├── src/                          # Core simulation engine (TypeScript)
│   ├── core/                     # Runtime KV Memory OS modules
│   │   ├── GlobalStateStore.ts
│   │   ├── RuntimeScheduler.ts
│   │   ├── SemanticAgent.ts
│   │   ├── ReuseAgent.ts
│   │   ├── CommunicationAgent.ts
│   │   └── PlacementAgent.ts
│   ├── serving/                  # PD serving simulator
│   │   ├── EnhancedPDServingSimulator.ts
│   │   ├── ContinuousBatchingScheduler.ts
│   │   └── constants.ts
│   └── algorithms/               # KV cache management algorithms
├── gpu-experiments/              # GPU experiment scripts
│   ├── run_g1_baseline.py
│   ├── run_g2_pd_bench.py
│   ├── run_g3_taa.py
│   ├── run_g4_sws.py
│   ├── run_g5_eviction.py
│   └── run_g6_full_os.py
├── experiments/                  # Experiment results & scripts
│   ├── multimodel_locality/      # Multi-model locality characterization data
│   └── scripts/                  # Experiment scripts
├── tests/                        # Test suite
├── paper/                        # LaTeX paper source
│   ├── main.tex
│   └── references.bib
└── README.md
```

## Experiment Logs

- [Multi-Model Locality & Sink-Aware Eviction (2026-05-28)](experiment_logs/multimodel_locality_sink_aware_2026-05-28.md)

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

**Title**: Semantic Working Sets for KV Transfer in Prefill--Decode Disaggregated LLM Serving

**Status**: Empirical characterization + prototype preprint. Does not claim a production-quality end-to-end PD serving runtime.

**arXiv category**: cs.DC (primary), cs.LG (secondary)

**Latest commit**: `85c288e`

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact model identifiers, revisions, package versions, and hardware details.

See [TABLE_MAPPING.md](TABLE_MAPPING.md) for the mapping from each paper table/figure to the corresponding JSON result files and scripts.

See [CHANGELOG.md](CHANGELOG.md) for known issues, bugged runs, and obsolete data marked per experiment.

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Experiment Data (JSON)**: CC BY 4.0
- **Paper (LaTeX/PDF)**: CC BY 4.0
