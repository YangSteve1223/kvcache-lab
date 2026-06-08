# SpectrumKV

**SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer for Prefill-Decode Disaggregated LLM Serving**

This repository contains the official implementation and experimental data for the SpectrumKV paper.

## Overview

SpectrumKV is a per-token mixed-precision KV cache transfer policy for PD-disaggregated LLM serving. Instead of binary keep/drop decisions, SpectrumKV assigns a precision level (FP16, INT8, or INT4) to each token based on its importance, enabling better quality at the same transfer budget.

### Three Components

1. **SWS (Semantic Working Set)** — Importance scoring for each token's KV cache entry
2. **QCBM (Quality-Constrained Bandwidth Minimization)** — Greedy precision allocation under budget constraint
3. **Probe** — Lightweight deployment-time probe to detect INT4 tolerance (3 NIAH trials → 2-tier or 3-tier policy)

## Key Results

### GPU Verification — WikiText-2 PPL (b=0.5)

| Model | FP16 Baseline | SpectrumKV | PDTrim | Δ vs PDTrim |
|-------|--------------|-----------|--------|-------------|
| Qwen2.5-7B | 7.0415 | +1.97% | +25.85% | −23.88pp |
| Mistral-7B | 7.9568 | −0.06% | +22.07% | −22.13pp |
| Gemma-2-9B | 11.2042 | −0.44% | +35.63% | −36.07pp |

### NIAH Retrieval (b=0.3)

- SpectrumKV 2-tier on Qwen: 52.6% vs PDTrim 26.3% (+26pp)
- Mistral/Gemma 3-tier: 100% at b=0.3–0.7

### Three Attention Locality Patterns

| Pattern | Model | Key trait |
|---------|-------|-----------|
| Local-dominant | Qwen2.5-7B/14B | Recent tokens dominate |
| Sink-dominant | Mistral-7B | First few tokens (attention sinks) dominate |
| Hybrid | Gemma-2-9B | Both sink and recent matter; sink protection essential |

## Repository Structure

```
kvcache-lab/
├── paper/                          # Paper source (LaTeX + BibTeX)
│   ├── main_v2_final_arxiv.tex     # arXiv submission version
│   ├── main_v2_final.tex           # Latest paper (28 pages)
│   ├── main_v2_final.pdf           # Compiled PDF
│   ├── spectrumkv_v2.bib           # References
│   ├── figures/                    # Figure generation scripts + outputs
│   └── arxiv_metadata.txt          # arXiv submission metadata
│
├── gpu-experiments/                # GPU experiment scripts & results
│   ├── exp_spectrumkv_gpu.py       # Main SpectrumKV GPU experiment
│   ├── exp_qcbm_quantization.py    # QCBM quantization experiment
│   ├── spectrumkv_scripts/         # Helper scripts (exp1–exp4)
│   ├── results/                    # Aggregated result JSONs
│   └── experiment_results_new/     # Per-model experiment results (34 JSONs)
│
├── spectrumkv/                     # Python package
│   ├── core/
│   │   ├── sws.py                  # SWS implementation
│   │   ├── qcbm.py                 # QCBM implementation
│   │   ├── quantizer.py            # Quantization utilities
│   │   ├── probe.py                # Probe mechanism
│   │   └── baselines.py            # PDTrim and other baselines
│   ├── requirements.txt
│   └── tests/test_core.py
│
├── spectrumkv_data/                # Processed experiment data
│   ├── ALL_EXPERIMENT_DATA.md      # Comprehensive data document
│   ├── exp1_ppl_*.json             # PPL results per model
│   ├── exp2_niah_*.json            # NIAH results per model
│   ├── exp3_quant_error_*.json     # Quantization error per model
│   ├── exp4_layer_budget_*.json    # Layer budget allocation
│   └── raw/                        # Unprocessed experiment logs
│
├── data/                           # Simulation input data
├── docs/                           # Project documentation
├── osf-preprint/                   # OSF preprint source
│
├── run_gpu_experiments.sh          # GPU experiment runner
├── REPRODUCIBILITY.md              # Reproducibility information
├── TABLE_MAPPING.md                # Paper table → JSON mapping
└── LICENSE                         # MIT License
```

## Reproducing Experiments

### GPU Experiments (requires NVIDIA GPU + vLLM)

```bash
cd gpu-experiments
python exp_spectrumkv_gpu.py --model Qwen/Qwen2.5-7B-Instruct
python exp_spectrumkv_gpu.py --model mistralai/Mistral-7B-Instruct-v0.3
python exp_spectrumkv_gpu.py --model google/gemma-2-9b-it
```

### Using the Python Package

```bash
pip install -r spectrumkv/requirements.txt
python -c "from spectrumkv.core import sws, qcbm, probe; print('SpectrumKV loaded')"
```

## Models and Environment

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for full model IDs, software versions, and known environment issues.

## Citation

```bibtex
@article{yang2026spectrumkv,
  title={SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer for Prefill-Decode Disaggregated LLM Serving},
  author={Yang, Pengju},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
