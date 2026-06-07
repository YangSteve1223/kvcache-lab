# kvcache-lab

**SpectrumKV: Per-Token Mixed-Precision KV Cache Transfer for Prefill-Decode Disaggregated LLM Serving**

This project implements SpectrumKV, a per-token mixed-precision KV cache transfer policy for PD-disaggregated LLM serving. Instead of the binary keep/drop decision used by prior work, SpectrumKV assigns a precision level (FP16, INT8, or INT4) to each token based on its importance, enabling better quality at the same transfer budget.

## Core Idea

- **SWS (Semantic Working Set)**: Importance scoring for each token's KV cache entry
- **QCBM (Quality-Constrained Bandwidth Minimization)**: Greedy precision allocation under budget constraint
- **Probe**: Lightweight deployment-time probe to detect INT4 tolerance (3 NIAH trials → 2-tier or 3-tier)

## Key Results (GPU, WikiText-2 PPL)

| Model | Budget | SpectrumKV | PDTrim | SWS Original |
|-------|--------|-----------|--------|-------------|
| Qwen2.5-7B | b=0.5 | +1.97% | +25.85% | +30.9% |
| Mistral-7B | b=0.5 | -0.06% | +22.07% | +33.2% |
| Gemma-2-9B | b=0.5 | -0.44% | +35.63% | +43.8% |

**NIAH Retrieval (b=0.3)**: SpectrumKV 52.6% vs PDTrim 26.3% on Qwen; Mistral/Gemma reach 100% under 3-tier.

## Repository Structure

```
kvcache-lab/
├── paper/                          # Final paper (LaTeX + BibTeX)
│   ├── main_v2_final.tex           # ★ Latest paper source (28 pages)
│   ├── main_v2_final.bbl           # Compiled references (for arXiv)
│   ├── spectrumkv_v2.bib           # BibTeX source
│   ├── REVIEW_CRITERIA.md          # Review audit criteria
│   ├── main_v2_polish_a/b/c/d.tex  # Polishing iterations
│   └── main_v2.tex                 # Pre-polish version
│
├── gpu-experiments/                 # GPU experiment scripts & results
│   ├── exp_spectrumkv_gpu.py        # ★ Main SpectrumKV GPU experiment
│   ├── exp_qcbm_quantization.py     # QCBM quantization experiment
│   ├── spectrumkv_scripts/          # Helper scripts for SpectrumKV
│   ├── run_g3_v5f2.py              # G3: SpectrumKV variants
│   ├── run_g4_sws.py               # G4: SWS strategies
│   ├── niah_depth_scan.py          # NIAH depth scan
│   ├── pd_separation.py            # PD separation hook simulation
│   ├── results/                    # Raw result JSON files
│   └── README.md                   # Experiment documentation
│
├── src/                            # Core library code
│   ├── attention_analysis.py       # Attention locality analysis
│   ├── semantic_working_set.py     # SWS implementation
│   └── transmission_aware_attn.py  # TAA for KV selection
│
├── experiments/                    # Simulation experiments
│   ├── exp-formula-optimization.py
│   ├── exp-qcbm-corrected-v3.py
│   └── exp-spectrumkv-v3-fine.py
│
├── data/                           # Simulation data
├── docs/                           # Documentation & plans
├── logs/                           # Experiment logs
└── osf-preprint/                   # OSF preprint files
```

## Three Attention Locality Patterns

| Pattern | Model | Description |
|---------|-------|-------------|
| Local-dominant | Qwen2.5-7B/14B | Recent tokens dominate; sink protection less critical |
| Sink-dominant | Mistral-7B | First few tokens (attention sinks) receive most attention |
| Hybrid | Gemma-2-9B | Both sink and recent tokens matter; sink protection essential |

## Quantization Findings

- **INT8**: Safe across all models (cosine similarity > 0.9999)
- **INT4**: Model-dependent tolerance
  - Qwen2.5-7B: Catastrophic failure (PPL explodes)
  - Mistral-7B: Stable
  - Gemma-2-9B: Stable
- **Per-layer variation**: INT4 cosine similarity varies ~2.5x across layers in Qwen → supports per-layer budget allocation

## Reproducing Experiments

### GPU Experiments (requires NVIDIA GPU with vLLM)

```bash
cd gpu-experiments
# Run main SpectrumKV experiment
python exp_spectrumkv_gpu.py --model Qwen/Qwen2.5-7B-Instruct
python exp_spectrumkv_gpu.py --model mistralai/Mistral-7B-Instruct-v0.3
python exp_spectrumkv_gpu.py --model google/gemma-2-9b-it
```

### Simulation Experiments

```bash
cd experiments
python exp-spectrumkv-v3-fine.py
```

## Paper

- **arXiv**: (to be submitted)
- **Preprint**: OSF (pending)
- **Target venue**: MLSys 2027

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
