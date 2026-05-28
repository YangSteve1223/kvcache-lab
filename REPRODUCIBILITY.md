# Reproducibility Information

## Models

| Paper Name | HuggingFace ID | Revision | Tokenizer |
|---|---|---|---|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | default (main branch) | AutoTokenizer from same repo |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | default (main branch) | AutoTokenizer from same repo |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | default (main branch) | AutoTokenizer from same repo |

### Model Notes
- Qwen2.5-7B uses eager attention (SDPA returns NaN for some configurations; fallback to eager mode required)
- Mistral-7B uses SWA (sliding window attention = 4096), limiting effective context to ~2048 tokens for some experiments
- Gemma-2-9B uses alternating local/global attention layers (hybrid pattern)
- All models loaded in BF16 precision

## Software Environment

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.1+ |
| Transformers | 5.9.0 (note: DynamicCache API changed — use `kv.layers[i].keys/values`) |
| CUDA | 12.x |
| Datasets | 4.8.5+ (use `Salesforce/wikitext` for WikiText-2) |

### Python Dependencies
- `torch`, `transformers`, `datasets`, `numpy`, `accelerate`

## Hardware

| Instance | GPU | VRAM | Role |
|---|---|---|---|
| Instance 1 | NVIDIA RTX 4080 SUPER | 32 GB | Qwen2.5-7B experiments |
| Instance 2 | NVIDIA RTX 4090 | 48 GB | Mistral-7B + Gemma-2-9B experiments |

Both instances are cloud GPU rentals (AutoDL/SeetaCloud).

## Data

| Dataset | Source | Notes |
|---|---|---|
| WikiText-2 | `Salesforce/wikitext`, `wikitext-2-raw-v1` | PPL evaluation (v5 runs, not v4) |
| NIAH needles | Custom (see `niah_depth_scan.py`) | Standard NIAH protocol with chat template |
| Multi-task | Code: `code_search_net`, Science: `scientific_papers`, etc. | See `supp_experiments.py` |

## Known Environment Issues

1. **Transformers 5.9.0 DynamicCache**: Access pattern changed to `kv.layers[i].keys/values` instead of legacy tuple interface
2. **Datasets 4.8.5+**: WikiText-2 must be loaded as `Salesforce/wikitext` (not `wikitext`)
3. **Qwen eager attention NaN**: SDPA kernel produces NaN for certain KV cache configurations; must fall back to `attn_implementation="eager"`
4. **Mistral system_message**: Requires `system_message` parameter in chat template for proper formatting

## Commit Reference

Primary experimental data: commit `85c288e` on `main` branch.
