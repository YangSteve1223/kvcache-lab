# Table-to-JSON Mapping

This document maps each table and figure in the preprint
"Semantic Working Sets for KV Transfer in Prefill--Decode Disaggregated LLM Serving"
to the corresponding raw JSON result files and experiment scripts.

## Paper Tables

| Paper Table | Description | JSON Files | Scripts |
|---|---|---|---|
| Tab 1: b=0.5 comparison | Strategy comparison at 50% KV budget on WikiText-2 | `exp1_pdtrim_ppl_*.json`, `exp_eviction_compare_*.json` | `p0_experiments.py`, `run_g5_eviction.py` |
| Tab 2: Long-context | Long-context PPL at 50% budget | `exp_longctx_ppl_*.json` | `supp_experiments.py` |
| Tab 3: TTFT/TPS | Latency summary (full vs 50% budget) | `exp8_latency_*.json` | `exp7_exp8_combined.py` |
| Tab 4: NIAH | Needle-in-a-haystack retrieval | `exp2_niah_*.json` | `run_all_experiments_v2.py` |
| Tab 5: First-ratio | PDTrim first-ratio sensitivity | `exp_fratio_scan_*.json` | `supp_experiments.py` |
| Tab 6: Multi-task | Cross-domain PPL (Mistral) | `exp_multitask_ppl_mistral7b.json` | `supp_experiments.py` |

## Supplementary Tables (not in preprint, available in data)

| Description | JSON Files | Scripts |
|---|---|---|
| Budget scan (0.3–0.9) | `exp7_budget_scan_*.json` | `exp7_exp8_combined.py` |
| Bandwidth simulation | `exp5_bandwidth_sim.json` | `run_all_experiments_v2.py` |
| Decay-rate sensitivity | `exp_decay_scan_*.json` | `exp_decay_scan_fix5.py` |
| Robustness (multi-run) | `exp3_robustness_*.json` | `run_all_experiments_v2.py` |
| Multi-seed (determinism) | `exp_multiseed_*.json` | `supp_experiments.py` |
| Eviction comparison | `exp_eviction_compare_*.json` | `run_g5_eviction.py` |

## Bugged / Obsolete Runs

| File | Status | Reason |
|---|---|---|
| `niah_depth_scan_qwen7b.json` | ⚠️ OBSOLETE | Full KV also returns 0% retrieval — implementation bug confirmed. Do NOT use as evidence. |
| `niah_depth_scan_mistral7b.json` | ⚠️ OBSOLETE | Same bug as above. |
| `niah_depth_scan_gemma9b.json` | ⚠️ OBSOLETE | Same bug as above. |
| `niah_debug_qwen7b.json` | ⚠️ DEBUG | Debug run, not part of main experimental suite. |

All other 37 JSON files are MD5-verified against GitHub commit `85c288e` and match the paper data.

## Notes

- `*_qwen7b.json` = Qwen2.5-7B-Instruct
- `*_mistral7b.json` = Mistral-7B-Instruct-v0.3
- `*_gemma9b.json` = Gemma-2-9B-it
- All JSON files are in `gpu-experiments/experiment_results_new/`
