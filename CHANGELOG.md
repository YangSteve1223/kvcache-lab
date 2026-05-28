# Changelog

## v1.0 — 2026-05-28

### Known Issues (OBSOLETE runs marked with ⚠️)

#### NIAH Depth Scan Bug
- **Files affected**: `niah_depth_scan_qwen7b.json`, `niah_depth_scan_mistral7b.json`, `niah_depth_scan_gemma9b.json`
- **Issue**: Full KV control also returns 0% retrieval in depth scan, while the main NIAH experiment (exp2) shows 12/12 = 100% for full KV. This confirms an implementation bug in the depth scan script, NOT in the model or KV policy.
- **Root cause**: The depth scan script does not apply chat template or has a different NIAH protocol than the main exp2 script.
- **Status**: These files are marked OBSOLETE. They are NOT used in the preprint. The main NIAH sanity check (exp2) is used instead.
- **Fix plan**: Re-implement depth scan with same protocol as exp2 (apply_chat_template, disable "secret/hidden/magic" keywords, needle truncation fix).

#### Qwen Eager Attention NaN
- **Issue**: Qwen2.5-7B SDPA kernel produces NaN for some KV cache masking configurations.
- **Workaround**: Fall back to `attn_implementation="eager"`.
- **Impact on data**: Qwen decay-scan results use KV-norm fallback instead of attention scores. Interpretation is limited (overlap ≥ 98% between decay rates).

#### PPL Below Baseline (Metric Caveat)
- **Issue**: SWS with sink=16 on Mistral/Gemma produces PPL below full-KV baseline (e.g., -4.8%, -12.5%).
- **Explanation**: This is a metric artifact — compressed runs evaluate on a subset of retained positions, removing difficult long-range context. Sub-baseline PPL does NOT mean improved generation quality.
- **How reported**: Treated as metric caveat in the preprint, not as a quality improvement claim.

#### Missing Position IDs (Fixed in v5)
- **Issue**: Early runs (v1–v4) did not pass `position_ids` when concatenating sink + window tokens, causing RoPE misalignment.
- **Impact**: Mistral PPL=179 in early runs was entirely a bug.
- **Fix**: v5 runs include explicit `position_ids`. All data in commit `85c288e` is from v5+ runs.

#### V4 Results Invalid
- **Issue**: v4 used synthetic WikiText data and produced PPL ~1.69 (unrealistically low).
- **Fix**: v5 reran with proper `Salesforce/wikitext` dataset. All current data is v5.

### Data Verification
- 37/37 JSON files MD5-verified against GitHub commit `85c288e`
- 4 depth-scan files marked OBSOLETE (not counted in the 37 verified)
- 1 debug file (`niah_debug_qwen7b.json`) not part of main suite

### Commit History
- `85c288e` — Full experimental data + scripts upload (37 JSON + 5 core scripts)
- `df5cd91` — Add NIAH depth scan (Qwen)
- `0240200` — Add NIAH depth scan (Mistral)
- `86f0022` — Add NIAH depth scan (Gemma)
- `7f95f8a` — Calibrated quality model + exp runner v2
- `e52838b` — Progress report update
