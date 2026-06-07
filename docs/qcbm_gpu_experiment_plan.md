# QCBM GPU Experiment Plan

**Project**: kvcache-lab — PD-Disaggregated LLM Serving with KV Cache Management  
**Date**: 2026-05-30  
**Status**: Ready for GPU execution  
**Script**: `gpu-experiments/exp_qcbm_quantization.py`

---

## 1. Background & Motivation

### The Problem with Selection-Based Approaches

Current KV cache compression (SWS, PDTrim) operates by **selecting** which tokens to keep and **dropping** the rest. At low bandwidth budgets (30–50%), these methods converge to the same strategy: keep front (sink) tokens + back (recent) tokens. Middle tokens are discarded entirely.

Our existing GPU experiments confirm this:
- **Qwen2.5-7B** at b=0.5: SWS PPL +23–26% (baseline 7.04), essentially same as PDTrim
- **NIAH at b=0.3/0.5**: Both SWS and PDTrim fail completely (0% retrieval) because the needle is in the discarded middle

The fundamental issue: **dropping tokens is an irreversible information loss**. You cannot recover what you never transmitted.

### The QCBM Insight

**Quality-Constrained Budget Minimization (QCBM)** goes beyond selection. Instead of dropping tokens, keep ALL tokens but at different precision levels:

| Tier | Precision | BW per token | Quality factor |
|------|-----------|-------------|----------------|
| Hot  | FP16      | 1.0×        | 1.00 (lossless) |
| Warm | INT8      | 0.5×        | ~0.95 |
| Cold | INT4      | 0.25×       | ~0.80 |

**Key math**: At bandwidth budget b=0.5, instead of keeping 50% of tokens at FP16:
- QCBM keeps 100% of tokens with ~50% → FP16, ~30% → INT8, ~20% → INT4
- Effective bandwidth = 0.50×1.0 + 0.30×0.5 + 0.20×0.25 = **0.60** (slightly over, adjusted by greedy fill)
- The actual split is computed greedily to match exactly b=0.5

**Result**: Middle tokens that SWS would discard are now kept at INT4/INT8, providing degraded but non-zero context. This should especially help:
- NIAH (needle in middle tokens)
- PPL on documents with important mid-sequence information
- Task types that rely on global context (QA, reasoning)

### Prior Simulation Evidence

The TypeScript QCBM simulation (`experiments/exp-qcbm-extended.ts`) showed:
- At quality target 0.95, QCBM saves **97–98% bandwidth** while maintaining 95% quality
- Per-layer allocation: low layers get ~0.25 budget, high layers get ~0.64
- Ablation: precision optimization alone contributes ~60% of bandwidth savings, retention optimization ~40%

However, this was a *simulation* using the CalibratedQualityModel. We need **real GPU validation**.

---

## 2. Experimental Design

### 2.1 Test Matrix

| # | Configuration | Token Retention | Precision | Bandwidth | Rationale |
|---|--------------|----------------|-----------|-----------|-----------|
| 1 | Full KV Baseline | 100% | FP16 | 1.0× | Upper bound |
| 2 | SWS b=0.5 FP16 | 50% | FP16 | 0.5× | Current best approach |
| 3 | SWS b=0.5 Mixed | 100% | sel=FP16, rest=INT8 | 0.75× | SWS + quantize dropped tokens |
| 4 | QCBM b=0.5 | 100% | tiered (FP16/INT8/INT4) | 0.5× | **Our method** |
| 5 | SWS b=0.3 FP16 | 30% | FP16 | 0.3× | Low-budget baseline |
| 6 | SWS b=0.3 Mixed | 100% | sel=FP16, rest=INT8 | 0.65× | SWS + quantize at low budget |
| 7 | QCBM b=0.3 | 100% | tiered (FP16/INT8/INT4) | 0.3× | **Our method at low budget** |

### 2.2 Evaluation Metrics

#### PPL (Perplexity) on WikiText-2
- **What it measures**: Language modeling quality under compressed KV
- **How**: Standard causal LM evaluation, seq_len=2048, 3-5 samples
- **Key comparison**: PPL delta vs baseline for same bandwidth budget
- **Expected**: QCBM should have lower PPL than SWS at same bandwidth, because middle tokens provide context even at INT4

#### NIAH (Needle-in-a-Haystack)
- **What it measures**: Retrieval capability — can the model find information in the middle of a long context?
- **How**: Insert a needle (e.g., "confirmation number is 7294") at varying depths (10%, 25%, 50%, 75%, 90%) in a 4K–8K context
- **Key comparison**: Retrieval rate at each depth
- **Expected**: QCBM should significantly outperform SWS at mid-range depths (25–75%), where SWS drops the needle entirely

#### Effective Bandwidth
- **What it measures**: Total FP16-equivalent tokens transmitted
- **How**: Σ (tokens_per_tier × precision_BW_factor) / total_tokens
- **Why important**: Ensures fair comparison — all methods compared at same bandwidth budget

### 2.3 Quantization Implementation

#### Symmetric Affine Quantization (Per-Token)

```
For INT8:
  scale = max(|kv_channel|) / 127
  quantized = round(kv / scale), clamped to [-127, 127]
  dequantized = quantized × scale

For INT4:
  scale = max(|kv_channel|) / 7
  quantized = round(kv / scale), clamped to [-7, 7]
  dequantized = quantized × scale
```

**Granularity**: Per-head, per-token (i.e., each `(head_idx, token_idx)` pair gets its own scale). This is critical because:
- Different attention heads capture different patterns
- Different tokens have different magnitude ranges
- Per-token granularity minimizes quantization error

**Why not per-channel/per-tensor**: Coarser granularity would introduce significant bias, especially for INT4 where the dynamic range is only 15 distinct values.

### 2.4 Two-Pass Architecture

The experiment uses a two-pass approach for PPL with modified KV:

1. **Pass 1 (Prefill)**: Run full forward pass to compute KV cache at all layers
2. **Modification**: Apply quantization/dequantization to KV cache based on tier assignment
3. **Pass 2 (Evaluate)**: Re-run forward with modified KV cache to compute logits and loss

This approach is chosen because:
- It avoids fragile model hooking that breaks across different model architectures
- It faithfully simulates the PD-separated scenario (P-side computes KV, transmits quantized version, D-side receives and uses it)
- The double computation cost is acceptable for an offline experiment

### 2.5 Tier Assignment Strategy

For QCBM, tokens are assigned to tiers using **greedy bandwidth filling**:

1. Sort tokens by importance score (descending)
2. Assign top tokens to FP16 until adding more would exceed budget
3. Assign next tokens to INT8 until adding more would exceed budget
4. Remaining tokens get INT4

Importance scores use the same SWS heuristic (sink + exponential decay + recency), ensuring a fair comparison — the only difference is precision vs. dropping.

### 2.6 Per-Layer Budget Differentiation

From the QCBM simulation, we know that not all layers need the same precision. The experiment applies a sigmoid-based layer budget curve:

```
budget(l) = 0.25 + 0.40 × sigmoid(10 × (l/(L-1) - 0.5))
```

This gives:
- Low layers (0–30%): budget ~0.25 → mostly INT4/INT8
- Mid layers (30–70%): budget ~0.45 → mixed
- High layers (70–100%): budget ~0.64 → mostly FP16/INT8

This is based on the observation that high layers capture semantic information (more precision-sensitive) while low layers capture syntactic patterns (more quantization-tolerant).

---

## 3. Expected Results

### 3.1 PPL Predictions

| Configuration | Expected PPL (Qwen2.5-7B) | Rationale |
|--------------|---------------------------|-----------|
| Full KV FP16 | ~7.04 | Baseline from existing experiments |
| SWS b=0.5 FP16 | ~8.7 (+24%) | Existing data |
| SWS b=0.5 Mixed | ~7.8 (+11%) | Dropped tokens at INT8 provide partial context |
| **QCBM b=0.5** | **~7.5 (+6%)** | All tokens present, tiered precision |
| SWS b=0.3 FP16 | ~10.6 (+50%) | Existing data |
| SWS b=0.3 Mixed | ~9.0 (+28%) | Partial context from INT8 tokens |
| **QCBM b=0.3** | **~8.2 (+16%)** | Significant advantage from keeping all tokens |

**Key prediction**: QCBM at b=0.5 should approach or beat SWS at b=0.7 in PPL, while using ~30% less bandwidth.

### 3.2 NIAH Predictions

| Depth | Full KV | SWS b=0.5 | QCBM b=0.5 | SWS b=0.3 | QCBM b=0.3 |
|-------|---------|-----------|-------------|-----------|-------------|
| 10% | 100% | 100% (sink) | 100% (FP16) | 100% | ~80% (INT8) |
| 25% | 100% | 0% (dropped) | ~70% (INT8) | 0% | ~40% (INT4) |
| 50% | 100% | 0% (dropped) | ~60% (INT8/INT4) | 0% | ~20% (INT4) |
| 75% | 100% | ~50% (partial) | ~80% (INT8/FP16) | 0% | ~50% |
| 90% | 100% | 100% (recent) | 100% (FP16) | ~80% | ~90% |

**Key prediction**: QCBM should achieve non-trivial retrieval at mid-depths where SWS completely fails. Even INT4-quality KV provides enough signal for the model to locate the needle.

### 3.3 Bandwidth Efficiency

At b=0.5:
- SWS: 50% tokens × FP16 = 1024 FP16-equivalent tokens (out of 2048)
- QCBM: 100% tokens × mixed = ~1024 FP16-equivalent tokens (but all 2048 tokens present)
- **QCBM provides 2× the token coverage at the same bandwidth cost**

---

## 4. How to Interpret Results

### 4.1 Positive Outcome (QCBM works as predicted)

If QCBM shows:
- **PPL improvement over SWS at same bandwidth**: This validates the core hypothesis that keeping tokens at reduced precision is better than dropping them.
- **NIAH retrieval at mid-depths**: This demonstrates practical value — QCBM enables retrieval that selection-based methods cannot.
- **Layer differentiation helps**: If per-layer budgets outperform uniform budgets, this validates the QCBM optimization insight.

**Action**: Proceed to implement QCBM in the production PD-separated serving system.

### 4.2 Partial Outcome (QCBM helps NIAH but not PPL)

If QCBM improves NIAH significantly but PPL only marginally:
- This suggests that quantization noise primarily affects **local** token prediction (PPL) but preserves **global** context (retrieval).
- The PPL metric may be too coarse to capture the quality improvement.
- **Action**: Focus QCBM deployment on retrieval-heavy workloads. Consider hybrid approaches (SWS for PPL-sensitive tasks, QCBM for retrieval tasks).

### 4.3 Negative Outcome (INT4/INT8 quantization degrades quality severely)

If QCBM performs worse than SWS at same bandwidth:
- This would indicate that KV cache is highly sensitive to quantization noise.
- INT4 may be too aggressive — even with per-token scaling, 15 distinct values may be insufficient for attention computation.
- **Action**: Reduce to 2-tier (FP16 + INT8 only), or use fine-grained grouping (per-group-of-4-channels).

### 4.4 Surprising Outcome (QCBM beats full FP16)

If QCBM at reduced bandwidth somehow achieves lower PPL than full FP16:
- This would be a form of **regularization through quantization**.
- Analogous to dropout: removing precision acts as noise that prevents overfitting.
- This has been observed in weight quantization literature but would be surprising for KV cache.
- **Action**: Investigate which layers/tiers benefit from quantization and why.

---

## 5. Potential Pitfalls

### 5.1 Quantization Granularity

**Risk**: Per-token quantization may be too fine-grained, incurring high metadata overhead (storing scales per token).

**Mitigation**: The experiment measures effective bandwidth including scale storage. Each token's K and V each need one fp16 scale value per head, which is `(2 * num_heads * 2 bytes) / (2 * num_heads * head_dim * 2 bytes) = 1/head_dim ≈ 0.8%` overhead for head_dim=128. This is negligible.

### 5.2 Dequantization Error Accumulation

**Risk**: In the two-pass approach, quantization + dequantization introduces round-trip error. If this error is large, the modified KV may be worse than expected.

**Mitigation**: We measure the actual quantization error by comparing original vs. dequantized KV tensors. This diagnostic helps determine if INT4 is viable.

**To check**: Add a diagnostic mode that reports `||original - dequantized|| / ||original||` per tier.

### 5.3 Position Encoding Sensitivity

**Risk**: Qwen2.5 uses RoPE (Rotary Position Embedding). Quantization of K/V may interact poorly with rotary embeddings if the quantization happens before position encoding is applied.

**Mitigation**: Our approach quantizes the *stored* KV cache (after RoPE is applied), so the quantization noise affects the already-position-encoded values. This is the correct approach for PD separation where KV is transmitted post-prefill.

### 5.4 Memory Pressure

**Risk**: The two-pass approach requires storing the full KV cache in memory, then a modified copy. For 7B models at 2K–4K context, this is ~4–8GB per copy. With a 96GB GPU, this is fine, but at 32K context it could OOM.

**Mitigation**: 
- Start with seq_len=2048 for PPL (safe)
- For NIAH at 4K/8K, monitor VRAM carefully
- If needed, process layers one at a time (modify KV, immediately free original)

### 5.5 Fairness of Comparison

**Risk**: SWS-Mixed uses more bandwidth than SWS-FP16 (because it keeps dropped tokens at INT8). We must compare at **equal bandwidth**, not equal configuration.

**Mitigation**: The primary comparison is:
- **Same bandwidth**: SWS b=0.5 FP16 (0.5× BW) vs. QCBM b=0.5 (0.5× BW)
- SWS-Mixed at 0.75× BW is shown as an **intermediate** datapoint

### 5.6 Importance Score Quality

**Risk**: The heuristic importance scores (sink + decay + recency) may not perfectly reflect true attention importance. If important tokens are misassigned to INT4, QCBM could underperform.

**Mitigation**: 
- The experiment also runs with **real attention extraction** (when feasible) as an ablation
- The heuristic is the same one used by SWS, so any misassignment equally affects both methods
- Future: extract actual attention weights from the model for data-driven tier assignment

### 5.7 INT4 Dynamic Range

**Risk**: INT4 has only 15 distinct values ([-7, 7]). For KV channels with large dynamic range, this may cause severe clipping.

**Mitigation**: 
- Per-token scaling ensures the max value maps to ±7
- The quantization error is bounded by `scale * 0.5` (rounding error)
- For tokens where INT4 is assigned, they're low-importance by definition — the model can tolerate more noise there
- **Fallback**: If INT4 proves too aggressive, we can replace INT4 tier with INT4-grouped (quantize groups of 4 channels together, using shared scale)

---

## 6. Extension Experiments (Future Work)

If the core experiment validates QCBM, these follow-ups would strengthen the paper:

1. **Multi-model validation**: Run on Mistral-7B and Gemma-2-9B to test generalizability
2. **Scaling to 14B**: Test on Qwen2.5-14B to validate at larger model size
3. **Longer contexts (8K–32K)**: QCBM advantage should grow with context length (more middle tokens)
4. **Per-layer quantization ablation**: Uniform vs. differentiated layer budgets
5. **Adaptive tier assignment**: Use actual attention weights instead of heuristic importance
6. **Serving benchmark**: End-to-end latency measurement with simulated PD separation
7. **INT4 grouping study**: Compare per-token INT4 vs. per-group-of-4 INT4 vs. per-head INT4

---

## 7. Reproducibility

### Hardware Requirements
- 1× GPU with ≥40GB VRAM (for 7B model + KV cache)
- Tested on: RTX PRO 6000 (96GB), should work on A100-40GB

### Software Requirements
```
torch >= 2.0
transformers >= 4.36
datasets
numpy
```

### Running the Experiment
```bash
# Full experiment (PPL + NIAH)
python exp_qcbm_quantization.py --model qwen7b --seq_len 2048 --budgets 0.3 0.5

# PPL only
python exp_qcbm_quantization.py --model qwen7b --ppl_only --seq_len 2048

# NIAH only
python exp_qcbm_quantization.py --model qwen7b --niah_only --niah_seq_len 4096

# With custom output directory
python exp_qcbm_quantization.py --model qwen7b --output_dir my_results
```

### Expected Runtime
- PPL (3 samples × 7 configs): ~15 minutes on RTX PRO 6000
- NIAH (3 needles × 5 depths × 7 configs): ~30 minutes
- Total: ~45 minutes

---

## 8. Success Criteria

| Criterion | Metric | Threshold |
|-----------|--------|-----------|
| QCBM PPL ≤ SWS PPL at same BW | PPL delta | ≤ 80% of SWS delta |
| QCBM NIAH > SWS NIAH at mid-depths | Retrieval rate at 25-75% depth | > 0% (SWS = 0%) |
| QCBM retains all tokens | Token retention | 100% |
| Bandwidth budget matched | Effective BW | Within 5% of target |

If all four criteria are met, QCBM is validated for production deployment.
