#!/usr/bin/env python3
"""
[DEPRECATED] This script has been superseded by spectrumkv_scripts/exp1_ppl_fine_budget.py
and spectrumkv_scripts/exp2_niah_fine_depth.py. Use those instead.
===============================================================================
SpectrumKV GPU Experiment: Per-Token Mixed-Precision KV Cache under Bandwidth Constraints
================================================================================

Core idea: Instead of dropping tokens (PDTrim/SWS), keep ALL tokens but assign
different precision levels (FP16/INT8/INT4) based on per-token importance scores.
At the SAME bandwidth budget, SpectrumKV retains information from ALL tokens,
including middle-depth tokens that selection-based methods irreversibly discard.

Methods compared:
  1. Full FP16 (oracle, b=1.0)
  2. PDTrim: keep first_k + last_w tokens at FP16, drop rest
  3. SWS: score-based top-K selection at FP16, drop rest
  4. SpectrumKV_Greedy: ALL tokens retained, greedy precision tier assignment
  5. Random_Tier: ALL tokens retained, RANDOM precision assignment (ablation)

Evaluations:
  - PPL on WikiText-2
  - NIAH (Needle-in-a-Haystack) at multiple depths

Usage:
  python exp_spectrumkv_gpu.py --model qwen7b --budgets 0.3 0.5 0.7
  python exp_spectrumkv_gpu.py --model qwen7b --ppl_only --budgets 0.5
  python exp_spectrumkv_gpu.py --model qwen7b --niah_only --budgets 0.3 0.5

Requires GPU with sufficient VRAM for model loading.
===============================================================================
"""

import argparse
import json
import math
import os
import string
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral/Mistral-7B-Instruct-v0.3",
    "gemma9b": "google/gemma-2-9b-it",
}

# Bandwidth multipliers per precision (relative to FP16)
PRECISION_BW = {"fp16": 1.0, "int8": 0.5, "int4": 0.25}

# NIAH quality factor per precision (1.0 = lossless)
# These are priors validated by the simulation; GPU experiment will verify
PRECISION_QUALITY = {"fp16": 1.0, "int8": 0.95, "int4": 0.78}

DEFAULT_OUTPUT_DIR = "experiment_results_spectrumkv"
SINK_COUNT = 16  # Number of sink tokens to protect


# =============================================================================
# KV Cache Quantization
# =============================================================================

class KVQuantizer:
    """Symmetric affine quantization for KV cache tensors."""

    QMAX = {"int8": 127, "int4": 7}

    @staticmethod
    def quantize_dequantize(kv_tensor: torch.Tensor, precision: str) -> torch.Tensor:
        """
        Quantize then dequantize a KV cache slice (simulate quantization error).
        
        Args:
            kv_tensor: shape (batch, heads, seq_len, head_dim) or similar
            precision: "int8" or "int4"
        
        Returns:
            Dequantized tensor (same shape & dtype as input)
        """
        if precision == "fp16":
            return kv_tensor

        assert precision in ("int8", "int4"), f"Unknown precision: {precision}"
        qmax = KVQuantizer.QMAX[precision]

        original_dtype = kv_tensor.dtype
        kv_float = kv_tensor.float()

        # Per-token scale: max absolute value per (..., seq_len) position
        # Reduce over head_dim to get per-token scale
        abs_max = kv_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / qmax

        # Quantize → dequantize
        quantized = torch.clamp(torch.round(kv_float / scale), -qmax, qmax)
        dequantized = quantized * scale

        return dequantized.to(original_dtype)


# =============================================================================
# Tier Assignment Strategies
# =============================================================================

def spectrumkv_greedy(budget: float, importance_scores: np.ndarray) -> np.ndarray:
    """
    SpectrumKV Greedy: Assign precision tiers greedily by benefit/cost ratio.
    
    ALL tokens retained. Strategy:
      1. Start all tokens at INT4 (cheapest)
      2. Upgrade highest-importance tokens to INT8 (benefit/cost = 0.17/0.25 = 0.68)
      3. Upgrade top-importance tokens to FP16 (benefit/cost = 0.05/0.50 = 0.10)
    
    At b=0.5: all tokens → INT8 (0.5*1.0 = budget exactly)
    At b=0.3: 20% INT8 + 80% INT4
    At b=0.7: 40% FP16 + 60% INT8
    """
    n = len(importance_scores)
    tiers = np.full(n, "int4", dtype=object)

    if budget <= 0.25:
        return tiers  # Can't afford any upgrades

    ranked = np.argsort(-importance_scores)
    remaining_budget = (budget - 0.25) * n  # Budget above INT4 baseline (FP16 units)

    # Phase 1: Upgrade to INT8 (cost 0.25 FP16-units per token)
    n_int8 = min(n, int(remaining_budget / 0.25))
    for i in range(n_int8):
        tiers[ranked[i]] = "int8"
    remaining_budget -= n_int8 * 0.25

    # Phase 2: Upgrade top INT8 tokens to FP16 (cost 0.50 per token)
    n_fp16 = min(n_int8, max(0, int(remaining_budget / 0.50 + 0.5)))
    for i in range(n_fp16):
        tiers[ranked[i]] = "fp16"

    # Verify budget
    actual = sum(PRECISION_BW[t] for t in tiers) / n
    assert abs(actual - budget) < 0.03, f"Budget mismatch: {actual:.3f} vs {budget:.3f}"

    return tiers


def random_tier(budget: float, n_tokens: int, seed: int = 42) -> np.ndarray:
    """
    Random baseline: randomly assign precision tiers within budget.
    
    Same constraint (total bandwidth = budget), but no importance-based ordering.
    Used to validate that importance-based tier assignment actually matters.
    """
    rng = np.random.RandomState(seed)
    tiers = np.full(n_tokens, "int4", dtype=object)

    if budget <= 0.25:
        return tiers

    remaining_budget = (budget - 0.25) * n_tokens
    perm = rng.permutation(n_tokens)

    n_int8 = min(n_tokens, int(remaining_budget / 0.25))
    for i in range(n_int8):
        tiers[perm[i]] = "int8"
    remaining_budget -= n_int8 * 0.25

    n_fp16 = min(n_int8, max(0, int(remaining_budget / 0.50 + 0.5)))
    for i in range(n_fp16):
        tiers[perm[i]] = "fp16"

    return tiers


def pdtrim_select(n: int, budget: float, sink: int = SINK_COUNT) -> List[int]:
    """PDTrim baseline: keep first sink + last (budget*n - sink) tokens."""
    k = max(int(n * budget), sink + 1)
    selected = set(range(min(sink, n)))
    selected.update(range(max(n - (k - sink), 0), n))
    return sorted(selected)


def sws_select(n: int, budget: float, importance_scores: np.ndarray,
               sink: int = SINK_COUNT) -> List[int]:
    """SWS baseline: score-based top-K selection, sink always included."""
    k = max(int(n * budget), sink + 1)
    selected = set(range(min(sink, n)))
    remaining = k - len(selected)
    for idx in np.argsort(-importance_scores):
        if idx.item() not in selected:
            selected.add(idx.item())
            remaining -= 1
            if remaining <= 0:
                break
    return sorted(selected)


# =============================================================================
# Importance Score Computation
# =============================================================================

def compute_importance_from_attention(attention_weights: torch.Tensor) -> np.ndarray:
    """
    Compute per-token importance from attention weights.
    
    Args:
        attention_weights: (num_layers, batch, num_heads, q_len, k_len)
                          or simplified shapes
    
    Returns:
        scores: (seq_len,) importance score per token
    """
    if attention_weights.dim() == 1:
        return attention_weights.cpu().numpy()
    elif attention_weights.dim() == 2:
        return attention_weights.sum(dim=0).cpu().numpy()
    else:
        # Average across layers, batch, heads; sum across queries
        scores = attention_weights.mean(dim=(0, 1, 2)).sum(dim=0).cpu().numpy()

    # Boost sink tokens
    if len(scores) > SINK_COUNT:
        scores[:SINK_COUNT] = scores.max() * 10

    return scores


def compute_importance_heuristic(seq_len: int, sink: int = SINK_COUNT) -> np.ndarray:
    """
    Heuristic importance: sink + exponential decay + recency boost.
    Used when attention weights can't be extracted from the model.
    """
    scores = np.zeros(seq_len, dtype=np.float64)

    # Sink: high importance
    if sink > 0:
        scores[:sink] = 1.0

    # Exponential decay from sink
    decay = np.exp(-np.arange(seq_len - sink) * 0.01)
    scores[sink:] = decay

    # Recency: linearly increasing for last 20%
    rec_start = int(seq_len * 0.8)
    if rec_start < seq_len:
        recency = np.linspace(0.3, 0.8, seq_len - rec_start)
        scores[rec_start:] = np.maximum(scores[rec_start:], recency)

    # Normalize
    smin, smax = scores.min(), scores.max()
    if smax > smin:
        scores = (scores - smin) / (smax - smin)

    return scores


# =============================================================================
# PPL Evaluation
# =============================================================================

def load_model(model_key: str, hf_path: str):
    """Load model and tokenizer."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_key} from {hf_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[{time.strftime('%H:%M:%S')}] Loaded. VRAM={vram:.1f}GB")
    return model, tokenizer


def get_wikitext2(tokenizer, seq_len=2048, num_samples=5):
    """Load WikiText-2 test data."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        import urllib.request
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        zip_path = "/tmp/wikitext-2-raw.zip"
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('wikitext-2-raw/wiki.test.raw') as f:
                text = f.read().decode('utf-8')

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        if end <= len(input_ids):
            samples.append(input_ids[start:end])
    return samples


def ppl_full_fp16(model, input_ids) -> float:
    """Oracle PPL: full FP16, no modification."""
    device = model.device
    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0).to(device),
            labels=input_ids.unsqueeze(0).to(device),
        )
    return torch.exp(outputs.loss).item()


def ppl_selection_drop(model, input_ids, selected_indices: List[int]) -> float:
    """PPL for selection-based methods: only feed selected tokens with original position_ids."""
    device = model.device
    ix = sorted(selected_indices)
    selected_ids = input_ids[ix].to(device)
    position_ids = torch.tensor(ix, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            selected_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            labels=selected_ids.unsqueeze(0),
        )
    return torch.exp(outputs.loss).item()


def ppl_tiered(model, input_ids, precision_map: np.ndarray) -> float:
    """
    PPL for tiered-precision methods (SpectrumKV, Random_Tier).
    
    Two-pass approach:
      1. Run full forward to get KV cache
      2. Modify KV cache per precision tier (quantize → dequantize)
      3. Re-run forward with modified cache, compute loss
    """
    device = model.device
    n = len(input_ids)

    with torch.no_grad():
        # Pass 1: Get full KV cache
        outputs = model(
            input_ids.unsqueeze(0).to(device),
            use_cache=True,
        )
        past_kv = outputs.past_key_values

    # Modify KV cache per precision tier
    modified_kv = []
    for layer_idx, (key, value) in enumerate(past_kv):
        key_mod = key.clone()
        value_mod = value.clone()

        for t in range(n):
            prec = precision_map[t]
            if prec == "fp16":
                continue  # No modification needed

            # Quantize → dequantize this token's K and V
            key_mod[:, :, t:t+1, :] = KVQuantizer.quantize_dequantize(
                key_mod[:, :, t:t+1, :], prec
            )
            value_mod[:, :, t:t+1, :] = KVQuantizer.quantize_dequantize(
                value_mod[:, :, t:t+1, :], prec
            )

        modified_kv.append((key_mod, value_mod))

    # Pass 2: Compute PPL with modified KV cache
    with torch.no_grad():
        outputs2 = model(
            input_ids.unsqueeze(0).to(device),
            past_key_values=modified_kv,
            use_cache=True,
        )
        logits = outputs2.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[1:].unsqueeze(0).to(device)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        ppl = torch.exp(loss).item()

    # Free memory
    del past_kv, modified_kv, outputs, outputs2
    torch.cuda.empty_cache()

    return ppl


def run_ppl_experiment(model, tokenizer, budgets, seq_len=2048, num_samples=5):
    """Run PPL experiment for all methods and budgets."""
    samples = get_wikitext2(tokenizer, seq_len, num_samples)
    if not samples:
        print("  WARNING: No WikiText-2 samples loaded!")
        return {}

    results = {}

    # Baseline: Full FP16
    print("\n  [1/5] Full FP16 baseline...")
    ppls = []
    for sample in samples:
        ppl = ppl_full_fp16(model, sample)
        ppls.append(ppl)
    baseline_ppl = np.mean(ppls)
    results["full_fp16"] = {"ppl": baseline_ppl, "ppl_ratio": 1.0, "ppl_pct": 0.0}
    print(f"    PPL = {baseline_ppl:.4f}")

    for budget in budgets:
        print(f"\n  --- Budget = {budget} ---")

        for sample_idx, sample in enumerate(samples):
            n = len(sample)
            importance = compute_importance_heuristic(n)

            # 2. PDTrim
            print(f"  [2/5] PDTrim b={budget} sample={sample_idx}...")
            sel = pdtrim_select(n, budget)
            ppl_val = ppl_selection_drop(model, sample, sel)
            key = f"pdtrim_b{budget}"
            if key not in results:
                results[key] = {"ppls": [], "budget": budget}
            results[key]["ppls"].append(ppl_val)

            # 3. SWS
            print(f"  [3/5] SWS b={budget} sample={sample_idx}...")
            sel = sws_select(n, budget, importance)
            ppl_val = ppl_selection_drop(model, sample, sel)
            key = f"sws_b{budget}"
            if key not in results:
                results[key] = {"ppls": [], "budget": budget}
            results[key]["ppls"].append(ppl_val)

            # 4. SpectrumKV Greedy
            print(f"  [4/5] SpectrumKV_Greedy b={budget} sample={sample_idx}...")
            precision_map = spectrumkv_greedy(budget, importance)
            ppl_val = ppl_tiered(model, sample, precision_map)
            key = f"spectrumkv_greedy_b{budget}"
            if key not in results:
                results[key] = {"ppls": [], "budget": budget}
            results[key]["ppls"].append(ppl_val)

            # 5. Random Tier (ablation)
            print(f"  [5/5] Random_Tier b={budget} sample={sample_idx}...")
            precision_map = random_tier(budget, n, seed=42 + sample_idx)
            ppl_val = ppl_tiered(model, sample, precision_map)
            key = f"random_tier_b{budget}"
            if key not in results:
                results[key] = {"ppls": [], "budget": budget}
            results[key]["ppls"].append(ppl_val)

    # Aggregate
    for key, val in results.items():
        if "ppls" in val and val["ppls"]:
            mean_ppl = np.mean(val["ppls"])
            val["ppl"] = round(mean_ppl, 4)
            val["ppl_ratio"] = round(mean_ppl / baseline_ppl, 4)
            val["ppl_pct"] = round((mean_ppl / baseline_ppl - 1.0) * 100, 2)

    return results


# =============================================================================
# NIAH Evaluation
# =============================================================================

def build_niah_prompt(tokenizer, needle: str, depth_frac: float,
                      seq_len: int) -> torch.Tensor:
    """Build a NIAH prompt with needle at given depth fraction."""
    passage = (
        "The development of computational methods has transformed many fields of research. "
        "Researchers analyze large datasets to identify patterns and trends. "
        "Modern algorithms process information efficiently across distributed systems. "
        "Statistical models provide frameworks for understanding complex phenomena. "
        "The integration of technology continues to reshape professional practices. "
    )
    passage_ids = tokenizer(passage, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needle_ids = tokenizer(needle, return_tensors="pt", add_special_tokens=False).input_ids[0]

    available = seq_len - len(needle_ids) - 10
    if available < 100:
        print(f"  WARNING: Very short haystack ({available} tokens)")

    repeats = (available // len(passage_ids)) + 1
    full_passage = passage_ids.repeat(repeats)[:available]

    # Insert needle at depth
    insert_pos = int(depth_frac * len(full_passage))
    input_ids = torch.cat([
        full_passage[:insert_pos],
        needle_ids,
        full_passage[insert_pos:],
    ])

    # Truncate to seq_len
    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]

    return input_ids


def niah_evaluate(model, tokenizer, input_ids, needle_answer: str,
                  selected_indices=None, precision_map=None, max_new_tokens=32):
    """
    Evaluate NIAH: can the model retrieve the needle answer?
    
    Args:
        selected_indices: if provided, only these tokens are kept (selection mode)
        precision_map: if provided, per-token precision (tiered mode)
    """
    device = model.device
    n = len(input_ids)

    with torch.no_grad():
        # Get KV cache
        outputs = model(
            input_ids.unsqueeze(0).to(device),
            use_cache=True,
        )
        past_kv = outputs.past_key_values

        # Modify KV cache based on method
        if selected_indices is not None:
            # Selection mode: zero out non-selected tokens
            sel_set = set(selected_indices)
            modified_kv = []
            for key, value in past_kv:
                key_mod = key.clone()
                value_mod = value.clone()
                for t in range(n):
                    if t not in sel_set:
                        key_mod[:, :, t:t+1, :] = 0
                        value_mod[:, :, t:t+1, :] = 0
                modified_kv.append((key_mod, value_mod))
            past_kv = modified_kv

        elif precision_map is not None:
            # Tiered mode: quantize per precision
            modified_kv = []
            for key, value in past_kv:
                key_mod = key.clone()
                value_mod = value.clone()
                for t in range(n):
                    prec = precision_map[t]
                    if prec != "fp16":
                        key_mod[:, :, t:t+1, :] = KVQuantizer.quantize_dequantize(
                            key_mod[:, :, t:t+1, :], prec
                        )
                        value_mod[:, :, t:t+1, :] = KVQuantizer.quantize_dequantize(
                            value_mod[:, :, t:t+1, :], prec
                        )
                modified_kv.append((key_mod, value_mod))
            past_kv = modified_kv

        # Generate answer
        # Use the last token's position as the query
        query = input_ids[-1:].unsqueeze(0).to(device)
        generated = model.generate(
            query,
            past_key_values=past_kv,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and check if answer is in generated text
    gen_text = tokenizer.decode(generated[0], skip_special_tokens=True).lower()
    success = needle_answer.lower() in gen_text

    del past_kv, outputs
    torch.cuda.empty_cache()

    return success, gen_text


def run_niah_experiment(model, tokenizer, budgets, seq_len=4096):
    """Run NIAH experiment for all methods, budgets, and depths."""
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    needles = [
        ("The secret code is UNICORN7.", "UNICORN7"),
        ("The magic number is 42719.", "42719"),
        ("The hidden word is SPECTRUM.", "SPECTRUM"),
    ]

    results = {}

    for needle_text, needle_answer in needles:
        for depth in depths:
            input_ids = build_niah_prompt(tokenizer, needle_text, depth, seq_len)
            n = len(input_ids)
            importance = compute_importance_heuristic(n)

            for budget in budgets:
                config_key = f"d{depth}_b{budget}"

                # Full FP16
                key = f"full_fp16_{config_key}"
                success, _ = niah_evaluate(model, tokenizer, input_ids, needle_answer)
                if key not in results:
                    results[key] = {"successes": []}
                results[key]["successes"].append(int(success))

                # PDTrim
                sel = pdtrim_select(n, budget)
                key = f"pdtrim_{config_key}"
                success, _ = niah_evaluate(model, tokenizer, input_ids, needle_answer,
                                           selected_indices=sel)
                if key not in results:
                    results[key] = {"successes": []}
                results[key]["successes"].append(int(success))

                # SWS
                sel = sws_select(n, budget, importance)
                key = f"sws_{config_key}"
                success, _ = niah_evaluate(model, tokenizer, input_ids, needle_answer,
                                           selected_indices=sel)
                if key not in results:
                    results[key] = {"successes": []}
                results[key]["successes"].append(int(success))

                # SpectrumKV Greedy
                pmap = spectrumkv_greedy(budget, importance)
                key = f"spectrumkv_greedy_{config_key}"
                success, _ = niah_evaluate(model, tokenizer, input_ids, needle_answer,
                                           precision_map=pmap)
                if key not in results:
                    results[key] = {"successes": []}
                results[key]["successes"].append(int(success))

                # Random Tier (ablation)
                pmap = random_tier(budget, n, seed=int(depth * 1000))
                key = f"random_tier_{config_key}"
                success, _ = niah_evaluate(model, tokenizer, input_ids, needle_answer,
                                           precision_map=pmap)
                if key not in results:
                    results[key] = {"successes": []}
                results[key]["successes"].append(int(success))

    # Aggregate
    for key, val in results.items():
        if val["successes"]:
            val["niah_rate"] = round(np.mean(val["successes"]), 4)

    return results


# =============================================================================
# Tier Distribution Verification
# =============================================================================

def verify_tier_distribution(budget: float, n: int = 1000):
    """Verify that tier assignments match the expected bandwidth budget."""
    importance = compute_importance_heuristic(n)
    tiers = spectrumkv_greedy(budget, importance)

    counts = {"fp16": 0, "int8": 0, "int4": 0}
    for t in tiers:
        counts[t] += 1

    actual_bw = sum(counts[p] * PRECISION_BW[p] for p in counts) / n
    total_tokens = sum(counts.values())

    print(f"  Budget={budget}: FP16={counts['fp16']/n:.2f} "
          f"INT8={counts['int8']/n:.2f} INT4={counts['int4']/n:.2f} "
          f"Total={total_tokens} ActualBW={actual_bw:.3f}")

    assert abs(actual_bw - budget) < 0.03, f"Budget mismatch: {actual_bw:.3f} vs {budget:.3f}"
    assert total_tokens == n, f"Token count mismatch: {total_tokens} vs {n}"

    return counts


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SpectrumKV GPU Experiment")
    parser.add_argument("--model", type=str, default="qwen7b",
                        choices=list(MODELS.keys()))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--ppl_only", action="store_true")
    parser.add_argument("--niah_only", action="store_true")
    parser.add_argument("--niah_seq_len", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify tier distributions, skip GPU")
    args = parser.parse_args()

    print("=" * 70)
    print("SpectrumKV GPU Experiment")
    print("Per-Token Mixed-Precision KV Cache under Bandwidth Constraints")
    print("=" * 70)

    # Verify tier distributions (no GPU needed)
    print("\n### Tier Distribution Verification ###")
    for b in args.budgets:
        verify_tier_distribution(b)
    print("  ✅ All tier distributions verified!")

    if args.verify_only:
        print("\nVerification complete. Exiting.")
        return

    # Load model
    model_key = args.model
    hf_path = MODELS[model_key]
    model, tokenizer = load_model(model_key, hf_path)

    all_results = {
        "model": model_key,
        "hf_path": hf_path,
        "budgets": args.budgets,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # PPL experiment
    if not args.niah_only:
        print("\n### PPL Experiment ###")
        ppl_results = run_ppl_experiment(
            model, tokenizer, args.budgets,
            seq_len=args.seq_len,
        )
        all_results["ppl"] = ppl_results

        # Print PPL summary
        print("\n### PPL Summary ###")
        baseline_ppl = ppl_results.get("full_fp16", {}).get("ppl", 0)
        print(f"  {'Method':<30s} {'PPL':>8s} {'vs Baseline':>12s}")
        print(f"  {'-'*30} {'-'*8} {'-'*12}")
        print(f"  {'Full FP16 (baseline)':<30s} {baseline_ppl:>8.4f} {'—':>12s}")
        for key, val in sorted(ppl_results.items()):
            if key == "full_fp16":
                continue
            if "ppl" in val:
                pct = val.get("ppl_pct", 0)
                print(f"  {key:<30s} {val['ppl']:>8.4f} {ppl:>+10.2f}%")

    # NIAH experiment
    if not args.ppl_only:
        print("\n### NIAH Experiment ###")
        niah_results = run_niah_experiment(
            model, tokenizer, args.budgets,
            seq_len=args.niah_seq_len,
        )
        all_results["niah"] = niah_results

        # Print NIAH summary
        print("\n### NIAH Summary ###")
        for budget in args.budgets:
            print(f"\n  Budget = {budget}:")
            print(f"    {'Method':<20s} {'d=0.1':>6s} {'d=0.25':>6s} {'d=0.5':>6s} {'d=0.75':>6s} {'d=0.9':>6s}")
            for method in ["full_fp16", "pdtrim", "sws", "spectrumkv_greedy", "random_tier"]:
                row = []
                for d in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    key = f"{method}_d{d}_b{budget}"
                    rate = niah_results.get(key, {}).get("niah_rate", 0)
                    row.append(f"{rate:>5.1%}")
                print(f"    {method:<20s} {' '.join(row)}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(args.output_dir, f"spectrumkv_{model_key}_{timestamp}.json")

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\n{'='*70}")
    print(f"Results saved to {result_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
