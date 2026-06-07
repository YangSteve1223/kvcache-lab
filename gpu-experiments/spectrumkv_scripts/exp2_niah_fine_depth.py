#!/usr/bin/env python3
"""
SpectrumKV Experiment 2: NIAH Fine-Grained Depth Scan
======================================================
Evaluates retrieval accuracy of KV cache methods across fine-grained
depth positions and budget levels using the Needle-in-a-Haystack protocol.

Methods: Full_FP16, PDTrim, SWS, SpectrumKV_Greedy,
         SpectrumKV_Balanced, SpectrumKV_SinkProtect, Random_Tier

Budgets: [0.3, 0.4, 0.5, 0.6, 0.7]
Depths:  19 points from 0.05 to 0.95 (step 0.05)
Needles: 5 (from spectrumkv_utils.NEEDLES)
Seq len: 4096
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import shared components
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectrumkv_utils import (
    KVQuantizer,
    MODELS,
    NEEDLES,
    PRECISION_BW,
    SINK_COUNT,
    build_niah_prompt,
    compute_importance_heuristic,
    json_serialize,
    load_model,
    pdtrim_select,
    random_tier,
    spectrumkv_balanced,
    spectrumkv_greedy,
    spectrumkv_sink_protect,
    sws_select,
)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
METHODS = [
    "Full_FP16",
    "PDTrim",
    "SWS",
    "SpectrumKV_Greedy",
    "SpectrumKV_Balanced",
    "SpectrumKV_SinkProtect",
    "Random_Tier",
]

BUDGETS = [0.3, 0.4, 0.5, 0.6, 0.7]

DEPTHS = [round(i * 0.05, 2) for i in range(1, 20)]  # 0.05 .. 0.95


# ---------------------------------------------------------------------------
# NIAH evaluation
# ---------------------------------------------------------------------------
def _extract_kv_layer(past_kv, layer_idx):
    """Extract (key, value) from any cache format for a given layer."""
    if hasattr(past_kv, 'layers'):
        layer = past_kv.layers[layer_idx]
        # DynamicLayer has .keys and .values attributes
        if hasattr(layer, 'keys'):
            return layer.keys, layer.values
        elif isinstance(layer, (tuple, list)):
            return layer[0], layer[1]
        else:
            raise TypeError(f"Unknown cache layer type: {type(layer).__name__}")
    elif hasattr(past_kv, 'key_cache'):
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    else:
        item = past_kv[layer_idx]
        return item[0], item[1]


def _num_kv_layers(past_kv):
    """Get number of KV cache layers."""
    if hasattr(past_kv, 'layers'):
        return len(past_kv.layers)
    elif hasattr(past_kv, 'key_cache'):
        return len(past_kv.key_cache)
    else:
        return len(past_kv)


def niah_evaluate(model, tokenizer, input_ids, needle_answer,
                  selected_indices=None, precision_map=None,
                  max_new_tokens=32):
    """Run a single NIAH trial with optional KV cache modification.

    Args:
        model: The causal LM.
        tokenizer: Corresponding tokenizer.
        input_ids: 1-D token tensor (the haystack + needle).
        needle_answer: The answer string to look for in the generation.
        selected_indices: If provided, zero out non-selected KV entries
                          (for PDTrim / SWS).
        precision_map: If provided, per-token precision tiers for
                       quantize-dequantize (for SpectrumKV / Random_Tier).
        max_new_tokens: Generation length.

    Returns:
        (success: bool, gen_text: str)
    """
    device = model.device
    n = len(input_ids)

    # --- Forward pass to populate KV cache ---
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0).to(device), use_cache=True)
        past_kv = outputs.past_key_values

    # --- Modify KV cache in place (supports DynamicCache) ---
    num_layers = _num_kv_layers(past_kv)

    if selected_indices is not None:
        # Selection-based: zero out non-selected tokens
        sel_mask = np.zeros(n, dtype=bool)
        sel_mask[list(selected_indices)] = True
        drop_mask = np.where(~sel_mask)[0]
        if len(drop_mask) > 0:
            for layer_idx in range(num_layers):
                key, value = _extract_kv_layer(past_kv, layer_idx)
                key[:, :, drop_mask, :] = 0
                value[:, :, drop_mask, :] = 0

    elif precision_map is not None:
        # Tiered-precision: quantize-dequantize per token
        int8_mask = np.array([precision_map[t] == "int8" for t in range(n)])
        int4_mask = np.array([precision_map[t] == "int4" for t in range(n)])
        for layer_idx in range(num_layers):
            key, value = _extract_kv_layer(past_kv, layer_idx)
            if int8_mask.any():
                idx = np.where(int8_mask)[0]
                key[:, :, idx, :] = KVQuantizer.quantize_dequantize(key[:, :, idx, :], "int8")
                value[:, :, idx, :] = KVQuantizer.quantize_dequantize(value[:, :, idx, :], "int8")
            if int4_mask.any():
                idx = np.where(int4_mask)[0]
                key[:, :, idx, :] = KVQuantizer.quantize_dequantize(key[:, :, idx, :], "int4")
                value[:, :, idx, :] = KVQuantizer.quantize_dequantize(value[:, :, idx, :], "int4")
    # else: Full_FP16 — no modification

    # --- Generate answer via manual auto-regressive decoding ---
    # NOTE: model.generate() with past_key_values has compatibility issues
    # across transformers versions. Manual step-by-step with model() works.
    with torch.no_grad():
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)  # [1]
        generated_ids = [next_token.item()]

        for _ in range(max_new_tokens - 1):
            step_out = model(
                input_ids=next_token.unsqueeze(0),  # [1, 1]
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = step_out.past_key_values
            next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1)
            generated_ids.append(next_token.item())

            if next_token.item() == tokenizer.eos_token_id:
                break

    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True).lower()
    success = needle_answer.lower() in gen_text

    del past_kv, outputs
    torch.cuda.empty_cache()
    return success, gen_text


# ---------------------------------------------------------------------------
# Run one method for one (budget, needle, depth) combination
# ---------------------------------------------------------------------------
def run_trial(model, tokenizer, method, budget, needle_text, needle_answer,
              depth, seq_len):
    """Execute a single NIAH trial and return the result dict."""

    input_ids = build_niah_prompt(tokenizer, needle_text, depth, seq_len)
    n = len(input_ids)

    selected_indices = None
    precision_map = None

    if method == "Full_FP16":
        pass  # no modification

    elif method == "PDTrim":
        selected_indices = pdtrim_select(n, budget, sink=SINK_COUNT)

    elif method == "SWS":
        importance = compute_importance_heuristic(n, sink=SINK_COUNT)
        selected_indices = sws_select(n, budget, importance, sink=SINK_COUNT)

    elif method == "SpectrumKV_Greedy":
        importance = compute_importance_heuristic(n, sink=SINK_COUNT)
        precision_map = spectrumkv_greedy(budget, importance)

    elif method == "SpectrumKV_Balanced":
        importance = compute_importance_heuristic(n, sink=SINK_COUNT)
        precision_map = spectrumkv_balanced(budget, importance)

    elif method == "SpectrumKV_SinkProtect":
        importance = compute_importance_heuristic(n, sink=SINK_COUNT)
        precision_map = spectrumkv_sink_protect(budget, importance, sink=SINK_COUNT)

    elif method == "Random_Tier":
        # BUG FIX: seed must vary per trial for statistical independence
        seed = hash((budget, depth, needle_text)) % (2**31)
        precision_map = random_tier(budget, n, seed=seed)

    else:
        raise ValueError(f"Unknown method: {method}")

    success, gen_text = niah_evaluate(
        model, tokenizer, input_ids, needle_answer,
        selected_indices=selected_indices,
        precision_map=precision_map,
    )

    return {
        "needle": needle_text,
        "depth": depth,
        "budget": budget,
        "method": method,
        "success": success,
        "response": gen_text[:120],  # truncate for storage
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def _save_ckpt(trials, output_dir, model_key, tag="checkpoint"):
    """Save intermediate trials to allow resume."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"exp2_checkpoint_{model_key}.json")
    payload = {
        "experiment": "exp2_niah_fine_depth",
        "tag": tag,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": model_key,
        "trials": trials,
    }
    with open(ckpt_path, "w") as f:
        json.dump(payload, f, indent=2, default=json_serialize)
    return ckpt_path


def _load_ckpt(output_dir, model_key):
    """Load previous checkpoint if exists."""
    ckpt_path = os.path.join(output_dir, f"exp2_checkpoint_{model_key}.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            data = json.load(f)
        trials = data.get("trials", [])
        print(f"  Loaded checkpoint: {len(trials)} trials from {ckpt_path}")
        return trials, ckpt_path
    return [], None


def run_experiment(model_key, seq_len, output_dir):
    """Run the full NIAH fine-depth scan for one model."""

    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_model(model_key)

    # Load checkpoint if resuming
    all_trials, _ = _load_ckpt(output_dir, model_key)
    completed_keys = set()
    for t in all_trials:
        key = (t["method"], t["budget"], t["depth"], t["needle"])
        completed_keys.add(key)
    if completed_keys:
        print(f"  Resuming: {len(completed_keys)} trials already done")

    # Full_FP16 doesn't depend on budget — run once per (depth, needle)
    total_fp16 = len(DEPTHS) * len(NEEDLES)
    total_other = len(BUDGETS) * len(DEPTHS) * len(NEEDLES) * (len(METHODS) - 1)
    total = total_fp16 + total_other
    done = len(completed_keys)
    t0 = time.time() if done == 0 else time.time()
    ckpt_counter = 0

    # Full_FP16 (budget-independent, run once)
    for depth in DEPTHS:
        for needle_text, needle_answer in NEEDLES:
            trial_key = ("Full_FP16", 0.0, depth, needle_text)
            if trial_key in completed_keys:
                continue
            result = run_trial(
                model, tokenizer, "Full_FP16", budget=0.0,
                needle_text=needle_text, needle_answer=needle_answer,
                depth=depth, seq_len=seq_len,
            )
            all_trials.append(result)
            completed_keys.add(trial_key)
            done += 1
            ckpt_counter += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / max(done - len(completed_keys) + done, 1) * (total - done)
                print(f"  [{done}/{total}] {elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m")
            if ckpt_counter % 100 == 0:
                _save_ckpt(all_trials, output_dir, model_key)

    # Budget-dependent methods
    for budget in BUDGETS:
        for method in METHODS:
            if method == "Full_FP16":
                continue
            for depth in DEPTHS:
                for needle_text, needle_answer in NEEDLES:
                    trial_key = (method, budget, depth, needle_text)
                    if trial_key in completed_keys:
                        done += 1
                        continue
                    result = run_trial(
                        model, tokenizer, method, budget,
                        needle_text, needle_answer, depth, seq_len,
                    )
                    all_trials.append(result)
                    completed_keys.add(trial_key)
                    done += 1
                    ckpt_counter += 1
                    if done % 50 == 0:
                        elapsed = time.time() - t0
                        eta = elapsed / done * (total - done)
                        print(f"  [{done}/{total}] {elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m")
                    if ckpt_counter % 100 == 0:
                        ckpt_path = _save_ckpt(all_trials, output_dir, model_key,
                                               tag=f"progress_{done}")
                        print(f"  [CHECKPOINT] Saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Aggregate: retrieval_rate per (method, budget, depth)
    # ------------------------------------------------------------------
    # Group successes
    agg = {}  # (method, budget, depth) -> [successes]
    for t in all_trials:
        key = (t["method"], t["budget"], t["depth"])
        agg.setdefault(key, []).append(t["success"])

    # Compute rates
    retrieval_rates = {}
    for (method, budget, depth), succs in sorted(agg.items()):
        retrieval_rates[(method, budget, depth)] = sum(succs) / len(succs)

    # ------------------------------------------------------------------
    # Save raw + aggregated results
    # ------------------------------------------------------------------
    results = {
        "model": model_key,
        "seq_len": seq_len,
        "budgets": BUDGETS,
        "depths": DEPTHS,
        "methods": METHODS,
        "num_needles": len(NEEDLES),
        "trials": all_trials,
        "retrieval_rates": {
            f"{method}|{budget}|{depth}": json_serialize(rate)
            for (method, budget, depth), rate in retrieval_rates.items()
        },
    }

    out_path = os.path.join(output_dir, f"niah_fine_depth_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=json_serialize)
    print(f"\n[SAVE] Results -> {out_path}")

    # ------------------------------------------------------------------
    # Print summary table: grouped by budget
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"  NIAH Fine-Depth Scan Summary — {model_key}  (seq_len={seq_len})")
    print("=" * 100)

    for budget in BUDGETS:
        print(f"\n  Budget = {budget:.1f}")
        header = f"  {'Depth':>6s}"
        for method in METHODS:
            header += f"  {method:>20s}"
        print(header)
        print("  " + "-" * (6 + 22 * len(METHODS)))

        for depth in DEPTHS:
            row = f"  {depth:>6.2f}"
            for method in METHODS:
                rate = retrieval_rates.get((method, budget, depth), 0.0)
                row += f"  {rate:>20.3f}"
            print(row)

        # Budget-level average across depths
        avg_row = f"  {'AVG':>6s}"
        for method in METHODS:
            rates = [retrieval_rates.get((method, budget, d), 0.0)
                     for d in DEPTHS]
            avg_row += f"  {np.mean(rates):>20.3f}"
        print(avg_row)
        print()

    # Also compute overall average per method across all budgets & depths
    print("  --- Overall Averages (across all budgets & depths) ---")
    for method in METHODS:
        all_rates = [retrieval_rates.get((method, b, d), 0.0)
                     for b in BUDGETS for d in DEPTHS]
        print(f"    {method:>22s}: {np.mean(all_rates):.4f}")

    print("\n" + "=" * 100)

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SpectrumKV Exp2: NIAH Fine-Grained Depth Scan"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model key to evaluate (e.g. qwen7b, mistral7b, gemma9b). "
             "If omitted, runs all three.",
    )
    parser.add_argument(
        "--niah_seq_len", type=int, default=4096,
        help="Sequence length for NIAH haystack (default: 4096)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./kvcache-lab/gpu-experiments/spectrumkv_scripts/results",
        help="Directory for result JSON files",
    )
    args = parser.parse_args()

    if args.model is not None:
        models_to_run = [args.model]
    else:
        models_to_run = ["qwen7b", "mistral7b", "gemma9b"]

    print("=" * 60)
    print(" SpectrumKV Exp2: NIAH Fine-Grained Depth Scan")
    print("=" * 60)
    print(f"  Models     : {models_to_run}")
    print(f"  Seq len    : {args.niah_seq_len}")
    print(f"  Budgets    : {BUDGETS}")
    print(f"  Depths     : {DEPTHS}  ({len(DEPTHS)} points)")
    print(f"  Needles    : {len(NEEDLES)}")
    print(f"  Methods    : {METHODS}")
    total_per_model = (len(BUDGETS) * len(DEPTHS) * len(NEEDLES) * len(METHODS))
    print(f"  Trials/model: {total_per_model}")
    print(f"  Output     : {args.output_dir}")
    print("=" * 60)

    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"  WARNING: Unknown model '{model_key}', skipping.")
            continue
        print(f"\n{'#' * 60}")
        print(f"  Running model: {model_key}")
        print(f"{'#' * 60}")
        run_experiment(model_key, args.niah_seq_len, args.output_dir)

    print("\n[DONE] All models completed.")


if __name__ == "__main__":
    main()
