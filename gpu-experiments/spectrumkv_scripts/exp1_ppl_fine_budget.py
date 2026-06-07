#!/usr/bin/env python3
"""
Exp1: SpectrumKV PPL Fine-Budget Sweep
========================================
Scans budgets at 0.05 step over [0.25..0.75] for 8 methods × 2 seq_lens × 5 samples.
Output: per-config PPL + delta% + tier fractions, saved as JSON + pretty table.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from spectrumkv_utils import (
    KVQuantizer,
    spectrumkv_greedy,
    spectrumkv_balanced,
    spectrumkv_sink_protect,
    random_tier,
    pdtrim_select,
    sws_select,
    compute_importance_heuristic,
    load_model,
    get_wikitext2,
    ppl_full_fp16,
    ppl_selection_drop,
    ppl_tiered,
    PRECISION_BW,
    MODELS,
    SINK_COUNT,
    json_serialize,
)

# =============================================================================
# Value-Aware Importance (SWS_ValueAware)
# =============================================================================

def compute_importance_value_aware(seq_len: int, sink: int = SINK_COUNT) -> np.ndarray:
    """Value-aware importance: boosted recency for last 20% of tokens.

    Unlike the default heuristic which adds a mild recency boost,
    this variant multiplies the score of the most recent 20% tokens by 1.5,
    emphasizing that recent KV pairs are disproportionately important for
    next-token prediction.
    """
    scores = compute_importance_heuristic(seq_len, sink)
    rec_start = int(seq_len * 0.8)
    if rec_start < seq_len:
        scores[rec_start:] *= 1.5
    # Renormalize to [0, 1]
    smin, smax = scores.min(), scores.max()
    if smax > smin:
        scores = (scores - smin) / (smax - smin)
    return scores


# =============================================================================
# Method dispatch
# =============================================================================

METHODS = [
    "Full_FP16",
    "Uniform_INT4",      # All tokens INT4 (budget=0.25), no differentiation
    "Uniform_INT8",      # All tokens INT8 (budget=0.50), no differentiation
    "PDTrim",
    "SWS_Original",
    "SWS_ValueAware",
    "SpectrumKV_Greedy",
    "SpectrumKV_Balanced",
    "SpectrumKV_SinkProtect",
    "Random_Tier",
]

# Budget-independent methods: run once per seq_len (not per budget)
BUDGET_INDEPENDENT = {"Full_FP16", "Uniform_INT4", "Uniform_INT8"}


def _tier_fractions(precision_map: np.ndarray) -> dict:
    """Compute fraction of tokens at each precision tier."""
    n = len(precision_map)
    if n == 0:
        return {"fp16_frac": 0.0, "int8_frac": 0.0, "int4_frac": 0.0}
    fp16 = np.sum(precision_map == "fp16") / n
    int8 = np.sum(precision_map == "int8") / n
    int4 = np.sum(precision_map == "int4") / n
    return {"fp16_frac": float(fp16), "int8_frac": float(int8), "int4_frac": float(int4)}


def run_single(
    method: str,
    model,
    tokenizer,
    input_ids,
    budget: float,
    seq_len: int,
    sample_idx: int,
) -> dict:
    """Run one method on one sample, return {ppl, fp16_frac, int8_frac, int4_frac}."""
    n = len(input_ids)
    result = {"ppl": float("inf"), "fp16_frac": 0.0, "int8_frac": 0.0, "int4_frac": 0.0}

    if method == "Full_FP16":
        ppl = ppl_full_fp16(model, input_ids)
        result["ppl"] = ppl
        result["fp16_frac"] = 1.0
        result["int8_frac"] = 0.0
        result["int4_frac"] = 0.0
        return result

    # Uniform quantization baselines (no per-token differentiation)
    if method == "Uniform_INT4":
        # All tokens at INT4 → effective budget = 0.25
        precision_map = np.full(n, "int4", dtype=object)
        ppl = ppl_tiered(model, input_ids, precision_map)
        result["ppl"] = ppl
        result["fp16_frac"] = 0.0
        result["int8_frac"] = 0.0
        result["int4_frac"] = 1.0
        return result

    if method == "Uniform_INT8":
        # All tokens at INT8 → effective budget = 0.50
        precision_map = np.full(n, "int8", dtype=object)
        ppl = ppl_tiered(model, input_ids, precision_map)
        result["ppl"] = ppl
        result["fp16_frac"] = 0.0
        result["int8_frac"] = 1.0
        result["int4_frac"] = 0.0
        return result

    # Selection-based methods
    if method == "PDTrim":
        selected = pdtrim_select(n, budget, sink=SINK_COUNT)
        ppl = ppl_selection_drop(model, input_ids, selected)
        # For selection methods, retained tokens are effectively fp16
        kept_ratio = len(selected) / n
        result["ppl"] = ppl
        result["fp16_frac"] = kept_ratio
        result["int8_frac"] = 0.0
        result["int4_frac"] = 1.0 - kept_ratio
        return result

    if method == "SWS_Original":
        scores = compute_importance_heuristic(n, SINK_COUNT)
        selected = sws_select(n, budget, scores, sink=SINK_COUNT)
        ppl = ppl_selection_drop(model, input_ids, selected)
        kept_ratio = len(selected) / n
        result["ppl"] = ppl
        result["fp16_frac"] = kept_ratio
        result["int8_frac"] = 0.0
        result["int4_frac"] = 1.0 - kept_ratio
        return result

    if method == "SWS_ValueAware":
        scores = compute_importance_value_aware(n, SINK_COUNT)
        selected = sws_select(n, budget, scores, sink=SINK_COUNT)
        ppl = ppl_selection_drop(model, input_ids, selected)
        kept_ratio = len(selected) / n
        result["ppl"] = ppl
        result["fp16_frac"] = kept_ratio
        result["int8_frac"] = 0.0
        result["int4_frac"] = 1.0 - kept_ratio
        return result

    # Tiered-precision methods
    if method == "SpectrumKV_Greedy":
        scores = compute_importance_heuristic(n, SINK_COUNT)
        precision_map = spectrumkv_greedy(budget, scores)
        ppl = ppl_tiered(model, input_ids, precision_map)
        fracs = _tier_fractions(precision_map)
        result["ppl"] = ppl
        result.update(fracs)
        return result

    if method == "SpectrumKV_Balanced":
        scores = compute_importance_heuristic(n, SINK_COUNT)
        precision_map = spectrumkv_balanced(budget, scores)
        ppl = ppl_tiered(model, input_ids, precision_map)
        fracs = _tier_fractions(precision_map)
        result["ppl"] = ppl
        result.update(fracs)
        return result

    if method == "SpectrumKV_SinkProtect":
        scores = compute_importance_heuristic(n, SINK_COUNT)
        precision_map = spectrumkv_sink_protect(budget, scores, sink=SINK_COUNT)
        ppl = ppl_tiered(model, input_ids, precision_map)
        fracs = _tier_fractions(precision_map)
        result["ppl"] = ppl
        result.update(fracs)
        return result

    if method == "Random_Tier":
        precision_map = random_tier(budget, n, seed=42 + sample_idx)
        ppl = ppl_tiered(model, input_ids, precision_map)
        fracs = _tier_fractions(precision_map)
        result["ppl"] = ppl
        result.update(fracs)
        return result

    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Main experiment loop
# =============================================================================

def _save_checkpoint(results, output_dir, model_key, tag="checkpoint"):
    """Save intermediate results to allow resume after interruption."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"exp1_checkpoint_{model_key}.json")
    payload = {
        "experiment": "exp1_ppl_fine_budget",
        "tag": tag,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": model_key,
        "results": results,
    }
    with open(ckpt_path, "w") as f:
        json.dump(payload, f, indent=2, default=json_serialize)
    return ckpt_path


def _load_checkpoint(output_dir, model_key):
    """Load previous checkpoint if exists, return results list."""
    ckpt_path = os.path.join(output_dir, f"exp1_checkpoint_{model_key}.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            data = json.load(f)
        results = data.get("results", [])
        print(f"  Loaded checkpoint: {len(results)} results from {ckpt_path}")
        return results, ckpt_path
    return [], None


def run_ppl_experiment(model, tokenizer, model_key, budgets, seq_lens, num_samples,
                       output_dir="./results"):
    """Run full sweep with checkpoint support. Return results list."""
    # Load checkpoint if resuming
    results, ckpt_path = _load_checkpoint(output_dir, model_key)
    completed_keys = set()
    if results:
        for r in results:
            # Create unique key for each completed config
            key = (r.get("method"), r.get("budget"), r.get("seq_len"))
            completed_keys.add(key)
        print(f"  Resuming from checkpoint: {len(completed_keys)} configs already done")

    # Count total configs
    n_budget_dep = len(budgets) * len(seq_lens) * (len(METHODS) - len(BUDGET_INDEPENDENT)) * num_samples
    n_budget_indep = len(BUDGET_INDEPENDENT) * len(seq_lens) * num_samples
    total = n_budget_dep + n_budget_indep
    done = len(completed_keys)
    ckpt_counter = 0

    for seq_len in seq_lens:
        print(f"\n{'='*70}")
        print(f"  seq_len = {seq_len}")
        print(f"{'='*70}")

        samples = get_wikitext2(tokenizer, seq_len=seq_len, num_samples=num_samples)
        if len(samples) == 0:
            print(f"  WARNING: no samples for seq_len={seq_len}, skipping")
            continue

        # Budget-independent methods
        for method in sorted(BUDGET_INDEPENDENT):
            result_key = (method, 1.0 if method == "Full_FP16" else
                          0.25 if method == "Uniform_INT4" else 0.50, seq_len)
            if result_key in completed_keys:
                print(f"  SKIP (checkpoint): {method} seq={seq_len}")
                continue

            ppls = []
            for si, input_ids in enumerate(samples):
                done += 1
                print(f"  [{done}/{total}] {method:25s}  seq={seq_len} sample={si}")
                r = run_single(method, model, tokenizer, input_ids,
                               budget=0.0, seq_len=seq_len, sample_idx=si)
                ppls.append(r["ppl"])

            mean_ppl = float(np.mean(ppls))
            # Find FP16 baseline for delta calculation
            fp16_key = ("Full_FP16", 1.0, seq_len)
            fp16_mean = None
            for r in results:
                if (r.get("method") == "Full_FP16" and r.get("seq_len") == seq_len):
                    fp16_mean = r["mean_ppl"]
                    break
            if method == "Full_FP16":
                fp16_mean = mean_ppl
            ppl_delta_pct = ((mean_ppl / fp16_mean) - 1.0) * 100.0 if fp16_mean else 0.0

            budget_val = 1.0 if method == "Full_FP16" else (
                0.25 if method == "Uniform_INT4" else 0.50)
            frac_map = {
                "Full_FP16": (1.0, 0.0, 0.0),
                "Uniform_INT4": (0.0, 0.0, 1.0),
                "Uniform_INT8": (0.0, 1.0, 0.0),
            }
            fp16_f, int8_f, int4_f = frac_map[method]

            entry = {
                "method": method,
                "budget": budget_val,
                "seq_len": seq_len,
                "mean_ppl": mean_ppl,
                "ppl_delta_pct": ppl_delta_pct,
                "fp16_frac": fp16_f,
                "int8_frac": int8_f,
                "int4_frac": int4_f,
            }
            results.append(entry)
            completed_keys.add(result_key)
            ckpt_counter += 1

        # Find FP16 baseline for this seq_len
        fp16_mean = None
        for r in results:
            if r.get("method") == "Full_FP16" and r.get("seq_len") == seq_len:
                fp16_mean = r["mean_ppl"]
                break

        for budget in budgets:
            assert budget - 0.25 > -0.05, f"Budget {budget} below minimum 0.25"
            for method in METHODS:
                if method in BUDGET_INDEPENDENT:
                    continue  # Already done above

                result_key = (method, budget, seq_len)
                if result_key in completed_keys:
                    done += num_samples
                    print(f"  SKIP (checkpoint): {method} b={budget:.2f} seq={seq_len}")
                    continue

                ppls = []
                agg_fracs = {"fp16_frac": 0.0, "int8_frac": 0.0, "int4_frac": 0.0}

                for si, input_ids in enumerate(samples):
                    done += 1
                    print(
                        f"  [{done}/{total}] {method:25s}  "
                        f"budget={budget:.2f} seq={seq_len} sample={si}"
                    )
                    r = run_single(method, model, tokenizer, input_ids,
                                   budget, seq_len, si)
                    ppls.append(r["ppl"])
                    agg_fracs["fp16_frac"] += r["fp16_frac"]
                    agg_fracs["int8_frac"] += r["int8_frac"]
                    agg_fracs["int4_frac"] += r["int4_frac"]

                mean_ppl = float(np.mean(ppls))
                ppl_delta_pct = (mean_ppl / fp16_mean - 1.0) * 100.0 if fp16_mean else 0.0
                for k in agg_fracs:
                    agg_fracs[k] /= len(samples)

                results.append({
                    "method": method,
                    "budget": budget,
                    "seq_len": seq_len,
                    "mean_ppl": mean_ppl,
                    "ppl_delta_pct": ppl_delta_pct,
                    **agg_fracs,
                })
                completed_keys.add(result_key)
                ckpt_counter += 1

                # Save checkpoint every 20 configs
                if ckpt_counter % 20 == 0:
                    ckpt_path = _save_checkpoint(results, output_dir, model_key,
                                                 tag=f"progress_{done}")
                    print(f"  [CHECKPOINT] Saved to {ckpt_path}")

    # Final checkpoint
    _save_checkpoint(results, output_dir, model_key, tag="final")
    return results


# =============================================================================
# Pretty table
# =============================================================================

def print_summary_table(results):
    """Print a formatted summary table grouped by seq_len."""
    seq_lens = sorted(set(r["seq_len"] for r in results))

    for sl in seq_lens:
        rows = [r for r in results if r["seq_len"] == sl]
        print(f"\n{'='*100}")
        print(f"  PPL Results  |  seq_len = {sl}")
        print(f"{'='*100}")
        header = (
            f"{'Method':<25s} {'Budget':>7s} {'Mean PPL':>10s} "
            f"{'ΔPPL%':>8s} {'FP16%':>7s} {'INT8%':>7s} {'INT4%':>7s}"
        )
        print(header)
        print("-" * 100)

        # Full_FP16 first, then sorted by method then budget
        fp16_row = [r for r in rows if r["method"] == "Full_FP16"]
        other_rows = sorted(
            [r for r in rows if r["method"] != "Full_FP16"],
            key=lambda r: (r["method"], r["budget"]),
        )
        ordered = fp16_row + other_rows

        for r in ordered:
            budget_str = f"{r['budget']:.2f}" if r["method"] != "Full_FP16" else "1.00"
            print(
                f"{r['method']:<25s} {budget_str:>7s} {r['mean_ppl']:>10.4f} "
                f"{r['ppl_delta_pct']:>+7.2f}% {r['fp16_frac']:>6.1%} "
                f"{r['int8_frac']:>6.1%} {r['int4_frac']:>6.1%}"
            )

        print()


# =============================================================================
# CLI
# =============================================================================

DEFAULT_BUDGETS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
DEFAULT_SEQ_LENS = [2048, 4096]
DEFAULT_NUM_SAMPLES = 5
DEFAULT_MODEL = "qwen7b"


def main():
    parser = argparse.ArgumentParser(
        description="Exp1: SpectrumKV PPL fine-budget sweep across methods and seq lengths"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model key (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS,
        help="Budget values to sweep (default: 0.25..0.75 step 0.05)",
    )
    parser.add_argument(
        "--seq_lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS,
        help="Sequence lengths (default: 2048 4096)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=DEFAULT_NUM_SAMPLES,
        help=f"WikiText-2 samples per config (default: {DEFAULT_NUM_SAMPLES})",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Directory for JSON output (default: ./results)",
    )
    args = parser.parse_args()

    # Validate budgets
    for b in args.budgets:
        assert b - 0.25 > -0.05, f"Budget {b} is below minimum 0.25 (with 0.05 tolerance)"

    print(f"{'='*70}")
    print(f"  Exp1: SpectrumKV PPL Fine-Budget Sweep")
    print(f"{'='*70}")
    print(f"  Model:       {args.model} ({MODELS[args.model]})")
    print(f"  Budgets:     {args.budgets}")
    print(f"  Seq lens:    {args.seq_lens}")
    print(f"  Samples:     {args.num_samples}")
    print(f"  Methods:     {METHODS}")
    print(f"  Output dir:  {args.output_dir}")
    print()

    # Load model
    model, tokenizer = load_model(args.model)

    # Run experiment
    t0 = time.time()
    results = run_ppl_experiment(
        model, tokenizer, args.model,
        args.budgets, args.seq_lens, args.num_samples,
        output_dir=args.output_dir,
    )
    elapsed = time.time() - t0
    print(f"\nExperiment completed in {elapsed/60:.1f} minutes")

    # Print summary
    print_summary_table(results)

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir, f"exp1_ppl_fine_budget_{args.model}_{ts}.json"
    )
    payload = {
        "experiment": "exp1_ppl_fine_budget",
        "timestamp": ts,
        "model": args.model,
        "model_path": MODELS[args.model],
        "budgets": [float(b) for b in args.budgets],
        "seq_lens": args.seq_lens,
        "num_samples": args.num_samples,
        "methods": METHODS,
        "elapsed_sec": elapsed,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=json_serialize)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
