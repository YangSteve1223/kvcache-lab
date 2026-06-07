#!/usr/bin/env python3
"""
Experiment 3: Quantization Error Analysis for KV Cache
=======================================================
Measures INT8/INT4 quantization error across multiple dimensions:
  - Per-layer: which layers are most sensitive to quantization
  - Per-position: sink / middle / recent token groups
  - Per-importance-tier: high / medium / low importance groups
  - K vs V: Key vs Value error comparison
  - Per-head: variance of error across attention heads

No PPL or NIAH evaluation — pure error metrics.

Usage:
    python exp3_quant_error.py --model qwen7b --seq_len 2048 --num_samples 3
    python exp3_quant_error.py --model mistral7b --seq_len 4096
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectrumkv_utils import (
    KVQuantizer,
    load_model,
    get_wikitext2,
    compute_importance_heuristic,
    PRECISION_BW,
    MODELS,
    SINK_COUNT,
    json_serialize,
)

# =============================================================================
# Position / Importance Grouping Helpers
# =============================================================================

def classify_position(idx: int, seq_len: int, sink: int = SINK_COUNT) -> str:
    """Classify token position into sink / middle / recent."""
    if idx < sink:
        return "sink"
    recent_start = int(seq_len * 0.8)
    if idx >= recent_start:
        return "recent"
    return "middle"


def classify_importance(score: float) -> str:
    """Classify importance score into high / medium / low tier."""
    if score >= 0.66:
        return "high"
    elif score >= 0.33:
        return "medium"
    return "low"


# =============================================================================
# Core Analysis
# =============================================================================

def analyze_quantization(
    model,
    tokenizer,
    seq_len: int,
    num_samples: int,
    token_sample_step: int = 8,
    model_name: str = "unknown",
    output_dir: str = "experiment_results_quant",
) -> Dict:
    """
    Run quantization error analysis on WikiText-2 samples.

    Parameters
    ----------
    model, tokenizer : loaded model & tokenizer
    seq_len : sequence length for each sample
    num_samples : how many WikiText-2 samples to process
    token_sample_step : analyze every N-th token for per-position
                        (1 = all, higher = faster but coarser)

    Returns
    -------
    Dict with keys: per_layer, per_position, per_importance, kv_comparison,
                    per_head, sensitive_layers, importance_correlation, metadata
    """
    samples = get_wikitext2(tokenizer, seq_len, num_samples)
    actual_samples = len(samples)
    print(f"[{time.strftime('%H:%M:%S')}] Got {actual_samples} samples (requested {num_samples})")

    # Accumulators
    # per_layer:  {layer_idx: {prec: {kv: {metric: [values]}}}}
    per_layer_acc: Dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    # ── Per-position: {group: {prec: {kv: {metric: [values]}}}} ──────
    per_position_acc: Dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    # ── Per-importance: {tier: {prec: {kv: {metric: [values]}}}} ────
    per_importance_acc: Dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    # kv_comparison: {kv: {prec: {metric: [values]}}}
    kv_comparison_acc: Dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    # per_head: {layer_idx: {kv: {prec: {metric_per_head: list}}}}
    per_head_acc: Dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    num_layers = None

    for sample_idx, sample in enumerate(samples):
        print(f"\n[{time.strftime('%H:%M:%S')}] ── Sample {sample_idx + 1}/{actual_samples} ──")

        with torch.no_grad():
            outputs = model(sample.unsqueeze(0).to(model.device), use_cache=True)
            past_kv = outputs.past_key_values

        # Extract (key, value) pairs from cache (handles DynamicCache/DynamicLayer)
        kv_layers = []
        if hasattr(past_kv, 'layers'):
            for layer in past_kv.layers:
                if hasattr(layer, 'keys'):
                    kv_layers.append((layer.keys, layer.values))
                elif isinstance(layer, (tuple, list)):
                    kv_layers.append((layer[0], layer[1]))
        elif hasattr(past_kv, 'key_cache'):
            kv_layers = [(past_kv.key_cache[i], past_kv.value_cache[i])
                         for i in range(len(past_kv.key_cache))]
        else:
            kv_layers = [(item[0], item[1]) for item in past_kv]

        if num_layers is None:
            num_layers = len(kv_layers)
            print(f"  Model has {num_layers} layers")

        importance = compute_importance_heuristic(len(sample))

        for layer_idx, (key, value) in enumerate(kv_layers):
            # key/value shape: (batch=1, num_heads, seq_len, head_dim)
            n_tokens = key.shape[2]
            n_heads = key.shape[1]
            head_dim = key.shape[3]

            # Ensure token count matches importance array length
            imp = importance[:n_tokens] if len(importance) >= n_tokens else importance

            for prec in ["int8", "int4"]:
                # ── Whole-tensor errors (per-layer & K/V comparison) ───────
                k_err = KVQuantizer.compute_error(key, prec)
                v_err = KVQuantizer.compute_error(value, prec)

                for kv_label, err in [("key", k_err), ("value", v_err)]:
                    for metric, val in err.items():
                        per_layer_acc[layer_idx][prec][kv_label][metric].append(val)
                        kv_comparison_acc[kv_label][prec][metric].append(val)

                # ── CPU copy: shared across per-head and per-position ──────
                # BUG FIX: Only copy once, reuse for per-head AND per-position
                need_per_head = (sample_idx < 2)
                if need_per_head:
                    key_cpu = key.float().cpu()
                    value_cpu = value.float().cpu()

                    for h in range(n_heads):
                        k_h_err = KVQuantizer.compute_error(key_cpu[:, h:h+1, :, :], prec)
                        v_h_err = KVQuantizer.compute_error(value_cpu[:, h:h+1, :, :], prec)
                        for kv_label, err in [("key", k_h_err), ("value", v_h_err)]:
                            for metric, val in err.items():
                                per_head_acc[layer_idx][kv_label][prec][metric].append(val)

                # Reuse or create CPU copy for per-position analysis
                if not need_per_head:
                    key_cpu = key.float().cpu()
                    value_cpu = value.float().cpu()

                # ── Per-position analysis (sampled tokens) ────────────────
                token_indices = list(range(0, n_tokens, token_sample_step))
                if (n_tokens - 1) not in token_indices:
                    token_indices.append(n_tokens - 1)

                for t in token_indices:
                    pos_group = classify_position(t, n_tokens)

                    k_t_err = KVQuantizer.compute_error(key_cpu[:, :, t:t+1, :], prec)
                    v_t_err = KVQuantizer.compute_error(value_cpu[:, :, t:t+1, :], prec)

                    # BUG FIX: Separate K and V errors (consistent with per_layer)
                    for kv_label, err in [("key", k_t_err), ("value", v_t_err)]:
                        for metric, val in err.items():
                            per_position_acc[pos_group][prec][kv_label][metric].append(val)

                    # ── Per-importance analysis ────────────────────────────
                    imp_t = imp[t] if t < len(imp) else 0.0
                    imp_tier = classify_importance(imp_t)

                    for kv_label, err in [("key", k_t_err), ("value", v_t_err)]:
                        for metric, val in err.items():
                            per_importance_acc[imp_tier][prec][kv_label][metric].append(val)

                del key_cpu, value_cpu

            # Progress
            if (layer_idx + 1) % 10 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx + 1}/{num_layers} done")

        del past_kv, outputs
        torch.cuda.empty_cache()

    # ── Save checkpoint before aggregation ──────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"exp3_checkpoint_{model_name}.json")
    ckpt_payload = {
        "tag": "raw_accumulators",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "model": model_name,
        "seq_len": seq_len,
        "num_samples": actual_samples,
        "num_layers": num_layers,
    }
    with open(ckpt_path, "w") as f:
        json.dump({"metadata": ckpt_payload}, f, indent=2, default=json_serialize)
    print(f"  [CHECKPOINT] Raw accumulators saved to {ckpt_path}")

    # ── Aggregate results ──────────────────────────────────────────────────
    def _mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def _std(vals: List[float]) -> float:
        return float(np.std(vals)) if vals else 0.0

    # Per-layer summary
    per_layer_summary = []
    for layer_idx in sorted(per_layer_acc.keys()):
        entry = {"layer_idx": layer_idx}
        for prec in ["int8", "int4"]:
            for kv_label in ["key", "value"]:
                for metric in ["rel_l2", "max_abs", "cosine_sim"]:
                    key = f"{prec}_{kv_label}_{metric}"
                    entry[key] = _mean(per_layer_acc[layer_idx][prec][kv_label][metric])
        per_layer_summary.append(entry)

    # Per-position summary (with K/V separation)
    per_position_summary = []
    for group in ["sink", "middle", "recent"]:
        entry = {"position_group": group}
        for prec in ["int8", "int4"]:
            for kv_label in ["key", "value"]:
                for metric in ["rel_l2", "max_abs", "cosine_sim"]:
                    key = f"{prec}_{kv_label}_{metric}"
                    entry[key] = _mean(per_position_acc[group][prec][kv_label][metric])
        per_position_summary.append(entry)

    # Per-importance summary (with K/V separation)
    per_importance_summary = []
    for tier in ["high", "medium", "low"]:
        entry = {"importance_tier": tier}
        for prec in ["int8", "int4"]:
            for kv_label in ["key", "value"]:
                for metric in ["rel_l2", "max_abs", "cosine_sim"]:
                    key = f"{prec}_{kv_label}_{metric}"
                    entry[key] = _mean(per_importance_acc[tier][prec][kv_label][metric])
        per_importance_summary.append(entry)

    # K vs V comparison
    kv_comparison_summary = []
    for kv_label in ["key", "value"]:
        entry = {"kv_type": kv_label}
        for prec in ["int8", "int4"]:
            for metric in ["rel_l2", "max_abs", "cosine_sim"]:
                key = f"{prec}_{metric}"
                entry[key] = _mean(kv_comparison_acc[kv_label][prec][metric])
        kv_comparison_summary.append(entry)

    # Per-head variance
    per_head_summary = []
    for layer_idx in sorted(per_head_acc.keys()):
        entry = {"layer_idx": layer_idx}
        for kv_label in ["key", "value"]:
            for prec in ["int8", "int4"]:
                for metric in ["rel_l2", "max_abs", "cosine_sim"]:
                    vals = per_head_acc[layer_idx][kv_label][prec][metric]
                    key_mean = f"{prec}_{kv_label}_{metric}_mean"
                    key_std = f"{prec}_{kv_label}_{metric}_std"
                    entry[key_mean] = _mean(vals)
                    entry[key_std] = _std(vals)
        per_head_summary.append(entry)

    # ── Sensitive layers: INT4 rel_l2 > threshold ─────────────────────────
    SENSITIVITY_THRESHOLD = 0.15
    sensitive_layers = []
    for entry in per_layer_summary:
        int4_k_l2 = entry.get("int4_key_rel_l2", 0)
        int4_v_l2 = entry.get("int4_value_rel_l2", 0)
        max_l2 = max(int4_k_l2, int4_v_l2)
        if max_l2 > SENSITIVITY_THRESHOLD:
            sensitive_layers.append({
                "layer_idx": entry["layer_idx"],
                "int4_key_rel_l2": int4_k_l2,
                "int4_value_rel_l2": int4_v_l2,
                "max_rel_l2": max_l2,
            })

    # ── Importance–error correlation ───────────────────────────────────────
    # Collect per-token importance & error from per_importance_acc aggregated
    # We re-derive a simple correlation: avg error in each tier vs tier rank
    importance_error_corr = {}
    for prec in ["int8", "int4"]:
        tier_order = {"high": 3, "medium": 2, "low": 1}
        tier_errors = []
        tier_ranks = []
        for tier in ["high", "medium", "low"]:
            avg_err = _mean(per_importance_acc[tier][prec]["rel_l2"])
            tier_errors.append(avg_err)
            tier_ranks.append(tier_order[tier])
        if len(tier_errors) >= 2 and np.std(tier_errors) > 0:
            corr = float(np.corrcoef(tier_ranks, tier_errors)[0, 1])
            # BUG FIX: Handle NaN from zero-variance inputs
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        importance_error_corr[prec] = {
            "pearson_r": corr,
            "tier_errors": dict(zip(["high", "medium", "low"], tier_errors)),
        }

    # ── Assemble final results ────────────────────────────────────────────
    results = {
        "metadata": {
            "model": model_name,
            "seq_len": seq_len,
            "num_samples": actual_samples,
            "num_layers": num_layers,
            "sink_count": SINK_COUNT,
            "sensitivity_threshold": SENSITIVITY_THRESHOLD,
            "token_sample_step": token_sample_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_layer": per_layer_summary,
        "per_position": per_position_summary,
        "per_importance": per_importance_summary,
        "kv_comparison": kv_comparison_summary,
        "per_head": per_head_summary,
        "sensitive_layers": sensitive_layers,
        "importance_error_correlation": importance_error_corr,
    }

    return results


# =============================================================================
# Pretty Printing
# =============================================================================

def print_summary(results: Dict):
    """Print human-readable summary tables."""

    SEP = "=" * 90
    THIN = "-" * 90

    # ── Per-layer table ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PER-LAYER QUANTIZATION ERROR (averaged over samples)")
    print(THIN)
    header = f"{'Layer':>5} │ {'INT8 K L2':>10} {'INT4 K L2':>10} │ {'INT8 V L2':>10} {'INT4 V L2':>10} │ {'INT8 K cos':>10} {'INT4 K cos':>10}"
    print(header)
    print(THIN)
    for e in results["per_layer"]:
        row = (
            f"{e['layer_idx']:>5} │ "
            f"{e['int8_key_rel_l2']:>10.4f} {e['int4_key_rel_l2']:>10.4f} │ "
            f"{e['int8_value_rel_l2']:>10.4f} {e['int4_value_rel_l2']:>10.4f} │ "
            f"{e['int8_key_cosine_sim']:>10.4f} {e['int4_key_cosine_sim']:>10.4f}"
        )
        print(row)

    # ── Per-position table ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PER-POSITION-GROUP ERROR (K and V separated)")
    print(THIN)
    header = f"{'Group':>8} │ {'INT8 K L2':>10} {'INT4 K L2':>10} │ {'INT8 V L2':>10} {'INT4 V L2':>10} │ {'INT4 K cos':>10} {'INT4 V cos':>10}"
    print(header)
    print(THIN)
    for e in results["per_position"]:
        row = (
            f"{e['position_group']:>8} │ "
            f"{e['int8_key_rel_l2']:>10.5f} {e['int4_key_rel_l2']:>10.5f} │ "
            f"{e['int8_value_rel_l2']:>10.5f} {e['int4_value_rel_l2']:>10.5f} │ "
            f"{e['int4_key_cosine_sim']:>10.5f} {e['int4_value_cosine_sim']:>10.5f}"
        )
        print(row)

    # ── K vs V comparison ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("KEY vs VALUE QUANTIZATION ERROR")
    print(THIN)
    header = f"{'Type':>6} │ {'INT8 L2':>10} {'INT4 L2':>10} │ {'INT8 MaxAbs':>12} {'INT4 MaxAbs':>12} │ {'INT8 Cos':>10} {'INT4 Cos':>10}"
    print(header)
    print(THIN)
    for e in results["kv_comparison"]:
        row = (
            f"{e['kv_type']:>6} │ "
            f"{e['int8_rel_l2']:>10.5f} {e['int4_rel_l2']:>10.5f} │ "
            f"{e['int8_max_abs']:>12.6f} {e['int4_max_abs']:>12.6f} │ "
            f"{e['int8_cosine_sim']:>10.5f} {e['int4_cosine_sim']:>10.5f}"
        )
        print(row)

    # ── Per-importance table ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PER-IMPORTANCE-TIER ERROR (K and V separated)")
    print(THIN)
    header = f"{'Tier':>8} │ {'INT8 K L2':>10} {'INT4 K L2':>10} │ {'INT8 V L2':>10} {'INT4 V L2':>10} │ {'INT4 K cos':>10} {'INT4 V cos':>10}"
    print(header)
    print(THIN)
    for e in results["per_importance"]:
        row = (
            f"{e['importance_tier']:>8} │ "
            f"{e['int8_key_rel_l2']:>10.5f} {e['int4_key_rel_l2']:>10.5f} │ "
            f"{e['int8_value_rel_l2']:>10.5f} {e['int4_value_rel_l2']:>10.5f} │ "
            f"{e['int4_key_cosine_sim']:>10.5f} {e['int4_value_cosine_sim']:>10.5f}"
        )
        print(row)

    # ── Sensitive layers ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    threshold = results["metadata"]["sensitivity_threshold"]
    sens = results["sensitive_layers"]
    if sens:
        print(f"SENSITIVE LAYERS (INT4 rel_l2 > {threshold})")
        print(THIN)
        header = f"{'Layer':>5} │ {'INT4 Key L2':>12} {'INT4 Value L2':>14} │ {'Max L2':>10}"
        print(header)
        print(THIN)
        for s in sens:
            row = (
                f"{s['layer_idx']:>5} │ "
                f"{s['int4_key_rel_l2']:>12.4f} {s['int4_value_rel_l2']:>14.4f} │ "
                f"{s['max_rel_l2']:>10.4f}"
            )
            print(row)
    else:
        print(f"No layers exceed INT4 rel_l2 threshold of {threshold}")

    # ── Importance–error correlation ───────────────────────────────────────
    print(f"\n{SEP}")
    print("IMPORTANCE–ERROR CORRELATION (Pearson r)")
    print(THIN)
    for prec, info in results["importance_error_correlation"].items():
        print(f"  {prec.upper()}: r = {info['pearson_r']:.4f}")
        for tier, err in info["tier_errors"].items():
            print(f"    {tier:>8}: avg rel_l2 = {err:.6f}")

    print(SEP)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3: KV Cache Quantization Error Analysis"
    )
    parser.add_argument(
        "--model", type=str, default="qwen7b",
        choices=list(MODELS.keys()),
        help="Model key (default: qwen7b)",
    )
    parser.add_argument(
        "--seq_len", type=int, default=2048,
        help="Sequence length per sample (default: 2048)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3,
        help="Number of WikiText-2 samples (default: 3)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment_results_quant",
        help="Directory for JSON output (default: experiment_results_quant)",
    )
    parser.add_argument(
        "--token_sample_step", type=int, default=8,
        help="Analyze every N-th token for per-position (default: 8, 1=all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Experiment 3: KV Cache Quantization Error Analysis")
    print("=" * 60)
    print(f"  Model       : {args.model} ({MODELS[args.model]})")
    print(f"  Seq length  : {args.seq_len}")
    print(f"  Num samples : {args.num_samples}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Token step  : {args.token_sample_step}")
    print()

    # Load model
    model, tokenizer = load_model(args.model)

    # Run analysis
    results = analyze_quantization(
        model, tokenizer,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        token_sample_step=args.token_sample_step,
        model_name=args.model,
        output_dir=args.output_dir,
    )

    # Print summary
    print_summary(results)

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = f"quant_error_{args.model}_seqlen{args.seq_len}_ns{args.num_samples}.json"
    out_path = os.path.join(args.output_dir, out_name)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=json_serialize)

    print(f"\n[{time.strftime('%H:%M:%S')}] Results saved to: {out_path}")
