#!/usr/bin/env python3
"""
Experiment 4: Per-Layer Budget Differentiation Scan
=====================================================
Core hypothesis: Lower layers (syntactic) tolerate lower precision,
higher layers (semantic) require higher precision.

Sweeps beta parameter controlling differentiation strength:
  beta=0.0  → uniform (no differentiation, baseline)
  beta=0.02 → moderate (default from simulation)
  beta=0.05 → aggressive differentiation

Total transmission bandwidth is preserved across all configs:
  mean(layer_budgets) ≈ base_budget
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Ensure the shared utils module is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from spectrumkv_utils import (
    KVQuantizer,
    PRECISION_BW,
    compute_importance_heuristic,
    compute_layer_budgets,
    get_wikitext2,
    json_serialize,
    load_model,
    ppl_full_fp16,
    spectrumkv_greedy,
)


# =============================================================================
# Per-Layer Differentiated PPL
# =============================================================================

def ppl_per_layer_tiered(model, input_ids, base_budget, beta,
                         importance_scores):
    """PPL with per-layer differentiated budget.

    Key difference from ppl_tiered: each layer gets its own budget
    derived from compute_layer_budgets(), and therefore its own
    precision_map.  The total bandwidth (mean of per-layer budgets)
    is constrained to ≈ base_budget.

    Uses k_proj/v_proj hooks to inject quantized KV during a single
    forward pass (avoids the two-pass context-doubling bug).

    Returns:
        ppl: float – perplexity
        layer_budgets: list[float] – budget assigned to each layer
        tier_fractions: list[dict] – {fp16, int8, int4} fraction per layer
    """
    device = model.device
    n = len(input_ids)
    num_layers = model.config.num_hidden_layers

    # 1. Compute per-layer budgets
    layer_budgets = compute_layer_budgets(num_layers, base_budget, beta)

    # 2. Compute per-layer precision maps
    layer_precision_maps = []
    tier_fractions = []

    for layer_idx in range(num_layers):
        layer_budget = layer_budgets[layer_idx]
        layer_precision_map = spectrumkv_greedy(layer_budget, importance_scores)
        layer_precision_maps.append(layer_precision_map)

        # Record tier fraction stats
        n_tokens = len(layer_precision_map)
        frac = {"fp16": 0, "int8": 0, "int4": 0}
        for t in layer_precision_map:
            frac[t] += 1
        frac = {k: round(v / n_tokens, 4) for k, v in frac.items()}
        tier_fractions.append(frac)

    # 3. Find all k_proj and v_proj modules and register hooks
    # Each layer's k_proj/v_proj gets its own precision_map
    hook_handles = []
    
    # Map module name → layer index
    layer_kv_modules = {}  # {module: layer_idx}
    for name, module in model.named_modules():
        if name.endswith('.k_proj') or name.endswith('.v_proj'):
            # Extract layer index from name like "model.layers.5.self_attn.k_proj"
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_kv_modules[module] = layer_idx
                    except ValueError:
                        pass
    
    if not layer_kv_modules:
        raise RuntimeError(
            "Could not find k_proj/v_proj modules for per-layer PPL. "
            "Model architecture may not be supported."
        )
    
    def make_per_layer_quantize_hook(prec_map):
        """Hook that applies per-token QDQ using a specific layer's precision map."""
        int8_indices = [t for t in range(len(prec_map)) if prec_map[t] == "int8"]
        int4_indices = [t for t in range(len(prec_map)) if prec_map[t] == "int4"]
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                main = output[0]
            else:
                main = output
            
            seq_len = main.shape[1]
            
            for prec, indices in [("int8", int8_indices), ("int4", int4_indices)]:
                valid_idx = [i for i in indices if i < seq_len]
                if not valid_idx:
                    continue
                idx_tensor = torch.tensor(valid_idx, device=main.device)
                token_slice = main[:, idx_tensor, :]
                main[:, idx_tensor, :] = KVQuantizer.quantize_dequantize(
                    token_slice, prec
                )
            
            return output
        return hook_fn
    
    # Register hooks — each k_proj/v_proj module gets its layer's precision map
    for module, layer_idx in layer_kv_modules.items():
        prec_map = layer_precision_maps[layer_idx]
        handle = module.register_forward_hook(make_per_layer_quantize_hook(prec_map))
        hook_handles.append(handle)
    
    try:
        with torch.no_grad():
            outputs = model(
                input_ids.unsqueeze(0).to(device),
                labels=input_ids.unsqueeze(0).to(device),
            )
        ppl = torch.exp(outputs.loss).item()
    finally:
        for h in hook_handles:
            h.remove()
        torch.cuda.empty_cache()

    return ppl, layer_budgets.tolist(), tier_fractions


# =============================================================================
# Main experiment driver
# =============================================================================

def run_experiment(args):
    """Run per-layer budget differentiation sweep."""

    # ---- Load checkpoint if resuming --------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"exp4_checkpoint_{args.model}.json")
    all_results = []
    completed_keys = set()
    baseline_ppl = None  # Initialize before checkpoint loading
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt_data = json.load(f)
        all_results = ckpt_data.get("results", [])
        baseline_ppl = ckpt_data.get("baseline_ppl", None)
        for r in all_results:
            completed_keys.add((r["beta"], r["budget"]))
        print(f"  Loaded checkpoint: {len(completed_keys)} configs done, baseline_ppl={baseline_ppl}")

    # ---- Load model & data ------------------------------------------------
    model, tokenizer = load_model(args.model)
    samples = get_wikitext2(tokenizer, seq_len=args.seq_len,
                            num_samples=args.num_samples)
    num_layers = model.config.num_hidden_layers

    # ---- Baseline: full FP16 PPL ------------------------------------------
    if baseline_ppl is None:
        print(f"\n[Phase 0] Computing FP16 baseline PPL ...")
        baseline_ppls = []
        for si, sample in enumerate(samples):
            bppl = ppl_full_fp16(model, sample)
            baseline_ppls.append(bppl)
            print(f"  sample {si}: FP16 PPL = {bppl:.4f}")
        baseline_ppl = float(np.mean(baseline_ppls))
        print(f"  >>> Mean FP16 baseline PPL = {baseline_ppl:.4f}\n")
    else:
        print(f"  Baseline PPL from checkpoint: {baseline_ppl:.4f}\n")

    # ---- Compute importance scores (shared across layers) ------------------
    importance_scores = compute_importance_heuristic(args.seq_len)

    # ---- Sweep beta × budget ----------------------------------------------
    betas = [float(b) for b in args.betas]
    budgets = [float(b) for b in args.budgets]

    summary_grid = {}  # (beta, budget) -> ppl

    total_configs = len(betas) * len(budgets)
    cfg_idx = 0

    for beta in betas:
        for budget in budgets:
            cfg_idx += 1

            # Skip if already completed
            if (beta, budget) in completed_keys:
                print(f"[{cfg_idx}/{total_configs}] SKIP (checkpoint): beta={beta:.3f} budget={budget:.2f}")
                continue

            tag = f"beta={beta:.3f} budget={budget:.2f}"
            print(f"[{cfg_idx}/{total_configs}] {tag}")

            # Compute per-layer budgets for info
            layer_bds = compute_layer_budgets(num_layers, budget, beta)
            mean_bd = float(np.mean(layer_bds))
            print(f"  layer budgets: min={layer_bds.min():.3f}  "
                  f"max={layer_bds.max():.3f}  mean={mean_bd:.3f}")

            # Per-sample PPL
            ppls = []
            all_tier_fracs = None

            for si, sample in enumerate(samples):
                ppl_val, lb_list, tier_fracs = ppl_per_layer_tiered(
                    model, sample, budget, beta, importance_scores,
                )
                ppls.append(ppl_val)
                # BUG FIX: Average tier fractions across samples
                if all_tier_fracs is None:
                    all_tier_fracs = tier_fracs
                else:
                    # Accumulate fractions for averaging
                    for li, tf in enumerate(tier_fracs):
                        for k in all_tier_fracs[li]:
                            all_tier_fracs[li][k] += tf[k]

                print(f"  sample {si}: PPL = {ppl_val:.4f}")

            # Average tier fractions across samples
            if all_tier_fracs is not None and len(samples) > 1:
                for li in range(len(all_tier_fracs)):
                    for k in all_tier_fracs[li]:
                        all_tier_fracs[li][k] /= len(samples)

            mean_ppl = float(np.mean(ppls))
            ppl_delta_pct = (mean_ppl - baseline_ppl) / baseline_ppl * 100.0

            print(f"  >>> Mean PPL = {mean_ppl:.4f}  "
                  f"(Δ = {ppl_delta_pct:+.2f}%)\n")

            summary_grid[(beta, budget)] = mean_ppl

            result = {
                "beta": beta,
                "budget": budget,
                "model": args.model,
                "ppl": round(mean_ppl, 4),
                "ppl_delta_pct": round(ppl_delta_pct, 2),
                "layer_budgets": [round(x, 4) for x in lb_list],
                "tier_fractions_per_layer": all_tier_fracs,
                "baseline_ppl": round(baseline_ppl, 4),
                "num_layers": num_layers,
                "seq_len": args.seq_len,
                "num_samples": args.num_samples,
            }
            all_results.append(result)

            # Save checkpoint after each config
            ckpt_payload = {
                "results": all_results,
                "baseline_ppl": baseline_ppl,
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            }
            with open(ckpt_path, "w") as f:
                json.dump(ckpt_payload, f, indent=2, default=json_serialize)

    # ---- Save detailed JSON results ---------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "exp4_layer_budget_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=json_serialize)
    print(f"Results saved to {out_path}")

    # ---- Print summary table ----------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY TABLE: PPL by (beta, budget)")
    print("=" * 72)

    # Header
    header = f"{'beta':>8s}"
    for b in budgets:
        header += f" | budget={b:.2f}"
    print(header)
    print("-" * len(header))

    for beta in betas:
        row = f"{beta:8.3f}"
        for budget in budgets:
            ppl_val = summary_grid.get((beta, budget), float("nan"))
            row += f" | {ppl_val:11.4f}"
        print(row)

    print("=" * 72)

    # ---- PPL delta table --------------------------------------------------
    print("\n  SUMMARY TABLE: PPL Δ% by (beta, budget)")
    print("=" * 72)

    header2 = f"{'beta':>8s}"
    for b in budgets:
        header2 += f" | budget={b:.2f}"
    print(header2)
    print("-" * len(header2))

    for beta in betas:
        row = f"{beta:8.3f}"
        for budget in budgets:
            key = (beta, budget)
            match = [r for r in all_results
                     if r["beta"] == beta and r["budget"] == budget]
            if match:
                delta = match[0]["ppl_delta_pct"]
                row += f" | {delta:+10.2f}%"
            else:
                row += f" | {'N/A':>10s}"
        print(row)

    print("=" * 72)

    # ---- Find optimal beta ------------------------------------------------
    print("\n  OPTIMAL BETA SELECTION")
    print("-" * 40)

    for budget in budgets:
        best_beta = None
        best_ppl = float("inf")
        for beta in betas:
            key = (beta, budget)
            if key in summary_grid and summary_grid[key] < best_ppl:
                best_ppl = summary_grid[key]
                best_beta = beta
        if best_beta is not None:
            delta = (best_ppl - baseline_ppl) / baseline_ppl * 100.0
            print(f"  budget={budget:.2f}:  best_beta={best_beta:.3f}  "
                  f"PPL={best_ppl:.4f}  (Δ={delta:+.2f}%)")

    # ---- Overall best config ----------------------------------------------
    overall_best = min(all_results, key=lambda r: r["ppl"])
    print(f"\n  >>> Overall best: beta={overall_best['beta']:.3f}  "
          f"budget={overall_best['budget']:.2f}  "
          f"PPL={overall_best['ppl']:.4f}  "
          f"(Δ={overall_best['ppl_delta_pct']:+.2f}%)")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return all_results


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp4: Per-Layer Budget Differentiation Scan"
    )
    parser.add_argument("--model", type=str, default="qwen7b",
                        help="Model key (default: qwen7b)")
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
                        help="Beta values to sweep (default: 0.0..0.05 step 0.01)")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7],
                        help="Budget values to test (default: 0.3 0.5 0.7)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length (default: 2048)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of WikiText-2 samples (default: 3)")
    parser.add_argument("--output_dir", type=str,
                        default="experiment_results_layer",
                        help="Output directory (default: experiment_results_layer)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
