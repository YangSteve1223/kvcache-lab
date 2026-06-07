#!/usr/bin/env python3
"""
SpectrumKV Simulation v3 (Corrected)
=====================================
SpectrumKV: ALL tokens retained with per-token mixed-precision tiers.
- Selection methods: keep budget% tokens at FP16, drop rest → NIAH=0 for dropped
- SpectrumKV: keep ALL tokens, assign FP16/INT8/INT4 tiers within budget → NIAH>0 for all

NIAH: precision-dependent quality (FP16=1.0, INT8=0.95, INT4=0.78).
      0.78 for INT4 is a simulation assumption — actual INT4 KV quality
      depends on model architecture and must be validated on GPU.
PPL: selection=evicted_mass model; SpectrumKV=calibrated quantization noise

All results are SIMULATION ONLY with synthetic attention distributions.
"""

import json
import numpy as np
import os
from collections import defaultdict

SEED = 42
np.random.seed(SEED)

# ============ Configuration ============
MODELS = {
    "qwen-7b":   {"layers": 28, "locality": "medium"},
    "mistral-7b": {"layers": 32, "locality": "low"},
    "gemma-9b":  {"layers": 42, "locality": "high"},
}
SEQ_LENS = [2048, 4096]
BUDGETS = [0.3, 0.5, 0.7]
DEPTH_FRACS = [0.1, 0.25, 0.5, 0.75, 0.9]
SINK = 4
N_RUNS = 10
SAMPLE_LAYERS = 5  # Sample 5 layers per model

# Precision quality factors (NIAH retrieval quality)
QUALITY = {16: 1.0, 8: 0.95, 4: 0.78}

# Precision bandwidth cost (relative to FP16)
COST = {16: 1.0, 8: 0.5, 4: 0.25}

# PPL calibration: % increase when ALL tokens at this precision
# Based on real quantization results: INT8 ~1-3%, INT4 ~8-15%
PPL_FULL_TIER = {8: 0.025, 4: 0.12}  # fraction increase

RESULTS_PATH = "experiment_logs/qcbm_corrected_v3.json"
FIGURES_DIR = "paper/figures"

# ============ Attention Generation ============
def gen_attn(n, sink=SINK):
    """Realistic attention: sink + recency + global + spikes."""
    s = np.zeros(n)
    s[:sink] = np.random.uniform(0.05, 0.12, size=sink)  # Realistic sink: 5-12% per token
    
    r = np.zeros(n)
    r_str = np.random.uniform(0.3, 0.6)
    r_tail = np.random.uniform(50, 120)
    for j in range(n):
        r[j] = r_str * np.exp(-(n - 1 - j) / r_tail)
    
    g = np.random.uniform(0.002, 0.008, n)
    n_sp = np.random.randint(3, 8)
    sp = np.random.choice(n, n_sp, replace=False)
    sp_mask = np.zeros(n)
    sp_mask[sp] = np.random.uniform(0.03, 0.12, n_sp)
    
    raw = np.maximum(s + r + g + sp_mask, 1e-12)
    return raw / raw.sum()


def gen_locality(n_layers, profile):
    if profile == "low":
        base = np.linspace(0.7, 0.2, n_layers)
    elif profile == "high":
        base = np.linspace(0.3, 0.8, n_layers)
    else:
        base = np.linspace(0.5, 0.5, n_layers)
    return np.clip(base + np.random.uniform(-0.05, 0.05, n_layers), 0.05, 0.95)


# ============ Scoring Functions ============
EPS = 1e-8

def score_original(attn, n, **kw):
    """Original SWS: I_j = A_j * exp(-λ * Δ_j) — KNOWN TO COLLAPSE TO FRONT+BACK"""
    lam = kw.get('lam', 0.005)
    delta = np.arange(n, dtype=np.float64)
    delta = n - 1 - delta
    return attn * np.exp(-lam * delta)

def score_value_aware(attn, n, **kw):
    """CAOTE-style: odds ratio of attention with weaker decay"""
    lam = kw.get('lam', 0.005)
    a_norm = attn / (attn.max() + EPS)
    delta = np.arange(n, dtype=np.float64)
    delta = n - 1 - delta
    return (a_norm / (1.0 - a_norm + EPS)) * np.exp(-lam * 0.3 * delta)

def score_attn_only(attn, n, **kw):
    """Pure attention weight — no distance decay"""
    return attn.copy()

def score_layer_mod(attn, n, li=0, nl=1, loc=None, **kw):
    """Attention-only + per-layer locality modulation"""
    l_val = loc[li] if loc is not None and li < len(loc) else 0.5
    jn = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    bonus = np.sin(np.pi * jn) * 0.3 * (1.0 - l_val)
    return attn * (1.0 + bonus)


# ============ Selection-Based Methods ============
def sel_pdtrim(n, budget):
    """PDTrim baseline: sink + recent window"""
    k = max(int(n * budget), SINK + 1)
    selected = set(range(SINK))
    selected.update(range(n - (k - SINK), n))
    return sorted(selected)

def sel_topk(scores, budget):
    """Score-based top-K selection"""
    n = len(scores)
    k = max(int(n * budget), SINK + 1)
    selected = set(range(SINK))
    remaining = k - SINK
    for idx in np.argsort(-scores):
        if idx.item() not in selected:
            selected.add(idx.item())
            remaining -= 1
            if remaining <= 0:
                break
    return sorted(selected)


# ============ SpectrumKV Tier Assignment ============
def spectrumkv_greedy(scores, budget):
    """
    SpectrumKV: ALL tokens retained. Greedy tier assignment.
    
    Strategy: Start all at INT4, upgrade highest-score tokens:
    1. INT4→INT8 first (best benefit/cost ratio: 0.17/0.25 = 0.68)
    2. INT8→FP16 second (benefit/cost: 0.05/0.50 = 0.10)
    
    At b=0.5: all tokens → INT8 (exact budget match: 0.5*1.0 = 0.5)
    At b=0.3: 20% INT8 + 80% INT4
    At b=0.7: 40% FP16 + 60% INT8
    """
    n = len(scores)
    tiers = np.full(n, 4)  # All start at INT4
    
    if budget <= 0.25:
        return tiers  # Edge case: can't afford upgrades
    
    remaining = (budget - 0.25) * n  # Budget above INT4 baseline (in FP16-token units)
    ranked = np.argsort(-scores)
    
    # Phase 1: Upgrade to INT8 (cost 0.25 per token in FP16 units)
    n_int8 = min(n, int(remaining / 0.25))
    for i in range(n_int8):
        tiers[ranked[i]] = 8
    remaining -= n_int8 * 0.25
    
    # Phase 2: Upgrade top INT8 tokens to FP16 (cost 0.50 per token)
    n_fp16 = min(n_int8, max(0, int(remaining / 0.50 + 0.5)))
    for i in range(n_fp16):
        tiers[ranked[i]] = 16
    
    return tiers


def spectrumkv_balanced(scores, budget):
    """
    SpectrumKV Balanced: ALL tokens retained. Proportional tier split.
    
    Tries to spread across FP16/INT8/INT4 proportionally.
    Falls back to FP16+INT8 when budget is too high for INT4.
    """
    n = len(scores)
    ranked = np.argsort(-scores)
    tiers = np.full(n, 4)
    
    if budget <= 0.25:
        return tiers
    
    # Constraint: f*1.0 + m*0.5 + q*0.25 = budget, f+m+q = 1
    # → 3f + m = 4*budget - 1
    R = 4 * budget - 1  # target for 3f + m
    
    if R <= 0:
        return tiers  # All INT4
    
    # Ideal balanced: f:m:q ≈ 1:2:2 (based on b=0.5 → 0.2:0.4:0.4)
    f = R / 5
    m = 2 * R / 5
    q = 1 - f - m
    
    if q < 0:
        # Budget too high for three tiers → FP16 + INT8 only
        # f + 0.5m = budget, f+m=1 → f = 2*budget - 1
        q = 0
        f = 2 * budget - 1
        m = 1 - f
    elif f < 0:
        f = 0
        m = 4 * budget - 1
        q = 1 - m
    
    # Ensure non-negative
    f = max(0, min(1, f))
    m = max(0, min(1 - f, m))
    q = 1 - f - m
    
    n_fp16 = min(int(n * f), n)
    n_int8 = min(int(n * m), n - n_fp16)
    
    for i in range(n_fp16):
        tiers[ranked[i]] = 16
    for i in range(n_fp16, n_fp16 + n_int8):
        tiers[ranked[i]] = 8
    
    return tiers


def spectrumkv_sink_protect(scores, budget):
    """SpectrumKV greedy but sink tokens guaranteed FP16."""
    tiers = spectrumkv_greedy(scores, budget)
    # Force sink to FP16 (may slightly exceed budget, but negligible)
    for i in range(min(SINK, len(tiers))):
        tiers[i] = 16
    return tiers


# ============ NIAH Computation ============
def niah_base_success(depth, budget_frac=1.0):
    """Base NIAH success rate for a given needle depth and budget coverage."""
    # Parabolic depth penalty (middle harder)
    depth_penalty = 1.0 - 0.15 * 4 * depth * (1 - depth)
    # Budget bonus: more coverage → slightly higher success
    budget_bonus = 0.85 + 0.15 * min(1.0, budget_frac)
    return max(0.3, budget_bonus * depth_penalty)

def compute_niah_selection(selected, n, depths):
    """NIAH for selection-based methods: 0% if needle dropped."""
    sel_set = set(selected)
    b_frac = len(selected) / n
    results = {}
    for d in depths:
        pos = int(d * (n - 1))
        if pos in sel_set:
            results[d] = niah_base_success(d, b_frac)
        else:
            results[d] = 0.0  # DROPPED = NO RETRIEVAL
    return results

def compute_niah_qcbm(tiers, n, depths, budget):
    """NIAH for SpectrumKV: quality depends on precision tier. ALL tokens retained.
    
    Uses actual budget (not hardcoded 0.95) for fairness.
    All tokens are retained, but at reduced precision — this means the model
    still sees all KV pairs, just at lower fidelity. Budget reflects the
    effective information bandwidth, not token coverage.
    """
    results = {}
    for d in depths:
        pos = int(d * (n - 1))
        tier = int(tiers[pos])
        quality = QUALITY.get(tier, 0.78)
        # Use actual budget — all tokens retained, quality from tier
        results[d] = niah_base_success(d, budget) * quality
    return results


# ============ PPL Computation ============
def compute_ppl_selection(selected, attn, n):
    """PPL for selection: evicted attention mass model."""
    sel_set = set(selected)
    evicted_mass = 0.0
    evicted_entropy = 0.0
    for j in range(n):
        if j not in sel_set:
            evicted_mass += attn[j]
            if attn[j] > EPS:
                evicted_entropy += attn[j] * np.log(1.0 / attn[j])
    if evicted_mass > EPS:
        avg_surprise = evicted_entropy / evicted_mass
        return max(1.0, 1.0 + evicted_mass * avg_surprise * 0.12)
    return 1.0

def compute_ppl_qcbm(tiers, attn, n):
    """PPL for SpectrumKV: calibrated quantization noise model."""
    n_total = n
    n_fp16 = np.sum(tiers == 16)
    n_int8 = np.sum(tiers == 8)
    n_int4 = np.sum(tiers == 4)
    
    # Base PPL increase from quantization (calibrated to real results)
    # All-INT8: ~2.5% increase; All-INT4: ~12% increase
    base_increase = (n_int8 / n_total) * PPL_FULL_TIER[8] + \
                    (n_int4 / n_total) * PPL_FULL_TIER[4]
    
    # Attention concentration penalty: quantizing high-attention tokens is worse
    total_mass = attn.sum()
    int4_mass = sum(attn[j] for j in range(n) if tiers[j] == 4) / total_mass
    int8_mass = sum(attn[j] for j in range(n) if tiers[j] == 8) / total_mass
    
    # Expected mass at each tier if uniform
    expected_int4 = n_int4 / n_total
    expected_int8 = n_int8 / n_total
    
    # Concentration factor: if attention is more concentrated on quantized tokens than expected
    if expected_int4 > 0.01:
        conc_int4 = int4_mass / expected_int4
    else:
        conc_int4 = 1.0
    if expected_int8 > 0.01:
        conc_int8 = int8_mass / expected_int8
    else:
        conc_int8 = 1.0
    
    # Penalty: up to 2x if all attention is on quantized tokens
    penalty = 1.0 + 0.5 * max(0, conc_int4 - 1.0) + 0.2 * max(0, conc_int8 - 1.0)
    penalty = min(penalty, 2.5)
    
    ppl_increase = base_increase * penalty
    
    # Sink token penalty: if sink tokens are at INT4, extra penalty
    sink_at_int4 = sum(1 for j in range(min(SINK, n)) if tiers[j] == 4)
    if sink_at_int4 > 0:
        ppl_increase += sink_at_int4 * 0.03  # Each sink at INT4 adds 3%
    
    return 1.0 + ppl_increase


# ============ Method Definitions ============
METHODS = [
    # Selection-based (drop tokens)
    ("PDTrim",              "selection", None),
    ("SWS_Original",        "selection", score_original),
    ("SWS_ValueAware",      "selection", score_value_aware),
    # SpectrumKV (keep ALL tokens, tiered precision)
    ("SpectrumKV_Greedy_AttnOnly",     "spectrumkv_greedy",      score_attn_only),
    ("SpectrumKV_Greedy_ValueAware",   "spectrumkv_greedy",      score_value_aware),
    ("SpectrumKV_Greedy_Original",     "spectrumkv_greedy",      score_original),
    ("SpectrumKV_Balanced_AttnOnly",   "spectrumkv_balanced",    score_attn_only),
    ("SpectrumKV_Balanced_ValueAware", "spectrumkv_balanced",    score_value_aware),
    ("SpectrumKV_SinkProtect_AttnOnly","spectrumkv_sink_protect", score_attn_only),
    ("SpectrumKV_LayerMod",            "spectrumkv_greedy",      score_layer_mod),
]


# ============ Main Simulation ============
def run_simulation():
    all_results = []
    
    for method_name, method_type, score_fn in METHODS:
        print(f"\n--- {method_name} ---")
        for model_name, mcfg in MODELS.items():
            nl = mcfg["layers"]
            loc = gen_locality(nl, mcfg["locality"])
            sample_layers = sorted(set([0, nl // 4, nl // 2, 3 * nl // 4, nl - 1]))
            
            for seq_len in SEQ_LENS:
                for budget in BUDGETS:
                    niah_by_depth = defaultdict(list)
                    attn_covs = []
                    ppl_mults = []
                    tier_fracs = defaultdict(list)
                    
                    for run in range(N_RUNS):
                        for li in sample_layers:
                            np.random.seed(SEED + run * 1000 + li)
                            attn = gen_attn(seq_len)
                            
                            # Compute importance scores
                            if score_fn is not None:
                                scores = score_fn(attn, seq_len, li=li, nl=nl, loc=loc)
                            else:
                                scores = attn.copy()
                            
                            if method_type == "selection":
                                if method_name == "PDTrim":
                                    selected = sel_pdtrim(seq_len, budget)
                                else:
                                    selected = sel_topk(scores, budget)
                                
                                niah = compute_niah_selection(selected, seq_len, DEPTH_FRACS)
                                ppl = compute_ppl_selection(selected, attn, seq_len)
                                cov = sum(attn[j] for j in selected)
                                tier_fracs[16].append(len(selected) / seq_len)
                                tier_fracs[4].append(1 - len(selected) / seq_len)
                                tier_fracs[8].append(0.0)
                            
                            elif method_type == "spectrumkv_greedy":
                                tiers = spectrumkv_greedy(scores, budget)
                                niah = compute_niah_qcbm(tiers, seq_len, DEPTH_FRACS, budget)
                                ppl = compute_ppl_qcbm(tiers, attn, seq_len)
                                cov = 1.0
                                for t in [16, 8, 4]:
                                    tier_fracs[t].append(np.sum(tiers == t) / seq_len)
                            
                            elif method_type == "spectrumkv_balanced":
                                tiers = spectrumkv_balanced(scores, budget)
                                niah = compute_niah_qcbm(tiers, seq_len, DEPTH_FRACS, budget)
                                ppl = compute_ppl_qcbm(tiers, attn, seq_len)
                                cov = 1.0
                                for t in [16, 8, 4]:
                                    tier_fracs[t].append(np.sum(tiers == t) / seq_len)
                            
                            elif method_type == "spectrumkv_sink_protect":
                                tiers = spectrumkv_sink_protect(scores, budget)
                                niah = compute_niah_qcbm(tiers, seq_len, DEPTH_FRACS, budget)
                                ppl = compute_ppl_qcbm(tiers, attn, seq_len)
                                cov = 1.0
                                for t in [16, 8, 4]:
                                    tier_fracs[t].append(np.sum(tiers == t) / seq_len)
                            
                            for d, acc in niah.items():
                                niah_by_depth[d].append(acc)
                            attn_covs.append(cov)
                            ppl_mults.append(ppl)
                    
                    result = {
                        "method": method_name,
                        "model": model_name,
                        "seq_len": seq_len,
                        "budget": budget,
                        "niah_by_depth": {str(d): round(np.mean(v), 4) for d, v in niah_by_depth.items()},
                        "niah_avg": round(np.mean([np.mean(v) for v in niah_by_depth.values()]), 4),
                        "niah_middle": round(np.mean([np.mean(niah_by_depth[d]) for d in [0.25, 0.5, 0.75]]), 4),
                        "attn_coverage": round(np.mean(attn_covs), 4),
                        "ppl_multiplier": round(np.mean(ppl_mults), 4),
                        "ppl_degradation_pct": round((np.mean(ppl_mults) - 1.0) * 100, 2),
                        "fp16_frac": round(np.mean(tier_fracs[16]), 4),
                        "int8_frac": round(np.mean(tier_fracs[8]), 4),
                        "int4_frac": round(np.mean(tier_fracs[4]), 4),
                    }
                    all_results.append(result)
                    print(f"  {model_name:12s} seq={seq_len} b={budget:.1f} → "
                          f"NIAH={result['niah_avg']:.4f} mid={result['niah_middle']:.4f} "
                          f"PPL%={result['ppl_degradation_pct']:+.2f} "
                          f"FP16={result['fp16_frac']:.2f} INT8={result['int8_frac']:.2f} INT4={result['int4_frac']:.2f}")
    
    return all_results


def analyze_results(all_results):
    analysis = {}
    
    # 1. Selection vs SpectrumKV comparison (averaged across models & seq_lens)
    comparison = {}
    for budget in BUDGETS:
        sel = [r for r in all_results if r["method"] in ["PDTrim", "SWS_Original", "SWS_ValueAware"] and r["budget"] == budget]
        qcbm = [r for r in all_results if "SpectrumKV" in r["method"] and r["budget"] == budget]
        
        if sel and qcbm:
            # Best selection method
            sel_best = max(sel, key=lambda r: r["niah_avg"])
            # Best SpectrumKV method
            qcbm_best = max(qcbm, key=lambda r: r["niah_avg"])
            
            comparison[str(budget)] = {
                "best_selection": {
                    "method": sel_best["method"],
                    "niah_avg": sel_best["niah_avg"],
                    "niah_middle": sel_best["niah_middle"],
                    "ppl_pct": sel_best["ppl_degradation_pct"],
                },
                "best_qcbm": {
                    "method": qcbm_best["method"],
                    "niah_avg": qcbm_best["niah_avg"],
                    "niah_middle": qcbm_best["niah_middle"],
                    "ppl_pct": qcbm_best["ppl_degradation_pct"],
                    "fp16_frac": qcbm_best["fp16_frac"],
                    "int8_frac": qcbm_best["int8_frac"],
                    "int4_frac": qcbm_best["int4_frac"],
                },
                "niah_delta": round(qcbm_best["niah_avg"] - sel_best["niah_avg"], 4),
                "niah_middle_delta": round(qcbm_best["niah_middle"] - sel_best["niah_middle"], 4),
            }
    analysis["selection_vs_qcbm_best"] = comparison
    
    # 2. Average improvement (across all configs)
    sel_all = [r for r in all_results if r["method"] in ["PDTrim", "SWS_Original", "SWS_ValueAware"]]
    qcbm_all = [r for r in all_results if "SpectrumKV" in r["method"]]
    
    for budget in BUDGETS:
        s = [r for r in sel_all if r["budget"] == budget]
        q = [r for r in qcbm_all if r["budget"] == budget]
        if s and q:
            s_niah = np.mean([r["niah_avg"] for r in s])
            q_niah = np.mean([r["niah_avg"] for r in q])
            s_ppl = np.mean([r["ppl_degradation_pct"] for r in s])
            q_ppl = np.mean([r["ppl_degradation_pct"] for r in q])
            print(f"\n  b={budget}: Selection NIAH={s_niah:.4f} PPL={s_ppl:.1f}% | SpectrumKV NIAH={q_niah:.4f} PPL={q_ppl:.1f}% | ΔNIAH={q_niah-s_niah:+.4f}")
    
    # 3. Best SpectrumKV variant per budget
    best_qcbm = {}
    for budget in BUDGETS:
        qcbm_budget = [r for r in qcbm_all if r["budget"] == budget]
        if qcbm_budget:
            # Composite: NIAH - 0.05 * PPL_deg
            best = max(qcbm_budget, key=lambda r: r["niah_avg"] - 0.05 * r["ppl_degradation_pct"])
            best_qcbm[str(budget)] = {
                "best_method": best["method"],
                "niah_avg": best["niah_avg"],
                "niah_middle": best["niah_middle"],
                "ppl_degradation_pct": best["ppl_degradation_pct"],
                "fp16_frac": best["fp16_frac"],
                "int8_frac": best["int8_frac"],
                "int4_frac": best["int4_frac"],
            }
    analysis["best_qcbm_per_budget"] = best_qcbm
    
    # 4. Pareto front
    all_pts = [(r["ppl_degradation_pct"], r["niah_avg"], r["method"]) for r in all_results]
    pareto = set()
    for p in all_pts:
        dominated = False
        for q in all_pts:
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.add(p[2])
    analysis["pareto_front_methods"] = sorted(pareto)
    
    # 5. Per-depth NIAH comparison (the key figure for the paper)
    depth_comparison = {}
    for budget in BUDGETS:
        depth_comparison[str(budget)] = {}
        for d in DEPTH_FRACS:
            ds = str(d)
            pdtrim_niah = [r["niah_by_depth"].get(ds, 0) for r in all_results 
                          if r["method"] == "PDTrim" and r["budget"] == budget]
            # Best SpectrumKV at this budget
            qcbm_best_method = best_qcbm.get(str(budget), {}).get("best_method", "SpectrumKV_Greedy_AttnOnly")
            qcbm_niah = [r["niah_by_depth"].get(ds, 0) for r in all_results 
                        if r["method"] == qcbm_best_method and r["budget"] == budget]
            
            depth_comparison[str(budget)][ds] = {
                "PDTrim": round(np.mean(pdtrim_niah), 4) if pdtrim_niah else 0,
                "SpectrumKV_best": round(np.mean(qcbm_niah), 4) if qcbm_niah else 0,
            }
    analysis["depth_comparison"] = depth_comparison
    
    # 6. Full summary table
    summary = {}
    method_names = [m[0] for m in METHODS]
    for mn in method_names:
        method_data = [r for r in all_results if r["method"] == mn]
        if method_data:
            summary[mn] = {
                "niah_avg_by_budget": {},
                "niah_middle_by_budget": {},
                "ppl_by_budget": {},
            }
            for b in BUDGETS:
                bs = str(b)
                bd = [r for r in method_data if r["budget"] == b]
                if bd:
                    summary[mn]["niah_avg_by_budget"][bs] = round(np.mean([r["niah_avg"] for r in bd]), 4)
                    summary[mn]["niah_middle_by_budget"][bs] = round(np.mean([r["niah_middle"] for r in bd]), 4)
                    summary[mn]["ppl_by_budget"][bs] = round(np.mean([r["ppl_degradation_pct"] for r in bd]), 2)
    analysis["summary"] = summary
    
    return analysis


def generate_plots(all_results):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # --- Plot 1: NIAH by budget (selection vs SpectrumKV) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, budget in enumerate(BUDGETS):
        ax = axes[idx]
        methods_to_plot = ["PDTrim", "SWS_Original", "SWS_ValueAware",
                          "SpectrumKV_Greedy_AttnOnly", "SpectrumKV_Greedy_ValueAware",
                          "SpectrumKV_Balanced_AttnOnly", "SpectrumKV_SinkProtect_AttnOnly"]
        
        x_positions = []
        labels = []
        niah_values = []
        colors = []
        
        sel_color = '#4ECDC4'
        qcbm_color = '#FF6B6B'
        
        for i, mn in enumerate(methods_to_plot):
            data = [r for r in all_results if r["method"] == mn and r["budget"] == budget]
            if data:
                x_positions.append(i)
                labels.append(mn.replace("SpectrumKV_", "Q:").replace("SWS_", "S:").replace("_AttnOnly", "_AO").replace("_ValueAware", "_VA").replace("_Original", "_Orig").replace("SinkProtect_", "SP_"))
                niah_values.append(np.mean([r["niah_avg"] for r in data]))
                colors.append(sel_color if "SWS" in mn or mn == "PDTrim" else qcbm_color)
        
        bars = ax.bar(x_positions, niah_values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('NIAH Avg')
        ax.set_title(f'Budget = {budget}')
        ax.set_ylim(0, 1.0)
        
        # Add value labels
        for bar, val in zip(bars, niah_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle('NIAH by Budget: Selection vs SpectrumKV (v3 Corrected)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_qcbm_v3_niah_by_budget.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_qcbm_v3_niah_by_budget.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: NIAH vs PPL Pareto ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sel_methods = ["PDTrim", "SWS_Original", "SWS_ValueAware"]
    qcbm_methods = [m[0] for m in METHODS if "SpectrumKV" in m[0]]
    
    for mn in sel_methods:
        data = [r for r in all_results if r["method"] == mn]
        if data:
            ax.scatter([r["ppl_degradation_pct"] for r in data],
                      [r["niah_avg"] for r in data],
                      marker='s', s=60, alpha=0.7, label=mn, zorder=3)
    
    for mn in qcbm_methods:
        data = [r for r in all_results if r["method"] == mn]
        if data:
            ax.scatter([r["ppl_degradation_pct"] for r in data],
                      [r["niah_avg"] for r in data],
                      marker='o', s=60, alpha=0.7, label=mn.replace("SpectrumKV_", "Q:").replace("_AttnOnly", "_AO").replace("_ValueAware", "_VA").replace("_Original", "_Orig").replace("SinkProtect_", "SP_").replace("Balanced_", "Bal_").replace("LayerMod", "LMod"),
                      zorder=3)
    
    ax.set_xlabel('PPL Degradation (%)', fontsize=12)
    ax.set_ylabel('NIAH Avg', fontsize=12)
    ax.set_title('Pareto: NIAH vs PPL Cost (v3 Corrected)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, max(30, max(r["ppl_degradation_pct"] for r in all_results) * 1.1))
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_qcbm_v3_pareto.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_qcbm_v3_pareto.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Depth-wise NIAH (the key figure) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    key_methods = ["PDTrim", "SpectrumKV_Greedy_AttnOnly", "SpectrumKV_Greedy_ValueAware"]
    depth_labels = ['10%', '25%', '50%', '75%', '90%']
    
    for idx, budget in enumerate(BUDGETS):
        ax = axes[idx]
        x = np.arange(len(DEPTH_FRACS))
        width = 0.25
        
        for mi, mn in enumerate(key_methods):
            data = [r for r in all_results if r["method"] == mn and r["budget"] == budget]
            if data:
                niah_by_d = [np.mean([r["niah_by_depth"].get(str(d), 0) for r in data]) for d in DEPTH_FRACS]
                ax.bar(x + mi * width, niah_by_d, width, label=mn.replace("SpectrumKV_", "Q:").replace("_AttnOnly", "_AO").replace("_ValueAware", "_VA"), alpha=0.8)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(depth_labels)
        ax.set_xlabel('Needle Depth')
        ax.set_ylabel('NIAH Success Rate')
        ax.set_title(f'Budget = {budget}')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('NIAH by Depth: PDTrim vs SpectrumKV (v3 Corrected)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_qcbm_v3_depth_niah.pdf'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_qcbm_v3_depth_niah.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {FIGURES_DIR}/fig_qcbm_v3_*.pdf/png")


# ============ Main ============
if __name__ == "__main__":
    print("=" * 70)
    print("SWS-SpectrumKV Corrected Simulation v3")
    print("KEY FIX: SpectrumKV keeps ALL tokens with tiered precision")
    print("=" * 70)
    
    all_results = run_simulation()
    analysis = analyze_results(all_results)
    
    # Save results
    output = {
        "metadata": {
            "seed": SEED,
            "n_runs": N_RUNS,
            "models": list(MODELS.keys()),
            "seq_lengths": SEQ_LENS,
            "budgets": BUDGETS,
            "needle_depths": DEPTH_FRACS,
            "quality_factors": QUALITY,
            "ppl_calibration": PPL_FULL_TIER,
            "note": "v3 CORRECTED - SpectrumKV models ALL tokens retained with tiered precision, "
                    "NIAH uses precision-dependent quality, PPL uses calibrated quantization noise",
        },
        "analysis": analysis,
        "results": all_results,
    }
    
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Saved {len(all_results)} results to {RESULTS_PATH}")
    print(f"{'=' * 70}")
    
    # Print analysis summary
    print(f"\n### PARETO FRONT: {analysis['pareto_front_methods']}")
    
    print(f"\n### SELECTION vs SpectrumKV (best of each):")
    for b, v in analysis.get('selection_vs_qcbm_best', {}).items():
        print(f"  b={b}:")
        print(f"    Best Selection: {v['best_selection']['method']} "
              f"NIAH={v['best_selection']['niah_avg']:.4f} PPL={v['best_selection']['ppl_pct']:.1f}%")
        print(f"    Best SpectrumKV:      {v['best_qcbm']['method']} "
              f"NIAH={v['best_qcbm']['niah_avg']:.4f} PPL={v['best_qcbm']['ppl_pct']:.1f}% "
              f"(FP16={v['best_qcbm']['fp16_frac']:.2f} INT8={v['best_qcbm']['int8_frac']:.2f} INT4={v['best_qcbm']['int4_frac']:.2f})")
        print(f"    ΔNIAH={v['niah_delta']:+.4f} Δmiddle={v['niah_middle_delta']:+.4f}")
    
    print(f"\n### BEST SpectrumKV PER BUDGET:")
    for b, v in analysis.get('best_qcbm_per_budget', {}).items():
        print(f"  b={b}: {v['best_method']} NIAH={v['niah_avg']:.4f} mid={v['niah_middle']:.4f} "
              f"PPL={v['ppl_degradation_pct']:.1f}% FP16={v['fp16_frac']:.2f} INT8={v['int8_frac']:.2f} INT4={v['int4_frac']:.2f}")
    
    print(f"\n### FULL SUMMARY:")
    for mn, data in analysis.get('summary', {}).items():
        print(f"  {mn}:")
        for b in BUDGETS:
            bs = str(b)
            niah = data['niah_avg_by_budget'].get(bs, 0)
            mid = data['niah_middle_by_budget'].get(bs, 0)
            ppl = data['ppl_by_budget'].get(bs, 0)
            print(f"    b={b}: NIAH={niah:.4f} mid={mid:.4f} PPL={ppl:.1f}%")
    
    # Generate plots
    generate_plots(all_results)
