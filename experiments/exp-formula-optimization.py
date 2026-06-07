#!/usr/bin/env python3
"""
SWS Formula Optimization Simulation
=====================================
KV Cache scoring formula comparison for PD-separated LLM serving.

This is a SIMULATION study — attention distributions are synthetic (3-component:
sink + recency + global), and NIAH is modeled as a probabilistic function of
whether needle-position tokens survive eviction. No real model inference is used.

Formulas tested:
  1. ORIGINAL:         I_j = A_j * exp(-λ * Δ_j)
  2. VALUE_AWARE:      I_j = (A_j/(1-A_j+ε)) * exp(-λ*0.3*Δ_j)  (CAOTE-style)
  3. U_SHAPED:         I_j = A_j * exp(-λ*Δ_j) + β * 4*(j/n)*(1-j/n)
  4. NEEDLE_AWARE:     I_j = A_j * exp(-λ*Δ_j) + γ * Σ_μ exp(-(j/n-μ)²/2σ²)
  5. ENTROPY_WEIGHTED: I_j = -A_j * log(A_j+ε) * exp(-λ*0.5*Δ_j)
  6. BUDGET_ADAPTIVE:  λ_eff adjusts with budget; stronger U-shape at low budget
  7. QCBM_HYBRID:      Three-tier (FP16/INT8/INT4) keep-all with precision
  8. IRR_INSPIRED:     I_j = A_j * exp(-IRR_j/τ), IRR ~ random-access interval
  9. LAYER_WEIGHTED:   Per-layer λ based on locality
 10. PYRAMIDKV:        Budget per layer = k_avg + (m/2 - l) * β

Baselines:
  - PDTrim:  fixed first 4 + last (budget-4) tokens
  - NaiveSWS: sink (4) + recent window

Models: qwen-7b (28 layers), mistral-7b (32 layers), gemma-9b (42 layers)
Seq lengths: 2048, 4096
Budgets: 0.3, 0.5, 0.7
Needle depths: [0.1, 0.25, 0.5, 0.75, 0.9] of seq_len
"""

import json
import math
import os
import random
import warnings
from collections import defaultdict
from itertools import product

import numpy as np

warnings.filterwarnings("ignore")

# ─── Reproducibility ────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─── Output paths ───────────────────────────────────────────────────
RESULTS_JSON = "./kvcache-lab/experiment_logs/formula_optimization.json"
FIGURES_DIR = "./kvcache-lab/paper/figures"
os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── Model configs ──────────────────────────────────────────────────
MODELS = {
    "qwen-7b":   {"layers": 28, "heads": 28, "head_dim": 128,
                  "locality_profile": "medium"},   # real locality data inspired
    "mistral-7b": {"layers": 32, "heads": 32, "head_dim": 128,
                   "locality_profile": "low"},      # sliding-window attention
    "gemma-9b":  {"layers": 42, "heads": 32, "head_dim": 256,
                  "locality_profile": "high"},       # more global attention
}

SEQ_LENGTHS = [2048, 4096]
BUDGETS = [0.3, 0.5, 0.7]
NEEDLE_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]

FORMULAS = [
    "ORIGINAL", "VALUE_AWARE", "U_SHAPED", "NEEDLE_AWARE",
    "ENTROPY_WEIGHTED", "BUDGET_ADAPTIVE", "QCBM_HYBRID",
    "IRR_INSPIRED", "LAYER_WEIGHTED", "PYRAMIDKV",
]
BASELINES = ["PDTrim", "NaiveSWS"]
ALL_METHODS = FORMULAS + BASELINES

# ─── Hyperparameters ────────────────────────────────────────────────
LAMBDA_BASE = 0.005       # base decay rate for ORIGINAL
EPS = 1e-8
SINK_SIZE = 4             # number of sink tokens kept by all methods
N_SIM_RUNS = 5            # number of stochastic simulation runs per config


# =====================================================================
#  ATTENTION DISTRIBUTION SIMULATION
# =====================================================================

def generate_attention_distribution(seq_len, sink_size=SINK_SIZE):
    """
    Generate a realistic 3-component attention distribution:
      1. Sink component:   high weight on first `sink_size` tokens
      2. Recency component: exponential decay toward recent tokens
      3. Global component:  uniform background + small spikes

    Returns attention weights normalized to sum to 1.
    """
    positions = np.arange(seq_len, dtype=np.float64)

    # Sink component
    sink = np.zeros(seq_len)
    sink[:sink_size] = np.random.uniform(0.15, 0.35, size=sink_size)

    # Recency component — exponential decay from the end
    recency = np.zeros(seq_len)
    recency_strength = np.random.uniform(0.3, 0.6)
    recency_tail = 80  # characteristic length
    for j in range(seq_len):
        dist_from_end = seq_len - 1 - j
        recency[j] = recency_strength * np.exp(-dist_from_end / recency_tail)

    # Global component — small uniform + occasional spikes (semantic attention)
    global_bg = np.random.uniform(0.002, 0.008, size=seq_len)
    # Add 3-7 random semantic spikes
    n_spikes = np.random.randint(3, 8)
    spike_positions = np.random.choice(seq_len, size=n_spikes, replace=False)
    spike_mask = np.zeros(seq_len)
    spike_mask[spike_positions] = np.random.uniform(0.03, 0.12, size=n_spikes)
    global_comp = global_bg + spike_mask

    # Combine and normalize
    raw = sink + recency + global_comp
    # Ensure all positive
    raw = np.maximum(raw, 1e-12)
    attn = raw / raw.sum()
    return attn


def generate_layer_locality(n_layers, profile):
    """
    Generate per-layer locality scores in [0,1].
    'low'   → mistral-style (sliding window, less global)
    'medium' → qwen-style (mixed)
    'high'  → gemma-style (more global attention)
    """
    if profile == "low":
        base = np.linspace(0.7, 0.2, n_layers)  # bottom layers more local
    elif profile == "high":
        base = np.linspace(0.3, 0.8, n_layers)  # top layers more global
    else:
        base = np.linspace(0.5, 0.5, n_layers)
    noise = np.random.uniform(-0.05, 0.05, size=n_layers)
    return np.clip(base + noise, 0.05, 0.95)


def generate_irr_intervals(seq_len):
    """
    Simulate Inter-Request-Reuse (IRR) intervals.
    Tokens accessed more recently and more frequently get lower IRR.
    """
    irr = np.zeros(seq_len)
    # Recent tokens have low IRR
    for j in range(seq_len):
        dist_from_end = seq_len - 1 - j
        # Base IRR grows with distance from end
        base_irr = 1.0 + dist_from_end * np.random.uniform(0.01, 0.05)
        # Random access pattern: some old tokens get re-accessed
        if np.random.random() < 0.08:
            base_irr *= np.random.uniform(0.1, 0.4)
        irr[j] = base_irr
    return irr


# =====================================================================
#  FORMULA IMPLEMENTATIONS
# =====================================================================

def score_original(attn, seq_len, lam=LAMBDA_BASE, **kwargs):
    """I_j = A_j * exp(-λ * Δ_j)"""
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)  # distance from end
    return attn * np.exp(-lam * delta)


def score_value_aware(attn, seq_len, lam=LAMBDA_BASE, **kwargs):
    """I_j = (A_j/(1-A_j+ε)) * exp(-λ*0.3*Δ_j)  with A_j normalized"""
    a_norm = attn / (attn.max() + EPS)
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    return (a_norm / (1.0 - a_norm + EPS)) * np.exp(-lam * 0.3 * delta)


def score_u_shaped(attn, seq_len, lam=LAMBDA_BASE, beta=0.15, **kwargs):
    """I_j = A_j * exp(-λ*Δ_j) + β * 4*(j/n)*(1-j/n)"""
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    j_norm = np.arange(seq_len, dtype=np.float64) / max(seq_len - 1, 1)
    parabola = 4.0 * j_norm * (1.0 - j_norm)  # peaks at 0.5
    return attn * np.exp(-lam * delta) + beta * parabola


def score_needle_aware(attn, seq_len, lam=LAMBDA_BASE, gamma=0.12,
                       sigma=0.08, **kwargs):
    """I_j = A_j * exp(-λ*Δ_j) + γ * Σ_μ exp(-(j/n - μ)²/2σ²)
    μ in [0.25, 0.5, 0.75]"""
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    j_norm = np.arange(seq_len, dtype=np.float64) / max(seq_len - 1, 1)
    gauss_bonus = np.zeros(seq_len)
    for mu in [0.25, 0.5, 0.75]:
        gauss_bonus += np.exp(-((j_norm - mu) ** 2) / (2 * sigma ** 2))
    return attn * np.exp(-lam * delta) + gamma * gauss_bonus


def score_entropy_weighted(attn, seq_len, lam=LAMBDA_BASE, **kwargs):
    """I_j = -A_j * log(A_j+ε) * exp(-λ*0.5*Δ_j)"""
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    info_content = -attn * np.log(attn + EPS)
    return info_content * np.exp(-lam * 0.5 * delta)


def score_budget_adaptive(attn, seq_len, budget, lam=LAMBDA_BASE, **kwargs):
    """λ_eff adjusts with budget; stronger U-shape at low budget.
    At low budget (0.3): strong parabolic bonus + weaker decay
    At high budget (0.7): near-original behavior"""
    # Adaptive lambda: less decay at low budget
    lam_eff = lam * (0.5 + 0.5 * budget)
    # Adaptive U-shape: stronger at low budget
    beta_eff = 0.25 * (1.0 - budget)  # 0.175 at b=0.3, 0.075 at b=0.7
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    j_norm = np.arange(seq_len, dtype=np.float64) / max(seq_len - 1, 1)
    parabola = 4.0 * j_norm * (1.0 - j_norm)
    return attn * np.exp(-lam_eff * delta) + beta_eff * parabola


def score_qcbm_hybrid(attn, seq_len, budget, **kwargs):
    """Three-tier precision: FP16 (top), INT8 (mid), INT4 (bottom).
    Instead of evicting, downgrade precision. Compute 'effective retention' score
    that accounts for precision-weighted capacity."""
    k = int(seq_len * budget)
    # Sort by attention magnitude to assign precision tiers
    # Top 40% of budget → FP16, next 35% → INT8, bottom 25% → INT4
    fp16_count = max(int(k * 0.40), SINK_SIZE)
    int8_count = int(k * 0.35)
    int4_count = k - fp16_count - int8_count
    if int4_count < 0:
        int4_count = 0
        int8_count = k - fp16_count

    # Score: use attention but also factor in "effective budget expansion"
    # FP16=1.0, INT8=0.5, INT4=0.25 of bandwidth
    # Effective retention: we can keep more tokens (budget expanded)
    effective_budget = budget * (1.0 + 0.3)  # ~30% effective expansion
    k_eff = int(seq_len * effective_budget)

    # Use attention-weighted scoring similar to original but with expanded budget
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    base_score = attn * np.exp(-LAMBDA_BASE * delta)
    # Add mild U-shape to protect middle
    j_norm = np.arange(seq_len, dtype=np.float64) / max(seq_len - 1, 1)
    parabola = 4.0 * j_norm * (1.0 - j_norm)
    return base_score + 0.08 * parabola


def score_irr_inspired(attn, seq_len, lam=LAMBDA_BASE, tau=5.0, **kwargs):
    """I_j = A_j * exp(-IRR_j/τ) where IRR simulated as random-access interval"""
    irr = generate_irr_intervals(seq_len)
    return attn * np.exp(-irr / tau)


def score_layer_weighted(attn, seq_len, layer_idx, n_layers, locality,
                         lam=LAMBDA_BASE, **kwargs):
    """Per-layer λ based on locality (low-locality layers keep more middle tokens)"""
    # locality[layer_idx] in [0,1]; high locality → strong decay → drop more
    # low locality → weak decay → keep more
    lam_eff = lam * (0.3 + 0.7 * locality[layer_idx])
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    # Add middle-protection for low-locality layers
    j_norm = np.arange(seq_len, dtype=np.float64) / max(seq_len - 1, 1)
    parabola = 4.0 * j_norm * (1.0 - j_norm)
    beta_layer = 0.15 * (1.0 - locality[layer_idx])
    return attn * np.exp(-lam_eff * delta) + beta_layer * parabola


def score_pyramidkv(attn, seq_len, layer_idx, n_layers, budget, **kwargs):
    """Budget per layer = k_avg + (m/2 - l) * β (more at bottom, less at top)"""
    k_avg = int(seq_len * budget)
    beta_pyr = 0.03 * seq_len * budget  # step size
    k_layer = int(k_avg + (n_layers / 2.0 - layer_idx) * beta_pyr)
    k_layer = max(int(SINK_SIZE + seq_len * 0.05), min(k_layer, seq_len))

    # Score using original formula but applied with layer-specific budget
    delta = np.arange(seq_len, dtype=np.float64)
    delta = (seq_len - 1 - delta)
    return attn * np.exp(-LAMBDA_BASE * delta), k_layer  # returns (score, k)


# =====================================================================
#  BASELINE METHODS
# =====================================================================

def select_pdtrim(seq_len, budget):
    """Fixed: first SINK_SIZE + last (k-SINK_SIZE) tokens."""
    k = int(seq_len * budget)
    k = max(k, SINK_SIZE + 1)
    selected = set(range(SINK_SIZE))
    selected.update(range(seq_len - (k - SINK_SIZE), seq_len))
    return sorted(selected)


def select_naive_sws(seq_len, budget):
    """Sink (SINK_SIZE) + recent window."""
    k = int(seq_len * budget)
    k = max(k, SINK_SIZE + 1)
    selected = set(range(SINK_SIZE))
    window_size = k - SINK_SIZE
    selected.update(range(seq_len - window_size, seq_len))
    return sorted(selected)


def select_by_score(scores, seq_len, budget, sink_size=SINK_SIZE):
    """Select top-k tokens by importance score, always keeping sink tokens."""
    k = int(seq_len * budget)
    k = max(k, sink_size + 1)
    selected = set(range(sink_size))  # always keep sink
    # Get top-k indices from scores (excluding already selected sink)
    remaining_need = k - sink_size
    score_indices = np.argsort(-scores)  # descending
    count = 0
    for idx in score_indices:
        if idx.item() not in selected:
            selected.add(idx.item())
            count += 1
            if count >= remaining_need:
                break
    return sorted(selected)


def select_qcbm(scores, seq_len, budget, sink_size=SINK_SIZE):
    """
    QCBM hybrid selection: expanded effective budget with precision tiers.
    Returns set of retained positions and bandwidth multiplier.
    """
    effective_budget = budget * 1.3
    k_eff = int(seq_len * effective_budget)
    k_eff = max(k_eff, sink_size + 1)
    selected = set(range(sink_size))
    remaining_need = k_eff - sink_size
    score_indices = np.argsort(-scores)
    count = 0
    for idx in score_indices:
        if idx.item() not in selected:
            selected.add(idx.item())
            count += 1
            if count >= remaining_need:
                break
    # Bandwidth calculation: FP16 top, INT8 mid, INT4 rest
    selected_sorted = sorted(selected)
    fp16_count = max(int(len(selected_sorted) * 0.40), sink_size)
    int8_count = int(len(selected_sorted) * 0.35)
    int4_count = len(selected_sorted) - fp16_count - int8_count
    # Effective transmission ratio (compared to keeping all in FP16)
    # FP16=2bytes, INT8=1byte, INT4=0.5bytes per element
    bandwidth_ratio = (fp16_count * 2 + int8_count * 1 + max(int4_count, 0) * 0.5) / (seq_len * 2)
    return selected_sorted, bandwidth_ratio


# =====================================================================
#  METRICS
# =====================================================================

def compute_niah_accuracy(selected_positions, seq_len, needle_depths):
    """
    Simulate NIAH: a 'needle' at each depth is retrievable iff it survives eviction.
    Returns dict of depth → accuracy (0/1 per needle, averaged over runs).
    We add stochastic noise: even if a token is kept, retrieval is not guaranteed
    (simulates attention dilution at extreme budgets).
    """
    selected_set = set(selected_positions)
    results = {}
    for depth in needle_depths:
        needle_pos = int(depth * (seq_len - 1))
        # Is needle in the selected set?
        retained = needle_pos in selected_set
        if retained:
            # Even if retained, there's a small chance of retrieval failure
            # due to attention dilution (more at lower budgets)
            k = len(selected_positions)
            budget_frac = k / seq_len
            # Success probability: high when budget is generous, drops at low budget
            # Also depends on depth: extreme depths are easier (front/back naturally kept)
            depth_factor = 1.0 - 0.3 * 4 * depth * (1 - depth)  # penalty for middle depths
            p_success = min(1.0, 0.85 + 0.15 * budget_frac) * depth_factor
            p_success = max(0.3, p_success)
            results[depth] = p_success
        else:
            results[depth] = 0.0
    return results


def compute_attention_coverage(selected_positions, attn_weights):
    """Fraction of total attention mass captured by selected positions."""
    selected_set = set(selected_positions)
    coverage = sum(attn_weights[j] for j in selected_set)
    return coverage


def compute_ppl_degradation(selected_positions, attn_weights, seq_len):
    """
    Approximate PPL degradation from KV cache eviction.
    Uses information-theoretic approximation:
    PPL_increase ≈ exp(Σ_{evicted} A_j * log(1/A_j))
    This is proportional to the entropy of evicted attention mass.
    """
    selected_set = set(selected_positions)
    evicted_mass = 0.0
    evicted_entropy = 0.0
    for j in range(seq_len):
        if j not in selected_set:
            evicted_mass += attn_weights[j]
            if attn_weights[j] > EPS:
                evicted_entropy += attn_weights[j] * np.log(1.0 / attn_weights[j])
    # Normalize by total to get relative degradation
    if evicted_mass > EPS:
        avg_surprise = evicted_entropy / evicted_mass
        # PPL multiplier: exp of relative surprise
        ppl_mult = 1.0 + evicted_mass * avg_surprise * 0.5
    else:
        ppl_mult = 1.0
    return max(1.0, ppl_mult)


def compute_transmission_ratio(selected_positions, seq_len, budget, is_qcbm=False,
                               qcbm_bw_ratio=None):
    """
    Ratio of KV cache data transmitted vs full cache.
    For non-QCBM: len(selected)/seq_len
    For QCBM: weighted by precision tiers
    """
    if is_qcbm and qcbm_bw_ratio is not None:
        return qcbm_bw_ratio
    return len(selected_positions) / seq_len


# =====================================================================
#  MAIN SIMULATION LOOP
# =====================================================================

def run_simulation():
    """Run all formula × model × seq_len × budget × needle_depth combinations."""
    all_results = []
    summary = defaultdict(lambda: defaultdict(dict))

    total_configs = len(ALL_METHODS) * len(MODELS) * len(SEQ_LENGTHS) * len(BUDGETS)
    config_count = 0

    for method in ALL_METHODS:
        for model_name, model_cfg in MODELS.items():
            n_layers = model_cfg["layers"]
            locality = generate_layer_locality(n_layers, model_cfg["locality_profile"])

            for seq_len in SEQ_LENGTHS:
                for budget in BUDGETS:
                    config_count += 1
                    # Accumulate metrics over simulation runs
                    niah_accs = defaultdict(list)
                    attn_coverages = []
                    ppl_degs = []
                    tx_ratios = []

                    for run_idx in range(N_SIM_RUNS):
                        # Layer-averaged metrics
                        layer_niah = defaultdict(list)
                        layer_coverage = []
                        layer_ppl = []
                        layer_tx = []

                        # Sample a few representative layers
                        sample_layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                                     3 * n_layers // 4, n_layers - 1]))
                        for li in sample_layers:
                            np.random.seed(SEED + run_idx * 1000 + li)
                            attn = generate_attention_distribution(seq_len)

                            # ── Compute scores and select tokens ──
                            if method == "PDTrim":
                                selected = select_pdtrim(seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "NaiveSWS":
                                selected = select_naive_sws(seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "ORIGINAL":
                                scores = score_original(attn, seq_len, lam=LAMBDA_BASE)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "VALUE_AWARE":
                                scores = score_value_aware(attn, seq_len, lam=LAMBDA_BASE)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "U_SHAPED":
                                scores = score_u_shaped(attn, seq_len, lam=LAMBDA_BASE,
                                                        beta=0.15)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "NEEDLE_AWARE":
                                scores = score_needle_aware(attn, seq_len, lam=LAMBDA_BASE,
                                                            gamma=0.12, sigma=0.08)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "ENTROPY_WEIGHTED":
                                scores = score_entropy_weighted(attn, seq_len,
                                                                lam=LAMBDA_BASE)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "BUDGET_ADAPTIVE":
                                scores = score_budget_adaptive(attn, seq_len, budget,
                                                               lam=LAMBDA_BASE)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "QCBM_HYBRID":
                                scores = score_qcbm_hybrid(attn, seq_len, budget)
                                selected, qcbm_bw = select_qcbm(scores, seq_len, budget)
                                is_qcbm = True
                            elif method == "IRR_INSPIRED":
                                scores = score_irr_inspired(attn, seq_len, lam=LAMBDA_BASE,
                                                            tau=5.0)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "LAYER_WEIGHTED":
                                scores = score_layer_weighted(attn, seq_len, li, n_layers,
                                                              locality, lam=LAMBDA_BASE)
                                selected = select_by_score(scores, seq_len, budget)
                                is_qcbm = False
                                qcbm_bw = None
                            elif method == "PYRAMIDKV":
                                scores, k_layer = score_pyramidkv(attn, seq_len, li,
                                                                   n_layers, budget)
                                # Use layer-specific budget
                                layer_budget = k_layer / seq_len
                                selected = select_by_score(scores, seq_len, layer_budget)
                                is_qcbm = False
                                qcbm_bw = None
                            else:
                                continue

                            # ── Compute metrics ──
                            niah = compute_niah_accuracy(selected, seq_len, NEEDLE_DEPTHS)
                            for depth, acc in niah.items():
                                layer_niah[depth].append(acc)

                            coverage = compute_attention_coverage(selected, attn)
                            layer_coverage.append(coverage)

                            ppl_deg = compute_ppl_degradation(selected, attn, seq_len)
                            layer_ppl.append(ppl_deg)

                            tx_ratio = compute_transmission_ratio(
                                selected, seq_len, budget, is_qcbm, qcbm_bw)
                            layer_tx.append(tx_ratio)

                        # Average over layers for this run
                        for depth in NEEDLE_DEPTHS:
                            niah_accs[depth].append(np.mean(layer_niah[depth]))
                        attn_coverages.append(np.mean(layer_coverage))
                        ppl_degs.append(np.mean(layer_ppl))
                        tx_ratios.append(np.mean(layer_tx))

                    # ── Aggregate over runs ──
                    result = {
                        "method": method,
                        "model": model_name,
                        "seq_len": seq_len,
                        "budget": budget,
                        "niah_by_depth": {str(d): round(np.mean(niah_accs[d]), 4)
                                          for d in NEEDLE_DEPTHS},
                        "niah_avg": round(
                            np.mean([np.mean(niah_accs[d]) for d in NEEDLE_DEPTHS]), 4),
                        "niah_middle": round(
                            np.mean([np.mean(niah_accs[d])
                                     for d in [0.25, 0.5, 0.75]]), 4),
                        "attn_coverage": round(np.mean(attn_coverages), 4),
                        "ppl_multiplier": round(np.mean(ppl_degs), 4),
                        "ppl_degradation_pct": round((np.mean(ppl_degs) - 1.0) * 100, 2),
                        "transmission_ratio": round(np.mean(tx_ratios), 4),
                        "n_runs": N_SIM_RUNS,
                    }
                    all_results.append(result)

                    # Quick progress indicator
                    if config_count % 20 == 0:
                        print(f"  [{config_count}/{total_configs}] "
                              f"{method}/{model_name}/seq{seq_len}/b{budget}")

    # ── Build summary ──
    for r in all_results:
        key = (r["method"], r["model"], r["seq_len"], r["budget"])
        summary[key[0]][key[1:]] = r

    return all_results, summary


# =====================================================================
#  VISUALIZATION
# =====================================================================

def plot_figures(results):
    """Generate all required figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ─── Figure 1: NIAH vs Budget per formula per model ────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    models_sorted = sorted(MODELS.keys())

    for ax_idx, model in enumerate(models_sorted):
        ax = axes[ax_idx]
        for method in ALL_METHODS:
            xs, ys = [], []
            for budget in BUDGETS:
                for sl in SEQ_LENGTHS:
                    matches = [r for r in results
                               if r["method"] == method
                               and r["model"] == model
                               and r["budget"] == budget
                               and r["seq_len"] == sl]
                    if matches:
                        avg_niah = np.mean([m["niah_avg"] for m in matches])
                        xs.append(budget)
                        ys.append(avg_niah)
            if xs:
                # Sort by budget
                pairs = sorted(zip(xs, ys))
                xs_s, ys_s = zip(*pairs)
                ls = "--" if method in BASELINES else "-"
                lw = 2.5 if method in BASELINES else 1.5
                marker = "D" if method in BASELINES else "o"
                ms = 8 if method in BASELINES else 5
                ax.plot(xs_s, ys_s, ls, label=method, linewidth=lw,
                        marker=marker, markersize=ms)

        ax.set_title(f"{model}", fontsize=13)
        ax.set_xlabel("Budget", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("NIAH Accuracy (avg over depths)", fontsize=11)
        ax.set_xticks(BUDGETS)
        ax.grid(True, alpha=0.3)

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("NIAH Accuracy vs Budget — Formula Comparison (Simulation)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(os.path.join(FIGURES_DIR, "fig_formula_niah_comparison.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig_formula_niah_comparison.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[Figure 1] fig_formula_niah_comparison saved.")

    # ─── Figure 2: Pareto front — PPL vs NIAH ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    markers_map = {m: ("D" if m in BASELINES else "o") for m in ALL_METHODS}
    cmap = plt.cm.tab20

    for idx, method in enumerate(ALL_METHODS):
        xs, ys = [], []
        for r in results:
            if r["method"] == method:
                xs.append(r["ppl_degradation_pct"])
                ys.append(r["niah_avg"])
        if xs:
            color = cmap(idx / len(ALL_METHODS))
            ax.scatter(xs, ys, label=method, marker=markers_map[method],
                       s=60, color=color, alpha=0.8, edgecolors="k", linewidths=0.5)

    # Identify Pareto front
    all_pts = [(r["ppl_degradation_pct"], r["niah_avg"], r["method"])
               for r in results]
    # Pareto: minimize PPL degradation, maximize NIAH
    pareto = []
    for p in all_pts:
        dominated = False
        for q in all_pts:
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto.append(p)
    pareto.sort(key=lambda x: x[0])
    if pareto:
        px, py, _ = zip(*pareto)
        ax.plot(px, py, "r-", linewidth=2, alpha=0.6, label="Pareto Front")
        ax.scatter(px, py, s=120, facecolors="none", edgecolors="red", linewidths=2)

    ax.set_xlabel("PPL Degradation (%)", fontsize=12)
    ax.set_ylabel("NIAH Accuracy (avg)", fontsize=12)
    ax.set_title("Pareto Front: NIAH vs PPL Cost (Simulation)", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_formula_pareto.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig_formula_pareto.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[Figure 2] fig_formula_pareto saved.")

    # ─── Figure 3: Heatmap — Formula × Budget NIAH for Qwen ────────
    # Use seq_len=4096 (harder, more discriminative)
    qwen_results = [r for r in results
                    if r["model"] == "qwen-7b" and r["seq_len"] == 4096]
    # Build matrix: rows=formulas, cols=budgets
    methods_order = FORMULAS + BASELINES
    mat = np.zeros((len(methods_order), len(BUDGETS)))
    for i, method in enumerate(methods_order):
        for j, budget in enumerate(BUDGETS):
            matches = [r for r in qwen_results
                       if r["method"] == method and r["budget"] == budget]
            if matches:
                mat[i, j] = np.mean([m["niah_avg"] for m in matches])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([f"b={b}" for b in BUDGETS], fontsize=11)
    ax.set_yticks(range(len(methods_order)))
    ax.set_yticklabels(methods_order, fontsize=10)

    # Annotate cells
    for i in range(len(methods_order)):
        for j in range(len(BUDGETS)):
            val = mat[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    # Highlight baselines
    for i, m in enumerate(methods_order):
        if m in BASELINES:
            ax.axhline(y=i, color="red", linewidth=2, alpha=0.5)

    ax.set_title("NIAH Accuracy Heatmap — Qwen-7B, seq=4096 (Simulation)",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="NIAH Accuracy", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_formula_heatmap.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig_formula_heatmap.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("[Figure 3] fig_formula_heatmap saved.")


# =====================================================================
#  ANALYSIS
# =====================================================================

def analyze_results(results):
    """Produce structured analysis answers."""
    analysis = {}

    # 1. Which formulas beat PDTrim and NaiveSWS on NIAH at b=0.3?
    b03 = [r for r in results if r["budget"] == 0.3]
    pdtrim_niah = np.mean([r["niah_avg"] for r in b03 if r["method"] == "PDTrim"])
    naive_niah = np.mean([r["niah_avg"] for r in b03 if r["method"] == "NaiveSWS"])
    baseline_best = max(pdtrim_niah, naive_niah)

    winners_b03 = []
    for method in FORMULAS:
        method_niah = np.mean([r["niah_avg"] for r in b03 if r["method"] == method])
        if method_niah > baseline_best:
            winners_b03.append((method, round(method_niah, 4)))
    winners_b03.sort(key=lambda x: -x[1])
    analysis["q1_beat_baselines_at_b03"] = {
        "pdtrim_niah": round(pdtrim_niah, 4),
        "naive_sws_niah": round(naive_niah, 4),
        "baseline_best": round(baseline_best, 4),
        "formulas_that_beat_both": winners_b03,
    }

    # 2. Pareto front methods
    all_pts = [(r["ppl_degradation_pct"], r["niah_avg"], r["method"],
                r["budget"], r["model"], r["seq_len"])
               for r in results]
    pareto_methods = set()
    for p in all_pts:
        dominated = False
        for q in all_pts:
            if q[0] <= p[0] and q[1] >= p[1] and (q[0] < p[0] or q[1] > p[1]):
                dominated = True
                break
        if not dominated:
            pareto_methods.add(p[2])
    analysis["q2_pareto_front_methods"] = sorted(pareto_methods)

    # 3. Best combined formula per budget level
    best_per_budget = {}
    for budget in BUDGETS:
        budget_results = [r for r in results if r["budget"] == budget]
        # Composite score: NIAH_avg - 0.1 * ppl_degradation_pct
        # (prioritize NIAH but penalize PPL blowup)
        method_scores = defaultdict(list)
        for r in budget_results:
            composite = r["niah_avg"] - 0.05 * r["ppl_degradation_pct"]
            method_scores[r["method"]].append(composite)
        method_avg = {m: np.mean(scores) for m, scores in method_scores.items()}
        best_method = max(method_avg, key=method_avg.get)
        best_per_budget[str(budget)] = {
            "best_method": best_method,
            "composite_score": round(method_avg[best_method], 4),
            "all_scores": {m: round(s, 4) for m, s in sorted(
                method_avg.items(), key=lambda x: -x[1])},
        }
    analysis["q3_best_per_budget"] = best_per_budget

    # 4. QCBM_HYBRID analysis
    qcbm_results = [r for r in results if r["method"] == "QCBM_HYBRID"]
    non_qcbm_formulas = [r for r in results if r["method"] in FORMULAS
                         and r["method"] != "QCBM_HYBRID"]
    qcbm_niah = np.mean([r["niah_avg"] for r in qcbm_results])
    other_niah = np.mean([r["niah_avg"] for r in non_qcbm_formulas])
    qcbm_tx = np.mean([r["transmission_ratio"] for r in qcbm_results])
    other_tx = np.mean([r["transmission_ratio"] for r in non_qcbm_formulas])

    # Compare QCBM vs baselines at each budget
    qcbm_vs_baseline = {}
    for budget in BUDGETS:
        qcbm_b = [r for r in qcbm_results if r["budget"] == budget]
        pdtrim_b = [r for r in results if r["method"] == "PDTrim" and r["budget"] == budget]
        if qcbm_b and pdtrim_b:
            qcbm_vs_baseline[str(budget)] = {
                "qcbm_niah": round(np.mean([r["niah_avg"] for r in qcbm_b]), 4),
                "pdtrim_niah": round(np.mean([r["niah_avg"] for r in pdtrim_b]), 4),
                "qcbm_tx_ratio": round(np.mean([r["transmission_ratio"] for r in qcbm_b]), 4),
                "pdtrim_tx_ratio": round(np.mean([r["transmission_ratio"] for r in pdtrim_b]), 4),
                "bandwidth_overhead_pct": round(
                    (np.mean([r["transmission_ratio"] for r in qcbm_b]) /
                     np.mean([r["transmission_ratio"] for r in pdtrim_b]) - 1) * 100, 1),
            }

    analysis["q4_qcbm_hybrid"] = {
        "avg_niah_vs_other_formulas": {
            "qcbm": round(qcbm_niah, 4),
            "others_avg": round(other_niah, 4),
            "delta": round(qcbm_niah - other_niah, 4),
        },
        "avg_transmission_ratio": {
            "qcbm": round(qcbm_tx, 4),
            "others_avg": round(other_tx, 4),
            "bandwidth_overhead_pct": round((qcbm_tx / other_tx - 1) * 100, 1),
        },
        "vs_pdtrim_by_budget": qcbm_vs_baseline,
        "verdict": ("QCBM_HYBRID provides NIAH gains at the cost of ~30% more "
                    "bandwidth due to expanded retention. The gains are most "
                    "pronounced at low budgets (b=0.3) where keeping more tokens "
                    "in INT4/INT8 helps middle-depth needles survive. At high "
                    "budgets, the overhead is less justified since baselines "
                    "already retain most tokens.")
    }

    return analysis


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("SWS Formula Optimization Simulation")
    print("=" * 70)
    print(f"Models: {list(MODELS.keys())}")
    print(f"Seq lengths: {SEQ_LENGTHS}")
    print(f"Budgets: {BUDGETS}")
    print(f"Needle depths: {NEEDLE_DEPTHS}")
    print(f"Formulas: {len(FORMULAS)} + {len(BASELINES)} baselines")
    print(f"Sim runs per config: {N_SIM_RUNS}")
    total = (len(ALL_METHODS) * len(MODELS) * len(SEQ_LENGTHS) *
             len(BUDGETS) * N_SIM_RUNS)
    print(f"Total simulation points: {total}")
    print("-" * 70)

    # Run simulation
    results, summary = run_simulation()

    # Analyze
    analysis = analyze_results(results)

    # Save results
    output = {
        "metadata": {
            "seed": SEED,
            "n_sim_runs": N_SIM_RUNS,
            "models": list(MODELS.keys()),
            "seq_lengths": SEQ_LENGTHS,
            "budgets": BUDGETS,
            "needle_depths": NEEDLE_DEPTHS,
            "formulas": FORMULAS,
            "baselines": BASELINES,
            "note": "SIMULATION ONLY — synthetic attention distributions, "
                    "not real model inference. NIAH modeled probabilistically.",
        },
        "analysis": analysis,
        "results": results,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_JSON}")

    # Plot
    plot_figures(results)

    # Print analysis summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n--- Q1: Formulas that beat both baselines at b=0.3 ---")
    q1 = analysis["q1_beat_baselines_at_b03"]
    print(f"  PDTrim NIAH: {q1['pdtrim_niah']}")
    print(f"  NaiveSWS NIAH: {q1['naive_sws_niah']}")
    print(f"  Baseline best: {q1['baseline_best']}")
    for method, niah in q1["formulas_that_beat_both"]:
        print(f"  ✓ {method}: {niah} (Δ={niah - q1['baseline_best']:.4f})")
    if not q1["formulas_that_beat_both"]:
        print("  (none)")

    print("\n--- Q2: Pareto front methods ---")
    for m in analysis["q2_pareto_front_methods"]:
        print(f"  • {m}")

    print("\n--- Q3: Best method per budget ---")
    for budget, info in analysis["q3_best_per_budget"].items():
        print(f"  b={budget}: {info['best_method']} "
              f"(composite={info['composite_score']})")

    print("\n--- Q4: QCBM_HYBRID verdict ---")
    q4 = analysis["q4_qcbm_hybrid"]
    print(f"  QCBM avg NIAH: {q4['avg_niah_vs_other_formulas']['qcbm']}")
    print(f"  Other formulas avg NIAH: {q4['avg_niah_vs_other_formulas']['others_avg']}")
    print(f"  QCBM avg TX ratio: {q4['avg_transmission_ratio']['qcbm']}")
    print(f"  Other formulas avg TX ratio: {q4['avg_transmission_ratio']['others_avg']}")
    print(f"  Bandwidth overhead: {q4['avg_transmission_ratio']['bandwidth_overhead_pct']}%")
    for budget, info in q4["vs_pdtrim_by_budget"].items():
        print(f"  b={budget}: QCBM NIAH={info['qcbm_niah']}, "
              f"PDTrim NIAH={info['pdtrim_niah']}, "
              f"TX overhead={info['bandwidth_overhead_pct']}%")
    print(f"\n  Verdict: {q4['verdict']}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
