#!/usr/bin/env python3
"""
SWS Sink/Recent Split Ratio Sweep
====================================
Sweep the allocation between sink (front) and recent (back) tokens
within a fixed budget, with step size 0.01.

Current naive SWS uses sink=16 (nearly 0% of budget for long seqs).
This sweep finds the optimal split ratio.

Parameterization:
  - sink_ratio: fraction of kept tokens allocated to the front (sink)
  - recent_ratio = 1.0 - sink_ratio: fraction allocated to the back (recent)
  - sink_ratio=0.0 → pure recent window (no sink tokens)
  - sink_ratio=1.0 → pure sink (no recent window)
  - sink_ratio=0.5 → 50/50 split (same as PDTrim first_ratio=0.5)

Quality metrics:
  - attention_coverage: fraction of attention mass on retained tokens
  - NIAH: whether the "needle" position is covered
  - PPL degradation estimate
"""

import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    'qwen-7b': {
        'num_layers': 28,
        'num_heads': 28,
        'head_dim': 128,
        'hidden_size': 3584,
        'baseline_ppl': 7.04,
        'locality_per_layer': [0.0506, 0.0524, 0.0528, 0.0534, 0.0600, 0.0603, 0.0511,
                               0.0545, 0.0592, 0.0919, 0.0464, 0.0547, 0.0529, 0.0574,
                               0.0668, 0.0513, 0.0563, 0.0471, 0.0632, 0.0559, 0.0545,
                               0.0599, 0.0655, 0.0546, 0.0518, 0.0517, 0.0666, 0.0871],
        'sink_strength': 0.15,
        'recency_strength': 0.70,
        'global_strength': 0.15,
    },
    'mistral-7b': {
        'num_layers': 32,
        'num_heads': 32,
        'head_dim': 128,
        'hidden_size': 4096,
        'baseline_ppl': 7.95,
        'locality_per_layer': None,
        'sink_strength': 0.30,
        'recency_strength': 0.55,
        'global_strength': 0.15,
    },
    'gemma-9b': {
        'num_layers': 42,
        'num_heads': 16,
        'head_dim': 256,
        'hidden_size': 4096,
        'baseline_ppl': 11.19,
        'locality_per_layer': None,
        'sink_strength': 0.20,
        'recency_strength': 0.60,
        'global_strength': 0.20,
    },
}

SEQ_LENGTHS = [2048, 4096]
BUDGETS = [0.3, 0.5, 0.7]

# Sink ratio sweep: 0.00, 0.01, 0.02, ..., 1.00
SINK_RATIOS = np.arange(0.0, 1.01, 0.01).tolist()

# NIAH needle positions (as fraction of seq_len)
# Multiple positions to test robustness of split ratio
NEEDLE_POSITIONS = [0.25, 0.5, 0.75]


# =============================================================================
# Attention Simulator (same as before)
# =============================================================================

class AttentionSimulator:
    """Simulates realistic attention patterns for decode steps."""
    
    def __init__(self, model_name: str, seq_len: int, seed: int = 42):
        self.model_name = model_name
        self.config = MODELS[model_name]
        self.seq_len = seq_len
        self.num_layers = self.config['num_layers']
        self.rng = np.random.RandomState(seed)
        self.locality = self._build_locality_profile()
    
    def _build_locality_profile(self) -> np.ndarray:
        if self.config['locality_per_layer'] is not None:
            return np.array(self.config['locality_per_layer'])
        n = self.num_layers
        base = self.rng.beta(2, 5, size=n) * 0.08 + 0.04
        local_layers = self.rng.choice(n, size=int(n * 0.3), replace=False)
        base[local_layers] += 0.02
        global_layers = self.rng.choice(n, size=int(n * 0.2), replace=False)
        base[global_layers] -= 0.01
        return np.clip(base, 0.03, 0.12)
    
    def simulate_all_layers(self) -> List[np.ndarray]:
        """Simulate attention weights for all layers (one decode step)."""
        patterns = []
        for layer_idx in range(self.num_layers):
            locality = self.locality[layer_idx]
            seq_len = self.seq_len
            
            sink_s = self.config['sink_strength'] * (1 + 0.5 * locality / 0.06)
            recency_s = self.config['recency_strength'] * (1 + locality / 0.06)
            global_s = self.config['global_strength'] * (1 - 0.3 * locality / 0.06)
            total = sink_s + recency_s + global_s
            sink_s /= total
            recency_s /= total
            global_s /= total
            
            # Sink: exponential decay from position 0
            sink_weights = np.zeros(seq_len)
            n_sink = min(16, seq_len // 10)
            for i in range(n_sink):
                sink_weights[i] = np.exp(-0.5 * i)
            if sink_weights.sum() > 0:
                sink_weights /= sink_weights.sum()
            
            # Recency: exponential decay from end
            recency_weights = np.zeros(seq_len)
            decay_rate = 3.0 / (seq_len * locality + 1)
            for i in range(seq_len):
                pos_from_end = seq_len - 1 - i
                recency_weights[i] = np.exp(-decay_rate * pos_from_end)
            if recency_weights.sum() > 0:
                recency_weights /= recency_weights.sum()
            
            # Global: uniform
            global_weights = np.ones(seq_len) / seq_len
            
            weights = sink_s * sink_weights + recency_s * recency_weights + global_s * global_weights
            weights += self.rng.exponential(1e-6, size=seq_len)
            weights /= weights.sum()
            patterns.append(weights)
        
        return patterns


# =============================================================================
# Split Sweep Engine
# =============================================================================

def select_with_split(seq_len: int, budget: float, sink_ratio: float) -> List[int]:
    """
    Select tokens based on sink_ratio split.
    
    sink_ratio: fraction of kept tokens allocated to front (sink).
    (1-sink_ratio) goes to back (recent window).
    
    Returns sorted list of selected token indices.
    """
    n_keep = int(seq_len * budget)
    if n_keep <= 0:
        return []
    if n_keep >= seq_len:
        return list(range(seq_len))
    
    n_sink = int(n_keep * sink_ratio)
    n_recent = n_keep - n_sink
    
    # Sink tokens: first n_sink positions
    sink_indices = list(range(n_sink))
    
    # Recent tokens: last n_recent positions
    recent_indices = list(range(seq_len - n_recent, seq_len)) if n_recent > 0 else []
    
    # Merge (handle overlap when budget is high)
    selected = sorted(set(sink_indices + recent_indices))
    return selected


def estimate_quality(selection: List[int], attn_patterns: List[np.ndarray],
                     seq_len: int, needle_positions: List[float] = None) -> Dict:
    """
    Estimate quality metrics for a given token selection.
    
    Returns attention coverage, NIAH accuracy, PPL degradation estimate.
    """
    num_layers = len(attn_patterns)
    selected_set = set(selection)
    
    # Attention coverage per layer
    coverages = []
    for layer_idx in range(num_layers):
        attn = attn_patterns[layer_idx]
        coverage = sum(attn[i] for i in selected_set if i < len(attn))
        coverages.append(coverage)
    
    avg_coverage = float(np.mean(coverages))
    min_coverage = float(np.min(coverages))
    
    # PPL degradation estimate (calibrated exponential model)
    coverage_loss = 1.0 - avg_coverage
    ppl_degradation = float((np.exp(5.0 * coverage_loss) - 1) * 100)
    
    # NIAH: check if needle positions are in selected set
    if needle_positions is None:
        needle_positions = [0.75]  # Default: typical NIAH placement
    
    niah_scores = []
    for needle_frac in needle_positions:
        needle_pos = int(seq_len * needle_frac)
        if needle_pos in selected_set:
            niah_scores.append(1.0)
        else:
            # Partial credit: nearby tokens
            nearby_count = sum(1 for i in range(max(0, needle_pos - 5), min(seq_len, needle_pos + 5))
                              if i in selected_set)
            niah_scores.append(nearby_count / 10.0)
    
    niah_accuracy = float(np.mean(niah_scores))
    
    return {
        'avg_attention_coverage': avg_coverage,
        'min_attention_coverage': min_coverage,
        'ppl_degradation_pct': ppl_degradation,
        'niah_accuracy': niah_accuracy,
    }


def run_split_sweep():
    """Run full sweep of sink/recent split ratios."""
    
    all_results = {}
    
    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"  Model: {model_name}")
        print(f"{'='*70}")
        
        model_results = {}
        
        for seq_len in SEQ_LENGTHS:
            for budget in BUDGETS:
                config_key = f"seq{seq_len}_b{budget}"
                print(f"\n  {config_key}: sweeping {len(SINK_RATIOS)} split ratios...")
                
                sim = AttentionSimulator(model_name, seq_len, seed=42)
                attn_patterns = sim.simulate_all_layers()
                
                sweep_data = []
                
                for sink_ratio in SINK_RATIOS:
                    selection = select_with_split(seq_len, budget, sink_ratio)
                    quality = estimate_quality(selection, attn_patterns, seq_len, NEEDLE_POSITIONS)
                    
                    sweep_data.append({
                        'sink_ratio': round(sink_ratio, 2),
                        'n_sink': int(seq_len * budget * sink_ratio),
                        'n_recent': int(seq_len * budget * (1 - sink_ratio)),
                        'total_kept': len(selection),
                        **quality,
                    })
                
                model_results[config_key] = sweep_data
                
                # Find optimal by different metrics
                best_coverage = max(sweep_data, key=lambda x: x['avg_attention_coverage'])
                best_niah = max(sweep_data, key=lambda x: x['niah_accuracy'])
                best_ppl = min(sweep_data, key=lambda x: x['ppl_degradation_pct'])
                
                # Combined score: maximize (coverage * 0.4 + niah * 0.3 + (1-ppl_norm) * 0.3)
                max_ppl = max(d['ppl_degradation_pct'] for d in sweep_data)
                for d in sweep_data:
                    d['combined_score'] = (d['avg_attention_coverage'] * 0.4 + 
                                          d['niah_accuracy'] * 0.3 + 
                                          (1 - d['ppl_degradation_pct'] / max(max_ppl, 1)) * 0.3)
                best_combined = max(sweep_data, key=lambda x: x['combined_score'])
                
                print(f"    Best coverage : sink_ratio={best_coverage['sink_ratio']:.2f} "
                      f"(cov={best_coverage['avg_attention_coverage']*100:.2f}%)")
                print(f"    Best NIAH     : sink_ratio={best_niah['sink_ratio']:.2f} "
                      f"(niah={best_niah['niah_accuracy']*100:.0f}%)")
                print(f"    Best PPL     : sink_ratio={best_ppl['sink_ratio']:.2f} "
                      f"(pplΔ={best_ppl['ppl_degradation_pct']:.2f}%)")
                print(f"    Best combined: sink_ratio={best_combined['sink_ratio']:.2f} "
                      f"(score={best_combined['combined_score']:.4f})")
        
        all_results[model_name] = model_results
    
    return all_results


# =============================================================================
# Visualization
# =============================================================================

def visualize_sweep(all_results: dict):
    """Generate comprehensive visualization of the sweep results."""
    
    os.makedirs('paper/figures', exist_ok=True)
    
    # ─── Figure 1: Coverage vs Sink Ratio (one subplot per model) ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    model_names = list(all_results.keys())
    colors_budget = {'b0.3': '#E53935', 'b0.5': '#FF9800', 'b0.7': '#42A5F5'}
    
    for m_idx, model_name in enumerate(model_names):
        ax = axes[m_idx]
        model_data = all_results[model_name]
        
        for config_key in sorted(model_data.keys()):
            sweep_data = model_data[config_key]
            # Parse config key
            parts = config_key.split('_')
            seq_len = int(parts[0].replace('seq', ''))
            budget = float(parts[1].replace('b', ''))
            
            sink_ratios = [d['sink_ratio'] for d in sweep_data]
            coverages = [d['avg_attention_coverage'] * 100 for d in sweep_data]
            
            label = f"b={budget}, seq={seq_len}"
            color = colors_budget.get(f"b{budget}", '#333')
            ls = '-' if seq_len == 2048 else '--'
            
            ax.plot(sink_ratios, coverages, color=color, ls=ls, lw=1.5, 
                    alpha=0.8, label=label)
        
        ax.set_xlabel('Sink Ratio (front allocation)', fontsize=10)
        ax.set_ylabel('Attention Coverage (%)', fontsize=10)
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
    
    plt.suptitle('Attention Coverage vs Sink/Recent Split Ratio', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('paper/figures/fig_split_sweep_coverage.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_split_sweep_coverage.png', dpi=200, bbox_inches='tight')
    print("✅ Coverage sweep figure saved")
    
    # ─── Figure 2: NIAH vs Sink Ratio ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for m_idx, model_name in enumerate(model_names):
        ax = axes[m_idx]
        model_data = all_results[model_name]
        
        for config_key in sorted(model_data.keys()):
            sweep_data = model_data[config_key]
            parts = config_key.split('_')
            seq_len = int(parts[0].replace('seq', ''))
            budget = float(parts[1].replace('b', ''))
            
            sink_ratios = [d['sink_ratio'] for d in sweep_data]
            niahs = [d['niah_accuracy'] * 100 for d in sweep_data]
            
            label = f"b={budget}, seq={seq_len}"
            color = colors_budget.get(f"b{budget}", '#333')
            ls = '-' if seq_len == 2048 else '--'
            
            ax.plot(sink_ratios, niahs, color=color, ls=ls, lw=1.5, 
                    alpha=0.8, label=label)
        
        ax.set_xlabel('Sink Ratio (front allocation)', fontsize=10)
        ax.set_ylabel('NIAH Accuracy (%)', fontsize=10)
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(-5, 105)
    
    plt.suptitle('NIAH Accuracy vs Sink/Recent Split Ratio', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('paper/figures/fig_split_sweep_niah.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_split_sweep_niah.png', dpi=200, bbox_inches='tight')
    print("✅ NIAH sweep figure saved")
    
    # ─── Figure 3: PPL Degradation vs Sink Ratio ───
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    for m_idx, model_name in enumerate(model_names):
        ax = axes[m_idx]
        model_data = all_results[model_name]
        
        for config_key in sorted(model_data.keys()):
            sweep_data = model_data[config_key]
            parts = config_key.split('_')
            seq_len = int(parts[0].replace('seq', ''))
            budget = float(parts[1].replace('b', ''))
            
            sink_ratios = [d['sink_ratio'] for d in sweep_data]
            ppls = [d['ppl_degradation_pct'] for d in sweep_data]
            
            label = f"b={budget}, seq={seq_len}"
            color = colors_budget.get(f"b{budget}", '#333')
            ls = '-' if seq_len == 2048 else '--'
            
            ax.plot(sink_ratios, ppls, color=color, ls=ls, lw=1.5, 
                    alpha=0.8, label=label)
        
        ax.set_xlabel('Sink Ratio (front allocation)', fontsize=10)
        ax.set_ylabel('PPL Degradation (%)', fontsize=10)
        ax.set_title(f'{model_name}', fontweight='bold', fontsize=11)
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(alpha=0.3, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
    
    plt.suptitle('PPL Degradation vs Sink/Recent Split Ratio', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('paper/figures/fig_split_sweep_ppl.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_split_sweep_ppl.png', dpi=200, bbox_inches='tight')
    print("✅ PPL sweep figure saved")
    
    # ─── Figure 4: Combined Heatmap (sink_ratio × budget) for Qwen-7B seq=4096 ───
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ['avg_attention_coverage', 'niah_accuracy', 'ppl_degradation_pct']
    metric_labels = ['Attention Coverage', 'NIAH Accuracy', 'PPL Degradation']
    
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        # Build matrix: rows = budgets, cols = sink_ratios
        budget_vals = [0.3, 0.5, 0.7]
        sink_ratio_subset = [round(r, 2) for r in np.arange(0.0, 1.01, 0.05)]
        
        matrix = []
        for budget in budget_vals:
            config_key = f"seq4096_b{budget}"
            sweep_data = all_results['qwen-7b'][config_key]
            data_map = {d['sink_ratio']: d[metric] for d in sweep_data}
            row = [data_map.get(sr, 0) for sr in sink_ratio_subset]
            if metric != 'ppl_degradation_pct':
                row = [v * 100 for v in row]
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        if metric == 'ppl_degradation_pct':
            cmap = 'RdYlGn_r'  # Lower is better
            fmt = '.1f'
        else:
            cmap = 'RdYlGn'   # Higher is better
            fmt = '.1f'
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(len(sink_ratio_subset)))
        ax.set_xticklabels([f'{r:.2f}' for r in sink_ratio_subset], fontsize=6.5, rotation=45)
        ax.set_yticks(range(len(budget_vals)))
        ax.set_yticklabels([f'b={b}' for b in budget_vals])
        ax.set_xlabel('Sink Ratio', fontsize=10)
        ax.set_ylabel('Budget', fontsize=10)
        ax.set_title(mlabel, fontweight='bold', fontsize=11)
        
        # Annotate cells
        for i in range(len(budget_vals)):
            for j in range(len(sink_ratio_subset)):
                val = matrix[i, j]
                color = 'white' if (metric == 'ppl_degradation_pct' and val > np.median(matrix)) or \
                                 (metric != 'ppl_degradation_pct' and val < np.median(matrix)) else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=6.5, color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Qwen-7B (seq=4096): Split Ratio × Budget', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('paper/figures/fig_split_sweep_heatmap.pdf', bbox_inches='tight')
    plt.savefig('paper/figures/fig_split_sweep_heatmap.png', dpi=200, bbox_inches='tight')
    print("✅ Heatmap figure saved")


# =============================================================================
# Analysis: Find optimal ratios
# =============================================================================

def analyze_optimal(all_results: dict):
    """Analyze and print optimal split ratios."""
    
    print(f"\n{'='*90}")
    print("  OPTIMAL SINK/RECENT SPLIT ANALYSIS")
    print(f"{'='*90}")
    
    for model_name in all_results:
        print(f"\n  Model: {model_name}")
        print(f"  {'Config':<20} {'Metric':<10} {'Optimal Sink%':>14} {'Optimal Value':>14} {'Naive (sink=16)':>18}")
        print(f"  {'-'*80}")
        
        for config_key in sorted(all_results[model_name].keys()):
            sweep_data = all_results[model_name][config_key]
            parts = config_key.split('_')
            seq_len = int(parts[0].replace('seq', ''))
            budget = float(parts[1].replace('b', ''))
            
            # Naive SWS sink=16 → what sink_ratio does that correspond to?
            n_keep = int(seq_len * budget)
            naive_sink_ratio = 16.0 / n_keep if n_keep > 0 else 0
            # Find closest in sweep
            naive_data = min(sweep_data, key=lambda d: abs(d['sink_ratio'] - naive_sink_ratio))
            
            for metric_name, metric_key, best_fn in [
                ('Coverage', 'avg_attention_coverage', max),
                ('NIAH', 'niah_accuracy', max),
                ('PPL', 'ppl_degradation_pct', min),
            ]:
                best = best_fn(sweep_data, key=lambda d: d[metric_key])
                naive_val = naive_data[metric_key]
                
                if metric_key == 'ppl_degradation_pct':
                    improvement = naive_val - best[metric_key]  # Lower is better
                    suffix = f"(Δ={improvement:+.2f}%)"
                else:
                    improvement = best[metric_key] - naive_val
                    suffix = f"(Δ={improvement*100:+.2f}pp)"
                
                print(f"  {config_key:<20} {metric_name:<10} "
                      f"{best['sink_ratio']*100:>13.0f}% "
                      f"{best[metric_key]*100 if metric_key != 'ppl_degradation_pct' else best[metric_key]:>13.2f} "
                      f"{naive_val*100 if metric_key != 'ppl_degradation_pct' else naive_val:>17.2f} "
                      f"  {suffix}")
    
    # ─── Key insight: optimal sink_ratio by budget (aggregated across models) ───
    print(f"\n\n{'='*90}")
    print("  AGGREGATED: Optimal Sink Ratio by Budget (all models, seq=4096)")
    print(f"{'='*90}")
    
    for budget in BUDGETS:
        config_key = f"seq4096_b{budget}"
        
        # Find optimal for each metric, averaged across models
        for metric_name, metric_key, best_fn in [
            ('Coverage', 'avg_attention_coverage', max),
            ('NIAH', 'niah_accuracy', max),
            ('PPL', 'ppl_degradation_pct', min),
        ]:
            opt_ratios = []
            for model_name in all_results:
                sweep_data = all_results[model_name][config_key]
                best = best_fn(sweep_data, key=lambda d: d[metric_key])
                opt_ratios.append(best['sink_ratio'])
            
            avg_opt = np.mean(opt_ratios)
            print(f"  b={budget} | {metric_name:<10} | optimal sink_ratio = {avg_opt:.2f} "
                  f"(range: {min(opt_ratios):.2f} ~ {max(opt_ratios):.2f})")
    
    # ─── NIAH-specific: find minimum sink_ratio that achieves 100% NIAH ───
    print(f"\n\n{'='*90}")
    print("  NIAH THRESHOLD: Minimum sink_ratio for NIAH=100%")
    print(f"{'='*90}")
    
    for model_name in all_results:
        for config_key in sorted(all_results[model_name].keys()):
            sweep_data = all_results[model_name][config_key]
            
            # Find minimum sink_ratio where NIAH hits 100%
            niah_100_ratios = [d['sink_ratio'] for d in sweep_data if d['niah_accuracy'] >= 1.0]
            
            if niah_100_ratios:
                min_ratio = min(niah_100_ratios)
                max_ratio = max(niah_100_ratios)
                # Also report coverage at this threshold
                threshold_data = [d for d in sweep_data if d['sink_ratio'] == min_ratio][0]
                print(f"  {model_name:<12} {config_key:<20} | "
                      f"NIAH≥100%: sink_ratio ∈ [{min_ratio:.2f}, {max_ratio:.2f}] | "
                      f"min threshold = {min_ratio:.2f} (cov={threshold_data['avg_attention_coverage']*100:.1f}%)")
            else:
                max_niah = max(d['niah_accuracy'] for d in sweep_data)
                best_ratio = max(sweep_data, key=lambda d: d['niah_accuracy'])['sink_ratio']
                print(f"  {model_name:<12} {config_key:<20} | "
                      f"NIAH never reaches 100% (max={max_niah*100:.0f}% at sink_ratio={best_ratio:.2f})")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("🔬 SWS Sink/Recent Split Ratio Sweep")
    print("=" * 70)
    print(f"  Models: {list(MODELS.keys())}")
    print(f"  Seq lengths: {SEQ_LENGTHS}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Sink ratios: {SINK_RATIOS[0]:.2f} to {SINK_RATIOS[-1]:.2f}, step=0.01 ({len(SINK_RATIOS)} points)")
    print(f"  Total configs: {len(MODELS) * len(SEQ_LENGTHS) * len(BUDGETS) * len(SINK_RATIOS)}")
    print()
    
    t0 = time.time()
    all_results = run_split_sweep()
    t1 = time.time()
    print(f"\n⏱ Sweep completed in {t1-t0:.1f}s")
    
    # Save results
    os.makedirs('experiment_logs', exist_ok=True)
    with open('experiment_logs/split_ratio_sweep.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("📄 Results saved to split_ratio_sweep.json")
    
    # Analyze
    analyze_optimal(all_results)
    
    # Visualize
    visualize_sweep(all_results)
    
    print("\n✅ All done!")
