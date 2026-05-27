#!/usr/bin/env python3
"""
Figure 4: Eviction Strategy Comparison
Grouped bar chart comparing Random, LRU, and TAA-guided eviction strategies
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Set up academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})

# Load data
with open('../../gpu-experiments/results/g5_all_results.json', 'r') as f:
    g5_data = json.load(f)

eviction_data = g5_data['results']
baseline_ppl = g5_data['baseline_ppl']

# Extract data
evict_ratios = [e['evict_ratio'] for e in eviction_data]
random_ppls = [e['random_ppl'] for e in eviction_data]
random_deltas = [e['random_delta'] for e in eviction_data]
lru_ppls = [e['lru_ppl'] for e in eviction_data]
lru_deltas = [e['lru_delta'] for e in eviction_data]
taa_ppls = [e['taa_guided_ppl'] for e in eviction_data]
taa_deltas = [e['taa_guided_delta'] for e in eviction_data]

# Figure setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Color palette
colors = {
    'random': '#e74c3c',       # Red
    'lru': '#3498db',          # Blue
    'taa': '#27ae60',         # Green
    'baseline': '#2c3e50',    # Dark
}

bar_width = 0.25
x = np.arange(len(evict_ratios))

# Left plot: Absolute PPL (log scale)
bars1 = ax1.bar(x - bar_width, random_ppls, bar_width, label='Random', 
                color=colors['random'], alpha=0.8)
bars2 = ax1.bar(x, lru_ppls, bar_width, label='LRU', 
                color=colors['lru'], alpha=0.8)
bars3 = ax1.bar(x + bar_width, taa_ppls, bar_width, label='TAA-Guided', 
                color=colors['taa'], alpha=0.8)

# Baseline reference
ax1.axhline(y=baseline_ppl, color=colors['baseline'], 
            linestyle='--', linewidth=1.5, label=f'Baseline={baseline_ppl:.2f}')

ax1.set_xlabel('Eviction Ratio')
ax1.set_ylabel('PPL (log scale)')
ax1.set_title('(a) Absolute PPL')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{int(r*100)}%' for r in evict_ratios])
ax1.set_yscale('log')
ax1.legend(loc='upper left', framealpha=0.95)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Add value labels on bars (for significant points)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 10:  # Only label significant values
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

# Right plot: ΔPPL percentage
bars4 = ax2.bar(x - bar_width, random_deltas, bar_width, label='Random', 
                color=colors['random'], alpha=0.8)
bars5 = ax2.bar(x, lru_deltas, bar_width, label='LRU', 
                color=colors['lru'], alpha=0.8)
bars6 = ax2.bar(x + bar_width, taa_deltas, bar_width, label='TAA-Guided', 
                color=colors['taa'], alpha=0.8)

ax2.set_xlabel('Eviction Ratio')
ax2.set_ylabel('PPL Change (%)')
ax2.set_title('(b) Quality Degradation')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{int(r*100)}%' for r in evict_ratios])
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

# Add value labels
for bars in [bars4, bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        if height > 50:  # Only label significant values
            ax2.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

# Main title
fig.suptitle('Figure 4: Eviction Strategy Comparison', fontweight='bold', y=1.02)

plt.tight_layout()

# Save
plt.savefig('fig4_eviction_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig4_eviction_comparison.png', format='png', bbox_inches='tight')
print("Figure 4 saved: fig4_eviction_comparison.pdf/png")

plt.close()
