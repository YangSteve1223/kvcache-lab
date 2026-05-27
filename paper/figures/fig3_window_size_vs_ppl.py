#!/usr/bin/env python3
"""
Figure 3: SWS Window Size vs PPL
Shows PPL for SWS only vs SWS+TAA with different window sizes,
标注50% memory budget对应的window size
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
with open('../../gpu-experiments/results/g4_all_results.json', 'r') as f:
    g4_data = json.load(f)

with open('../../gpu-experiments/results/g7g8_all_results.json', 'r') as f:
    g7g8_data = json.load(f)

# G4: quality vs window size data
quality_data = g4_data['part2_quality_vs_window']
baseline_ppl_g4 = quality_data['baseline_ppl']
seq_len_g4 = quality_data['seq_len']
results_g4 = quality_data['results']

window_sizes_g4 = [r['window_size'] for r in results_g4]
sws_ppls_g4 = [r['sws_ppl'] for r in results_g4]
sws_taa_ppls_g4 = [r['sws_taa_ppl'] for r in results_g4]

# G8: memory budget vs quality tradeoff
tradeoff = g7g8_data['g8_tradeoff']
baseline_ppl_g8 = tradeoff['baseline_ppl']
results_g8 = tradeoff['results']

# Memory budget data: budget_pct, window_size
g8_budgets = [r['budget_pct'] * 100 for r in results_g8]
g8_window_sizes = [r['window_size'] for r in results_g8]

# Find 50% budget window size
idx_50 = g8_budgets.index(50.0)
ws_50 = g8_window_sizes[idx_50]

# Figure setup
fig, ax = plt.subplots(figsize=(6, 4))

# Color palette
colors = {
    'baseline': '#2c3e50',     # Dark blue-gray
    'sws_only': '#e74c3c',     # Red
    'sws_taa': '#27ae60',      # Green
}

# Plot G4 data (seq_len=511)
ax.plot(window_sizes_g4, sws_ppls_g4, 's--', color=colors['sws_only'],
        linewidth=1.5, markersize=7, label='SWS Only', alpha=0.8)
ax.plot(window_sizes_g4, sws_taa_ppls_g4, '^-.', color=colors['sws_taa'],
        linewidth=2, markersize=7, label='SWS + TAA')

# Baseline reference line
ax.axhline(y=baseline_ppl_g4, color=colors['baseline'], 
           linestyle=':', linewidth=1.5, label=f'Baseline PPL={baseline_ppl_g4:.2f}')

# Mark 50% memory budget point (ws_50)
# At 50% budget with seq_len=511, window_size should be ~255
# From g8_tradeoff: budget_pct=0.5 gives window_size=255
# Note: G4 uses seq_len=511, G8 uses different sequence length
# We'll annotate where 50% budget would be

# Find where the curves approach baseline
ax.axvline(x=ws_50, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.plot(ws_50, baseline_ppl_g4, '*', color='#f39c12', markersize=15, zorder=5)

# Annotation for 50% budget
ax.annotate(f'50% Memory Budget\nWindow Size={ws_50}',
            xy=(ws_50, baseline_ppl_g4),
            xytext=(ws_50 + 50, baseline_ppl_g4 + 2),
            fontsize=9,
            ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                      edgecolor='gray', alpha=0.9))

# Labels
ax.set_xlabel('Window Size (tokens)')
ax.set_ylabel('PPL')
ax.set_title('Figure 3: PPL vs Sliding Window Size', fontweight='bold', pad=10)

# Log scale for x-axis
ax.set_xscale('log', base=2)
ax.set_xlim([20, 400])

# Grid
ax.grid(True, linestyle='--', alpha=0.4, which='both')

# Legend
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')

plt.tight_layout()

# Save
plt.savefig('fig3_window_size_vs_ppl.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig3_window_size_vs_ppl.png', format='png', bbox_inches='tight')
print("Figure 3 saved: fig3_window_size_vs_ppl.pdf/png")

plt.close()
