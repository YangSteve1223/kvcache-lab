#!/usr/bin/env python3
"""
Figure 5: Concurrent Request Scaling
Shows how SWS enables 8x more concurrent requests compared to Full KV
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
with open('../../gpu-experiments/results/g7g8_all_results.json', 'r') as f:
    g7g8_data = json.load(f)

concurrency_data = g7g8_data['g7_concurrency']

# Extract data
seq_lens = [c['seq_len'] for c in concurrency_data]
full_kv_max_req = [c['full_kv_max_req'] for c in concurrency_data]
sws_256_max_req = [c['sws_256_max_req'] for c in concurrency_data]
sws_512_max_req = [c['sws_512_max_req'] for c in concurrency_data]
sws_1024_max_req = [c['sws_1024_max_req'] for c in concurrency_data]

# Calculate improvement ratios
improvement_ratios = [s/f for s, f in zip(sws_512_max_req, full_kv_max_req)]

# Figure setup
fig, ax = plt.subplots(figsize=(6.5, 4))

# Color palette
colors = {
    'full_kv': '#e74c3c',      # Red
    'sws_256': '#27ae60',      # Green
    'sws_512': '#3498db',      # Blue
    'sws_1024': '#9b59b6',     # Purple
}

# Plot concurrent request capacity
ax.plot(seq_lens, full_kv_max_req, 'o-', color=colors['full_kv'],
        linewidth=2, markersize=8, label='Full KV')
ax.plot(seq_lens, sws_256_max_req, 's--', color=colors['sws_256'],
        linewidth=1.5, markersize=6, label='SWS (ws=256)', alpha=0.7)
ax.plot(seq_lens, sws_512_max_req, '^-', color=colors['sws_512'],
        linewidth=2, markersize=7, label='SWS (ws=512)')
ax.plot(seq_lens, sws_1024_max_req, 'D-.', color=colors['sws_1024'],
        linewidth=1.5, markersize=6, label='SWS (ws=1024)', alpha=0.7)

# Log scale for both axes
ax.set_xscale('log', base=2)
ax.set_yscale('log')

# Mark the 8x improvement point
# At seq_len=4096: Full KV = 372, SWS-512 = 2978, ratio = 8x
idx_4k = seq_lens.index(4096)
ax.annotate(f'8x improvement\nat 4K context',
            xy=(4096, sws_512_max_req[idx_4k]),
            xytext=(6000, 4000),
            fontsize=9,
            ha='center',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                          connectionstyle='arc3,rad=0.2'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                      edgecolor='gray', alpha=0.9))

# Draw bracket showing 8x
ax.annotate('',
            xy=(4096, full_kv_max_req[idx_4k]),
            xytext=(4096, sws_512_max_req[idx_4k]),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))

# Labels
ax.set_xlabel('Context Length (tokens)')
ax.set_ylabel('Max Concurrent Requests')
ax.set_title('Figure 5: Concurrent Request Capacity', fontweight='bold', pad=10)

# Format x-axis labels
ax.set_xticks(seq_lens)
ax.set_xticklabels([f'{s//1024}K' for s in seq_lens])

# Grid
ax.grid(True, linestyle='--', alpha=0.4, which='both')

# Legend
ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', ncol=2)

plt.tight_layout()

# Save
plt.savefig('fig5_concurrent_scaling.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig5_concurrent_scaling.png', format='png', bbox_inches='tight')
print("Figure 5 saved: fig5_concurrent_scaling.pdf/png")

plt.close()
