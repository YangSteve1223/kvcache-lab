#!/usr/bin/env python3
"""
Figure 6: KV Cache Memory Breakdown
Stacked bar chart showing local vs remote KV sizes at different context lengths
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

memory_data = g4_data['part1_memory_reduction']

# Extract data - use window size 512 as the representative SWS setting
# At ws=512: 50% memory reduction regardless of sequence length
seq_lens = [m['seq_len'] for m in memory_data]
full_kv_mb = [m['full_kv_mb'] for m in memory_data]
ws_512_mb = [m['ws_512_mb'] for m in memory_data]
ws_256_mb = [m['ws_256_mb'] for m in memory_data]
ws_128_mb = [m['ws_128_mb'] for m in memory_data]

# Calculate local vs remote breakdown for SWS
# Local KV = window size, Remote KV = full - window
local_kv_512 = [m['ws_512_mb'] for m in memory_data]
remote_kv_512 = [f - l for f, l in zip(full_kv_mb, local_kv_512)]

local_kv_256 = [m['ws_256_mb'] for m in memory_data]
remote_kv_256 = [f - l for f, l in zip(full_kv_mb, local_kv_256)]

# Figure setup
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Color palette
colors = {
    'local': '#27ae60',        # Green (local/hot)
    'remote': '#e74c3c',       # Red (remote/cold)
    'full_kv': '#2c3e50',      # Dark blue-gray
}

x = np.arange(len(seq_lens))
bar_width = 0.35

# Left plot: Full KV vs SWS breakdown
ax1.bar(x - bar_width/2, full_kv_mb, bar_width, label='Full KV', 
        color=colors['full_kv'], alpha=0.9)
ax1.bar(x + bar_width/2, local_kv_512, bar_width, label='Local KV (SWS)', 
        color=colors['local'], alpha=0.9)
ax1.bar(x + bar_width/2, remote_kv_512, bar_width, bottom=local_kv_512,
        label='Remote KV (SWS)', color=colors['remote'], alpha=0.9)

ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('KV Cache Size (MB)')
ax1.set_title('(a) Memory Usage Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{s//1024}K' for s in seq_lens])
ax1.legend(loc='upper left', framealpha=0.95)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

# Add memory saving annotation
saving_pct = [(f-l)/f*100 for f, l in zip(full_kv_mb, local_kv_512)]
for i, (f, l, s) in enumerate(zip(full_kv_mb, local_kv_512, saving_pct)):
    ax1.annotate(f'-{s:.0f}%',
                xy=(x[i] + bar_width/2, l/2),
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='white')

# Right plot: Stacked bar showing SWS savings
ax2.bar(x, local_kv_512, bar_width, label='Local (SWS-512)', 
        color=colors['local'], alpha=0.9)
ax2.bar(x, remote_kv_512, bar_width, bottom=local_kv_512, label='Remote', 
        color=colors['remote'], alpha=0.9)

# Add line showing full KV for reference
ax2.plot(x, full_kv_mb, 'o--', color=colors['full_kv'], 
         linewidth=2, markersize=8, label='Full KV Reference')

ax2.set_xlabel('Sequence Length')
ax2.set_ylabel('KV Cache Size (MB)')
ax2.set_title('(b) SWS Memory Savings')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{s//1024}K' for s in seq_lens])
ax2.legend(loc='upper left', framealpha=0.95)
ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

# Add savings text
for i, (f, l) in enumerate(zip(full_kv_mb, local_kv_512)):
    savings = f - l
    ax2.annotate(f'Save\n{savings:.0f}MB',
                xy=(x[i], f + 50),
                ha='center', fontsize=8, color='gray')

# Main title
fig.suptitle('Figure 6: KV Cache Memory Breakdown', fontweight='bold', y=1.02)

plt.tight_layout()

# Save
plt.savefig('fig6_memory_breakdown.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig6_memory_breakdown.png', format='png', bbox_inches='tight')
print("Figure 6 saved: fig6_memory_breakdown.pdf/png")

plt.close()
