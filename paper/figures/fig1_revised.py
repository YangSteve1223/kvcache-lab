#!/usr/bin/env python3
"""
Figure 1 (Revised): Eviction Strategy Comparison — ONLY real experimental data
SWS vs PDTrim at budget b=0.5 across 3 models
No fabricated data.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12.5,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

import os
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

models = ['Qwen2.5-7B', 'Mistral-7B', 'Gemma-2-9B']

# ─── ONLY real experimental data from GPU experiments (exp1, budget=0.5) ───
# Source: kvcache-lab/gpu-experiments/experiment_results_merged/exp1_pdtrim_ppl_*.json

# SWS sink=0
sws_sink0 = [24.52, 19.8, 32.1]
# SWS sink=16
sws_sink16 = [27.48, 26.1, 35.3]
# PDTrim FR=0.1 (best PDTrim)
pdtrim_fr10 = [23.6, 23.8, 32.7]
# PDTrim FR=0.5 (default PDTrim)
pdtrim_fr50 = [26.2, 29.3, 44.8]

fig, ax = plt.subplots(figsize=(6.5, 4))
x = np.arange(len(models))
w = 0.18

b1 = ax.bar(x - 1.5*w, sws_sink0, w, label='SWS (sink=0)',
           color='#42A5F5', ec='white', lw=0.5, zorder=3)
b2 = ax.bar(x - 0.5*w, sws_sink16, w, label='SWS (sink=16)',
           color='#1565C0', ec='white', lw=0.5, zorder=3)
b3 = ax.bar(x + 0.5*w, pdtrim_fr10, w, label='PDTrim (FR=0.1)',
           color='#FF7043', ec='white', lw=0.5, zorder=3)
b4 = ax.bar(x + 1.5*w, pdtrim_fr50, w, label='PDTrim (FR=0.5)',
           color='#D32F2F', ec='white', lw=0.5, zorder=3)

for bars in [b1, b2, b3, b4]:
    for bar in bars:
        h = bar.get_height()
        if h > 2:
            ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7, fontweight='medium')

ax.set_ylabel('PPL Degradation (%)', fontweight='medium')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='medium')
ax.set_ylim(0, 55)
ax.legend(loc='upper left', framealpha=0.95, edgecolor='#ddd', fontsize=9,
         borderpad=0.4, handlelength=1.3)
ax.grid(axis='y', alpha=0.2, linestyle='--', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.98, 0.97, 'Budget $b$ = 0.5', transform=ax.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#90CAF9', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_eviction_comparison.pdf'))
plt.savefig(os.path.join(OUT_DIR, 'fig_eviction_comparison.png'))
plt.close()
print("✅ Fig 1 (revised): Only real data, no LRU/Random")
