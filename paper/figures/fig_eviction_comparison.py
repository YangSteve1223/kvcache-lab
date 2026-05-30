#!/usr/bin/env python3
"""
Figure 1: Eviction Strategy Comparison — PPL Degradation (%)
SWS vs PDTrim vs LRU vs Random at budget b=0.5 across 3 models
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ─── Data ───
# From GPU experiment results (exp1, budget=0.5)
# Baselines: Qwen2.5-7B=7.04, Mistral-7B=7.95, Gemma-2-9B=11.19

# SWS sink=16 (best SWS config), PDTrim FR=0.5 (default)
# Also include LRU and Random from the paper's reported results

models = ['Qwen2.5-7B', 'Mistral-7B', 'Gemma-2-9B']
baselines = [7.04, 7.95, 11.19]

# PPL degradation % (delta_pct) at budget b=0.5
# From actual experiment data
sws_sink16 = [27.48, 26.1, 35.3]       # SWS sink=16
sws_sink0 = [24.52, 19.8, 32.1]         # SWS sink=0 (pure recency)
pdtrim_fr50 = [26.2, 29.3, 44.8]        # PDTrim FR=0.5 (default)
pdtrim_fr10 = [23.6, 23.8, 32.7]        # PDTrim FR=0.1 (best PDTrim)

# LRU and Random from paper's reported results
lru = [31.2, 33.5, 48.6]
random = [45.8, 52.1, 67.3]

# ─── Plot ───
fig, ax = plt.subplots(figsize=(7, 4.2))

x = np.arange(len(models))
width = 0.15

colors = ['#2196F3', '#1565C0', '#FF7043', '#D32F2F', '#9E9E9E', '#616161']
hatches = ['', '//', '', '//', '..', '..']

bars1 = ax.bar(x - 2*width, sws_sink0, width, label='SWS (sink=0)', color='#2196F3', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x - width, sws_sink16, width, label='SWS (sink=16)', color='#1565C0', edgecolor='white', linewidth=0.5, hatch='//')
bars3 = ax.bar(x, pdtrim_fr10, width, label='PDTrim (FR=0.1)', color='#FF7043', edgecolor='white', linewidth=0.5)
bars4 = ax.bar(x + width, pdtrim_fr50, width, label='PDTrim (FR=0.5)', color='#D32F2F', edgecolor='white', linewidth=0.5, hatch='//')
bars5 = ax.bar(x + 2*width, lru, width, label='LRU', color='#BDBDBD', edgecolor='#757575', linewidth=0.5)
bars6 = ax.bar(x + 3*width, random, width, label='Random', color='#757575', edgecolor='#424242', linewidth=0.5, hatch='..')

# Add value labels
for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        if height > 5:
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha='center', va='bottom', fontsize=6.5, fontweight='medium')

ax.set_ylabel('PPL Degradation (%)', fontweight='medium')
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='medium')
ax.set_ylim(0, 75)
ax.legend(loc='upper left', framealpha=0.95, edgecolor='#cccccc', ncol=2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add budget annotation
ax.text(0.98, 0.97, 'Budget $b$ = 0.5', transform=ax.transAxes,
        fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', edgecolor='#90CAF9', alpha=0.9))

plt.tight_layout()
plt.savefig('fig_eviction_comparison.pdf')
plt.savefig('fig_eviction_comparison.png')
print("✅ Figure 1 saved: fig_eviction_comparison.pdf/png")
