#!/usr/bin/env python3
"""
Figure 3: Budget vs Quality Tradeoff — PPL degradation as budget varies
SWS vs PDTrim across 3 budgets (0.3, 0.5, 0.7)
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ─── Data from GPU experiments (exp1) ───
# PPL delta_pct for SWS (sink=16) vs PDTrim (FR=0.5) at budgets 0.3/0.5/0.7
budgets = [0.3, 0.5, 0.7]

# SWS sink=16
sws_qwen = [48.0, 27.48, 17.45]
sws_mistral = [35.3, 26.1, 17.2]
sws_gemma = [94.0, 35.3, 19.8]

# PDTrim FR=0.5 (default)
pdtrim_qwen = [57.33, 26.2, 13.7]
pdtrim_mistral = [48.8, 29.3, 22.9]
pdtrim_gemma = [103.4, 44.8, 17.4]

# PDTrim FR=0.1 (best)
pdtrim_best_qwen = [50.17, 23.6, 14.1]
pdtrim_best_mistral = [40.6, 23.8, 17.2]
pdtrim_best_gemma = [94.9, 32.7, 16.9]

# ─── Plot ───
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.subplots_adjust(wspace=0.3)

model_names = ['Qwen2.5-7B', 'Mistral-7B', 'Gemma-2-9B']
sws_data = [sws_qwen, sws_mistral, sws_gemma]
pdtrim_data = [pdtrim_qwen, pdtrim_mistral, pdtrim_gemma]
pdtrim_best_data = [pdtrim_best_qwen, pdtrim_best_mistral, pdtrim_best_gemma]
baselines_ppl = [7.04, 7.95, 11.19]

for idx, ax in enumerate(axes):
    b = np.array(budgets)
    
    # Plot lines
    ax.plot(b, sws_data[idx], 'o-', color='#1565C0', linewidth=2.2, markersize=7,
            label='SWS (sink=16)', zorder=5)
    ax.plot(b, pdtrim_data[idx], 's--', color='#D32F2F', linewidth=1.8, markersize=6,
            label='PDTrim (FR=0.5)', alpha=0.85, zorder=4)
    ax.plot(b, pdtrim_best_data[idx], '^:', color='#FF7043', linewidth=1.5, markersize=5.5,
            label='PDTrim (FR=0.1)', alpha=0.75, zorder=3)
    
    # Fill area between SWS and PDTrim
    ax.fill_between(b, sws_data[idx], pdtrim_data[idx], 
                    where=[s < p for s, p in zip(sws_data[idx], pdtrim_data[idx])],
                    alpha=0.1, color='#1565C0', interpolate=True)
    
    # Annotate specific points
    for i, budget in enumerate(budgets):
        # SWS value
        ax.annotate(f'{sws_data[idx][i]:.1f}%',
                   xy=(budget, sws_data[idx][i]),
                   xytext=(0, 8), textcoords='offset points',
                   fontsize=7.5, ha='center', color='#1565C0', fontweight='medium')
    
    ax.set_title(model_names[idx], fontweight='bold', fontsize=12)
    ax.set_xlabel('Budget $b$ (KV retention ratio)', fontweight='medium')
    if idx == 0:
        ax.set_ylabel('PPL Degradation (%)', fontweight='medium')
    
    ax.set_xticks(budgets)
    ax.set_xticklabels(['0.3', '0.5', '0.7'])
    ax.set_ylim(0, max(max(sws_data[idx]), max(pdtrim_data[idx])) * 1.15)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if idx == 0:
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc', fontsize=8.5)

    # Add baseline PPL reference
    ax.text(0.97, 0.03, f'Baseline PPL = {baselines_ppl[idx]:.2f}',
           transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
           color='#757575', style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5', edgecolor='#E0E0E0'))

plt.tight_layout()
plt.savefig('kvcache-lab/paper/figures/fig_budget_tradeoff.pdf')
plt.savefig('kvcache-lab/paper/figures/fig_budget_tradeoff.png')
print("✅ Figure 3 saved: fig_budget_tradeoff.pdf/png")
