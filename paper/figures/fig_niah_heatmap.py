#!/usr/bin/env python3
"""
Figure 2: NIAH Depth Scan Heatmap
3 models × 3 context lengths × multiple strategies
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.colors as mcolors

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ─── Data from NIAH Depth Scan (9-group, exact percentages) ───
# Format: [2K, 4K, 8K] for each model × strategy combination

data = {
    'Qwen2.5-7B': {
        'Full KV':   [100, 100, 100],
        'PDTrim 0.5': [100, 60,  60],
        'PDTrim 0.7': [100, 86,  86],
        'PDTrim 0.9': [100, 86,  86],
        'SWS 0.5':    [100, 60,  60],
        'SWS 0.7':    [80,  73,  73],
        'SWS 0.9':    [100, 100, 100],
    },
    'Mistral-7B': {
        'Full KV':   [100, 100, 100],
        'PDTrim 0.5': [80,  53,  40],
        'PDTrim 0.7': [80,  86,  86],
        'PDTrim 0.9': [80,  86,  86],
        'SWS 0.5':    [40,  40,  40],
        'SWS 0.7':    [60,  60,  73],
        'SWS 0.9':    [100, 100, 100],
    },
    'Gemma-2-9B': {
        'Full KV':   [100, 86,  100],
        'PDTrim 0.5': [100, 53,  53],
        'PDTrim 0.7': [100, 73,  73],
        'PDTrim 0.9': [100, 73,  86],
        'SWS 0.5':    [100, 60,  60],
        'SWS 0.7':    [86,  66,  73],
        'SWS 0.9':    [100, 80,  100],
    },
}

ctx_labels = ['2K', '4K', '8K']
strategies = ['Full KV', 'PDTrim 0.5', 'PDTrim 0.7', 'PDTrim 0.9', 'SWS 0.5', 'SWS 0.7', 'SWS 0.9']

# Color map: white=0%, deep blue=100%
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'custom_blue', ['#FFFFFF', '#E3F2FD', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1']
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

for idx, (model_name, model_data) in enumerate(data.items()):
    ax = axes[idx]
    
    # Build matrix: rows=strategies, cols=ctx lengths
    matrix = np.array([model_data[s] for s in strategies]) / 100.0
    
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    
    # Annotate cells
    for i in range(len(strategies)):
        for j in range(len(ctx_labels)):
            val = int(matrix[i, j] * 100)
            text_color = 'white' if val > 70 else '#333333'
            fontweight = 'bold' if val == 100 else 'normal'
            ax.text(j, i, f'{val}%', ha='center', va='center',
                   fontsize=9, color=text_color, fontweight=fontweight)
    
    ax.set_xticks(range(len(ctx_labels)))
    ax.set_xticklabels(ctx_labels, fontweight='medium')
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies if idx == 0 else ['' for _ in strategies])
    ax.set_title(model_name, fontweight='bold', fontsize=12, pad=8)
    ax.tick_params(length=0)
    
    # Add borders between strategy groups
    for y_line in [0.5, 3.5]:
        ax.axhline(y=y_line, color='#9E9E9E', linewidth=0.8, linestyle='-')
    
    # Add context label only on bottom
    if idx == 1:
        ax.set_xlabel('Context Length', fontweight='medium', labelpad=6)

# Shared colorbar
cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, shrink=0.85)
cbar.set_label('Retrieval Accuracy', fontweight='medium')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])

# Add group labels on left
fig.text(0.065, 0.72, 'Baseline', fontsize=8, ha='center', va='center', rotation=90, color='#616161', style='italic')
fig.text(0.065, 0.45, 'PDTrim', fontsize=8, ha='center', va='center', rotation=90, color='#616161', style='italic')
fig.text(0.065, 0.18, 'SWS', fontsize=8, ha='center', va='center', rotation=90, color='#616161', style='italic')

plt.tight_layout()
plt.savefig('kvcache-lab/paper/figures/fig_niah_heatmap.pdf')
plt.savefig('kvcache-lab/paper/figures/fig_niah_heatmap.png')
print("✅ Figure 2 saved: fig_niah_heatmap.pdf/png")
