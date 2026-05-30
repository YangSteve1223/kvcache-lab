#!/usr/bin/env python3
"""
Master script v2: Generate all publication-quality figures for kvcache-lab paper
Refined based on review
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os

# ─── Global style ───
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
    'axes.edgecolor': '#333333',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.06,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Color palette
SWS_BLUE = '#1565C0'
SWS_LIGHT = '#42A5F5'
PDTRIM_RED = '#D32F2F'
PDTRIM_ORANGE = '#FF7043'
NEUTRAL_GRAY = '#9E9E9E'


# ═══════════════════════════════════════════════════════════
# Figure 1: Eviction Strategy Comparison
# ═══════════════════════════════════════════════════════════
def fig1_eviction():
    models = ['Qwen2.5-7B', 'Mistral-7B', 'Gemma-2-9B']
    
    # PPL degradation % at budget b=0.5
    sws_sink0 = [24.52, 19.8, 32.1]
    sws_sink16 = [27.48, 26.1, 35.3]
    pdtrim_fr10 = [23.6, 23.8, 32.7]
    pdtrim_fr50 = [26.2, 29.3, 44.8]
    lru = [31.2, 33.5, 48.6]
    random = [45.8, 52.1, 67.3]
    
    fig, ax = plt.subplots(figsize=(7.2, 4))
    x = np.arange(len(models))
    w = 0.12
    
    b1 = ax.bar(x - 2.5*w, sws_sink0, w, label='SWS (sink=0)', color=SWS_LIGHT, ec='white', lw=0.5, zorder=3)
    b2 = ax.bar(x - 1.5*w, sws_sink16, w, label='SWS (sink=16)', color=SWS_BLUE, ec='white', lw=0.5, zorder=3)
    b3 = ax.bar(x - 0.5*w, pdtrim_fr10, w, label='PDTrim (FR=0.1)', color=PDTRIM_ORANGE, ec='white', lw=0.5, zorder=3)
    b4 = ax.bar(x + 0.5*w, pdtrim_fr50, w, label='PDTrim (FR=0.5)', color=PDTRIM_RED, ec='white', lw=0.5, zorder=3)
    b5 = ax.bar(x + 1.5*w, lru, w, label='LRU', color='#BDBDBD', ec='#9E9E9E', lw=0.5, zorder=3)
    b6 = ax.bar(x + 2.5*w, random, w, label='Random', color='#757575', ec='#616161', lw=0.5, zorder=3)
    
    for bars in [b1, b2, b3, b4, b5, b6]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 2.5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=6, fontweight='medium')
    
    ax.set_ylabel('PPL Degradation (%)', fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='medium')
    ax.set_ylim(0, 78)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#ddd', ncol=2, fontsize=8.5,
             borderpad=0.4, handlelength=1.2)
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
    print("✅ Fig 1: Eviction Comparison")


# ═══════════════════════════════════════════════════════════
# Figure 2: NIAH Depth Scan Heatmap
# ═══════════════════════════════════════════════════════════
def fig2_niah():
    data = {
        'Qwen2.5-7B': {
            'Full KV':    [100, 100, 100],
            'PDTrim 0.5': [100, 60,  60],
            'PDTrim 0.7': [100, 86,  86],
            'PDTrim 0.9': [100, 86,  86],
            'SWS 0.5':    [100, 60,  60],
            'SWS 0.7':    [80,  73,  73],
            'SWS 0.9':    [100, 100, 100],
        },
        'Mistral-7B': {
            'Full KV':    [100, 100, 100],
            'PDTrim 0.5': [80,  53,  40],
            'PDTrim 0.7': [80,  86,  86],
            'PDTrim 0.9': [80,  86,  86],
            'SWS 0.5':    [40,  40,  40],
            'SWS 0.7':    [60,  60,  73],
            'SWS 0.9':    [100, 100, 100],
        },
        'Gemma-2-9B': {
            'Full KV':    [100, 86,  100],
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
    
    # Custom colormap: warm yellow → cool blue
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'niah', ['#FFF8E1', '#FFE082', '#90CAF9', '#42A5F5', '#1565C0', '#0D47A1']
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    
    for idx, (model_name, model_data) in enumerate(data.items()):
        ax = axes[idx]
        matrix = np.array([model_data[s] for s in strategies]) / 100.0
        
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        
        for i in range(len(strategies)):
            for j in range(len(ctx_labels)):
                val = int(matrix[i, j] * 100)
                text_color = 'white' if val > 65 else '#333333'
                fontweight = 'bold' if val == 100 else 'normal'
                ax.text(j, i, f'{val}%', ha='center', va='center',
                       fontsize=9, color=text_color, fontweight=fontweight)
        
        ax.set_xticks(range(len(ctx_labels)))
        ax.set_xticklabels(ctx_labels, fontweight='medium')
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies if idx == 0 else ['' for _ in strategies])
        ax.set_title(model_name, fontweight='bold', fontsize=11.5, pad=8)
        ax.tick_params(length=0)
        
        # Group separators
        for y_line in [0.5, 3.5]:
            ax.axhline(y=y_line, color='#666666', linewidth=0.7, linestyle='-')
        
        if idx == 1:
            ax.set_xlabel('Context Length', fontweight='medium', labelpad=6)
    
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, shrink=0.85)
    cbar.set_label('Retrieval Accuracy', fontweight='medium')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_niah_heatmap.pdf'))
    plt.savefig(os.path.join(OUT_DIR, 'fig_niah_heatmap.png'))
    plt.close()
    print("✅ Fig 2: NIAH Heatmap")


# ═══════════════════════════════════════════════════════════
# Figure 3: Budget vs Quality Tradeoff
# ═══════════════════════════════════════════════════════════
def fig3_budget():
    budgets = [0.3, 0.5, 0.7]
    
    sws_data = {
        'Qwen2.5-7B': [48.0, 27.48, 17.45],
        'Mistral-7B': [35.3, 26.1, 17.2],
        'Gemma-2-9B': [94.0, 35.3, 19.8],
    }
    pdtrim_data = {
        'Qwen2.5-7B': [57.33, 26.2, 13.7],
        'Mistral-7B': [48.8, 29.3, 22.9],
        'Gemma-2-9B': [103.4, 44.8, 17.4],
    }
    baselines = {'Qwen2.5-7B': 7.04, 'Mistral-7B': 7.95, 'Gemma-2-9B': 11.19}
    model_names = ['Qwen2.5-7B', 'Mistral-7B', 'Gemma-2-9B']
    
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    fig.subplots_adjust(wspace=0.28)
    
    for idx, model in enumerate(model_names):
        ax = axes[idx]
        b = np.array(budgets)
        sws = np.array(sws_data[model])
        pdr = np.array(pdtrim_data[model])
        
        # Plot
        ax.plot(b, sws, 'o-', color=SWS_BLUE, linewidth=2.2, markersize=7,
                label='SWS (sink=16)', zorder=5, markeredgecolor='white', markeredgewidth=0.5)
        ax.plot(b, pdr, 's--', color=PDTRIM_RED, linewidth=1.8, markersize=6,
                label='PDTrim (FR=0.5)', alpha=0.85, zorder=4, markeredgecolor='white', markeredgewidth=0.5)
        
        # Shade SWS advantage
        mask = sws < pdr
        if mask.any():
            ax.fill_between(b, sws, pdr, where=mask, alpha=0.10, color=SWS_BLUE, interpolate=True)
        
        # Annotate key crossover
        for i, budget in enumerate(budgets):
            offset_y = 8 if sws[i] > pdr[i] else -14
            ax.annotate(f'{sws[i]:.1f}%', xy=(budget, sws[i]),
                       xytext=(0, offset_y), textcoords='offset points',
                       fontsize=7.5, ha='center', color=SWS_BLUE, fontweight='medium')
        
        ax.set_title(model, fontweight='bold', fontsize=11)
        ax.set_xlabel('Budget $b$', fontweight='medium')
        if idx == 0:
            ax.set_ylabel('PPL Degradation (%)', fontweight='medium')
        ax.set_xticks(budgets)
        ax.set_xticklabels(['0.3', '0.5', '0.7'])
        ymax = max(max(sws), max(pdr)) * 1.18
        ax.set_ylim(0, ymax)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if idx == 0:
            ax.legend(loc='upper right', framealpha=0.95, edgecolor='#ddd', fontsize=8.5,
                     borderpad=0.4, handlelength=1.5)
        
        ax.text(0.97, 0.04, f'Baseline = {baselines[model]:.2f}',
               transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
               color='#9E9E9E', style='italic',
               bbox=dict(boxstyle='round,pad=0.3', fc='#FAFAFA', ec='#E0E0E0', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_budget_tradeoff.pdf'))
    plt.savefig(os.path.join(OUT_DIR, 'fig_budget_tradeoff.png'))
    plt.close()
    print("✅ Fig 3: Budget Tradeoff")


# ═══════════════════════════════════════════════════════════
# Figure 4: SWS + QCBM Joint Effect
# ═══════════════════════════════════════════════════════════
def fig4_joint():
    schemes = ['Full FP16\n(Baseline)', 'SWS Only\n(50% FP16)', 'QCBM Only\n(K4V4)',
               'SWS+QCBM\n(50% K4V4)', 'SWS+QCBM\n(30% K4V4)']
    volumes = [54.69, 27.34, 3.42, 1.71, 1.03]
    savings_pct = [0, 50.0, 93.8, 96.9, 98.1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1.15, 1]})
    
    # ─── Left: Volume ───
    c1 = ['#EF5350', '#FF9800', '#42A5F5', '#1E88E5', '#0D47A1']
    bars1 = ax1.bar(range(len(schemes)), volumes, color=c1, ec='white', lw=0.8, width=0.58, zorder=3)
    
    for bar, vol in zip(bars1, volumes):
        fmt = f'{vol:.1f}' if vol > 5 else f'{vol:.2f}'
        ax1.text(bar.get_x() + bar.get_width()/2, vol + 1.2, f'{fmt} MB',
                ha='center', va='bottom', fontsize=8.5, fontweight='medium')
    
    ax1.set_xticks(range(len(schemes)))
    ax1.set_xticklabels(schemes, fontsize=7.5)
    ax1.set_ylabel('KV Transfer Volume (MB)', fontweight='medium')
    ax1.set_ylim(0, 65)
    ax1.grid(axis='y', alpha=0.2, linestyle='--', zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Transfer Volume (4K tokens, 7B)', fontweight='bold', fontsize=10.5)
    
    # ─── Right: Savings ───
    c2 = ['#E0E0E0', '#A5D6A7', '#81C784', '#4CAF50', '#2E7D32']
    bars2 = ax2.bar(range(len(schemes)), savings_pct, color=c2, ec='white', lw=0.8, width=0.55, zorder=3)
    
    for bar, save in zip(bars2, savings_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, save + 1.5, f'{save:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='medium')
    
    # Incremental labels
    for i in range(1, len(savings_pct)):
        diff = savings_pct[i] - savings_pct[i-1]
        if diff > 0:
            ax2.annotate(f'+{diff:.1f}pp', xy=(i-0.5, savings_pct[i-1]+3),
                        fontsize=7, color='#33691E', fontweight='medium', ha='center')
    
    ax2.set_xticks(range(len(schemes)))
    ax2.set_xticklabels(schemes, fontsize=7.5)
    ax2.set_ylabel('Bandwidth Saving (%)', fontweight='medium')
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.2, linestyle='--', zorder=0)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Cumulative Bandwidth Saving', fontweight='bold', fontsize=10.5)
    
    ax2.axhline(y=90, color='#F44336', linewidth=0.8, linestyle='--', alpha=0.45)
    ax2.text(4.6, 91.5, '90% target', fontsize=7.5, color='#F44336', va='bottom', alpha=0.65)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_sws_qcbm_joint.pdf'))
    plt.savefig(os.path.join(OUT_DIR, 'fig_sws_qcbm_joint.png'))
    plt.close()
    print("✅ Fig 4: SWS+QCBM Joint")


# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🎨 Generating publication-quality figures v2...")
    print()
    fig1_eviction()
    fig2_niah()
    fig3_budget()
    fig4_joint()
    print()
    print("🎉 All figures generated! Output dir:", OUT_DIR)
