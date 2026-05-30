#!/usr/bin/env python3
"""
Figure 4: SWS + QCBM Joint Effect — Bandwidth Savings Breakdown
Stacked bar showing transmission volume reduction
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ─── Data from QCBM+SWS joint simulation (4K tokens, 7B model) ───
# Transmission volume in MB
schemes = [
    'Full FP16\n(baseline)',
    'SWS only\n(50% FP16)',
    'QCBM only\n(100% K4V4)',
    'SWS+QCBM\n(50% K4V4)',
    'SWS+QCBM\n(30% K4V4)',
]

volumes = [54.69, 27.34, 3.42, 1.71, 1.03]
savings_pct = [0, 50.0, 93.8, 96.9, 98.1]

# Colors - gradient from warm to cool
colors = ['#E53935', '#FF7043', '#42A5F5', '#1565C0', '#0D47A1']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.2, 1]})

# ─── Left: Absolute volume ───
bars = ax1.bar(range(len(schemes)), volumes, color=colors, edgecolor='white', linewidth=0.8, width=0.65)

# Add value labels on bars
for i, (bar, vol, save) in enumerate(zip(bars, volumes, savings_pct)):
    if vol > 5:
        ax1.text(bar.get_x() + bar.get_width()/2, vol + 1.5, f'{vol:.1f} MB',
                ha='center', va='bottom', fontsize=9, fontweight='medium')
    else:
        ax1.text(bar.get_x() + bar.get_width()/2, vol + 1.5, f'{vol:.2f} MB',
                ha='center', va='bottom', fontsize=9, fontweight='medium')

ax1.set_xticks(range(len(schemes)))
ax1.set_xticklabels(schemes, fontsize=8.5)
ax1.set_ylabel('KV Transfer Volume (MB)', fontweight='medium')
ax1.set_ylim(0, 65)
ax1.grid(axis='y', alpha=0.25, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Absolute Volume (4K tokens, 7B model)', fontweight='bold', fontsize=11)

# ─── Right: Savings percentage waterfall ───
x = range(len(schemes))
bar_colors = ['#BDBDBD', '#81C784', '#66BB6A', '#43A047', '#2E7D32']

bars2 = ax2.bar(x, savings_pct, color=bar_colors, edgecolor='white', linewidth=0.8, width=0.6)

# Add percentage labels
for bar, save in zip(bars2, savings_pct):
    ax2.text(bar.get_x() + bar.get_width()/2, save + 1, f'{save:.1f}%',
            ha='center', va='bottom', fontsize=9, fontweight='medium')

# Add step arrows between bars
for i in range(1, len(savings_pct)):
    diff = savings_pct[i] - savings_pct[i-1]
    if diff > 0:
        mid_x = (i-0.5 + i) / 2  # Between bars
        mid_y = (savings_pct[i] + savings_pct[i-1]) / 2
        ax2.annotate(f'+{diff:.1f}pp', xy=(i-0.5, savings_pct[i-1]+2),
                    fontsize=7, color='#558B2F', fontweight='medium', ha='center')

ax2.set_xticks(x)
ax2.set_xticklabels(schemes, fontsize=8.5)
ax2.set_ylabel('Bandwidth Saving (%)', fontweight='medium')
ax2.set_ylim(0, 110)
ax2.grid(axis='y', alpha=0.25, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Cumulative Bandwidth Saving', fontweight='bold', fontsize=11)

# Add dashed reference line at 90%
ax2.axhline(y=90, color='#F44336', linewidth=0.8, linestyle='--', alpha=0.5)
ax2.text(4.7, 91, '90% target', fontsize=7.5, color='#F44336', va='bottom', alpha=0.7)

plt.tight_layout()
plt.savefig('kvcache-lab/paper/figures/fig_sws_qcbm_joint.pdf')
plt.savefig('kvcache-lab/paper/figures/fig_sws_qcbm_joint.png')
print("✅ Figure 4 saved: fig_sws_qcbm_joint.pdf/png")
