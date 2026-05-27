#!/usr/bin/env python3
"""
Figure 7: System Architecture Overview
PD-disaggregated LLM serving architecture with KV tiering
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.lines as mlines

# Set up academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')

# Color palette
colors = {
    'prefill': '#3498db',      # Blue
    'decode': '#27ae60',        # Green
    'network': '#f39c12',      # Orange
    'local_kv': '#2ecc71',     # Light green
    'remote_kv': '#e74c3c',    # Red
    'arrow': '#7f8c8d',        # Gray
}

# ============ Prefill Instance ============
prefill_box = FancyBboxPatch((0.3, 3.5), 2.5, 1.8, 
                              boxstyle="round,pad=0.1",
                              facecolor=colors['prefill'], alpha=0.2,
                              edgecolor=colors['prefill'], linewidth=2)
ax.add_patch(prefill_box)
ax.text(1.55, 4.7, 'Prefill Instance', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=colors['prefill'])
ax.text(1.55, 4.2, '• Full KV Cache\n• Prompt Processing', ha='center', va='center', 
        fontsize=9)

# Input arrow to prefill
ax.annotate('', xy=(0.5, 4.4), xytext=(0, 4.4),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(0, 4.6, 'Input\nPrompt', ha='center', fontsize=8)

# ============ Network Component ============
network_box = FancyBboxPatch((4, 3.8), 2, 1.2, 
                              boxstyle="round,pad=0.1",
                              facecolor=colors['network'], alpha=0.2,
                              edgecolor=colors['network'], linewidth=2)
ax.add_patch(network_box)
ax.text(5, 4.5, 'Network', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=colors['network'])

# Arrow: Prefill -> Network (KV Transfer)
ax.annotate('', xy=(4.2, 4.4), xytext=(2.8, 4.4),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                          connectionstyle='arc3,rad=0'))
ax.text(3.5, 4.7, 'KV\nTransfer', ha='center', fontsize=8, color='gray')

# ============ Decode Instance ============
decode_box = FancyBboxPatch((6.8, 3.5), 2.5, 1.8, 
                            boxstyle="round,pad=0.1",
                            facecolor=colors['decode'], alpha=0.2,
                            edgecolor=colors['decode'], linewidth=2)
ax.add_patch(decode_box)
ax.text(8.05, 4.7, 'Decode Instance', ha='center', va='center', 
        fontsize=11, fontweight='bold', color=colors['decode'])
ax.text(8.05, 4.2, '• Selective KV Cache\n• Token Generation', ha='center', va='center', 
        fontsize=9)

# Arrow: Network -> Decode
ax.annotate('', xy=(6.8, 4.4), xytext=(6, 4.4),
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2,
                          connectionstyle='arc3,rad=0'))
ax.text(6.4, 4.7, 'Selected\nKV', ha='center', fontsize=8, color='gray')

# Output arrow from decode
ax.annotate('', xy=(9.7, 4.4), xytext=(9.3, 4.4),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.text(9.9, 4.4, 'Output', ha='left', va='center', fontsize=8)

# ============ KV Tiering Below ============
# Local KV box (Hot)
local_box = FancyBboxPatch((6.8, 1), 2.5, 1.5, 
                           boxstyle="round,pad=0.1",
                           facecolor=colors['local_kv'], alpha=0.3,
                           edgecolor=colors['local_kv'], linewidth=2)
ax.add_patch(local_box)
ax.text(8.05, 2.2, 'Local KV Cache', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='#1e8449')
ax.text(8.05, 1.7, '(Hot - GPU Memory)', ha='center', va='center', fontsize=8)
ax.text(8.05, 1.3, 'SWS Window (256-512 tokens)', ha='center', va='center', 
        fontsize=8, color='gray')

# Remote KV box (Cold)
remote_box = FancyBboxPatch((0.3, 1), 2.5, 1.5, 
                             boxstyle="round,pad=0.1",
                             facecolor=colors['remote_kv'], alpha=0.3,
                             edgecolor=colors['remote_kv'], linewidth=2)
ax.add_patch(remote_box)
ax.text(1.55, 2.2, 'Remote KV Cache', ha='center', va='center', 
        fontsize=10, fontweight='bold', color='#922b21')
ax.text(1.55, 1.7, '(Cold - Disaggregated)', ha='center', va='center', fontsize=8)
ax.text(1.55, 1.3, 'Full Context on Demand', ha='center', va='center', 
        fontsize=8, color='gray')

# Connection boxes with decode instance
ax.annotate('', xy=(2.8, 1.75), xytext=(2.8, 3.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5,
                          connectionstyle='arc3,rad=0.2'))
ax.annotate('', xy=(6.8, 1.75), xytext=(6.8, 3.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5,
                          connectionstyle='arc3,rad=0.2'))

# ============ TAA Component ============
taa_box = FancyBboxPatch((3.5, 0.3), 3, 0.8, 
                         boxstyle="round,pad=0.1",
                         facecolor='#9b59b6', alpha=0.2,
                         edgecolor='#9b59b6', linewidth=2)
ax.add_patch(taa_box)
ax.text(5, 0.7, 'TAA: Token-level Attention Allocation', ha='center', va='center', 
        fontsize=9, fontweight='bold', color='#7d3c98')

# ============ Title ============
ax.text(5, 5.6, 'Figure 7: PD-Disaggregated LLM Serving Architecture', 
        ha='center', va='center', fontsize=13, fontweight='bold')

# ============ Legend ============
# Create legend manually
legend_elements = [
    mpatches.Patch(facecolor=colors['prefill'], alpha=0.3, edgecolor=colors['prefill'],
                   label='Prefill Instance'),
    mpatches.Patch(facecolor=colors['decode'], alpha=0.3, edgecolor=colors['decode'],
                   label='Decode Instance'),
    mpatches.Patch(facecolor=colors['local_kv'], alpha=0.3, edgecolor=colors['local_kv'],
                   label='Local (Hot) KV'),
    mpatches.Patch(facecolor=colors['remote_kv'], alpha=0.3, edgecolor=colors['remote_kv'],
                   label='Remote (Cold) KV'),
    mpatches.Patch(facecolor='#9b59b6', alpha=0.2, edgecolor='#9b59b6',
                   label='TAA Controller'),
]

ax.legend(handles=legend_elements, loc='lower center', ncol=5, 
          fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()

# Save
plt.savefig('fig7_architecture_overview.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig7_architecture_overview.png', format='png', bbox_inches='tight')
print("Figure 7 saved: fig7_architecture_overview.pdf/png")

plt.close()
