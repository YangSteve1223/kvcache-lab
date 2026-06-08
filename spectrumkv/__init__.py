#!/usr/bin/env python3
"""
SpectrumKV: Per-Token Mixed-Precision KV Cache Transmission
for PD-Disaggregated LLM Serving

Core components:
  - SWS (Semantic Working Set): token-level importance scoring
  - QCBM (Quality-Constrained Budget Mapper): precision tier assignment
  - Probe: adaptive bandwidth/quality monitoring
"""

__version__ = "1.0.0"
__author__ = "Yang Pengju"

from spectrumkv.core.sws import sws_select, compute_importance_heuristic
from spectrumkv.core.qcbm import spectrumkv_greedy, spectrumkv_balanced, spectrumkv_sink_protect
from spectrumkv.core.probe import ProbeMonitor
