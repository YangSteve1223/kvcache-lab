#!/usr/bin/env python3
"""Quick smoke test for SpectrumKV package."""

import numpy as np
import sys
sys.path.insert(0, ".")

from spectrumkv.core.sws import compute_importance_heuristic, sws_select
from spectrumkv.core.qcbm import spectrumkv_greedy, spectrumkv_balanced, spectrumkv_sink_protect, random_tier
from spectrumkv.core.baselines import pdtrim_select
from spectrumkv.core.probe import ProbeMonitor


def test_sws():
    scores = compute_importance_heuristic(1000, sink=4)
    assert scores.shape == (1000,), f"Expected (1000,), got {scores.shape}"
    assert scores[:4].sum() == 4.0, "Sink tokens should have importance 1.0"
    selected = sws_select(1000, 0.5, scores)
    assert len(selected) > 0, "Should select some tokens"
    print(f"  SWS: {len(selected)} tokens selected from 1000 at budget=0.5")


def test_qcbm():
    scores = compute_importance_heuristic(1000, sink=4)
    for func_name, func in [("greedy", spectrumkv_greedy), 
                             ("balanced", spectrumkv_balanced),
                             ("sink_protect", spectrumkv_sink_protect)]:
        pmap = func(0.5, scores)
        assert pmap.shape == (1000,), f"{func_name}: Expected (1000,), got {pmap.shape}"
        unique = np.unique(pmap)
        assert all(u in [0, 1, 2] for u in unique), f"{func_name}: Invalid tier values"
        print(f"  QCBM-{func_name}: tiers={dict(zip(*np.unique(pmap, return_counts=True)))}")


def test_baselines():
    selected = pdtrim_select(1000, 0.5, sink=4)
    assert len(selected) > 0, "PDTrim should select tokens"
    assert 0 in selected and 1 in selected, "PDTrim should keep sink tokens"
    print(f"  PDTrim: {len(selected)} tokens selected from 1000 at budget=0.5")


def test_probe():
    probe = ProbeMonitor(budget=0.5)
    assert probe.suggest_budget() == 0.5, "Initial budget should be 0.5"
    probe.update_bandwidth(0.3)
    new_budget = probe.suggest_budget()
    assert new_budget < 0.5, "Budget should decrease with lower bandwidth"
    print(f"  Probe: budget adjusted from 0.5 to {new_budget:.3f}")


if __name__ == "__main__":
    print("SpectrumKV Smoke Test")
    print("-" * 40)
    test_sws()
    test_qcbm()
    test_baselines()
    test_probe()
    print("-" * 40)
    print("All tests passed!")
