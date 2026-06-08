#!/usr/bin/env python3
"""
Probe - Adaptive bandwidth/quality monitoring for SpectrumKV.

Monitors runtime conditions (available bandwidth, observed quality degradation)
and adjusts precision tier assignments dynamically.
"""

import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ProbeState:
    """Current state of the Probe monitor."""
    bandwidth_available: float = 1.0  # Fraction of full bandwidth
    quality_score: float = 1.0        # Recent quality metric
    last_update: float = 0.0          # Timestamp of last update
    update_count: int = 0             # Number of updates received


class ProbeMonitor:
    """Adaptive monitoring for SpectrumKV precision tier adjustment.

    Usage:
        probe = ProbeMonitor(budget=0.5)
        probe.update_bandwidth(observed_bw)
        probe.update_quality(observed_ppl / baseline_ppl)
        new_budget = probe.suggest_budget()
        new_map = spectrumkv_greedy(new_budget, importance_scores)
    """

    def __init__(
        self,
        budget: float = 0.5,
        min_budget: float = 0.25,
        max_budget: float = 0.75,
        bw_threshold: float = 0.1,
        quality_threshold: float = 0.05,
        smoothing: float = 0.8,
    ):
        self.budget = budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.bw_threshold = bw_threshold
        self.quality_threshold = quality_threshold
        self.smoothing = smoothing
        self.state = ProbeState(last_update=time.time())

    def update_bandwidth(self, observed_bw: float) -> bool:
        """Update observed bandwidth availability.

        Returns True if budget should be re-optimized.
        """
        prev = self.state.bandwidth_available
        self.state.bandwidth_available = (
            self.smoothing * prev + (1 - self.smoothing) * observed_bw
        )
        self.state.last_update = time.time()
        self.state.update_count += 1
        return abs(self.state.bandwidth_available - prev) > self.bw_threshold

    def update_quality(self, quality_ratio: float) -> bool:
        """Update observed quality ratio.

        Returns True if quality degradation exceeds threshold.
        """
        prev = self.state.quality_score
        self.state.quality_score = (
            self.smoothing * prev + (1 - self.smoothing) * quality_ratio
        )
        self.state.last_update = time.time()
        self.state.update_count += 1
        return (self.state.quality_score - prev) < -self.quality_threshold

    def suggest_budget(self) -> float:
        """Suggest adjusted budget based on current observations."""
        bw_factor = self.state.bandwidth_available
        quality_factor = 1.0 - (1.0 - self.state.quality_score) * 2

        adjusted = self.budget * bw_factor * quality_factor
        return max(self.min_budget, min(self.max_budget, adjusted))
