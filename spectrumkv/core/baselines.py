#!/usr/bin/env python3
"""
Baseline methods for comparison with SpectrumKV.
"""

from typing import List

SINK_COUNT = 4


def pdtrim_select(n: int, budget: float, sink: int = SINK_COUNT) -> List[int]:
    """PDTrim baseline: keep sink tokens + recent tokens, drop middle.

    Args:
        n: Total number of tokens.
        budget: Fraction of tokens to retain.
        sink: Number of sink tokens to keep.

    Returns:
        List of selected token indices (sorted).
    """
    n_keep = int(n * budget)
    n_keep = max(n_keep, sink + 1)
    n_keep = min(n_keep, n)

    n_recent = n_keep - sink
    selected = list(range(sink)) + list(range(n - n_recent, n))
    return sorted(selected)
