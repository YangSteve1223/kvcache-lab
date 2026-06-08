#!/usr/bin/env python3
"""
QCBM (Quality-Constrained Budget Mapper) - Precision tier assignment for SpectrumKV.

Assigns per-token precision tiers (FP16/INT8/INT4) based on SWS importance scores
and a bandwidth budget constraint.
"""

import numpy as np
from typing import Optional

# Bandwidth cost per precision tier (relative to FP16)
PRECISION_BW = {"fp16": 1.0, "int8": 0.5, "int4": 0.25}

# Quality factor per precision (simulation prior; GPU validates)
PRECISION_QUALITY = {"fp16": 1.0, "int8": 0.95, "int4": 0.78}

SINK_COUNT = 4


def spectrumkv_greedy(budget: float, importance_scores: np.ndarray) -> np.ndarray:
    """Greedy tier assignment: assign highest precision to most important tokens.

    Args:
        budget: Bandwidth budget as fraction of full FP16 transmission (0.0 to 1.0).
        importance_scores: Per-token importance scores, shape (n_tokens,).

    Returns:
        precision_map: array of shape (n_tokens,) with values in {2, 1, 0}
                       where 2=FP16, 1=INT8, 0=INT4.
    """
    n = len(importance_scores)
    precision_map = np.zeros(n, dtype=np.int32)  # Default INT4

    sorted_indices = np.argsort(importance_scores)[::-1]

    remaining_budget = budget * n

    for idx in sorted_indices:
        if remaining_budget >= 1.0:
            precision_map[idx] = 2  # FP16
            remaining_budget -= 1.0
        elif remaining_budget >= 0.5:
            precision_map[idx] = 1  # INT8
            remaining_budget -= 0.5
        else:
            precision_map[idx] = 0  # INT4
            remaining_budget -= 0.25

    return precision_map


def spectrumkv_balanced(budget: float, importance_scores: np.ndarray) -> np.ndarray:
    """Balanced tier assignment: distribute budget evenly across tiers.

    Args:
        budget: Bandwidth budget fraction (0.0 to 1.0).
        importance_scores: Per-token importance scores.

    Returns:
        precision_map: array with values in {2, 1, 0}.
    """
    n = len(importance_scores)
    precision_map = np.zeros(n, dtype=np.int32)

    sorted_indices = np.argsort(importance_scores)[::-1]

    total_budget = budget * n
    int4_cost = 0.25
    base_cost = n * int4_cost
    surplus = total_budget - base_cost

    n_int8_upgrades = min(int(surplus / 0.25), n)
    surplus -= n_int8_upgrades * 0.25

    n_fp16_upgrades = min(int(surplus / 0.25), n_int8_upgrades)

    n_fp16 = n_fp16_upgrades
    n_int8 = n_int8_upgrades - n_fp16_upgrades
    n_int4 = n - n_fp16 - n_int8

    for i, idx in enumerate(sorted_indices):
        if i < n_fp16:
            precision_map[idx] = 2
        elif i < n_fp16 + n_int8:
            precision_map[idx] = 1
        else:
            precision_map[idx] = 0

    return precision_map


def spectrumkv_sink_protect(
    budget: float,
    importance_scores: np.ndarray,
    sink: int = SINK_COUNT,
    window: int = 64,
) -> np.ndarray:
    """Sink-protected tier assignment: always keep sink tokens at FP16.

    Args:
        budget: Bandwidth budget fraction.
        importance_scores: Per-token importance scores.
        sink: Number of sink tokens to protect at FP16.
        window: Number of recent tokens to protect.

    Returns:
        precision_map: array with values in {2, 1, 0}.
    """
    n = len(importance_scores)
    precision_map = np.zeros(n, dtype=np.int32)

    # Sink tokens always FP16
    for i in range(min(sink, n)):
        precision_map[i] = 2

    # Recent window at least INT8
    for i in range(max(sink, n - window), n):
        precision_map[i] = max(precision_map[i], 1)

    # Allocate remaining budget to middle tokens
    sink_budget = sum(PRECISION_BW["fp16"] if precision_map[i] == 2
                      else PRECISION_BW["int8"] if precision_map[i] == 1
                      else PRECISION_BW["int4"] for i in range(n))
    remaining = budget * n - sink_budget

    middle_indices = [i for i in range(sink, max(sink, n - window))]
    if middle_indices and remaining > 0:
        mid_scores = importance_scores[middle_indices]
        sorted_mid = sorted(zip(mid_scores, middle_indices), reverse=True)
        for score, idx in sorted_mid:
            if remaining >= 0.5:
                precision_map[idx] = 1
                remaining -= 0.25
            else:
                break

    return precision_map


def random_tier(budget: float, n_tokens: int, seed: int = 42) -> np.ndarray:
    """Random baseline: randomly assign precision tiers respecting budget.

    Args:
        budget: Bandwidth budget fraction.
        n_tokens: Number of tokens.
        seed: Random seed.

    Returns:
        precision_map: array with values in {2, 1, 0}.
    """
    rng = np.random.RandomState(seed)
    random_scores = rng.rand(n_tokens).astype(np.float32)
    return spectrumkv_greedy(budget, random_scores)
