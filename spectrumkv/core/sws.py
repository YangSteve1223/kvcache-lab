#!/usr/bin/env python3
"""
SWS (Semantic Working Set) - Token-level importance scoring for SpectrumKV.

Determines which tokens are semantically important based on attention weights
and positional decay, with mandatory sink token protection.
"""

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

SINK_COUNT = 4


def compute_importance_heuristic(seq_len: int, sink: int = SINK_COUNT) -> np.ndarray:
    """Compute importance scores using heuristic (no attention weights needed).

    Uses exponential positional decay with sink token protection.
    Useful for simulation and quick experiments.

    Args:
        seq_len: Number of tokens in the sequence.
        sink: Number of sink tokens to protect at the start.

    Returns:
        importance_scores: array of shape (seq_len,) with values in [0, 1].
    """
    scores = np.zeros(seq_len, dtype=np.float32)
    scores[:sink] = 1.0
    decay = np.exp(-0.05 * np.arange(seq_len - sink))
    scores[sink:] = decay / decay.max() * 0.8
    return scores


def compute_importance_from_attention(
    attn_weights: "torch.Tensor",
    sink: int = SINK_COUNT,
    decay_alpha: float = 0.95,
) -> np.ndarray:
    """Compute per-token importance from actual attention weights.

    Args:
        attn_weights: (batch, heads, seq, seq) or (heads, seq, seq).
        sink: Number of sink tokens to protect.
        decay_alpha: Positional decay factor.

    Returns:
        importance_scores: array of shape (seq_len,) with values in [0, 1].
    """
    import torch  # lazy import: only needed for attention-based scoring

    if attn_weights.dim() == 4:
        attn = attn_weights[0].mean(dim=0).detach().cpu().numpy()
    elif attn_weights.dim() == 3:
        attn = attn_weights.mean(dim=0).detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")

    seq_len = attn.shape[-1]
    received_attention = attn.sum(axis=0)
    positions = np.arange(seq_len, dtype=np.float32)
    decay_weights = decay_alpha ** (seq_len - 1 - positions)
    scores = received_attention * decay_weights
    scores[:sink] = scores.max()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores.astype(np.float32)


def sws_select(
    n: int,
    budget: float,
    importance_scores: np.ndarray,
    sink: int = SINK_COUNT,
    window: int = 64,
) -> list:
    """Select token indices using SWS strategy.

    Combines sink tokens (always retained), top-k by importance,
    and a recent window of tokens.

    Args:
        n: Total number of tokens.
        budget: Fraction of tokens to retain (0.0 to 1.0).
        importance_scores: Per-token importance scores.
        sink: Number of sink tokens.
        window: Number of recent tokens to always retain.

    Returns:
        List of selected token indices (sorted).
    """
    n_keep = max(int(n * budget), sink + window)
    n_keep = min(n_keep, n)

    selected = set(range(min(sink, n))) | set(range(max(0, n - window), n))

    remaining_budget = n_keep - len(selected)
    if remaining_budget > 0:
        masked_scores = importance_scores.copy()
        for idx in selected:
            if idx < len(masked_scores):
                masked_scores[idx] = -1
        top_indices = np.argsort(masked_scores)[-remaining_budget:]
        selected.update(top_indices.tolist())

    return sorted(selected)
