#!/usr/bin/env python3
"""
[DEPRECATED] This script has been superseded by spectrumkv_scripts/exp3_quant_error.py
and spectrumkv_scripts/exp4_layer_budget.py. Use those instead.
===============================================================================
QCBM GPU Experiment: Quality-Constrained Budget Minimization with Mixed-Precision
KV Cache Quantization

Hypothesis:
  Instead of dropping tokens (SWS/PDTrim), keep ALL tokens but at different
  precision levels. At the SAME bandwidth budget, QCBM retains 2-4x more
  tokens including middle tokens that selection-based methods discard.

  - Hot tier:   FP16  (1.0x bandwidth per token)
  - Warm tier:  INT8  (0.5x bandwidth per token)
  - Cold tier:  INT4  (0.25x bandwidth per token)

Experiment Matrix:
  1. PPL evaluation (WikiText-2):
     - Full KV FP16 (baseline)
     - SWS b=0.5 FP16 (current approach — 50% tokens, all FP16)
     - SWS b=0.5 + mixed precision (50% tokens FP16, rest INT8)
     - QCBM: ALL tokens, tiered precision at same bandwidth as SWS b=0.5

  2. NIAH (Needle-in-a-Haystack):
     - Full KV at b=1.0 (baseline)
     - SWS b=0.3, b=0.5
     - QCBM b=0.3, b=0.5 (same bandwidth budget, all tokens kept)
     - Multiple needle depths (10%, 25%, 50%, 75%, 90%)

Usage:
  python exp_qcbm_quantization.py --model qwen7b --seq_len 2048 --budgets 0.3 0.5
  python exp_qcbm_quantization.py --model qwen7b --niah_only --budgets 0.3 0.5
  python exp_qcbm_quantization.py --model qwen7b --ppl_only --seq_len 2048

Requires GPU with sufficient VRAM for model loading.
===============================================================================
"""

import argparse
import json
import math
import os
import string
import time
import traceback
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# =============================================================================
# Constants
# =============================================================================

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral/Mistral-7B-Instruct-v0.3",
    "gemma9b": "google/gemma-2-9b-it",
}

# Bandwidth multipliers per precision level (relative to FP16)
PRECISION_BW_FACTOR = {
    "fp16": 1.0,
    "int8": 0.5,
    "int4": 0.25,
}

# Quality factor per precision (1.0 = lossless, lower = more degradation)
# These are priors; the experiment will validate them.
PRECISION_QUALITY_FACTOR = {
    "fp16": 1.0,
    "int8": 0.95,
    "int4": 0.80,
}

DEFAULT_OUTPUT_DIR = "experiment_results_qcbm"


# =============================================================================
# KV Cache Quantization
# =============================================================================

class KVCacheQuantizer:
    """
    Quantize KV cache tensors to INT8 or INT4, per-head per-token.
    
    Quantization scheme: symmetric affine
      scale = max(|x|) / qmax
      x_q  = round(x / scale)  (clamped to [-qmax, qmax])
      x_dq = x_q * scale       (dequantized)
    
    For INT8: qmax = 127
    For INT4: qmax = 7
    """

    QMAX = {"int8": 127, "int4": 7}

    @staticmethod
    def quantize_per_channel(
        kv_tensor: torch.Tensor,
        precision: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a KV cache slice.
        
        Args:
            kv_tensor: shape (batch, num_heads, seq_len, head_dim) or 
                       (num_heads, seq_len, head_dim) — we handle both
            precision: "int8" or "int4"
        
        Returns:
            (quantized, scale) where:
              quantized: same shape, dtype=float16 (stores integer values as floats)
              scale: per-token scale, shape broadcastable to kv_tensor
        """
        assert precision in ("int8", "int4"), f"Unsupported precision: {precision}"
        qmax = KVCacheQuantizer.QMAX[precision]

        original_dtype = kv_tensor.dtype
        kv_float = kv_tensor.float()

        # Per-token scale: reduce over head_dim
        # kv_float shape: (..., seq_len, head_dim)
        # We want scale per (num_heads, seq_len) — reduce last dim
        abs_max = kv_float.abs().amax(dim=-1, keepdim=True)
        abs_max = abs_max.clamp(min=1e-8)  # avoid division by zero
        scale = abs_max / qmax

        # Quantize
        quantized = torch.clamp(torch.round(kv_float / scale), -qmax, qmax)

        return quantized.half(), scale.half()

    @staticmethod
    def dequantize(
        quantized: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize back to FP16."""
        return (quantized.float() * scale.float()).half()

    @staticmethod
    def quantize_kv_cache(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        precision: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize both K and V caches.
        
        Returns: (key_q, key_scale, value_q, value_scale)
        """
        key_q, key_scale = KVCacheQuantizer.quantize_per_channel(key_states, precision)
        value_q, value_scale = KVCacheQuantizer.quantize_per_channel(value_states, precision)
        return key_q, key_scale, value_q, value_scale


# =============================================================================
# QCBM Tier Assignment
# =============================================================================

@dataclass
class QCBMTierConfig:
    """Configuration for QCBM tiered precision assignment."""
    # Fraction of tokens in each tier (must sum to 1.0)
    hot_fraction: float = 0.50    # Top 50% → FP16
    warm_fraction: float = 0.30   # Next 30% → INT8
    cold_fraction: float = 0.20   # Bottom 20% → INT4
    
    # Precision per tier
    hot_precision: str = "fp16"
    warm_precision: str = "int8"
    cold_precision: str = "int4"
    
    # Target bandwidth budget (as fraction of full FP16)
    # Computed automatically if not set
    target_bandwidth: Optional[float] = None

    @property
    def effective_bandwidth(self) -> float:
        """Compute the effective bandwidth fraction for this tier config."""
        return (
            self.hot_fraction * PRECISION_BW_FACTOR[self.hot_precision] +
            self.warm_fraction * PRECISION_BW_FACTOR[self.warm_precision] +
            self.cold_fraction * PRECISION_BW_FACTOR[self.cold_precision]
        )


def compute_qcbm_tiers_for_budget(
    budget: float,
    importance_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign precision tiers to tokens based on importance scores,
    such that total bandwidth matches the given budget.
    
    Since ALL tokens are kept (100% retention), the bandwidth budget
    must be satisfied entirely through precision assignment.
    
    Strategy: greedy tier assignment
      1. Sort tokens by importance (descending)
      2. Assign top tokens to FP16 until adding more would exceed budget
      3. Assign next tokens to INT8 until adding more would exceed budget  
      4. Remaining tokens get INT4
    
    Args:
        budget: target bandwidth as fraction of full FP16 (e.g., 0.5)
        importance_scores: (seq_len,) importance score per token
    
    Returns:
        (precision_labels, tier_boundaries)
          precision_labels: (seq_len,) string array of "fp16"/"int8"/"int4"
          tier_boundaries: dict with hot_end, warm_end indices
    """
    n = len(importance_scores)
    sorted_indices = np.argsort(-importance_scores)  # descending importance

    precision_labels = np.full(n, "int4", dtype=object)
    
    # Greedy: fill FP16 first, then INT8, rest INT4
    # Total BW = n_fp16 * 1.0 + n_int8 * 0.5 + n_int4 * 0.25
    # Budget: n * budget = total BW we can spend
    
    total_bw_budget = n * budget
    
    # Try to fill FP16 tokens
    n_fp16 = 0
    bw_used = 0.0
    for i in range(n):
        if bw_used + 1.0 <= total_bw_budget:
            n_fp16 += 1
            bw_used += 1.0
        else:
            break
    
    # Try to fill INT8 tokens
    n_int8 = 0
    for i in range(n_fp16, n):
        if bw_used + 0.5 <= total_bw_budget:
            n_int8 += 1
            bw_used += 0.5
        else:
            break
    
    # Rest is INT4
    n_int4 = n - n_fp16 - n_int8
    
    # Assign labels
    for i in range(n_fp16):
        precision_labels[sorted_indices[i]] = "fp16"
    for i in range(n_fp16, n_fp16 + n_int8):
        precision_labels[sorted_indices[i]] = "int8"
    # remaining are already "int4"
    
    actual_bw = n_fp16 * 1.0 + n_int8 * 0.5 + n_int4 * 0.25
    
    tier_boundaries = {
        "n_fp16": n_fp16,
        "n_int8": n_int8,
        "n_int4": n_int4,
        "actual_bandwidth": actual_bw / n,
        "target_bandwidth": budget,
    }
    
    return precision_labels, tier_boundaries


def compute_importance_scores(
    attention_weights: torch.Tensor,
    strategy: str = "attention_sum",
    sink_count: int = 16,
) -> np.ndarray:
    """
    Compute per-token importance scores for tier assignment.
    
    Args:
        attention_weights: (num_layers, batch, num_heads, seq_len, seq_len)
                          or simplified (seq_len,) average
        strategy: scoring method
        sink_count: number of sink tokens to boost
    
    Returns:
        scores: (seq_len,) importance score per token
    """
    if attention_weights.dim() == 1:
        # Already aggregated
        scores = attention_weights.cpu().numpy()
    elif attention_weights.dim() == 2:
        # (seq_len, seq_len) attention matrix
        # Score = sum of attention received by each token
        scores = attention_weights.sum(dim=0).cpu().numpy()
    else:
        # Multi-layer, multi-head: aggregate
        # attention_weights: (layers, batch, heads, q_len, k_len)
        # Average across layers and heads, sum across queries
        scores = attention_weights.mean(dim=(0, 1, 2)).sum(dim=0).cpu().numpy()
    
    # Boost sink tokens
    if sink_count > 0 and len(scores) > sink_count:
        scores[:sink_count] = scores.max() * 10  # ensure sinks are top priority
    
    # Normalize to [0, 1]
    smin, smax = scores.min(), scores.max()
    if smax > smin:
        scores = (scores - smin) / (smax - smin)
    else:
        scores = np.ones_like(scores)
    
    return scores


def compute_importance_scores_heuristic(
    seq_len: int,
    sink_count: int = 16,
    decay_rate: float = 0.01,
) -> np.ndarray:
    """
    Heuristic importance scores when we can't extract attention weights.
    
    Uses the same logic as SWS: sink + exponential decay + recency.
    This is a fallback for models where attention extraction is difficult.
    """
    scores = np.zeros(seq_len)
    
    # Sink tokens: very high importance
    if sink_count > 0:
        scores[:sink_count] = 1.0
    
    # Middle tokens: exponential decay from sink
    decay = np.exp(-np.arange(seq_len - sink_count) * decay_rate)
    scores[sink_count:] = decay
    
    # Recent tokens: linearly increasing importance for last 20%
    recency_start = int(seq_len * 0.8)
    if recency_start < seq_len:
        recency = np.linspace(0.3, 0.8, seq_len - recency_start)
        scores[recency_start:] = np.maximum(scores[recency_start:], recency)
    
    return scores


# =============================================================================
# Model Hooks for KV Cache Manipulation
# =============================================================================

class QCBMHookManager:
    """
    Manages forward hooks on model layers to intercept and modify KV cache.
    
    Strategy:
      1. Register hooks on each attention layer's forward pass
      2. On forward, capture the K/V states
      3. Apply quantization/dequantization based on tier assignment
      4. For SWS: zero out dropped tokens' KV (masking)
      5. For QCBM: quantize different tokens at different precisions
    """

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.hooks = []
        self.mode = "baseline"  # "baseline", "sws_drop", "sws_mixed", "qcbm"
        
        # Configuration for current run
        self.selected_indices = None      # for SWS
        self.precision_map = None         # for QCBM: (seq_len,) array of "fp16"/"int8"/"int4"
        self.kv_scales = {}               # stored scales for dequantization
        
        # Per-layer config (from QCBM optimization)
        self.layer_budgets = None         # (num_layers,) per-layer bandwidth budget
        
    def clear(self):
        """Remove all hooks and reset state."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.kv_scales = {}
        self.mode = "baseline"
        self.selected_indices = None
        self.precision_map = None

    def set_sws_drop_mode(self, selected_indices: List[int]):
        """Configure for SWS token dropping (FP16 only)."""
        self.mode = "sws_drop"
        self.selected_indices = set(selected_indices)

    def set_sws_mixed_mode(self, selected_indices: List[int], seq_len: int):
        """
        Configure for SWS + mixed precision:
        - Selected tokens → FP16
        - Dropped tokens → INT8 (kept but quantized)
        """
        self.mode = "sws_mixed"
        self.selected_indices = set(selected_indices)
        # Build precision map
        self.precision_map = np.array(
            ["fp16" if i in self.selected_indices else "int8" for i in range(seq_len)],
            dtype=object,
        )

    def set_qcbm_mode(self, precision_map: np.ndarray, layer_budgets: Optional[np.ndarray] = None):
        """
        Configure for QCBM: all tokens kept, tiered precision.
        
        Args:
            precision_map: (seq_len,) array of "fp16"/"int8"/"int4"
            layer_budgets: optional (num_layers,) per-layer budget multipliers
        """
        self.mode = "qcbm"
        self.precision_map = precision_map
        self.layer_budgets = layer_budgets


# =============================================================================
# PPL Evaluation with KV Cache Manipulation
# =============================================================================

def load_model(model_key: str, hf_path: str):
    """Load model and tokenizer."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_key} from {hf_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[{time.strftime('%H:%M:%S')}] Loaded. VRAM={vram:.1f}GB")
    return model, tokenizer


def get_wikitext2_data(tokenizer, seq_len=2048, num_samples=5):
    """Load WikiText-2 test data."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        import urllib.request
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        zip_path = "/tmp/wikitext-2-raw.zip"
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('wikitext-2-raw/wiki.test.raw') as f:
                text = f.read().decode('utf-8')

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        if end <= len(input_ids):
            samples.append(input_ids[start:end])
    return samples


def compute_ppl_baseline(model, input_ids):
    """Full KV baseline PPL (no modification)."""
    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0).to(model.device),
            labels=input_ids.unsqueeze(0).to(model.device),
        )
    ppl = torch.exp(outputs.loss).item()
    return ppl


def compute_ppl_with_quantized_kv(
    model,
    input_ids,
    mode: str,
    precision_map: Optional[np.ndarray] = None,
    selected_indices: Optional[List[int]] = None,
    layer_budgets: Optional[np.ndarray] = None,
    sink_count: int = 16,
):
    """
    Compute PPL with modified KV cache.
    
    This works by running the model forward, intercepting KV cache,
    and applying quantization or masking.
    
    Implementation approach:
      Instead of hooking into the model (which is fragile), we use a
      two-pass approach:
        Pass 1: Run full forward, capture KV cache at each layer
        Pass 2: Modify KV cache, run decoder-only forward with modified cache
      
      However, for simplicity and reliability, we use a SINGLE-PASS approach
      that modifies the model's past_key_values between generation steps.
      
      Actually, the simplest reliable approach for PPL is:
        1. Run forward pass to get full KV cache
        2. Modify the KV cache (quantize/mask)
        3. Compute loss using the modified cache
    
    Since HuggingFace models don't easily support step-by-step PPL with
    modified KV, we use the following approach:
      - For "sws_drop": use position_ids trick (only selected positions)
      - For "sws_mixed" and "qcbm": use custom forward with KV modification
    """
    device = model.device
    n = len(input_ids)
    
    if mode == "sws_drop":
        # SWS token dropping: only feed selected tokens with original position_ids
        ix = sorted(selected_indices)
        selected_ids = input_ids[ix].to(device)
        position_ids = torch.tensor(ix, dtype=torch.long, device=device)
        
        with torch.no_grad():
            outputs = model(
                selected_ids.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                labels=selected_ids.unsqueeze(0),
            )
        return torch.exp(outputs.loss).item()
    
    elif mode in ("sws_mixed", "qcbm"):
        # Mixed-precision approach:
        # 1. Run full forward to get KV cache
        # 2. Modify KV cache per precision tier
        # 3. Re-run forward with modified past_key_values
        
        with torch.no_grad():
            # Pass 1: Get full KV cache
            outputs = model(
                input_ids.unsqueeze(0).to(device),
                use_cache=True,
            )
            past_kv = outputs.past_key_values
        
        # Modify KV cache
        modified_kv = []
        for layer_idx, (key, value) in enumerate(past_kv):
            # key/value shape: (batch, num_heads, seq_len, head_dim)
            key_mod = key.clone()
            value_mod = value.clone()
            
            if mode == "sws_mixed":
                # Selected tokens: FP16 (unchanged)
                # Dropped tokens: quantize to INT8 then dequantize
                for t in range(n):
                    if t not in set(selected_indices):
                        # Quantize this token's K and V to INT8
                        key_t = key_mod[:, :, t:t+1, :]
                        value_t = value_mod[:, :, t:t+1, :]
                        
                        kq, k_scale = KVCacheQuantizer.quantize_per_channel(key_t, "int8")
                        vq, v_scale = KVCacheQuantizer.quantize_per_channel(value_t, "int8")
                        
                        key_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(kq, k_scale)
                        value_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(vq, v_scale)
            
            elif mode == "qcbm":
                # Per-token precision based on importance
                for t in range(n):
                    prec = precision_map[t]
                    if prec == "fp16":
                        continue  # No modification
                    
                    # Apply per-layer budget multiplier if available
                    # (lower layers can use lower precision)
                    actual_prec = prec
                    if layer_budgets is not None:
                        layer_budget = layer_budgets[layer_idx]
                        # If this layer has budget < 0.3, downgrade precision
                        if layer_budget < 0.3 and prec == "int8":
                            actual_prec = "int4"
                    
                    key_t = key_mod[:, :, t:t+1, :]
                    value_t = value_mod[:, :, t:t+1, :]
                    
                    kq, k_scale = KVCacheQuantizer.quantize_per_channel(key_t, actual_prec)
                    vq, v_scale = KVCacheQuantizer.quantize_per_channel(value_t, actual_prec)
                    
                    key_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(kq, k_scale)
                    value_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(vq, v_scale)
            
            modified_kv.append((key_mod, value_mod))
        
        # Pass 2: Compute PPL with modified KV cache
        # We feed the full input_ids but use the modified past_kv
        # Since past_kv already has all positions, we compute loss
        # by re-running with the modified cache
        with torch.no_grad():
            # Reconstruct logits from modified KV
            outputs2 = model(
                input_ids.unsqueeze(0).to(device),
                past_key_values=modified_kv,
                use_cache=True,
            )
            logits = outputs2.logits
            
            # Compute cross-entropy loss (shift)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[1:].unsqueeze(0).to(device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            ppl = torch.exp(loss).item()
        
        # Free memory
        del past_kv, modified_kv, outputs, outputs2
        torch.cuda.empty_cache()
        
        return ppl
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# NIAH (Needle-in-a-Haystack) Evaluation
# =============================================================================

def build_haystack_with_needle(tokenizer, target_tokens, needle_text, 
                                insert_fraction, max_haystack):
    """Build a haystack text with a needle inserted at a given depth."""
    passage = (
        "The development of computational methods has transformed many fields of research. "
        "Researchers analyze large datasets to identify patterns and trends. "
        "Modern algorithms process information efficiently across distributed systems. "
        "Statistical models provide frameworks for understanding complex phenomena. "
        "The integration of technology continues to reshape professional practices. "
    )
    passage_ids = tokenizer(passage, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needle_ids = tokenizer(needle_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    
    available = max_haystack - len(needle_ids) - 10
    if available < 100:
        print(f"  WARNING: Very short haystack ({available} tokens)")
    
    repeats = (available // len(passage_ids)) + 1
    full_passage = passage_ids.repeat(repeats)[:available]
    insert_pos = int(len(full_passage) * insert_fraction)
    
    haystack_with_needle = torch.cat([
        full_passage[:insert_pos],
        needle_ids,
        full_passage[insert_pos:]
    ])
    haystack_with_needle = haystack_with_needle[:max_haystack]
    
    return haystack_with_needle, insert_pos


def run_niah_single(
    model, tokenizer, input_ids, needle_text,
    strategy: str, budget: float,
    insert_fraction: float,
    precision_map: Optional[np.ndarray] = None,
    selected_indices: Optional[List[int]] = None,
    sink_count: int = 16,
):
    """
    Run a single NIAH trial.
    
    Args:
        strategy: "full", "sws_drop", "sws_mixed", "qcbm"
        budget: bandwidth budget fraction
        precision_map: for qcbm strategy
        selected_indices: for sws strategies
    """
    n = len(input_ids)
    device = model.device
    has_chat = hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None
    
    if strategy == "full" or budget >= 1.0:
        # Full context, no modification
        pass
    elif strategy == "sws_drop":
        # Only keep selected tokens
        ix = sorted(selected_indices)
        input_ids = input_ids[torch.tensor(ix)]
    elif strategy in ("sws_mixed", "qcbm"):
        # All tokens kept, but KV will be modified during generation
        # For NIAH, we still feed all tokens but apply quantization
        pass
    
    # Build prompt
    messages = [
        {"role": "system", "content": 
         "You are a helpful assistant that answers questions based on provided text."},
        {"role": "user", "content": 
         f"Read the following text carefully and answer the question.\n\n"
         f"{tokenizer.decode(input_ids)}\n\n"
         f"What is the confirmation number mentioned in the text? "
         f"Reply with only the number."},
    ]
    
    if has_chat:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        max_input_len = min(n + 200, 16384)
        input_tensor = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_len
        ).input_ids.to(device)
    else:
        input_tensor = input_ids.unsqueeze(0).to(device)
    
    try:
        # For QCBM/SWS-mixed: apply KV quantization during generation
        if strategy in ("sws_mixed", "qcbm") and budget < 1.0:
            # Two-phase approach: prefill with full tokens, then modify KV
            with torch.no_grad():
                # Prefill phase: get full KV cache
                outputs = model(input_tensor, use_cache=True)
                past_kv = outputs.past_key_values
            
            # Modify KV cache
            seq_len_kv = past_kv[0][0].shape[2]
            modified_kv = []
            for layer_idx, (key, value) in enumerate(past_kv):
                key_mod = key.clone()
                value_mod = value.clone()
                
                if strategy == "sws_mixed" and selected_indices is not None:
                    sel_set = set(selected_indices)
                    for t in range(min(seq_len_kv, n)):
                        if t not in sel_set:
                            key_t = key_mod[:, :, t:t+1, :]
                            value_t = value_mod[:, :, t:t+1, :]
                            kq, ks = KVCacheQuantizer.quantize_per_channel(key_t, "int8")
                            vq, vs = KVCacheQuantizer.quantize_per_channel(value_t, "int8")
                            key_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(kq, ks)
                            value_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(vq, vs)
                
                elif strategy == "qcbm" and precision_map is not None:
                    for t in range(min(seq_len_kv, len(precision_map))):
                        prec = precision_map[t]
                        if prec == "fp16":
                            continue
                        key_t = key_mod[:, :, t:t+1, :]
                        value_t = value_mod[:, :, t:t+1, :]
                        kq, ks = KVCacheQuantizer.quantize_per_channel(key_t, prec)
                        vq, vs = KVCacheQuantizer.quantize_per_channel(value_t, prec)
                        key_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(kq, ks)
                        value_mod[:, :, t:t+1, :] = KVCacheQuantizer.dequantize(vq, vs)
                
                modified_kv.append((key_mod, value_mod))
            
            # Generate with modified KV
            with torch.no_grad():
                out = model.generate(
                    input_tensor,
                    past_key_values=modified_kv,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id or 0,
                )
            
            del past_kv, modified_kv
            torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                out = model.generate(
                    input_tensor,
                    max_new_tokens=30,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id or 0,
                )
        
        generated = out[0][input_tensor.shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        
        # Check if needle was found
        needle_value = needle_text.strip()
        found = False
        
        # Word-level matching
        needle_words = needle_value.lower().split()
        for word in needle_words:
            cleaned = word.rstrip(string.punctuation)
            if len(cleaned) >= 4 and cleaned.isalnum() and not cleaned.startswith("the"):
                if cleaned in response.lower():
                    found = True
                    break
        
        # Full phrase matching
        needle_stripped = needle_value.lower().rstrip(string.punctuation)
        if needle_stripped in response.lower():
            found = True
        
        # Check if needle was in selected indices (for coverage)
        needle_coverage = 1.0
        if strategy == "sws_drop" and selected_indices is not None:
            insert_pos = int(n * insert_fraction)
            needle_len_est = max(5, int(n * 0.02))
            needle_range = set(range(max(0, insert_pos - 2), min(n, insert_pos + needle_len_est + 2)))
            sel_set = set(selected_indices)
            overlap = needle_range & sel_set
            needle_coverage = len(overlap) / max(len(needle_range), 1)
        elif strategy in ("sws_mixed", "qcbm"):
            # All tokens are present; coverage depends on precision
            if precision_map is not None:
                insert_pos = int(n * insert_fraction)
                needle_len_est = max(5, int(n * 0.02))
                needle_range = range(max(0, insert_pos - 2), min(n, insert_pos + needle_len_est + 2))
                # Coverage weighted by precision quality factor
                total_quality = sum(PRECISION_QUALITY_FACTOR.get(precision_map[t], 1.0) for t in needle_range)
                needle_coverage = total_quality / max(len(needle_range), 1)
        
        return found, response[:100], needle_coverage
    
    except Exception as e:
        traceback.print_exc()
        return False, f"ERROR: {e}", 0.0


# =============================================================================
# SWS Token Selection
# =============================================================================

def select_sws_tokens(n: int, budget: float, sink_count: int = 16) -> List[int]:
    """SWS token selection: sink + exponential decay scoring, top-k selection."""
    keep = max(sink_count, int(n * budget))
    scores = np.zeros(n)
    scores[:sink_count] = 1e6  # ensure sink tokens are selected
    decay = np.exp(-np.arange(n - sink_count) * 0.01)
    scores[sink_count:] = decay
    
    # Boost recent tokens
    recency_start = int(n * 0.8)
    if recency_start < n:
        recency = np.linspace(0.3, 0.9, n - recency_start)
        scores[recency_start:] = np.maximum(scores[recency_start:], recency)
    
    selected = np.argsort(scores)[-keep:]
    return sorted(selected.tolist())


def select_pdtrim_tokens(n: int, budget: float, first_ratio: float = 0.5) -> List[int]:
    """PDTrim token selection: first_k + last_w."""
    budget_tokens = int(n * budget)
    first_k = int(budget_tokens * first_ratio)
    last_w = budget_tokens - first_k
    first_indices = list(range(first_k))
    last_indices = list(range(n - last_w, n))
    return sorted(set(first_indices + last_indices))


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_ppl_experiment(model, tokenizer, model_key, seq_len, budgets, 
                       num_samples, sink_count):
    """Run all PPL configurations and return results."""
    
    print(f"\n{'='*80}")
    print(f"PPL Experiment: {model_key}, seq_len={seq_len}")
    print(f"{'='*80}")
    
    samples = get_wikitext2_data(tokenizer, seq_len, num_samples)
    print(f"Loaded {len(samples)} WikiText-2 samples (seq_len={seq_len})")
    
    results = {
        "model": model_key,
        "seq_len": seq_len,
        "num_samples": len(samples),
        "configurations": [],
    }
    
    # --- Baseline: Full KV FP16 ---
    print("\n[1/4] Baseline: Full KV FP16")
    baseline_ppls = []
    for i, sample in enumerate(samples):
        ppl = compute_ppl_baseline(model, sample)
        baseline_ppls.append(ppl)
        print(f"  Sample {i}: PPL = {ppl:.4f}")
    baseline_ppl = float(np.mean(baseline_ppls))
    results["baseline_ppl"] = baseline_ppl
    print(f"  Mean baseline PPL: {baseline_ppl:.4f}")
    
    # For each budget, test 4 configurations
    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Budget = {budget}")
        print(f"{'='*60}")
        
        n = seq_len
        
        # Compute importance scores for QCBM
        importance = compute_importance_scores_heuristic(n, sink_count=sink_count)
        
        # --- SWS b=X, FP16 (current approach) ---
        print(f"\n[2/4] SWS b={budget}, FP16 only (selected tokens)")
        sws_indices = select_sws_tokens(n, budget, sink_count=sink_count)
        
        sws_fp16_ppls = []
        for i, sample in enumerate(samples):
            ppl = compute_ppl_with_quantized_kv(
                model, sample, mode="sws_drop", selected_indices=sws_indices,
            )
            delta = (ppl / baseline_ppl - 1) * 100
            sws_fp16_ppls.append(ppl)
            print(f"  Sample {i}: PPL = {ppl:.4f} (Δ={delta:+.2f}%)")
        
        mean_sws_fp16 = float(np.mean(sws_fp16_ppls))
        delta_sws_fp16 = (mean_sws_fp16 / baseline_ppl - 1) * 100
        
        # Compute effective bandwidth for SWS
        sws_bw = budget * 1.0  # selected tokens * FP16 = budget fraction
        sws_fp16_eq_tokens = int(n * budget)  # FP16-equivalent token count
        
        results["configurations"].append({
            "name": f"SWS b={budget} FP16",
            "mode": "sws_drop",
            "budget": budget,
            "precision": "fp16",
            "retention": budget,
            "effective_bandwidth": sws_bw,
            "fp16_eq_tokens": sws_fp16_eq_tokens,
            "mean_ppl": mean_sws_fp16,
            "delta_pct": round(delta_sws_fp16, 2),
            "sample_ppls": sws_fp16_ppls,
        })
        print(f"  Mean PPL: {mean_sws_fp16:.4f} (Δ={delta_sws_fp16:+.2f}%)")
        print(f"  Effective BW: {sws_bw:.2f}x, FP16-eq tokens: {sws_fp16_eq_tokens}")
        
        # --- SWS b=X + Mixed Precision ---
        # Selected tokens FP16, dropped tokens INT8
        # BW = budget * 1.0 + (1 - budget) * 0.5
        print(f"\n[3/4] SWS b={budget} + Mixed Precision (selected=FP16, rest=INT8)")
        sws_mixed_bw = budget * 1.0 + (1 - budget) * 0.5
        sws_mixed_fp16_eq = int(n * sws_mixed_bw)
        
        sws_mixed_ppls = []
        for i, sample in enumerate(samples):
            ppl = compute_ppl_with_quantized_kv(
                model, sample, mode="sws_mixed",
                selected_indices=sws_indices,
                precision_map=None,
            )
            delta = (ppl / baseline_ppl - 1) * 100
            sws_mixed_ppls.append(ppl)
            print(f"  Sample {i}: PPL = {ppl:.4f} (Δ={delta:+.2f}%)")
        
        mean_sws_mixed = float(np.mean(sws_mixed_ppls))
        delta_sws_mixed = (mean_sws_mixed / baseline_ppl - 1) * 100
        
        results["configurations"].append({
            "name": f"SWS b={budget} Mixed (sel=FP16, rest=INT8)",
            "mode": "sws_mixed",
            "budget": budget,
            "precision": "mixed_fp16_int8",
            "retention": 1.0,  # all tokens kept
            "effective_bandwidth": round(sws_mixed_bw, 4),
            "fp16_eq_tokens": sws_mixed_fp16_eq,
            "mean_ppl": mean_sws_mixed,
            "delta_pct": round(delta_sws_mixed, 2),
            "sample_ppls": sws_mixed_ppls,
        })
        print(f"  Mean PPL: {mean_sws_mixed:.4f} (Δ={delta_sws_mixed:+.2f}%)")
        print(f"  Effective BW: {sws_mixed_bw:.2f}x, FP16-eq tokens: {sws_mixed_fp16_eq}")
        
        # --- QCBM: ALL tokens, tiered precision, matching SWS bandwidth ---
        print(f"\n[4/4] QCBM: ALL tokens, tiered precision (BW budget={budget})")
        precision_map, tier_info = compute_qcbm_tiers_for_budget(budget, importance)
        
        # Per-layer budgets from QCBM simulation
        # Low layers: ~0.25, high layers: ~0.65 (from TS simulation)
        num_layers = model.config.num_hidden_layers
        layer_budgets = np.zeros(num_layers)
        for l in range(num_layers):
            # Sigmoid curve: low layers get low budget, high layers get high budget
            t = l / (num_layers - 1) if num_layers > 1 else 0.5
            layer_budgets[l] = 0.25 + 0.40 * (1 / (1 + np.exp(-10 * (t - 0.5))))
        
        qcbm_ppls = []
        for i, sample in enumerate(samples):
            ppl = compute_ppl_with_quantized_kv(
                model, sample, mode="qcbm",
                precision_map=precision_map,
                selected_indices=None,
                layer_budgets=layer_budgets,
            )
            delta = (ppl / baseline_ppl - 1) * 100
            qcbm_ppls.append(ppl)
            print(f"  Sample {i}: PPL = {ppl:.4f} (Δ={delta:+.2f}%)")
        
        mean_qcbm = float(np.mean(qcbm_ppls))
        delta_qcbm = (mean_qcbm / baseline_ppl - 1) * 100
        
        qcbm_fp16_eq = int(n * tier_info["actual_bandwidth"])
        
        results["configurations"].append({
            "name": f"QCBM (all tokens, tiered precision, BW={budget})",
            "mode": "qcbm",
            "budget": budget,
            "precision": "tiered_fp16_int8_int4",
            "retention": 1.0,
            "effective_bandwidth": round(tier_info["actual_bandwidth"], 4),
            "fp16_eq_tokens": qcbm_fp16_eq,
            "tier_info": tier_info,
            "mean_ppl": mean_qcbm,
            "delta_pct": round(delta_qcbm, 2),
            "sample_ppls": qcbm_ppls,
        })
        print(f"  Mean PPL: {mean_qcbm:.4f} (Δ={delta_qcbm:+.2f}%)")
        print(f"  Tiers: FP16={tier_info['n_fp16']}, INT8={tier_info['n_int8']}, INT4={tier_info['n_int4']}")
        print(f"  Actual BW: {tier_info['actual_bandwidth']:.4f}x, FP16-eq tokens: {qcbm_fp16_eq}")
    
    return results


def run_niah_experiment(model, tokenizer, model_key, budgets, seq_len, sink_count):
    """Run NIAH experiment across budgets and strategies."""
    
    print(f"\n{'='*80}")
    print(f"NIAH Experiment: {model_key}, seq_len={seq_len}")
    print(f"{'='*80}")
    
    needles = [
        "The confirmation number is 7294.",
        "The passcode value is 5831.",
        "The reference code is 4168.",
    ]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Strategy configurations
    strategies = []
    strategies.append(("full", 1.0, {}))
    for budget in budgets:
        strategies.append(("sws_drop", budget, {"sink": sink_count}))
        strategies.append(("sws_mixed", budget, {"sink": sink_count}))
        strategies.append(("qcbm", budget, {}))
    
    results = {
        "model": model_key,
        "seq_len": seq_len,
        "experiments": [],
    }
    
    for ni, needle in enumerate(needles):
        for depth in depths:
            input_ids, insert_pos = build_haystack_with_needle(
                tokenizer, seq_len, needle, depth, seq_len
            )
            n = len(input_ids)
            print(f"\n  Needle {ni+1}/{len(needles)} at {depth*100:.0f}% depth "
                  f"(pos={insert_pos}), n_tokens={n}")
            
            for strat, budget, kwargs in strategies:
                # Prepare strategy-specific parameters
                selected_indices = None
                precision_map = None
                
                if strat == "sws_drop":
                    selected_indices = select_sws_tokens(n, budget, sink_count=sink_count)
                elif strat == "sws_mixed":
                    selected_indices = select_sws_tokens(n, budget, sink_count=sink_count)
                elif strat == "qcbm":
                    importance = compute_importance_scores_heuristic(n, sink_count=sink_count)
                    precision_map, tier_info = compute_qcbm_tiers_for_budget(budget, importance)
                
                found, response, coverage = run_niah_single(
                    model, tokenizer, input_ids, needle,
                    strategy=strat, budget=budget,
                    insert_fraction=depth,
                    precision_map=precision_map,
                    selected_indices=selected_indices,
                    sink_count=sink_count,
                )
                
                status = "✓" if found else "✗"
                print(f"    {strat} b={budget}: {status} cov={coverage:.2f} "
                      f"resp='{response[:40]}'")
                
                results["experiments"].append({
                    "needle": needle,
                    "depth": depth,
                    "insert_pos": insert_pos,
                    "n_tokens": n,
                    "strategy": strat,
                    "budget": budget,
                    "retrieved": found,
                    "response": response,
                    "needle_coverage": round(coverage, 2),
                })
                
                torch.cuda.empty_cache()
    
    return results


# =============================================================================
# Reporting
# =============================================================================

def print_ppl_summary(results):
    """Print a formatted PPL comparison table."""
    baseline = results["baseline_ppl"]
    
    print(f"\n{'='*90}")
    print(f"PPL COMPARISON: {results['model']} (baseline PPL = {baseline:.4f})")
    print(f"{'='*90}")
    print(f"{'Configuration':<45} {'BW':>6} {'Retain':>7} {'FP16-eq':>8} "
          f"{'Mean PPL':>10} {'Δ%':>8}")
    print("-" * 90)
    
    for cfg in results["configurations"]:
        print(f"{cfg['name']:<45} {cfg['effective_bandwidth']:>6.2f} "
              f"{cfg['retention']:>7.2f} {cfg['fp16_eq_tokens']:>8d} "
              f"{cfg['mean_ppl']:>10.4f} {cfg['delta_pct']:>+8.2f}%")


def print_niah_summary(results):
    """Print a formatted NIAH comparison table."""
    print(f"\n{'='*90}")
    print(f"NIAH COMPARISON: {results['model']}")
    print(f"{'='*90}")
    
    # Group by strategy and budget
    grouped = defaultdict(list)
    for exp in results["experiments"]:
        key = f"{exp['strategy']}_b{exp['budget']}"
        grouped[key].append(exp)
    
    print(f"{'Strategy':<25} {'Budget':>6} {'Retrieval Rate':>15} "
          f"{'Avg Coverage':>13}")
    print("-" * 65)
    
    for key in sorted(grouped.keys()):
        exps = grouped[key]
        strat = exps[0]["strategy"]
        budget = exps[0]["budget"]
        retrieval_rate = sum(1 for e in exps if e["retrieved"]) / len(exps) * 100
        avg_coverage = np.mean([e["needle_coverage"] for e in exps])
        
        print(f"{strat:<25} {budget:>6.1f} {retrieval_rate:>14.1f}% "
              f"{avg_coverage:>13.2f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="QCBM GPU Experiment: Mixed-Precision KV Cache Quantization"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()),
                        help="Model to evaluate")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.3, 0.5],
                        help="Bandwidth budgets to test (as fraction of FP16)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of WikiText-2 samples for PPL")
    parser.add_argument("--sink_count", type=int, default=16,
                        help="Number of sink tokens")
    parser.add_argument("--ppl_only", action="store_true",
                        help="Run only PPL evaluation")
    parser.add_argument("--niah_only", action="store_true",
                        help="Run only NIAH evaluation")
    parser.add_argument("--niah_seq_len", type=int, default=4096,
                        help="Sequence length for NIAH evaluation")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for results")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    hf_path = MODELS[args.model]
    model, tokenizer = load_model(args.model, hf_path)
    
    all_results = {
        "model": args.model,
        "hf_path": hf_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ppl_results": None,
        "niah_results": None,
    }
    
    # Run PPL experiment
    if not args.niah_only:
        ppl_results = run_ppl_experiment(
            model, tokenizer, args.model,
            args.seq_len, args.budgets, args.num_samples, args.sink_count,
        )
        print_ppl_summary(ppl_results)
        all_results["ppl_results"] = ppl_results
        
        # Save PPL results
        ppl_path = os.path.join(
            args.output_dir,
            f"qcbm_ppl_{args.model}_seq{args.seq_len}.json"
        )
        with open(ppl_path, "w") as f:
            json.dump(ppl_results, f, indent=2)
        print(f"\nPPL results saved to {ppl_path}")
    
    # Run NIAH experiment
    if not args.ppl_only:
        niah_results = run_niah_experiment(
            model, tokenizer, args.model,
            args.budgets, args.niah_seq_len, args.sink_count,
        )
        print_niah_summary(niah_results)
        all_results["niah_results"] = niah_results
        
        # Save NIAH results
        niah_path = os.path.join(
            args.output_dir,
            f"qcbm_niah_{args.model}_seq{args.niah_seq_len}.json"
        )
        with open(niah_path, "w") as f:
            json.dump(niah_results, f, indent=2)
        print(f"\nNIAH results saved to {niah_path}")
    
    # Save combined results
    combined_path = os.path.join(
        args.output_dir,
        f"qcbm_combined_{args.model}.json"
    )
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nCombined results saved to {combined_path}")
    
    # Print key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS SUMMARY")
    print(f"{'='*80}")
    
    if all_results["ppl_results"]:
        baseline = all_results["ppl_results"]["baseline_ppl"]
        for cfg in all_results["ppl_results"]["configurations"]:
            bw = cfg["effective_bandwidth"]
            ppl = cfg["mean_ppl"]
            delta = cfg["delta_pct"]
            fp16_eq = cfg["fp16_eq_tokens"]
            retain = cfg["retention"]
            print(f"  {cfg['name']}")
            print(f"    BW={bw:.2f}x | Retention={retain:.0%} | "
                  f"FP16-eq={fp16_eq} | PPL={ppl:.4f} (Δ={delta:+.2f}%)")
    
    print(f"\nDone! {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
