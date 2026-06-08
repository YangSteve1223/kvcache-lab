#!/usr/bin/env python3
"""KV Cache Quantizer - Symmetric affine quantization for KV cache tensors."""

import numpy as np
import torch
from typing import Dict

class KVQuantizer:
    """Symmetric affine quantization for KV cache tensors.
    
    Per-head, per-token granularity:
      scale = max(|x|) / qmax
      x_q   = round(x / scale), clamped to [-qmax, qmax]
      x_dq  = x_q * scale
    """

    QMAX = {"int8": 127, "int4": 7}

    @staticmethod
    def quantize_dequantize(kv_tensor: torch.Tensor, precision: str) -> torch.Tensor:
        """Quantize then dequantize a KV cache slice (simulate quantization error)."""
        if precision == "fp16":
            return kv_tensor

        assert precision in ("int8", "int4"), f"Unknown precision: {precision}"
        qmax = KVQuantizer.QMAX[precision]

        original_dtype = kv_tensor.dtype
        kv_float = kv_tensor.float()

        # Per-token scale: max absolute value per (..., seq_len) position
        abs_max = kv_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / qmax

        quantized = torch.clamp(torch.round(kv_float / scale), -qmax, qmax)
        dequantized = quantized * scale

        return dequantized.to(original_dtype)

    @staticmethod
    def quantize_with_scale(kv_tensor: torch.Tensor, precision: str):
        """Quantize and return both quantized values and scale."""
        if precision == "fp16":
            return kv_tensor, None

        qmax = KVQuantizer.QMAX[precision]
        original_dtype = kv_tensor.dtype
        kv_float = kv_tensor.float()

        abs_max = kv_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = abs_max / qmax

        quantized = torch.clamp(torch.round(kv_float / scale), -qmax, qmax)
        return quantized.half(), scale.squeeze(-1).half()

    @staticmethod
    def compute_error(kv_tensor: torch.Tensor, precision: str) -> Dict[str, float]:
        """Compute quantization error metrics for a KV tensor slice."""
        if precision == "fp16":
            return {"rel_l2": 0.0, "max_abs": 0.0, "cosine_sim": 1.0}

        deq = KVQuantizer.quantize_dequantize(kv_tensor, precision)
        diff = kv_tensor.float() - deq.float()

        orig_norm = kv_tensor.float().norm().item()
        diff_norm = diff.norm().item()
        rel_l2 = diff_norm / max(orig_norm, 1e-8)
        max_abs = diff.abs().max().item()

        # Cosine similarity
        orig_flat = kv_tensor.float().flatten()
        deq_flat = deq.float().flatten()
        cos_sim = F.cosine_similarity(orig_flat.unsqueeze(0), deq_flat.unsqueeze(0)).item()

        return {"rel_l2": rel_l2, "max_abs": max_abs, "cosine_sim": cos_sim}


# =============================================================================
# Tier Assignment Strategies
# =============================================================================

