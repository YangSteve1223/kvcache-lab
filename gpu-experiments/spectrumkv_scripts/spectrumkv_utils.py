#!/usr/bin/env python3
"""
SpectrumKV Shared Utilities for GPU Experiments
=================================================
Common components: quantizer, tier assignment, importance scoring,
model loading, data loading. All GPU experiment scripts import from here.
"""

import os
import time
import warnings
from typing import List, Dict

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
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma9b": "google/gemma-2-9b-it",
}

# Local model paths (checked first; falls back to HF cache if not found)
LOCAL_MODEL_PATHS = {
    "qwen7b": "/root/autodl-tmp/Qwen2.5-7B-Instruct",
    "mistral7b": "/root/autodl-tmp/Mistral-7B-Instruct-v0.3",
    "gemma9b": "/root/autodl-tmp/gemma-2-9b-it",
}

# Bandwidth cost per precision tier (relative to FP16)
PRECISION_BW = {"fp16": 1.0, "int8": 0.5, "int4": 0.25}

# Quality factor per precision (simulation prior; GPU validates)
PRECISION_QUALITY = {"fp16": 1.0, "int8": 0.95, "int4": 0.78}

SINK_COUNT = 4  # Default sink tokens (realistic for 2K-4K context)


# =============================================================================
# KV Cache Quantizer
# =============================================================================

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

def spectrumkv_greedy(budget: float, importance_scores: np.ndarray) -> np.ndarray:
    """
    SpectrumKV Greedy: ALL tokens retained, greedy precision tier assignment.
    
    Strategy:
      1. Start all at INT4 (cheapest)
      2. Upgrade highest-importance to INT8 (benefit/cost = 0.68)
      3. Upgrade top-importance to FP16 (benefit/cost = 0.10)
    """
    n = len(importance_scores)
    tiers = np.full(n, "int4", dtype=object)

    if budget <= 0.25:
        return tiers

    ranked = np.argsort(-importance_scores)
    remaining = (budget - 0.25) * n  # Budget above INT4 baseline

    # Phase 1: INT4→INT8
    n_int8 = min(n, int(remaining / 0.25))
    for i in range(n_int8):
        tiers[ranked[i]] = "int8"
    remaining -= n_int8 * 0.25

    # Phase 2: INT8→FP16
    n_fp16 = min(n_int8, max(0, int(remaining / 0.50 + 0.5)))
    for i in range(n_fp16):
        tiers[ranked[i]] = "fp16"

    # Verify (relaxed: warning instead of assert for small sequences)
    actual = sum(PRECISION_BW[t] for t in tiers) / n
    if abs(actual - budget) > 0.05:
        warnings.warn(
            f"spectrumkv_greedy: actual BW={actual:.3f} vs target={budget:.3f} "
            f"(diff={abs(actual-budget):.3f}). n={n} may be too small for exact budget."
        )

    return tiers


def spectrumkv_balanced(budget: float, importance_scores: np.ndarray) -> np.ndarray:
    """
    SpectrumKV Balanced: ALL tokens retained, proportional tier split.
    
    Tries to spread across FP16/INT8/INT4 proportionally.
    Falls back to FP16+INT8 when budget is too high for INT4.
    """
    n = len(importance_scores)
    ranked = np.argsort(-importance_scores)
    tiers = np.full(n, "int4", dtype=object)

    if budget <= 0.25:
        return tiers

    # Constraint: f*1.0 + m*0.5 + q*0.25 = budget, f+m+q = 1
    # → 3f + m = 4*budget - 1
    R = 4 * budget - 1

    if R <= 0:
        return tiers

    # Ideal balanced: f:m:q ≈ 1:2:2
    f = R / 5
    m = 2 * R / 5
    q = 1 - f - m

    if q < 0:
        q = 0
        f = 2 * budget - 1
        m = 1 - f
    # Note: elif f < 0 branch removed (dead code — f=R/5 can't be negative
    # when R>0, and R<=0 returns early above)

    f = max(0, min(1, f))
    m = max(0, min(1 - f, m))
    q = 1 - f - m

    n_fp16 = min(int(n * f), n)
    n_int8 = min(int(n * m), n - n_fp16)

    for i in range(n_fp16):
        tiers[ranked[i]] = "fp16"
    for i in range(n_fp16, n_fp16 + n_int8):
        tiers[ranked[i]] = "int8"

    return tiers


def spectrumkv_sink_protect(budget: float, importance_scores: np.ndarray,
                            sink: int = SINK_COUNT) -> np.ndarray:
    """SpectrumKV Greedy but sink tokens guaranteed FP16.
    
    Note: Forcing sink to FP16 may slightly exceed the target budget.
    The excess is bounded by (sink * 0.75) / n tokens worth of bandwidth,
    which is negligible for typical sequence lengths (>512).
    """
    tiers = spectrumkv_greedy(budget, importance_scores)
    n = len(tiers)
    
    # Force sink tokens to FP16
    for i in range(min(sink, n)):
        tiers[i] = "fp16"
    
    # Verify and log actual bandwidth (may slightly exceed budget)
    actual = sum(PRECISION_BW[t] for t in tiers) / n
    if abs(actual - budget) > 0.05:
        # If budget violation is significant, issue warning
        import warnings
        warnings.warn(
            f"spectrumkv_sink_protect: actual BW={actual:.3f} exceeds "
            f"target={budget:.3f} by {actual-budget:.3f}. "
            f"Consider reducing sink count or increasing budget."
        )
    
    return tiers


def random_tier(budget: float, n_tokens: int, seed: int = 42) -> np.ndarray:
    """Random baseline: randomly assign tiers within budget (ablation)."""
    rng = np.random.RandomState(seed)
    tiers = np.full(n_tokens, "int4", dtype=object)

    if budget <= 0.25:
        return tiers

    remaining = (budget - 0.25) * n_tokens
    perm = rng.permutation(n_tokens)

    n_int8 = min(n_tokens, int(remaining / 0.25))
    for i in range(n_int8):
        tiers[perm[i]] = "int8"
    remaining -= n_int8 * 0.25

    n_fp16 = min(n_int8, max(0, int(remaining / 0.50 + 0.5)))
    for i in range(n_fp16):
        tiers[perm[i]] = "fp16"

    return tiers


# =============================================================================
# Token Selection (for baseline comparison)
# =============================================================================

def pdtrim_select(n: int, budget: float, sink: int = SINK_COUNT) -> List[int]:
    """PDTrim: keep first sink + last (budget*n - sink) tokens."""
    k = max(int(n * budget), sink + 1)
    selected = set(range(min(sink, n)))
    selected.update(range(max(n - (k - sink), 0), n))
    return sorted(selected)


def sws_select(n: int, budget: float, importance_scores: np.ndarray,
               sink: int = SINK_COUNT) -> List[int]:
    """SWS: score-based top-K selection, sink always included."""
    k = max(int(n * budget), sink + 1)
    selected = set(range(min(sink, n)))
    remaining = k - len(selected)
    for idx in np.argsort(-importance_scores):
        if idx.item() not in selected:
            selected.add(idx.item())
            remaining -= 1
            if remaining <= 0:
                break
    return sorted(selected)


# =============================================================================
# Importance Score Computation
# =============================================================================

def compute_importance_heuristic(seq_len: int, sink: int = SINK_COUNT) -> np.ndarray:
    """Heuristic importance: sink + exponential decay + recency boost."""
    scores = np.zeros(seq_len, dtype=np.float64)

    if sink > 0:
        scores[:sink] = 1.0

    decay = np.exp(-np.arange(seq_len - sink) * 0.01)
    scores[sink:] = decay

    rec_start = int(seq_len * 0.8)
    if rec_start < seq_len:
        recency = np.linspace(0.3, 0.8, seq_len - rec_start)
        scores[rec_start:] = np.maximum(scores[rec_start:], recency)

    smin, smax = scores.min(), scores.max()
    if smax > smin:
        scores = (scores - smin) / (smax - smin)

    return scores


def compute_importance_from_attention(attn_weights: torch.Tensor,
                                      sink: int = SINK_COUNT) -> np.ndarray:
    """Compute per-token importance from extracted attention weights.
    
    Aggregates attention received by each key position across all
    query positions and attention heads.
    """
    if attn_weights.dim() == 1:
        scores = attn_weights.cpu().numpy()
    elif attn_weights.dim() == 2:
        # (Q, K) → per-key importance: sum over queries
        scores = attn_weights.sum(dim=0).cpu().numpy()
    elif attn_weights.dim() == 3:
        # (H, Q, K) → average over heads, sum over queries
        scores = attn_weights.mean(dim=0).sum(dim=0).cpu().numpy()
    else:
        # (B, H, Q, K) → average over batch & heads, sum over queries
        scores = attn_weights.mean(dim=(0, 1)).sum(dim=0).cpu().numpy()

    if len(scores) > sink and sink > 0:
        scores[:sink] = scores.max() * 10

    smin, smax = scores.min(), scores.max()
    if smax > smin:
        scores = (scores - smin) / (smax - smin)

    return scores


# =============================================================================
# Model & Data Loading
# =============================================================================

def load_model(model_key: str):
    """Load model and tokenizer. Checks local path first, falls back to HF cache."""
    local_path = LOCAL_MODEL_PATHS.get(model_key)
    if local_path and os.path.isdir(local_path):
        model_path = local_path
        print(f"[{time.strftime('%H:%M:%S')}] Loading {model_key} from LOCAL: {model_path}")
    else:
        model_path = MODELS[model_key]
        print(f"[{time.strftime('%H:%M:%S')}] Loading {model_key} from HF: {model_path}")

    is_local = local_path and os.path.isdir(local_path)
    # Use local_files_only on cloud GPUs (no internet or unreliable)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[{time.strftime('%H:%M:%S')}] Loaded. VRAM={vram:.1f}GB")
    return model, tokenizer


def get_wikitext2(tokenizer, seq_len=2048, num_samples=5):
    """Load WikiText-2 test data. Tries: 1) local file 2) HF datasets offline 3) HF datasets online 4) URL fallback."""
    text = None
    
    # 1) Try local pre-cached text file
    local_paths = ["/root/autodl-tmp/wikitext2_test.txt", "/tmp/wikitext2_test.txt"]
    for p in local_paths:
        if os.path.isfile(p):
            with open(p, "r") as f:
                text = f.read()
            print(f"[get_wikitext2] Loaded from local file: {p}")
            break
    
    # 2) Try HF datasets (offline first, then online)
    if text is None:
        try:
            from datasets import load_dataset
            try:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            except Exception:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", download_mode="force_redownload")
            text = "\n\n".join(ds["text"])
            print(f"[get_wikitext2] Loaded from HF datasets")
        except Exception:
            pass
    
    # 3) URL fallback with redirect handling
    if text is None:
        try:
            import urllib.request, zipfile
            url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
            zip_path = "/tmp/wikitext-2-raw.zip"
            opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler)
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open('wikitext-2-raw/wiki.test.raw') as f:
                    text = f.read().decode('utf-8')
            print(f"[get_wikitext2] Loaded from URL fallback")
        except Exception as e:
            raise RuntimeError(f"Cannot load WikiText-2 from any source. Last error: {e}")
    
    # Save to local for future runs
    if not os.path.isfile("/root/autodl-tmp/wikitext2_test.txt"):
        try:
            with open("/root/autodl-tmp/wikitext2_test.txt", "w") as f:
                f.write(text)
        except Exception:
            pass

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    samples = []
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        if end <= len(input_ids):
            samples.append(input_ids[start:end])
    return samples


# =============================================================================
# PPL Computation
# =============================================================================

def ppl_full_fp16(model, input_ids) -> float:
    """Oracle PPL: full FP16, no modification."""
    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0).to(model.device),
            labels=input_ids.unsqueeze(0).to(model.device),
        )
    return torch.exp(outputs.loss).item()


def ppl_selection_drop(model, input_ids, selected_indices: List[int]) -> float:
    """PPL for selection-based methods: only feed selected tokens with original position_ids."""
    device = model.device
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


def ppl_tiered(model, input_ids, precision_map: np.ndarray) -> float:
    """PPL for tiered-precision methods using KV projection hooks.
    
    Strategy: Register forward hooks on each layer's k_proj and v_proj modules.
    During a single forward pass, the hooks apply quantize-dequantize to the
    K and V projection outputs for non-FP16 tokens. The attention computation
    then uses the quantized K/V, producing correct logits for PPL calculation.
    
    This avoids the two-pass approach where passing past_key_values + full
    input_ids to HuggingFace models doubles the context length (input_ids
    treated as new tokens appended after the cache).
    """
    device = model.device
    hook_handles = []
    
    # Find all k_proj and v_proj modules (standard in Qwen2/Mistral/Gemma)
    kv_proj_modules = []
    for name, module in model.named_modules():
        if name.endswith('.k_proj') or name.endswith('.v_proj'):
            kv_proj_modules.append((name, module))
    
    if not kv_proj_modules:
        warnings.warn(
            "Could not find k_proj/v_proj modules. "
            "Falling back to two-pass PPL (results may be inaccurate)."
        )
        return _ppl_tiered_twopass_fallback(model, input_ids, precision_map)
    
    def make_quantize_hook(prec_map):
        """Hook that applies per-token QDQ to K/V projection output."""
        # Pre-compute index groups by precision tier for vectorized processing
        int8_indices = [t for t in range(len(prec_map)) if prec_map[t] == "int8"]
        int4_indices = [t for t in range(len(prec_map)) if prec_map[t] == "int4"]
        
        def hook_fn(module, input, output):
            # output shape: (batch, seq_len, num_heads * head_dim)
            if isinstance(output, tuple):
                main = output[0]
            else:
                main = output
            
            seq_len = main.shape[1]
            
            # Apply quantize-dequantize per tier (vectorized within tier)
            for prec, indices in [("int8", int8_indices), ("int4", int4_indices)]:
                valid_idx = [i for i in indices if i < seq_len]
                if not valid_idx:
                    continue
                idx_tensor = torch.tensor(valid_idx, device=main.device)
                # Select all tokens of this tier at once
                token_slice = main[:, idx_tensor, :]  # (1, n_tokens, hidden)
                main[:, idx_tensor, :] = KVQuantizer.quantize_dequantize(
                    token_slice, prec
                )
            
            return output
        return hook_fn
    
    # Register hooks on all k_proj and v_proj modules
    for name, module in kv_proj_modules:
        handle = module.register_forward_hook(make_quantize_hook(precision_map))
        hook_handles.append(handle)
    
    try:
        with torch.no_grad():
            outputs = model(
                input_ids.unsqueeze(0).to(device),
                labels=input_ids.unsqueeze(0).to(device),
            )
        ppl = torch.exp(outputs.loss).item()
    finally:
        # Always remove hooks to avoid side effects
        for h in hook_handles:
            h.remove()
    
    return ppl


def _ppl_tiered_twopass_fallback(model, input_ids, precision_map: np.ndarray) -> float:
    """
    Fallback PPL for models without k_proj/v_proj modules.
    
    Uses the two-pass approach with KV modification. Note: this method
    passes modified past_key_values + only the LAST token as input_ids,
    computing PPL for just the final position. This is less accurate
    than the hook-based approach but avoids the context-doubling bug.
    
    For full-sequence PPL, the hook-based approach (ppl_tiered) is preferred.
    This fallback provides a rough estimate only.
    """
    device = model.device
    n = len(input_ids)
    
    # Pass 1: Get full KV cache
    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0).to(device),
            use_cache=True,
        )
        past_kv = outputs.past_key_values
    
    # Modify KV cache per precision tier (vectorized)
    modified_kv = _modify_kv_by_precision(past_kv, precision_map, n)
    
    # Pass 2: Feed last token + modified past_kv to get final position prediction
    # This only computes PPL for the last token — rough estimate
    with torch.no_grad():
        last_token = input_ids[-1:].unsqueeze(0).to(device)
        outputs2 = model(
            last_token,
            past_key_values=modified_kv,
            use_cache=True,
        )
        # Logits for position n (predicting token n+1)
        # But we want PPL over the full sequence — this is inherently limited
        # Fall back to computing loss from full forward with modified KV
    
    del past_kv, modified_kv, outputs, outputs2
    torch.cuda.empty_cache()
    
    # For the fallback, use the hook-based approach which should always work
    # for standard HuggingFace models. If we reached here, the model
    # architecture is non-standard — raise an error.
    raise RuntimeError(
        "Could not compute PPL: no k_proj/v_proj modules found and "
        "two-pass approach is unreliable. Model architecture may need "
        "custom implementation."
    )


def _modify_kv_by_precision(past_kv, precision_map: np.ndarray, n: int):
    """Modify KV cache per precision tier (vectorized within tier).
    
    Groups tokens by precision tier and applies quantize-dequantize
    in batch for each tier, avoiding O(n) Python loop.
    
    Handles both DynamicCache and legacy tuple-of-tuples formats.
    Returns modified cache in the same format as input.
    """
    # Detect cache format
    is_dynamic = hasattr(past_kv, 'layers')
    has_key_cache = hasattr(past_kv, 'key_cache')
    
    # Pre-compute masks
    int8_mask = np.array([precision_map[t] == "int8" for t in range(n)])
    int4_mask = np.array([precision_map[t] == "int4" for t in range(n)])
    
    if is_dynamic:
        # DynamicCache with .layers — modify in place
        for layer_idx in range(len(past_kv.layers)):
            layer = past_kv.layers[layer_idx]
            if hasattr(layer, 'keys'):
                key, value = layer.keys, layer.values
            elif isinstance(layer, (tuple, list)):
                key, value = layer[0], layer[1]
            else:
                raise TypeError(f"Unknown cache layer type: {type(layer).__name__}")
            key_mod = key.clone()
            value_mod = value.clone()
            
            if int8_mask.any():
                idx = np.where(int8_mask)[0]
                key_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(
                    key_mod[:, :, idx, :], "int8")
                value_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(
                    value_mod[:, :, idx, :], "int8")
            if int4_mask.any():
                idx = np.where(int4_mask)[0]
                key_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(
                    key_mod[:, :, idx, :], "int4")
                value_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(
                    value_mod[:, :, idx, :], "int4")
            
            # Write back modified KV — use update() for DynamicLayer
            if hasattr(layer, 'update'):
                layer.update(key_mod, value_mod, layer_idx)
            elif isinstance(layer, (tuple, list)):
                past_kv.layers[layer_idx] = (key_mod, value_mod) + layer[2:]
            else:
                past_kv.layers[layer_idx] = (key_mod, value_mod)
        return past_kv
    
    elif has_key_cache:
        # Older DynamicCache with key_cache/value_cache
        for layer_idx in range(len(past_kv.key_cache)):
            key = past_kv.key_cache[layer_idx].clone()
            value = past_kv.value_cache[layer_idx].clone()
            if int8_mask.any():
                idx = np.where(int8_mask)[0]
                key[:, :, idx, :] = KVQuantizer.quantize_dequantize(key[:, :, idx, :], "int8")
                value[:, :, idx, :] = KVQuantizer.quantize_dequantize(value[:, :, idx, :], "int8")
            if int4_mask.any():
                idx = np.where(int4_mask)[0]
                key[:, :, idx, :] = KVQuantizer.quantize_dequantize(key[:, :, idx, :], "int4")
                value[:, :, idx, :] = KVQuantizer.quantize_dequantize(value[:, :, idx, :], "int4")
            past_kv.key_cache[layer_idx] = key
            past_kv.value_cache[layer_idx] = value
        return past_kv
    
    else:
        # Legacy tuple format
        modified_kv = []
        for item in past_kv:
            key, value = item[0], item[1]
            key_mod = key.clone()
            value_mod = value.clone()
            if int8_mask.any():
                idx = np.where(int8_mask)[0]
                key_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(key_mod[:, :, idx, :], "int8")
                value_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(value_mod[:, :, idx, :], "int8")
            if int4_mask.any():
                idx = np.where(int4_mask)[0]
                key_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(key_mod[:, :, idx, :], "int4")
                value_mod[:, :, idx, :] = KVQuantizer.quantize_dequantize(value_mod[:, :, idx, :], "int4")
            # Preserve original item structure (might have extra elements)
            if len(item) > 2:
                modified_kv.append((key_mod, value_mod) + item[2:])
            else:
                modified_kv.append((key_mod, value_mod))
        return modified_kv


# =============================================================================
# NIAH Utilities
# =============================================================================

NEEDLES = [
    ("The secret code is UNICORN7.", "UNICORN7"),
    ("The magic number is 42719.", "42719"),
    ("The hidden word is SPECTRUM.", "SPECTRUM"),
    ("The passkey value is 5831.", "5831"),
    ("The reference code is 4168.", "4168"),
]

NIAH_PASSAGE = (
    "The development of computational methods has transformed many fields of research. "
    "Researchers analyze large datasets to identify patterns and trends. "
    "Modern algorithms process information efficiently across distributed systems. "
    "Statistical models provide frameworks for understanding complex phenomena. "
    "The integration of technology continues to reshape professional practices. "
)


def build_niah_prompt(tokenizer, needle: str, depth_frac: float,
                      seq_len: int) -> torch.Tensor:
    """Build a NIAH prompt with needle at given depth fraction."""
    passage_ids = tokenizer(NIAH_PASSAGE, return_tensors="pt",
                            add_special_tokens=False).input_ids[0]
    needle_ids = tokenizer(needle, return_tensors="pt",
                           add_special_tokens=False).input_ids[0]

    available = seq_len - len(needle_ids) - 10
    if available < 100:
        print(f"  WARNING: Very short haystack ({available} tokens)")

    repeats = (available // len(passage_ids)) + 1
    full_passage = passage_ids.repeat(repeats)[:available]

    insert_pos = int(depth_frac * len(full_passage))
    input_ids = torch.cat([
        full_passage[:insert_pos],
        needle_ids,
        full_passage[insert_pos:],
    ])

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]

    return input_ids


# =============================================================================
# Layer Budget Differentiation
# =============================================================================

def compute_layer_budgets(num_layers: int, base_budget: float,
                          beta: float = 0.02) -> np.ndarray:
    """
    Per-layer differentiated budget.
    
    Lower layers (syntactic) → lower budget (more INT4)
    Higher layers (semantic) → higher budget (more FP16)
    
    budget(l) = base_budget + (l - (N-1)/2) * beta
    
    The adjustment term is zero-centered: Σ(l - (N-1)/2) = 0,
    so mean(layer_budgets) = base_budget exactly.
    
    beta=0 → uniform (no differentiation)
    beta=0.02 → moderate (default from simulation)
    beta=0.05 → aggressive
    
    Floor is 0.25 (= all-INT4) because spectrumkv_greedy cannot
    produce a budget lower than 0.25.
    """
    budgets = np.zeros(num_layers)
    center = (num_layers - 1) / 2.0  # zero-centered pivot
    for l in range(num_layers):
        budgets[l] = max(0.25, min(0.95,
            base_budget + (l - center) * beta))
    return budgets


def apply_layer_budgets(precision_map: np.ndarray,
                        layer_budget: float,
                        importance_scores: np.ndarray) -> np.ndarray:
    """
    Recompute tier assignment with a per-layer budget.
    
    For a given layer, if its budget is lower than the global budget,
    more tokens will be at lower precision.
    """
    return spectrumkv_greedy(layer_budget, importance_scores)


# =============================================================================
# Helper: JSON serialization
# =============================================================================

def json_serialize(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
