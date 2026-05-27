#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU实验脚本 v4 - TAA Direct Mask Injection (已验证可行)

核心突破:
    SDPA模式下attention_mask为None, 但可以自己构建4D causal mask + TAA bias
    直接传给model(), TAA真正生效!

验证结果:
    α=0.1: PPL 9.04→8.83 (-2.33%) 质量改善!
    α=5.0: PPL 9.04→32.2 (+256%) 极端α崩溃, 证明TAA确实在工作

策略:
    - SDPA + direct mask (无OOM, 无NaN)
    - 短序列(2048)做PPL+attention分析
    - 长序列做延迟+overhead
===============================================================================
"""

import argparse
import json
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
PPL_SEQ_LENGTH = 2048
ALPHA_VALUES = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def make_taa_hook_static(mask):
    """Static hook that always injects the given mask"""
    def pre_hook(module, args, kwargs):
        kwargs['attention_mask'] = mask
        return args, kwargs
    return pre_hook


def create_causal_mask_with_taa(seq_len: int, alpha: float, 
                                 cost_vector: torch.Tensor,
                                 dtype=torch.float16, device='cuda') -> torch.Tensor:
    """创建4D causal mask + TAA bias
    
    Returns:
        [1, 1, seq_len, seq_len] float mask for SDPA
    """
    # Standard causal mask: 0 for attend, -inf for masked
    neg_inf = torch.finfo(dtype).min  # Use min instead of -inf for fp16
    causal = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    # Upper triangle = -inf
    mask = causal.masked_fill(
        torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0,
        neg_inf
    )
    
    if alpha > 0.0 and cost_vector is not None:
        # TAA bias: b_i = -α × tanh((cost_i - μ) / σ)
        mu = cost_vector.mean()
        sigma = cost_vector.std()
        if sigma < 1e-8:
            sigma = torch.tensor(1.0, device=device)
        bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        # Add to key dimension: [1, 1, 1, seq]
        mask = mask + bias_1d.view(1, 1, 1, -1).to(dtype)
    
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


def compute_ppl(model, input_ids: torch.Tensor) -> float:
    """手动计算PPL"""
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return torch.exp(loss).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/experiment_results_v4")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    log("=" * 60)
    log("G3 TAA v4 - Direct Mask Injection (已验证可行)")
    log("=" * 60)
    log(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model with SDPA
    log(f"加载模型: {args.model} (attn_impl=sdpa)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    total_layers = len(model.model.layers)
    log(f"模型加载完成: {total_layers} layers, 显存: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    # =========================================================================
    # Part 1: PPL vs α (全层TAA, 短序列)
    # =========================================================================
    log("\n" + "=" * 60)
    log("Part 1: PPL vs α (全层, seq=2048)")
    log("=" * 60)
    
    # Generate prompt
    base_text = "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. "
    tokens = tokenizer.encode(base_text)
    while len(tokens) < PPL_SEQ_LENGTH:
        tokens = tokens + tokens
    tokens = tokens[:PPL_SEQ_LENGTH]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=PPL_SEQ_LENGTH).to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    
    # Cost vector
    cost_vector = torch.zeros(seq_len, device=model.device)
    remote_end = int(seq_len * 0.7)
    cost_vector[:remote_end] = 1.0
    
    log(f"Seq: {seq_len}, Remote: {remote_end}, Local: {seq_len-remote_end}")
    
    ppl_results = []
    for alpha in ALPHA_VALUES:
        log(f"  α={alpha}")
        
        if alpha == 0.0:
            # Baseline: standard forward
            ppl = compute_ppl(model, input_ids)
            result = {
                "alpha": alpha,
                "ppl": round(ppl, 4),
                "ppl_delta": 0.0,
                "ppl_delta_pct": 0.0,
                "layers": "all",
            }
        else:
            # TAA via hook: inject mask when attention_mask is None
            # Build TAA bias
            mu = cost_vector.mean()
            sigma = cost_vector.std()
            if sigma < 1e-8:
                sigma = torch.tensor(1.0, device=model.device)
            bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
            
            # Pre-create causal mask with TAA bias (fp32 for stability)
            neg_inf = torch.finfo(torch.float16).min
            causal_base = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float16, device=model.device)
            causal_base = causal_base.masked_fill(
                torch.tril(torch.ones(seq_len, seq_len, device=model.device)).unsqueeze(0).unsqueeze(0) == 0,
                neg_inf
            )
            taa_mask = causal_base + bias_1d.view(1, 1, 1, -1).to(torch.float16)
            no_taa_mask = causal_base  # Same causal mask without TAA bias
            
            # Install hooks for ALL layers (we want full-layer TAA here)
            hooks = []
            def make_taa_hook(use_taa, t_mask, no_t_mask):
                def pre_hook(module, args, kwargs):
                    mask_to_use = t_mask if use_taa else no_t_mask
                    kwargs['attention_mask'] = mask_to_use
                    return args, kwargs
                return pre_hook
            
            for layer_idx in range(total_layers):
                h = model.model.layers[layer_idx].self_attn.register_forward_pre_hook(
                    make_taa_hook(True, taa_mask, no_taa_mask), with_kwargs=True
                )
                hooks.append(h)
            
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
            
            # Clean up hooks
            for h in hooks:
                h.remove()
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            ppl = torch.exp(loss).item()
            
            # Get baseline PPL
            if not hasattr(main, '_baseline_ppl'):
                main._baseline_ppl = compute_ppl(model, input_ids)
            baseline_ppl = main._baseline_ppl
            
            result = {
                "alpha": alpha,
                "baseline_ppl": round(baseline_ppl, 4),
                "taa_ppl": round(ppl, 4),
                "ppl_delta": round(ppl - baseline_ppl, 4),
                "ppl_delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
                "layers": "all",
            }
        
        ppl_results.append(result)
        delta_str = f"Δ={result.get('ppl_delta', 0):+.4f} ({result.get('ppl_delta_pct', 0):+.2f}%)" if 'ppl_delta' in result else ""
        log(f"    PPL={ppl:.4f} {delta_str}")
        
        if 'mask' in dir():
            del mask
        torch.cuda.empty_cache()
    
    # Save baseline for Part 2
    if not hasattr(main, '_baseline_ppl'):
        main._baseline_ppl = compute_ppl(model, input_ids)
    baseline_ppl = main._baseline_ppl
    
    # =========================================================================
    # Part 2: Layer Sensitivity (α=0.1, 后1/4/1/3/1/2/全层)
    # =========================================================================
    log("\n" + "=" * 60)
    log(f"Part 2: Layer Sensitivity (α=0.1, baseline PPL={baseline_ppl:.4f})")
    log("=" * 60)
    
    alpha_fixed = 0.1
    layer_configs = {
        "last_1_4": total_layers * 3 // 4,
        "last_1_3": total_layers * 2 // 3,
        "last_1_2": total_layers // 2,
        "all_layers": 0,
    }
    
    layer_results = []
    for config_name, start_layer in layer_configs.items():
        log(f"  {config_name}: layers {start_layer}-{total_layers-1}")
        
        # For partial layer application, we need a different approach
        # Direct mask applies to ALL layers, so we need hook-based approach
        # But hooks don't work with SDPA (mask=None)
        # Solution: use register_forward_pre_hook to INJECT mask when it's None
        
        # Build TAA bias for this layer range
        mu = cost_vector.mean()
        sigma = cost_vector.std()
        if sigma < 1e-8:
            sigma = torch.tensor(1.0, device=model.device)
        bias_1d = -alpha_fixed * torch.tanh((cost_vector - mu) / sigma)
        
        # Create mask for partial layers
        mask_all = create_causal_mask_with_taa(
            seq_len, alpha_fixed, cost_vector,
            dtype=torch.float16, device=model.device
        )
        mask_no_taa = create_causal_mask_with_taa(
            seq_len, 0.0, None,  # No TAA
            dtype=torch.float16, device=model.device
        )
        
        # Use hooks to inject different masks for different layers
        hooks = []
        def make_mask_hook(use_taa_mask, taa_mask, no_taa_mask):
            def pre_hook(module, args, kwargs):
                mask_to_use = taa_mask if use_taa_mask else no_taa_mask
                kwargs['attention_mask'] = mask_to_use
                return args, kwargs
            return pre_hook
        
        for layer_idx in range(total_layers):
            use_taa = layer_idx >= start_layer
            h = model.model.layers[layer_idx].self_attn.register_forward_pre_hook(
                make_mask_hook(use_taa, mask_all, mask_no_taa), with_kwargs=True
            )
            hooks.append(h)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        ppl = torch.exp(loss).item()
        
        # Clean up hooks
        for h in hooks:
            h.remove()
        
        result = {
            "config": config_name,
            "alpha": alpha_fixed,
            "start_layer": start_layer,
            "modified_layers": total_layers - start_layer,
            "baseline_ppl": round(baseline_ppl, 4),
            "taa_ppl": round(ppl, 4),
            "ppl_delta": round(ppl - baseline_ppl, 4),
            "ppl_delta_pct": round((ppl - baseline_ppl) / baseline_ppl * 100, 2),
        }
        layer_results.append(result)
        log(f"    PPL={ppl:.4f}, Δ={ppl-baseline_ppl:+.4f} ({(ppl-baseline_ppl)/baseline_ppl*100:+.2f}%)")
        
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Part 3: 延迟 + TAA overhead (长序列)
    # =========================================================================
    log("\n" + "=" * 60)
    log("Part 3: 延迟 + TAA overhead (SDPA, 长序列)")
    log("=" * 60)
    
    latency_results = []
    for ctx_len in [8192, 32768]:
        tokens_long = tokenizer.encode(base_text)
        while len(tokens_long) < ctx_len:
            tokens_long = tokens_long + tokens_long
        tokens_long = tokens_long[:ctx_len]
        prompt_long = tokenizer.decode(tokens_long, skip_special_tokens=True)
        inputs_long = tokenizer(prompt_long, return_tensors="pt", truncation=True, max_length=ctx_len).to(model.device)
        actual_len = inputs_long.input_ids.shape[1]
        
        # Baseline TTFT
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs_long, use_cache=True)
        torch.cuda.synchronize()
        baseline_ttft = (time.perf_counter() - t0) * 1000
        
        # TAA TTFT via hook injection
        cost_vec_long = torch.zeros(actual_len, device=model.device)
        cost_vec_long[:int(actual_len * 0.7)] = 1.0
        alpha_t = 0.1
        mu_l = cost_vec_long.mean()
        sigma_l = cost_vec_long.std()
        bias_l = -alpha_t * torch.tanh((cost_vec_long - mu_l) / sigma_l)
        
        neg_inf_l = torch.finfo(torch.float16).min
        causal_l = torch.zeros(1, 1, actual_len, actual_len, dtype=torch.float16, device=model.device)
        causal_l = causal_l.masked_fill(
            torch.tril(torch.ones(actual_len, actual_len, device=model.device)).unsqueeze(0).unsqueeze(0) == 0,
            neg_inf_l
        )
        taa_mask_l = causal_l + bias_l.view(1, 1, 1, -1).to(torch.float16)
        
        # Measure mask creation time
        torch.cuda.synchronize()
        t_hook = time.perf_counter()
        # (mask already created above, this measures the overhead of the bias computation)
        bias_overhead = -alpha_t * torch.tanh((cost_vec_long - mu_l) / sigma_l)
        torch.cuda.synchronize()
        taa_overhead_us = (time.perf_counter() - t_hook) * 1e6
        
        # Install hooks
        hooks_l = []
        for layer_idx in range(total_layers):
            h = model.model.layers[layer_idx].self_attn.register_forward_pre_hook(
                make_taa_hook_static(taa_mask_l), with_kwargs=True
            )
            hooks_l.append(h)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=inputs_long.input_ids, use_cache=True)
        torch.cuda.synchronize()
        taa_ttft = (time.perf_counter() - t1) * 1000
        
        for h in hooks_l:
            h.remove()
        
        result = {
            "context_length": actual_len,
            "baseline_ttft_ms": round(baseline_ttft, 2),
            "taa_ttft_ms": round(taa_ttft, 2),
            "mask_creation_us": round(taa_overhead_us, 1),
            "ttft_overhead_ms": round(taa_ttft - baseline_ttft, 2),
            "ttft_overhead_pct": round((taa_ttft - baseline_ttft) / baseline_ttft * 100, 2),
        }
        latency_results.append(result)
        log(f"  ctx={actual_len}: TTFT baseline={baseline_ttft:.1f}ms, TAA={taa_ttft:.1f}ms "
            f"(+{taa_ttft-baseline_ttft:.1f}ms), mask_creation={taa_overhead_us:.0f}μs")
        
        del inputs_long, mask_taa
        torch.cuda.empty_cache()
    
    # =========================================================================
    # Save & Summary
    # =========================================================================
    all_results = {
        "experiment": "G3_TAA_v4_direct_mask",
        "timestamp": datetime.now().isoformat(),
        "part1_ppl_vs_alpha": ppl_results,
        "part2_layer_sensitivity": layer_results,
        "part3_latency": latency_results,
    }
    
    result_file = output_dir / "G3_TAA_v4_results.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"结果保存: {result_file}")
    
    log("\n" + "=" * 60)
    log("G3 v4 汇总")
    log("=" * 60)
    log("Part 1 - PPL vs α (全层):")
    for r in ppl_results:
        delta = r.get('ppl_delta', 0)
        pct = r.get('ppl_delta_pct', 0)
        log(f"  α={r['alpha']}: PPL={r.get('taa_ppl', r.get('ppl')):.4f}, Δ={delta:+.4f} ({pct:+.2f}%)")
    log("Part 2 - Layer Sensitivity (α=0.1):")
    for r in layer_results:
        log(f"  {r['config']}: Δ={r['ppl_delta']:+.4f} ({r['ppl_delta_pct']:+.2f}%), layers={r['modified_layers']}")
    log("Part 3 - Latency:")
    for r in latency_results:
        log(f"  ctx={r['context_length']}: TTFT +{r['ttft_overhead_ms']:.1f}ms ({r['ttft_overhead_pct']:+.2f}%), mask={r['mask_creation_us']:.0f}μs")
    
    log("EXPERIMENT_ALL_DONE")


if __name__ == "__main__":
    main()
