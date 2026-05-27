#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU实验脚本 v5f2 - TAA Final Verification (FIX: 6D mask bug)

BUG FIX: create_taa_mask 中 2D causal + 4D bias 广播成4D, 再unsqueeze变6D
FIX: 先在2D上加bias, 最后才unsqueeze到4D

实验内容:
  Part 1: PPL vs α（短序列64 tokens, 全层, 多样化文本）
  Part 2: Layer Sensitivity（短序列, α=0.1/0.2/0.5）
  Part 3: Long Sequence Quality（512/1024/2048 tokens, 多α）
  Part 4: Latency Overhead（8K/16K/32K, α=0.1）
  Part 5: Attention Bias Analysis（解析验证TAA偏移方向）
===============================================================================
"""

import json
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def create_taa_mask(seq_len: int, alpha: float, cost_vector: torch.Tensor,
                    device: str = 'cuda', dtype=torch.float16) -> torch.Tensor:
    """Create 4D causal mask + TAA bias. Returns [1, 1, seq_len, seq_len]."""
    neg_inf = torch.finfo(dtype).min
    # Step 1: Create 2D causal mask
    causal_2d = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    causal_2d = causal_2d.masked_fill(
        torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0,
        neg_inf
    )
    # Step 2: Add TAA bias in 2D space (avoid broadcasting issues)
    if alpha > 0.0 and cost_vector is not None:
        mu = cost_vector.mean()
        sigma = cost_vector.std()
        if sigma < 1e-8:
            sigma = torch.tensor(1.0, device=device)
        bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        # [seq, seq] + [1, seq] → [seq, seq] (broadcast along rows)
        causal_2d = causal_2d + bias_1d.unsqueeze(0).to(dtype)
    # Step 3: Unsqueeze to 4D (always correct now)
    return causal_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


def create_cost_vector(seq_len: int, remote_ratio: float = 0.7, device: str = 'cuda') -> torch.Tensor:
    cost = torch.zeros(seq_len, device=device)
    remote_end = int(seq_len * remote_ratio)
    cost[:remote_end] = 1.0
    return cost


def run_with_taa_hooks(model, input_ids: torch.Tensor, taa_mask: torch.Tensor,
                       target_layers: list = None) -> torch.Tensor:
    total_layers = len(model.model.layers)
    if target_layers is None:
        target_layers = list(range(total_layers))
    
    def make_hook(mask):
        def pre_hook(module, args, kwargs):
            kwargs['attention_mask'] = mask
            return args, kwargs
        return pre_hook
    
    hooks = []
    for idx in target_layers:
        h = model.model.layers[idx].self_attn.register_forward_pre_hook(
            make_hook(taa_mask), with_kwargs=True
        )
        hooks.append(h)
    
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
    finally:
        for h in hooks:
            h.remove()
    return logits


def compute_ppl(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return torch.exp(loss).item()


def prepare_input(tokenizer, text: str, max_len: int, device) -> tuple:
    tokens = tokenizer.encode(text)
    while len(tokens) < max_len:
        tokens = tokens + tokens
    tokens = tokens[:max_len]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    return inputs.input_ids, inputs.input_ids.shape[1]


DIVERSE_TEXT = """Artificial intelligence has transformed numerous industries in recent years. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping how we approach complex problems. The development of large language models represents a particularly significant advancement, enabling natural language understanding and generation at unprecedented scale.

In the realm of computer architecture, the shift toward specialized accelerators has been driven by the computational demands of deep learning. Graphics processing units, originally designed for rendering graphics, have become the workhorses of AI training and inference. The emergence of tensor processing units and neural network processors further illustrates this trend toward domain-specific hardware.

The concept of disaggregated computing has gained traction in data center architecture. By separating compute resources into specialized pools such as prefill and decode instances, systems can achieve better resource utilization and lower latency. This approach is particularly relevant for large language model serving, where the computational characteristics of prefill and decode phases differ significantly.

Memory management in distributed systems presents unique challenges. Key-value caches, which store intermediate attention states, must be efficiently transferred between compute nodes while minimizing latency. The design of cache eviction policies and compression strategies directly impacts both throughput and quality of service.

Network topology plays a crucial role in determining the performance of distributed training and inference systems. The bandwidth and latency characteristics of interconnects such as NVLink, PCIe, and Ethernet can become bottlenecks if not carefully managed. Recent advances in optical interconnects promise to alleviate some of these constraints.

The study of attention mechanisms has revealed interesting properties about how language models process information. Self-attention allows each token to attend to all other tokens, creating a rich contextual representation. However, this quadratic complexity in sequence length has motivated research into sparse attention patterns and efficient approximations.

Reinforcement learning from human feedback has emerged as a powerful technique for aligning language models with human preferences. By training reward models on human comparisons and using them to fine-tune policy models, researchers have achieved significant improvements in helpfulness and safety. The interplay between reward hacking and genuine capability improvement remains an active area of investigation."""


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    output_dir = Path("/root/autodl-tmp/experiment_results_v5f2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("G3 TAA v5f2 - Final Verification (6D mask bug FIXED)")
    log("=" * 60)
    log(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    log(f"Loading model: {DEFAULT_MODEL} (attn_impl=sdpa)")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    total_layers = len(model.model.layers)
    log(f"Model loaded: {total_layers} layers, VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    all_results = {}
    
    # =====================================================================
    # Part 1: PPL vs α (short seq=64, diverse text, all layers)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 1: PPL vs α (seq=64, all layers, diverse text)")
    log("=" * 60)
    
    input_ids, seq_len = prepare_input(tokenizer, DIVERSE_TEXT, 64, model.device)
    cost_vector = create_cost_vector(seq_len, 0.7, model.device)
    log(f"Seq: {seq_len}, Remote: {int(seq_len*0.7)}, Local: {seq_len-int(seq_len*0.7)}")
    
    with torch.no_grad():
        baseline_logits = model(input_ids=input_ids).logits
    baseline_ppl = compute_ppl(baseline_logits, input_ids)
    log(f"Baseline PPL: {baseline_ppl:.4f}")
    
    # Sanity check: verify mask shape
    test_mask = create_taa_mask(seq_len, 0.1, cost_vector, model.device)
    log(f"Mask shape check: {test_mask.shape} (should be [1,1,{seq_len},{seq_len}])")
    assert test_mask.dim() == 4, f"Mask is {test_mask.dim()}D, expected 4D!"
    
    alpha_values = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 5.0]
    part1_results = [{"alpha": 0.0, "ppl": round(baseline_ppl, 4), "ppl_delta_pct": 0.0}]
    
    for alpha in alpha_values[1:]:
        taa_mask = create_taa_mask(seq_len, alpha, cost_vector, model.device)
        assert taa_mask.dim() == 4, f"α={alpha}: mask is {taa_mask.dim()}D!"
        try:
            logits = run_with_taa_hooks(model, input_ids, taa_mask)
            ppl = compute_ppl(logits, input_ids)
            delta_pct = (ppl / baseline_ppl - 1) * 100
            result = {"alpha": alpha, "ppl": round(ppl, 4), "ppl_delta_pct": round(delta_pct, 2)}
            part1_results.append(result)
            log(f"  α={alpha:<5} PPL={ppl:.4f} ({delta_pct:+.2f}%)")
        except Exception as e:
            log(f"  α={alpha:<5} ERROR: {e}")
            part1_results.append({"alpha": alpha, "error": str(e)})
        del taa_mask
        torch.cuda.empty_cache()
    
    all_results["part1_ppl_vs_alpha"] = {"baseline_ppl": baseline_ppl, "seq_len": seq_len, "results": part1_results}
    
    # =====================================================================
    # Part 2: Layer Sensitivity (seq=64, α=0.1/0.2/0.5)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 2: Layer Sensitivity (seq=64)")
    log("=" * 60)
    
    layer_configs = {
        "last_1_4": list(range(total_layers * 3 // 4, total_layers)),
        "last_1_3": list(range(total_layers * 2 // 3, total_layers)),
        "last_1_2": list(range(total_layers // 2, total_layers)),
        "all_layers": list(range(total_layers)),
    }
    
    layer_results = {}
    for test_alpha in [0.1, 0.2, 0.5]:
        log(f"\n  --- α={test_alpha} ---")
        taa_mask = create_taa_mask(seq_len, test_alpha, cost_vector, model.device)
        alpha_results = []
        
        for config_name, layer_indices in layer_configs.items():
            log(f"    {config_name}: layers {layer_indices[0]}-{layer_indices[-1]}")
            try:
                logits = run_with_taa_hooks(model, input_ids, taa_mask, target_layers=layer_indices)
                ppl = compute_ppl(logits, input_ids)
                delta_pct = (ppl / baseline_ppl - 1) * 100
                result = {
                    "config": config_name, "start_layer": layer_indices[0],
                    "end_layer": layer_indices[-1], "num_layers": len(layer_indices),
                    "ppl": round(ppl, 4), "ppl_delta_pct": round(delta_pct, 2),
                }
                alpha_results.append(result)
                log(f"      PPL={ppl:.4f} ({delta_pct:+.2f}%)")
            except Exception as e:
                log(f"      ERROR: {e}")
                alpha_results.append({"config": config_name, "error": str(e)})
            torch.cuda.empty_cache()
        
        layer_results[str(test_alpha)] = alpha_results
        del taa_mask
        torch.cuda.empty_cache()
    
    all_results["part2_layer_sensitivity"] = layer_results
    
    # =====================================================================
    # Part 3: Long Sequence Quality (512/1024/2048)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 3: Long Sequence Quality Preservation")
    log("=" * 60)
    
    long_results = []
    for target_len in [512, 1024, 2048]:
        log(f"\n  Seq target: {target_len}")
        input_ids_l, seq_l = prepare_input(tokenizer, DIVERSE_TEXT, target_len, model.device)
        cost_l = create_cost_vector(seq_l, 0.7, model.device)
        
        with torch.no_grad():
            bl_logits = model(input_ids=input_ids_l).logits
        bl_ppl = compute_ppl(bl_logits, input_ids_l)
        log(f"    Baseline PPL: {bl_ppl:.4f} (seq={seq_l})")
        
        seq_data = {"seq_len": seq_l, "baseline_ppl": round(bl_ppl, 4), "alpha_tests": []}
        
        for alpha in [0.05, 0.1, 0.2, 0.5, 1.0]:
            taa_mask = create_taa_mask(seq_l, alpha, cost_l, model.device)
            try:
                logits = run_with_taa_hooks(model, input_ids_l, taa_mask)
                ppl = compute_ppl(logits, input_ids_l)
                delta_pct = (ppl / bl_ppl - 1) * 100
                seq_data["alpha_tests"].append({
                    "alpha": alpha, "ppl": round(ppl, 4), "ppl_delta_pct": round(delta_pct, 2),
                })
                log(f"    α={alpha:<5} PPL={ppl:.4f} ({delta_pct:+.2f}%)")
            except Exception as e:
                log(f"    α={alpha:<5} ERROR: {e}")
                seq_data["alpha_tests"].append({"alpha": alpha, "error": str(e)})
            del taa_mask
            torch.cuda.empty_cache()
        
        long_results.append(seq_data)
        del input_ids_l, bl_logits
        torch.cuda.empty_cache()
    
    all_results["part3_long_sequence_quality"] = long_results
    
    # =====================================================================
    # Part 4: Latency Overhead
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 4: Latency Overhead (α=0.1)")
    log("=" * 60)
    
    latency_results = []
    overhead_alpha = 0.1
    
    for target_len in [2048, 4096, 8192, 16384, 32768]:
        log(f"\n  Seq target: {target_len}")
        
        input_ids_t, seq_t = prepare_input(tokenizer, DIVERSE_TEXT, target_len, model.device)
        cost_t = create_cost_vector(seq_t, 0.7, model.device)
        
        # Baseline latency
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids_t).logits
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - t0) * 1000
        log(f"    Baseline TTFT: {baseline_ms:.1f} ms")
        
        # TAA latency
        taa_mask = create_taa_mask(seq_t, overhead_alpha, cost_t, model.device)
        _ = run_with_taa_hooks(model, input_ids_t, taa_mask)  # warmup
        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits_t = run_with_taa_hooks(model, input_ids_t, taa_mask)
        torch.cuda.synchronize()
        taa_ms = (time.perf_counter() - t1) * 1000
        
        overhead_ms = taa_ms - baseline_ms
        overhead_pct = (overhead_ms / baseline_ms) * 100
        
        result = {
            "seq_len": seq_t, "baseline_ttft_ms": round(baseline_ms, 1),
            "taa_ttft_ms": round(taa_ms, 1), "overhead_ms": round(overhead_ms, 2),
            "overhead_pct": round(overhead_pct, 3), "alpha": overhead_alpha,
        }
        latency_results.append(result)
        log(f"    TAA TTFT: {taa_ms:.1f} ms, Overhead: {overhead_ms:.2f} ms ({overhead_pct:.3f}%)")
        
        del taa_mask, logits_t, input_ids_t, cost_t
        torch.cuda.empty_cache()
    
    all_results["part4_latency_overhead"] = latency_results
    
    # =====================================================================
    # Part 5: Attention Bias Analysis
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 5: Attention Bias Analysis")
    log("=" * 60)
    
    # Analytical analysis (no need to run model)
    for remote_ratio in [0.5, 0.7, 0.9]:
        log(f"\n  Remote ratio: {remote_ratio}")
        cv = create_cost_vector(1024, remote_ratio)
        for alpha in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
            mu = cv.mean()
            sigma = cv.std()
            bias = -alpha * torch.tanh((cv - mu) / sigma)
            remote_bias = bias[:int(1024*remote_ratio)].mean().item()
            local_bias = bias[int(1024*remote_ratio):].mean().item()
            log(f"    α={alpha:<5} remote_bias={remote_bias:+.4f} local_bias={local_bias:+.4f} diff={local_bias-remote_bias:+.4f}")
    
    all_results["part5_bias_analysis"] = {
        "note": "TAA bias shifts attention toward local (cheap) tokens, away from remote (expensive) tokens",
        "formula": "b_i = -α × tanh((cost_i - μ) / σ), where cost=1.0 for remote, 0.0 for local",
    }
    
    # =====================================================================
    # Save and summarize
    # =====================================================================
    results_path = output_dir / "v5f2_all_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nAll results saved: {results_path}")
    
    # Summary
    log("\n" + "=" * 60)
    log("EXPERIMENT COMPLETE - Summary")
    log("=" * 60)
    
    log(f"\nPart 1: PPL vs α (seq={seq_len}, baseline PPL={baseline_ppl:.4f})")
    for r in part1_results:
        if 'ppl' in r:
            log(f"  α={r['alpha']:<5} PPL={r['ppl']:.4f} ({r.get('ppl_delta_pct', 0):+.2f}%)")
    
    log(f"\nPart 3: Long Sequence Quality")
    for seq_data in long_results:
        log(f"  seq={seq_data['seq_len']}, baseline PPL={seq_data['baseline_ppl']:.4f}")
        for a_test in seq_data['alpha_tests']:
            if 'ppl' in a_test:
                log(f"    α={a_test['alpha']:<5} PPL={a_test['ppl']:.4f} ({a_test['ppl_delta_pct']:+.2f}%)")
    
    log(f"\nPart 4: Latency Overhead (α={overhead_alpha})")
    for r in latency_results:
        log(f"  seq={r['seq_len']:<6} overhead={r['overhead_ms']:.2f}ms ({r['overhead_pct']:.3f}%)")


if __name__ == "__main__":
    main()
