#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU实验脚本 v5 - TAA Verification (Fix: diverse text + proper α range)

v4失败根因诊断:
  - Hook注入TAA确实生效 (diag_v5验证: α=5.0 PPL 9.04→32.22 +256%)
  - v4的PPL=1.0459是因为用重复文本，太低无法测量小α效果
  - v4 Part2崩溃: 直接model(input_ids)时4D mask维度不匹配

v5修复:
  1. 使用多样化文本(wiki混合)获取更高PPL基线
  2. α范围扩展: [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]
  3. 全部使用hook注入，不传attention_mask给model()
  4. 短序列(512/1024)做PPL，长序列(8K/16K)做延迟overhead
  5. Layer sensitivity用hook选择性注入
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
ALPHA_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0]
PPL_SEQ_LENGTH = 1024  # Short enough for fast test, long enough for meaningful PPL

# Diverse text for higher baseline PPL (not repeated)
DIVERSE_TEXT = """Artificial intelligence has transformed numerous industries in recent years. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping how we approach complex problems. The development of large language models represents a particularly significant advancement, enabling natural language understanding and generation at unprecedented scale.

In the realm of computer architecture, the shift toward specialized accelerators has been driven by the computational demands of deep learning. Graphics processing units, originally designed for rendering graphics, have become the workhorses of AI training and inference. The emergence of tensor processing units and neural network processors further illustrates this trend toward domain-specific hardware.

The concept of disaggregated computing has gained traction in data center architecture. By separating compute resources into specialized pools—such as prefill and decode instances—systems can achieve better resource utilization and lower latency. This approach is particularly relevant for large language model serving, where the computational characteristics of prefill and decode phases differ significantly.

Memory management in distributed systems presents unique challenges. Key-value caches, which store intermediate attention states, must be efficiently transferred between compute nodes while minimizing latency. The design of cache eviction policies and compression strategies directly impacts both throughput and quality of service.

Network topology plays a crucial role in determining the performance of distributed training and inference systems. The bandwidth and latency characteristics of interconnects such as NVLink, PCIe, and Ethernet can become bottlenecks if not carefully managed. Recent advances in optical interconnects promise to alleviate some of these constraints.

The study of attention mechanisms has revealed interesting properties about how language models process information. Self-attention allows each token to attend to all other tokens, creating a rich contextual representation. However, this quadratic complexity in sequence length has motivated research into sparse attention patterns and efficient approximations.

Reinforcement learning from human feedback has emerged as a powerful technique for aligning language models with human preferences. By training reward models on human comparisons and using them to fine-tune policy models, researchers have achieved significant improvements in helpfulness and safety. The interplay between reward hacking and genuine capability improvement remains an active area of investigation.

The physics of computation establishes fundamental limits on energy efficiency and speed. Landauer's principle tells us that erasing information necessarily dissipates heat, while the Bremermann-Bekenstein bound limits the maximum computational rate of a system of given mass. These theoretical constraints inform the design of practical computing systems.

Quantum computing offers a fundamentally different paradigm for certain computational problems. Shor's algorithm can factor large integers in polynomial time, threatening current cryptographic systems. Grover's algorithm provides quadratic speedup for unstructured search. However, the practical realization of large-scale quantum computers faces significant engineering challenges, including decoherence and error correction overhead.

The sociology of technology adoption reveals patterns that transcend specific innovations. Everett Rogers' diffusion of innovations theory identifies five adopter categories: innovators, early adopters, early majority, late majority, and laggards. Understanding these dynamics helps predict the trajectory of emerging technologies like autonomous vehicles and artificial general intelligence.

Climate modeling requires solving coupled partial differential equations across multiple scales. The atmosphere, oceans, ice sheets, and biosphere interact in complex ways that challenge even the most powerful supercomputers. Machine learning approaches, particularly physics-informed neural networks, are being explored as complementary tools for accelerating climate simulations.

The philosophy of consciousness remains one of the deepest unsolved problems. David Chalmers' hard problem asks why and how physical processes give rise to subjective experience. Functionalist approaches suggest that consciousness arises from information processing patterns rather than specific substrate, while biological naturalism argues that consciousness requires biological implementation. This debate has direct implications for assessing whether artificial systems can be truly conscious.

Evolutionary algorithms draw inspiration from natural selection to solve optimization problems. Genetic algorithms, evolution strategies, and genetic programming each represent different approaches to encoding and evolving solutions. Recent work on quality-diversity algorithms, which seek to discover diverse high-performing solutions rather than a single optimum, has found applications in robotics and game design.

Information theory, founded by Claude Shannon, provides the mathematical framework for quantifying communication. The concept of entropy measures the uncertainty in a random variable, while mutual information captures the dependence between variables. These tools have found applications far beyond their original communication context, including machine learning, neuroscience, and physics.

The development of the internet represents one of the most transformative technological achievements of the twentieth century. From its origins as ARPANET, a military research network, to the global information infrastructure we know today, the internet has fundamentally altered commerce, communication, and culture. The transition from web1 read-only pages to web2 social platforms and now toward decentralized web3 architectures continues to reshape digital interaction."""


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def compute_ppl_from_logits(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    """Compute perplexity from logits and input_ids."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return torch.exp(loss).item()


def create_taa_mask(seq_len: int, alpha: float, cost_vector: torch.Tensor,
                    device: str = 'cuda', dtype=torch.float16) -> torch.Tensor:
    """Create 4D causal mask + TAA bias for hook injection.
    
    Returns: [1, 1, seq_len, seq_len] mask
    """
    neg_inf = torch.finfo(dtype).min
    # Causal mask: lower triangle = 0, upper = -inf
    causal = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    causal = causal.masked_fill(
        torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0,
        neg_inf
    )
    
    if alpha > 0.0 and cost_vector is not None:
        mu = cost_vector.mean()
        sigma = cost_vector.std()
        if sigma < 1e-8:
            sigma = torch.tensor(1.0, device=device)
        bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        causal = causal + bias_1d.view(1, 1, 1, -1).to(dtype)
    
    return causal.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


def create_cost_vector(seq_len: int, remote_ratio: float = 0.7, device: str = 'cuda') -> torch.Tensor:
    """Create cost vector simulating PD separation: remote tokens cost more."""
    cost = torch.zeros(seq_len, device=device)
    remote_end = int(seq_len * remote_ratio)
    cost[:remote_end] = 1.0
    return cost


def run_with_taa_hooks(model, input_ids: torch.Tensor, taa_mask: torch.Tensor,
                       target_layers: list = None) -> torch.Tensor:
    """Run model with TAA mask injected via hooks on specified layers.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        taa_mask: 4D attention mask with TAA bias [1, 1, seq, seq]
        target_layers: List of layer indices to apply TAA. None = all layers.
    
    Returns:
        logits tensor
    """
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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    output_dir = Path("/root/autodl-tmp/experiment_results_v5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("G3 TAA v5 - Diverse Text + Extended α Range")
    log("=" * 60)
    log(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    log(f"Loading model: {DEFAULT_MODEL} (attn_impl=sdpa)")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    total_layers = len(model.model.layers)
    log(f"Model loaded: {total_layers} layers, VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    # =====================================================================
    # Prepare diverse text input
    # =====================================================================
    log("\n" + "=" * 60)
    log("Preparing diverse text input")
    log("=" * 60)
    
    tokens = tokenizer.encode(DIVERSE_TEXT)
    log(f"Diverse text tokens: {len(tokens)}")
    
    # Extend if needed
    while len(tokens) < PPL_SEQ_LENGTH:
        tokens = tokens + tokens
    tokens = tokens[:PPL_SEQ_LENGTH]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=PPL_SEQ_LENGTH).to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    cost_vector = create_cost_vector(seq_len, remote_ratio=0.7, device=model.device)
    
    log(f"Seq length: {seq_len}, Remote: {int(seq_len*0.7)}, Local: {seq_len-int(seq_len*0.7)}")
    
    # =====================================================================
    # Part 1: PPL vs α (all layers, diverse text)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 1: PPL vs α (all layers, diverse text)")
    log("=" * 60)
    
    # Baseline (no hooks)
    with torch.no_grad():
        baseline_logits = model(input_ids=input_ids).logits
    baseline_ppl = compute_ppl_from_logits(baseline_logits, input_ids)
    log(f"Baseline PPL: {baseline_ppl:.4f}")
    
    ppl_results = [{"alpha": 0.0, "ppl": round(baseline_ppl, 4), "ppl_delta": 0.0, "ppl_delta_pct": 0.0}]
    
    for alpha in ALPHA_VALUES[1:]:  # Skip 0.0
        log(f"  α={alpha}")
        taa_mask = create_taa_mask(seq_len, alpha, cost_vector, device=model.device)
        logits = run_with_taa_hooks(model, input_ids, taa_mask)
        ppl = compute_ppl_from_logits(logits, input_ids)
        delta = ppl - baseline_ppl
        delta_pct = (delta / baseline_ppl) * 100
        result = {
            "alpha": alpha,
            "ppl": round(ppl, 4),
            "baseline_ppl": round(baseline_ppl, 4),
            "ppl_delta": round(delta, 4),
            "ppl_delta_pct": round(delta_pct, 2),
        }
        ppl_results.append(result)
        log(f"    PPL={ppl:.4f} Δ={delta:+.4f} ({delta_pct:+.2f}%)")
        del taa_mask, logits
        torch.cuda.empty_cache()
    
    # Save Part 1
    part1_path = output_dir / "part1_ppl_vs_alpha.json"
    with open(part1_path, 'w') as f:
        json.dump({"baseline_ppl": baseline_ppl, "results": ppl_results}, f, indent=2)
    log(f"Part 1 saved: {part1_path}")
    
    # =====================================================================
    # Part 2: Layer Sensitivity (α=0.1 and α=0.2)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 2: Layer Sensitivity")
    log("=" * 60)
    
    layer_configs = {
        "last_1_4": list(range(total_layers * 3 // 4, total_layers)),
        "last_1_3": list(range(total_layers * 2 // 3, total_layers)),
        "last_1_2": list(range(total_layers // 2, total_layers)),
        "all_layers": list(range(total_layers)),
    }
    
    layer_results = {}
    for test_alpha in [0.1, 0.2]:
        log(f"\n  --- α={test_alpha} ---")
        taa_mask = create_taa_mask(seq_len, test_alpha, cost_vector, device=model.device)
        layer_results[str(test_alpha)] = []
        
        for config_name, layer_indices in layer_configs.items():
            log(f"    {config_name}: layers {layer_indices[0]}-{layer_indices[-1]} ({len(layer_indices)} layers)")
            try:
                logits = run_with_taa_hooks(model, input_ids, taa_mask, target_layers=layer_indices)
                ppl = compute_ppl_from_logits(logits, input_ids)
                delta = ppl - baseline_ppl
                delta_pct = (delta / baseline_ppl) * 100
                result = {
                    "config": config_name,
                    "layers": layer_indices,
                    "ppl": round(ppl, 4),
                    "ppl_delta": round(delta, 4),
                    "ppl_delta_pct": round(delta_pct, 2),
                }
                layer_results[str(test_alpha)].append(result)
                log(f"      PPL={ppl:.4f} Δ={delta:+.4f} ({delta_pct:+.2f}%)")
                del logits
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"      ERROR: {e}")
                traceback.print_exc()
        
        del taa_mask
        torch.cuda.empty_cache()
    
    # Save Part 2
    part2_path = output_dir / "part2_layer_sensitivity.json"
    with open(part2_path, 'w') as f:
        json.dump(layer_results, f, indent=2)
    log(f"Part 2 saved: {part2_path}")
    
    # =====================================================================
    # Part 3: Latency Overhead (longer sequences)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 3: Latency Overhead")
    log("=" * 60)
    
    # Use diverse text extended to longer sequences
    latency_results = []
    test_lengths = [2048, 4096, 8192, 16384]
    overhead_alpha = 0.1
    
    for target_len in test_lengths:
        log(f"\n  Seq length: {target_len}")
        
        # Build input
        tokens = tokenizer.encode(DIVERSE_TEXT)
        while len(tokens) < target_len:
            tokens = tokens + tokens
        tokens = tokens[:target_len]
        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_len).to(model.device)
        long_ids = inputs.input_ids
        long_seq = long_ids.shape[1]
        long_cost = create_cost_vector(long_seq, remote_ratio=0.7, device=model.device)
        
        # Baseline latency
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=long_ids).logits
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        baseline_ms = (t1 - t0) * 1000
        log(f"    Baseline TTFT: {baseline_ms:.1f} ms")
        
        # TAA overhead: just measure hook injection cost
        taa_mask = create_taa_mask(long_seq, overhead_alpha, long_cost, device=model.device)
        
        # Warmup
        _ = run_with_taa_hooks(model, long_ids, taa_mask)
        torch.cuda.synchronize()
        
        # Measure TAA latency
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        logits_taa = run_with_taa_hooks(model, long_ids, taa_mask)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        taa_ms = (t3 - t2) * 1000
        
        overhead_ms = taa_ms - baseline_ms
        overhead_pct = (overhead_ms / baseline_ms) * 100
        
        result = {
            "seq_len": long_seq,
            "baseline_ttft_ms": round(baseline_ms, 1),
            "taa_ttft_ms": round(taa_ms, 1),
            "overhead_ms": round(overhead_ms, 2),
            "overhead_pct": round(overhead_pct, 3),
            "alpha": overhead_alpha,
        }
        latency_results.append(result)
        log(f"    TAA TTFT: {taa_ms:.1f} ms, Overhead: {overhead_ms:.2f} ms ({overhead_pct:.3f}%)")
        
        del taa_mask, logits_taa, long_ids, long_cost
        torch.cuda.empty_cache()
    
    # Save Part 3
    part3_path = output_dir / "part3_latency_overhead.json"
    with open(part3_path, 'w') as f:
        json.dump(latency_results, f, indent=2)
    log(f"Part 3 saved: {part3_path}")
    
    # =====================================================================
    # Part 4: Cost Vector Sensitivity (vary remote_ratio)
    # =====================================================================
    log("\n" + "=" * 60)
    log("Part 4: Cost Vector Sensitivity (α=0.1)")
    log("=" * 60)
    
    remote_ratios = [0.3, 0.5, 0.7, 0.9]
    cost_results = []
    test_alpha = 0.1
    
    # Reuse PPL input
    tokens = tokenizer.encode(DIVERSE_TEXT)
    while len(tokens) < PPL_SEQ_LENGTH:
        tokens = tokens + tokens
    tokens = tokens[:PPL_SEQ_LENGTH]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=PPL_SEQ_LENGTH).to(model.device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    
    for ratio in remote_ratios:
        log(f"  remote_ratio={ratio}")
        cost_v = create_cost_vector(seq_len, remote_ratio=ratio, device=model.device)
        taa_mask = create_taa_mask(seq_len, test_alpha, cost_v, device=model.device)
        logits = run_with_taa_hooks(model, input_ids, taa_mask)
        ppl = compute_ppl_from_logits(logits, input_ids)
        delta = ppl - baseline_ppl
        delta_pct = (delta / baseline_ppl) * 100
        result = {
            "remote_ratio": ratio,
            "ppl": round(ppl, 4),
            "ppl_delta": round(delta, 4),
            "ppl_delta_pct": round(delta_pct, 2),
        }
        cost_results.append(result)
        log(f"    PPL={ppl:.4f} Δ={delta:+.4f} ({delta_pct:+.2f}%)")
        del taa_mask, logits, cost_v
        torch.cuda.empty_cache()
    
    # Save Part 4
    part4_path = output_dir / "part4_cost_sensitivity.json"
    with open(part4_path, 'w') as f:
        json.dump(cost_results, f, indent=2)
    log(f"Part 4 saved: {part4_path}")
    
    # =====================================================================
    # Summary
    # =====================================================================
    log("\n" + "=" * 60)
    log("EXPERIMENT COMPLETE - Summary")
    log("=" * 60)
    log(f"Part 1: PPL vs α (baseline PPL={baseline_ppl:.4f})")
    for r in ppl_results:
        alpha = r['alpha']
        ppl = r['ppl']
        delta_pct = r.get('ppl_delta_pct', 0)
        log(f"  α={alpha:<5} PPL={ppl:.4f} ({delta_pct:+.2f}%)")
    
    log(f"\nPart 3: Latency Overhead (α={overhead_alpha})")
    for r in latency_results:
        log(f"  seq={r['seq_len']:<6} overhead={r['overhead_ms']:.2f}ms ({r['overhead_pct']:.3f}%)")
    
    log("\nAll results saved to: " + str(output_dir))


if __name__ == "__main__":
    main()
