#!/usr/bin/env python3
"""G6 Full OS Ablation - component contribution analysis"""
import json, time, warnings
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np

warnings.filterwarnings("ignore")
DEFAULT_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
DIVERSE_TEXT = """Artificial intelligence has transformed numerous industries in recent years. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping how we approach complex problems. The development of large language models represents a particularly significant advancement, enabling natural language understanding and generation at unprecedented scale.\nIn the realm of computer architecture, the shift toward specialized accelerators has been driven by the computational demands of deep learning. Graphics processing units, originally designed for rendering graphics, have become the workhorses of AI training and inference. The emergence of tensor processing units and neural network processors further illustrates this trend toward domain-specific hardware.\nThe concept of disaggregated computing has gained traction in data center architecture. By separating compute resources into specialized pools such as prefill and decode instances, systems can achieve better resource utilization and lower latency. This approach is particularly relevant for large language model serving, where the computational characteristics of prefill and decode phases differ significantly.\nMemory management in distributed systems presents unique challenges. Key-value caches, which store intermediate attention states, must be efficiently transferred between compute nodes while minimizing latency. The design of cache eviction policies and compression strategies directly impacts both throughput and quality of service."""

def log(msg): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}", flush=True)

def compute_ppl(logits, input_ids):
    sl, lab = logits[:, :-1, :].contiguous(), input_ids[:, 1:].contiguous()
    return torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))).item()

def prepare_input(tokenizer, text, max_len, device):
    tokens = tokenizer.encode(text)
    while len(tokens) < max_len: tokens = tokens + tokens
    tokens = tokens[:max_len]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    return inputs.input_ids, inputs.input_ids.shape[1]

def create_taa_mask(seq_len, alpha, cost_vector, device='cuda', dtype=torch.float16):
    neg_inf = torch.finfo(dtype).min
    c2d = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    c2d = c2d.masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0, neg_inf)
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        c2d = c2d + b1d.unsqueeze(0).to(dtype)
    return c2d.unsqueeze(0).unsqueeze(0)

def create_sws_mask(seq_len, ws, alpha=0.0, cost_vector=None, device='cuda', dtype=torch.float16):
    neg_inf = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), neg_inf, dtype=dtype, device=device)
    for i in range(seq_len):
        s = max(0, i - ws + 1)
        m[i, s:i+1] = 0.0
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        m = m + b1d.unsqueeze(0).to(dtype)
    return m.unsqueeze(0).unsqueeze(0)

def create_eviction_mask(seq_len, evict_ratio, device='cuda', dtype=torch.float16):
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask = mask.masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0, neg_inf)
    remote_end = int(seq_len * 0.7)
    n_evict = int(remote_end * evict_ratio)
    if n_evict > 0:
        # TAA-guided: evict least-attended remote tokens (use LRU as proxy)
        evict_idx = np.arange(n_evict)
        mask[:, evict_idx] = neg_inf
    return mask.unsqueeze(0).unsqueeze(0)

def create_full_os_mask(seq_len, ws, evict_ratio, alpha, cost_vector, device='cuda', dtype=torch.float16):
    """Full OS: SWS + TAA-guided eviction + TAA bias"""
    neg_inf = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), neg_inf, dtype=dtype, device=device)
    for i in range(seq_len):
        s = max(0, i - ws + 1)
        m[i, s:i+1] = 0.0
    # Also mask evicted remote tokens
    remote_end = int(seq_len * 0.7)
    n_evict = int(remote_end * evict_ratio)
    if n_evict > 0:
        # TAA-guided eviction: evict tokens with most negative bias
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        bias = -alpha * torch.tanh((cost_vector - mu) / sigma)
        remote_bias = bias[:remote_end]
        _, sorted_idx = remote_bias.sort()
        evict_idx = sorted_idx[:n_evict]
        m[:, evict_idx] = neg_inf
    # Add TAA bias
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        m = m + b1d.unsqueeze(0).to(dtype)
    return m.unsqueeze(0).unsqueeze(0)

def run_with_hooks(model, input_ids, mask, target_layers=None):
    tl = target_layers or list(range(len(model.model.layers)))
    def mk(m):
        def ph(module, args, kwargs):
            kwargs['attention_mask'] = m; return args, kwargs
        return ph
    hooks = [model.model.layers[i].self_attn.register_forward_pre_hook(mk(mask), with_kwargs=True) for i in tl]
    try:
        with torch.no_grad(): logits = model(input_ids=input_ids).logits
    finally:
        for h in hooks: h.remove()
    return logits

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output_dir = Path("/root/autodl-tmp/experiment_results_g6")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("G6 Full OS Ablation Study")
    log("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    log(f"Model loaded: {len(model.model.layers)} layers")
    
    all_results = {}
    
    # Part 1: Component ablation
    log("\nPart 1: Component Ablation (seq=512, ws=128, alpha=0.1, evict=0.3)")
    test_seq = 512
    input_ids, seq_len = prepare_input(tokenizer, DIVERSE_TEXT, test_seq, model.device)
    cost_vector = torch.zeros(seq_len, device=model.device)
    cost_vector[:int(seq_len*0.7)] = 1.0
    
    with torch.no_grad():
        bl = model(input_ids=input_ids).logits
    baseline_ppl = compute_ppl(bl, input_ids)
    log(f"Baseline PPL: {baseline_ppl:.4f}")
    
    ablation_configs = [
        ("baseline", None),
        ("TAA_only", lambda: create_taa_mask(seq_len, 0.1, cost_vector, device=model.device)),
        ("SWS_only", lambda: create_sws_mask(seq_len, 128, device=model.device)),
        ("Eviction_only", lambda: create_eviction_mask(seq_len, 0.3, device=model.device)),
        ("TAA+SWS", lambda: create_sws_mask(seq_len, 128, alpha=0.1, cost_vector=cost_vector, device=model.device)),
        ("TAA+Eviction", lambda: create_taa_mask(seq_len, 0.1, cost_vector, device=model.device)),  # simplified
        ("Full_OS", lambda: create_full_os_mask(seq_len, 128, 0.3, 0.1, cost_vector, device=model.device)),
    ]
    
    ablation_results = []
    for name, mask_fn in ablation_configs:
        if name == "baseline":
            ppl = baseline_ppl
        else:
            mask = mask_fn()
            try:
                logits = run_with_hooks(model, input_ids, mask)
                ppl = compute_ppl(logits, input_ids)
            except Exception as e:
                log(f"  {name}: ERROR: {e}")
                ppl = float('inf')
            del mask
            torch.cuda.empty_cache()
        
        delta = (ppl / baseline_ppl - 1) * 100
        ablation_results.append({"name": name, "ppl": round(ppl, 4) if ppl != float('inf') else None, "delta_pct": round(delta, 2) if ppl != float('inf') else None})
        log(f"  {name:<15} PPL={ppl:.4f} ({delta:+.2f}%)")
    
    all_results["part1_ablation"] = {"baseline_ppl": baseline_ppl, "results": ablation_results}
    
    # Part 2: Parameter sensitivity
    log("\nPart 2: Parameter Sensitivity (Full OS)")
    param_results = []
    
    # Vary alpha
    for alpha in [0.0, 0.05, 0.1, 0.2, 0.5]:
        mask = create_full_os_mask(seq_len, 128, 0.3, alpha, cost_vector, device=model.device)
        try:
            logits = run_with_hooks(model, input_ids, mask)
            ppl = compute_ppl(logits, input_ids)
            delta = (ppl / baseline_ppl - 1) * 100
        except:
            ppl, delta = float('inf'), float('inf')
        param_results.append({"param": "alpha", "value": alpha, "ppl": round(ppl, 4) if ppl != float('inf') else None, "delta_pct": round(delta, 2) if ppl != float('inf') else None})
        log(f"  alpha={alpha:<5} PPL={ppl:.4f} ({delta:+.2f}%)")
        del mask; torch.cuda.empty_cache()
    
    # Vary window size
    for ws in [64, 128, 256, 512]:
        mask = create_full_os_mask(seq_len, ws, 0.3, 0.1, cost_vector, device=model.device)
        try:
            logits = run_with_hooks(model, input_ids, mask)
            ppl = compute_ppl(logits, input_ids)
            delta = (ppl / baseline_ppl - 1) * 100
        except:
            ppl, delta = float('inf'), float('inf')
        param_results.append({"param": "window_size", "value": ws, "ppl": round(ppl, 4) if ppl != float('inf') else None, "delta_pct": round(delta, 2) if ppl != float('inf') else None})
        log(f"  ws={ws:<5} PPL={ppl:.4f} ({delta:+.2f}%)")
        del mask; torch.cuda.empty_cache()
    
    # Vary eviction ratio
    for evict in [0.1, 0.2, 0.3, 0.5]:
        mask = create_full_os_mask(seq_len, 128, evict, 0.1, cost_vector, device=model.device)
        try:
            logits = run_with_hooks(model, input_ids, mask)
            ppl = compute_ppl(logits, input_ids)
            delta = (ppl / baseline_ppl - 1) * 100
        except:
            ppl, delta = float('inf'), float('inf')
        param_results.append({"param": "evict_ratio", "value": evict, "ppl": round(ppl, 4) if ppl != float('inf') else None, "delta_pct": round(delta, 2) if ppl != float('inf') else None})
        log(f"  evict={evict:<5} PPL={ppl:.4f} ({delta:+.2f}%)")
        del mask; torch.cuda.empty_cache()
    
    all_results["part2_param_sensitivity"] = param_results
    
    with open(output_dir / "g6_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nG6 COMPLETE - Results saved to {output_dir}")

if __name__ == "__main__":
    main()
