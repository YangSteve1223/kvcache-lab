#!/usr/bin/env python3
"""G5 Predictive Eviction - Fixed version"""
import json, warnings
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np

warnings.filterwarnings("ignore")
MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
TEXT = """Artificial intelligence has transformed numerous industries in recent years. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping how we approach complex problems. The development of large language models represents a particularly significant advancement, enabling natural language understanding and generation at unprecedented scale.\nIn the realm of computer architecture, the shift toward specialized accelerators has been driven by the computational demands of deep learning. Graphics processing units, originally designed for rendering graphics, have become the workhorses of AI training and inference. The emergence of tensor processing units and neural network processors further illustrates this trend toward domain-specific hardware.\nThe concept of disaggregated computing has gained traction in data center architecture. By separating compute resources into specialized pools such as prefill and decode instances, systems can achieve better resource utilization and lower latency. This approach is particularly relevant for large language model serving, where the computational characteristics of prefill and decode phases differ significantly.\nMemory management in distributed systems presents unique challenges. Key-value caches, which store intermediate attention states, must be efficiently transferred between compute nodes while minimizing latency. The design of cache eviction policies and compression strategies directly impacts both throughput and quality of service."""

def log(msg): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}", flush=True)

def compute_ppl(logits, input_ids):
    sl = logits[:, :-1, :].contiguous(); lab = input_ids[:, 1:].contiguous()
    return torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))).item()

def make_input(tokenizer, text, target_len, device):
    """Always return exact target_len tokens by padding."""
    tokens = tokenizer.encode(text)
    while len(tokens) < target_len + 100: tokens = tokens + tokens
    prompt = tokenizer.decode(tokens[:target_len], skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_len).to(device)
    ids = inputs.input_ids
    if ids.shape[1] < target_len:
        pad = torch.full((1, target_len - ids.shape[1]), tokenizer.eos_token_id or 0, dtype=ids.dtype, device=device)
        ids = torch.cat([ids, pad], dim=1)
    return ids

def make_causal(seq_len, device, dtype=torch.float16):
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    return m.masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0, neg_inf)

def make_eviction_mask(seq_len, evict_ratio, strategy, cost_vector=None, alpha=0.1, device='cuda'):
    """Create mask with evicted keys masked out. Always use exact seq_len."""
    neg_inf = torch.finfo(torch.float16).min
    mask = make_causal(seq_len, device)
    remote_end = int(seq_len * 0.7)
    n_evict = int(remote_end * evict_ratio)
    if n_evict > 0:
        if strategy == "random":
            evict_idx = np.random.choice(remote_end, n_evict, replace=False)
        elif strategy == "lru":
            evict_idx = np.arange(n_evict)
        elif strategy == "taa_guided":
            mu, sigma = cost_vector.mean(), cost_vector.std()
            if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
            bias = -alpha * torch.tanh((cost_vector - mu) / sigma)
            remote_bias = bias[:remote_end]
            _, sorted_idx = remote_bias.sort()
            evict_idx = sorted_idx[:n_evict].cpu().numpy()
        else:
            evict_idx = np.arange(n_evict)
        mask[:, evict_idx] = neg_inf
        if alpha > 0.0 and cost_vector is not None and strategy == "taa_guided":
            b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
            mask = mask + b1d.unsqueeze(0).to(torch.float16)
    return mask.unsqueeze(0).unsqueeze(0)

def run_hooked(model, input_ids, mask):
    def mk(m):
        def ph(_, args, kwargs):
            kwargs['attention_mask'] = m; return args, kwargs
        return ph
    hooks = [l.self_attn.register_forward_pre_hook(mk(mask), with_kwargs=True) for l in model.model.layers]
    try:
        with torch.no_grad(): return model(input_ids=input_ids).logits
    finally:
        for h in hooks: h.remove()

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path("/root/autodl-tmp/experiment_results_g5")
    out.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60); log("G5 Predictive Eviction (Fixed)"); log("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True, attn_implementation="sdpa")
    model.eval()
    log(f"Model loaded: {len(model.model.layers)} layers")
    
    SEQ = 512
    input_ids = make_input(tokenizer, TEXT, SEQ, model.device)
    assert input_ids.shape[1] == SEQ, f"Expected {SEQ}, got {input_ids.shape[1]}"
    cost_vector = torch.zeros(SEQ, device=model.device)
    cost_vector[:int(SEQ*0.7)] = 1.0
    
    with torch.no_grad(): bl = model(input_ids=input_ids).logits
    bp = compute_ppl(bl, input_ids)
    log(f"Baseline PPL (seq={SEQ}): {bp:.4f}")
    
    results = []
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        row = {"evict_ratio": ratio}
        for strat in ["random", "lru", "taa_guided"]:
            np.random.seed(42)
            mask = make_eviction_mask(SEQ, ratio, strat, cost_vector, 0.1, model.device)
            try:
                logits = run_hooked(model, input_ids, mask)
                ppl = compute_ppl(logits, input_ids)
                delta = (ppl/bp - 1)*100
                row[f"{strat}_ppl"] = round(ppl, 4)
                row[f"{strat}_delta"] = round(delta, 2)
                log(f"  evict={ratio:.0%} {strat:<12} PPL={ppl:.4f} ({delta:+.2f}%)")
            except Exception as e:
                log(f"  evict={ratio:.0%} {strat:<12} ERROR: {e}")
                row[f"{strat}_ppl"] = None; row[f"{strat}_delta"] = None
            del mask; torch.cuda.empty_cache()
        results.append(row)
    
    all_r = {"baseline_ppl": bp, "seq_len": SEQ, "results": results}
    with open(out / "g5_all_results.json", 'w') as f: json.dump(all_r, f, indent=2)
    log(f"\nG5 COMPLETE - saved to {out}")
    log("Summary:")
    for r in results:
        log(f"  evict={r['evict_ratio']:.0%}: random={r.get('random_ppl','ERR')} LRU={r.get('lru_ppl','ERR')} TAA={r.get('taa_guided_ppl','ERR')}")

if __name__ == "__main__":
    main()
