#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab 补充实验 - P0: Long Context Scaling + Dense α Scan + Pareto Curve

这是论文最关键的补充实验，产出3张核心图的数据：
1. Memory-Quality Pareto Frontier (不同seq_len)
2. Long Context Scaling (4K/8K/16K/32K)
3. Dense α Sensitivity Curve
===============================================================================
"""
import json, warnings, time
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np

warnings.filterwarnings("ignore")
MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"

TEXT = """Artificial intelligence has transformed numerous industries in recent years. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping how we approach complex problems. The development of large language models represents a particularly significant advancement, enabling natural language understanding and generation at unprecedented scale.

In the realm of computer architecture, the shift toward specialized accelerators has been driven by the computational demands of deep learning. Graphics processing units, originally designed for rendering graphics, have become the workhorses of AI training and inference. The emergence of tensor processing units and neural network processors further illustrates this trend toward domain-specific hardware.

The concept of disaggregated computing has gained traction in data center architecture. By separating compute resources into specialized pools such as prefill and decode instances, systems can achieve better resource utilization and lower latency. This approach is particularly relevant for large language model serving, where the computational characteristics of prefill and decode phases differ significantly.

Memory management in distributed systems presents unique challenges. Key-value caches, which store intermediate attention states, must be efficiently transferred between compute nodes while minimizing latency. The design of cache eviction policies and compression strategies directly impacts both throughput and quality of service.

Network topology plays a crucial role in determining the performance of distributed training and inference systems. The bandwidth and latency characteristics of interconnects such as NVLink, PCIe, and Ethernet can become bottlenecks if not carefully managed. Recent advances in optical interconnects promise to alleviate some of these constraints.

The study of attention mechanisms has revealed interesting properties about how language models process information. Self-attention allows each token to attend to all other tokens, creating a rich contextual representation. However, this quadratic complexity in sequence length has motivated research into sparse attention patterns and efficient approximations.

Reinforcement learning from human feedback has emerged as a powerful technique for aligning language models with human preferences. By training reward models on human comparisons and using them to fine-tune policy models, researchers have achieved significant improvements in helpfulness and safety. The interplay between reward hacking and genuine capability improvement remains an active area of investigation.

The philosophy of consciousness remains one of the deepest unsolved problems. David Chalmers hard problem asks why and how physical processes give rise to subjective experience. Functionalist approaches suggest that consciousness arises from information processing patterns rather than specific substrate, while biological naturalism argues that consciousness requires biological implementation.

Quantum computing represents a fundamentally different paradigm for information processing. Unlike classical bits that exist in definite states, quantum bits or qubits can exist in superposition of states, enabling parallel computation of certain problems. Shor's algorithm for factoring and Grover's search algorithm demonstrate potential quantum advantages, though practical quantum computers face significant challenges in error correction and coherence maintenance.

The intersection of neuroscience and artificial intelligence has proven remarkably fruitful. Deep neural networks were inspired by biological neural circuits, and in turn, AI models have become tools for understanding brain function. Convolutional networks mirror the hierarchical processing in visual cortex, while transformer attention mechanisms share computational principles with prefrontal cortex working memory systems. This bidirectional exchange continues to yield insights in both domains.

Sustainable computing has emerged as a critical concern as AI workloads grow exponentially. The energy consumption of training large language models can equal the lifetime carbon emissions of several automobiles. Techniques such as model distillation, pruning, and quantization aim to reduce computational requirements while preserving capability. Data center efficiency improvements through advanced cooling systems and renewable energy sourcing are equally important for responsible AI development."""

def log(msg): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}", flush=True)

def compute_ppl(logits, input_ids):
    sl = logits[:, :-1, :].contiguous()
    lab = input_ids[:, 1:].contiguous()
    return torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))).item()

def make_input(tokenizer, text, target_len, device):
    """Pad/truncate to exact target_len tokens."""
    tokens = tokenizer.encode(text)
    while len(tokens) < target_len + 200:
        tokens = tokens + tokens
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

def make_taa_mask(seq_len, alpha, cost_vector, device='cuda'):
    neg_inf = torch.finfo(torch.float16).min
    c2d = make_causal(seq_len, device)
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        c2d = c2d + b1d.unsqueeze(0).to(torch.float16)
    return c2d.unsqueeze(0).unsqueeze(0)

def make_sws_mask(seq_len, ws, alpha=0.0, cost_vector=None, device='cuda'):
    neg_inf = torch.finfo(torch.float16).min
    m = torch.full((seq_len, seq_len), neg_inf, dtype=torch.float16, device=device)
    for i in range(seq_len):
        s = max(0, i - ws + 1)
        m[i, s:i+1] = 0.0
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        m = m + b1d.unsqueeze(0).to(torch.float16)
    return m.unsqueeze(0).unsqueeze(0)

def make_cost_vector(seq_len, remote_ratio=0.7, device='cuda'):
    cv = torch.zeros(seq_len, device=device)
    cv[:int(seq_len * remote_ratio)] = 1.0
    return cv

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

BYTES_PER_TOKEN = 2 * 4 * 128 * 28 * 2  # Qwen2.5-7B: 57344

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out_dir = Path("/root/autodl-tmp/experiment_results_supplement")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("P0 Supplementary Experiments")
    log("1. Dense α Scan  2. Long Context Scaling  3. Pareto Curves")
    log("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa")
    model.eval()
    log(f"Model loaded: {len(model.model.layers)} layers, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    all_results = {}
    
    # ===================================================================
    # Experiment 1: Dense α Scan (seq=512)
    # ===================================================================
    log("\n" + "=" * 60)
    log("Experiment 1: Dense α Scan (seq=512)")
    log("=" * 60)
    
    SEQ1 = 512
    ids1 = make_input(tokenizer, TEXT, SEQ1, model.device)
    assert ids1.shape[1] == SEQ1
    cv1 = make_cost_vector(SEQ1, 0.7, model.device)
    
    with torch.no_grad():
        bl1 = model(input_ids=ids1).logits
    bp1 = compute_ppl(bl1, ids1)
    log(f"Baseline PPL (seq={SEQ1}): {bp1:.4f}")
    
    alphas = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.5, 0.8, 1.0]
    alpha_results = []
    for a in alphas:
        if a == 0.0:
            ppl = bp1
            t0 = 0
        else:
            mask = make_taa_mask(SEQ1, a, cv1, model.device)
            t0 = time.time()
            try:
                logits = run_hooked(model, ids1, mask)
                ppl = compute_ppl(logits, ids1)
            except Exception as e:
                log(f"  α={a}: ERROR: {e}")
                ppl = float('inf')
            t0 = time.time() - t0
            del mask; torch.cuda.empty_cache()
        delta = (ppl/bp1 - 1)*100 if ppl != float('inf') else None
        r = {"alpha": a, "ppl": round(ppl, 4) if ppl != float('inf') else None,
             "delta_pct": round(delta, 2) if delta is not None else None,
             "time_s": round(t0, 3)}
        alpha_results.append(r)
        log(f"  α={a:<5} PPL={ppl:.4f} ({delta:+.2f}%)" if ppl != float('inf') else f"  α={a}: ERROR")
    
    all_results["exp1_alpha_scan"] = {"baseline_ppl": bp1, "seq_len": SEQ1, "results": alpha_results}
    del ids1; torch.cuda.empty_cache()
    
    # ===================================================================
    # Experiment 2: Long Context Scaling
    # ===================================================================
    log("\n" + "=" * 60)
    log("Experiment 2: Long Context Scaling (Pareto at each length)")
    log("=" * 60)
    
    # For each seq_len, measure PPL at different memory budgets
    seq_lengths = [1024, 2048, 4096]  # 8K+ might OOM, try incrementally
    memory_budgets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    scaling_results = []
    for seq_len in seq_lengths:
        log(f"\n--- seq_len={seq_len} ---")
        ids = make_input(tokenizer, TEXT, seq_len, model.device)
        actual_len = ids.shape[1]
        cv = make_cost_vector(actual_len, 0.7, model.device)
        
        # Baseline
        with torch.no_grad():
            bl = model(input_ids=ids).logits
        bp = compute_ppl(bl, ids)
        log(f"  Baseline PPL: {bp:.4f}")
        
        seq_data = {"seq_len": actual_len, "baseline_ppl": round(bp, 4), "budgets": []}
        
        for budget in memory_budgets:
            ws = max(int(actual_len * budget), 32)
            if ws >= actual_len:
                # budget=1.0 → full KV, use baseline
                seq_data["budgets"].append({
                    "budget_pct": budget, "window_size": ws,
                    "sws_ppl": round(bp, 4), "sws_delta_pct": 0.0,
                    "sws_taa_ppl": round(bp, 4), "sws_taa_delta_pct": 0.0,
                    "kv_mb": round(BYTES_PER_TOKEN * ws / 1e6, 1)
                })
                continue
            
            # SWS only
            mask = make_sws_mask(actual_len, ws, device=model.device)
            try:
                logits = run_hooked(model, ids, mask)
                sws_ppl = compute_ppl(logits, ids)
                sws_d = (sws_ppl/bp - 1)*100
            except:
                sws_ppl, sws_d = float('inf'), float('inf')
            del mask; torch.cuda.empty_cache()
            
            # SWS + TAA
            mask_t = make_sws_mask(actual_len, ws, alpha=0.1, cost_vector=cv, device=model.device)
            try:
                logits_t = run_hooked(model, ids, mask_t)
                taa_ppl = compute_ppl(logits_t, ids)
                taa_d = (taa_ppl/bp - 1)*100
            except:
                taa_ppl, taa_d = float('inf'), float('inf')
            del mask_t; torch.cuda.empty_cache()
            
            r = {
                "budget_pct": budget, "window_size": ws,
                "sws_ppl": round(sws_ppl, 4) if sws_ppl != float('inf') else None,
                "sws_delta_pct": round(sws_d, 2) if sws_d != float('inf') else None,
                "sws_taa_ppl": round(taa_ppl, 4) if taa_ppl != float('inf') else None,
                "sws_taa_delta_pct": round(taa_d, 2) if taa_d != float('inf') else None,
                "kv_mb": round(BYTES_PER_TOKEN * ws / 1e6, 1)
            }
            seq_data["budgets"].append(r)
            sws_s = f"PPL={sws_ppl:.4f}({sws_d:+.2f}%)" if sws_ppl != float('inf') else "ERR"
            taa_s = f"PPL={taa_ppl:.4f}({taa_d:+.2f}%)" if taa_ppl != float('inf') else "ERR"
            log(f"  budget={budget:.0%} ws={ws:<5} SWS:{sws_s} SWS+TAA:{taa_s}")
        
        scaling_results.append(seq_data)
        del ids; torch.cuda.empty_cache()
    
    all_results["exp2_long_context"] = scaling_results
    
    # Try 8K if memory allows
    for seq_len in [8192]:
        log(f"\n--- seq_len={seq_len} (may OOM) ---")
        try:
            ids = make_input(tokenizer, TEXT, seq_len, model.device)
            actual_len = ids.shape[1]
            cv = make_cost_vector(actual_len, 0.7, model.device)
            
            with torch.no_grad():
                bl = model(input_ids=ids).logits
            bp = compute_ppl(bl, ids)
            log(f"  Baseline PPL: {bp:.4f}")
            
            seq_data = {"seq_len": actual_len, "baseline_ppl": round(bp, 4), "budgets": []}
            
            # Only test key budgets for 8K to save time
            for budget in [0.5, 0.7, 1.0]:
                ws = max(int(actual_len * budget), 32)
                if ws >= actual_len:
                    seq_data["budgets"].append({"budget_pct": budget, "window_size": ws,
                        "sws_ppl": round(bp, 4), "sws_delta_pct": 0.0,
                        "sws_taa_ppl": round(bp, 4), "sws_taa_delta_pct": 0.0,
                        "kv_mb": round(BYTES_PER_TOKEN * ws / 1e6, 1)})
                    continue
                mask = make_sws_mask(actual_len, ws, alpha=0.1, cost_vector=cv, device=model.device)
                try:
                    logits = run_hooked(model, ids, mask)
                    ppl = compute_ppl(logits, ids)
                    d = (ppl/bp - 1)*100
                except:
                    ppl, d = float('inf'), float('inf')
                del mask; torch.cuda.empty_cache()
                
                # SWS only
                mask_s = make_sws_mask(actual_len, ws, device=model.device)
                try:
                    logits_s = run_hooked(model, ids, mask_s)
                    sws_ppl = compute_ppl(logits_s, ids)
                    sws_d = (sws_ppl/bp - 1)*100
                except:
                    sws_ppl, sws_d = float('inf'), float('inf')
                del mask_s; torch.cuda.empty_cache()
                
                r = {"budget_pct": budget, "window_size": ws,
                     "sws_ppl": round(sws_ppl, 4) if sws_ppl != float('inf') else None,
                     "sws_delta_pct": round(sws_d, 2) if sws_d != float('inf') else None,
                     "sws_taa_ppl": round(ppl, 4) if ppl != float('inf') else None,
                     "sws_taa_delta_pct": round(d, 2) if d != float('inf') else None,
                     "kv_mb": round(BYTES_PER_TOKEN * ws / 1e6, 1)}
                seq_data["budgets"].append(r)
                log(f"  budget={budget:.0%} ws={ws:<5} SWS:PPL={sws_ppl:.4f} SWS+TAA:PPL={ppl:.4f}")
            
            scaling_results.append(seq_data)
            del ids; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            log(f"  8K OOM, skipping")
            torch.cuda.empty_cache()
    
    all_results["exp2_long_context"] = scaling_results
    
    # ===================================================================
    # Experiment 3: Pareto Curve (512 seq, dense budget points)
    # ===================================================================
    log("\n" + "=" * 60)
    log("Experiment 3: Dense Pareto Curve (seq=512)")
    log("=" * 60)
    
    SEQ3 = 512
    ids3 = make_input(tokenizer, TEXT, SEQ3, model.device)
    cv3 = make_cost_vector(SEQ3, 0.7, model.device)
    
    with torch.no_grad():
        bl3 = model(input_ids=ids3).logits
    bp3 = compute_ppl(bl3, ids3)
    log(f"Baseline PPL: {bp3:.4f}")
    
    dense_budgets = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    pareto_results = {"baseline_ppl": bp3, "seq_len": SEQ3, "curves": {}}
    
    # Curve 1: Pure sliding window (no re-fetch)
    sw_only = []
    for budget in dense_budgets:
        ws = max(int(SEQ3 * budget), 16)
        mask = make_sws_mask(SEQ3, ws, device=model.device)
        try:
            logits = run_hooked(model, ids3, mask)
            ppl = compute_ppl(logits, ids3)
        except: ppl = float('inf')
        sw_only.append({"budget": budget, "ws": ws, "ppl": round(ppl, 4) if ppl != float('inf') else None})
        del mask; torch.cuda.empty_cache()
    pareto_results["curves"]["sliding_window"] = sw_only
    
    # Curve 2: SWS + TAA
    sws_taa = []
    for budget in dense_budgets:
        ws = max(int(SEQ3 * budget), 16)
        mask = make_sws_mask(SEQ3, ws, alpha=0.1, cost_vector=cv3, device=model.device)
        try:
            logits = run_hooked(model, ids3, mask)
            ppl = compute_ppl(logits, ids3)
        except: ppl = float('inf')
        sws_taa.append({"budget": budget, "ws": ws, "ppl": round(ppl, 4) if ppl != float('inf') else None})
        del mask; torch.cuda.empty_cache()
    pareto_results["curves"]["sws_taa"] = sws_taa
    
    # Curve 3: SWS + TAA α=0.2
    sws_taa2 = []
    for budget in dense_budgets:
        ws = max(int(SEQ3 * budget), 16)
        mask = make_sws_mask(SEQ3, ws, alpha=0.2, cost_vector=cv3, device=model.device)
        try:
            logits = run_hooked(model, ids3, mask)
            ppl = compute_ppl(logits, ids3)
        except: ppl = float('inf')
        sws_taa2.append({"budget": budget, "ws": ws, "ppl": round(ppl, 4) if ppl != float('inf') else None})
        del mask; torch.cuda.empty_cache()
    pareto_results["curves"]["sws_taa_0.2"] = sws_taa2
    
    all_results["exp3_pareto"] = pareto_results
    
    # ===================================================================
    # Save
    # ===================================================================
    with open(out_dir / "supplement_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log(f"\nALL SUPPLEMENTARY EXPERIMENTS COMPLETE")
    log(f"Results saved to {out_dir}")
    
    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"\nExp1: Dense α scan ({len(alphas)} points)")
    for r in alpha_results:
        if r['ppl']: log(f"  α={r['alpha']:<5} PPL={r['ppl']:.4f} ({r['delta_pct']:+.2f}%)")
    
    log(f"\nExp3: Pareto curve ({len(dense_budgets)} budget points)")
    for i, budget in enumerate(dense_budgets):
        sw = sw_only[i]
        st = sws_taa[i]
        sw_s = f"PPL={sw['ppl']:.4f}" if sw['ppl'] else "ERR"
        st_s = f"PPL={st['ppl']:.4f}" if st['ppl'] else "ERR"
        log(f"  budget={budget:.0%}: SW={sw_s} SWS+TAA={st_s}")

if __name__ == "__main__":
    main()
