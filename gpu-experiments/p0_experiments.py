#!/usr/bin/env python3
"""
P0 Experiments: Addressing reviewer weaknesses
===============================================
P0-1: Eviction Strategy Comparison (Random vs LRU vs PDTrim vs SWS)
  - Random: randomly select budget tokens
  - LRU: keep most recent tokens (evict oldest first)
  - PDTrim: keep first + last tokens (first_ratio=0.5)
  - SWS: sliding window + attention sink (sink=16, exponential decay)
  - Budgets: 0.1~0.5, Seq: 2048/4096/8192

P0-2: Long Context PPL Scaling
  - Same strategies at 4K/8K context
  - Shows how strategies diverge at longer sequences

Output: exp_eviction_compare_{model}.json, exp_longctx_ppl_{model}.json
"""
import json, os, sys, time, gc, traceback, ssl
import torch, numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = "/root/autodl-tmp/kvcache-lab/gpu-experiments/experiment_results_new"
WIKITEXT_CACHE = "/root/autodl-tmp/wikitext2_test.txt"

MODELS = {
    "qwen7b": "/root/autodl-tmp/Qwen2.5-7B-Instruct",
    "mistral7b": "/root/autodl-tmp/Mistral-7B-Instruct-v0.3",
    "gemma9b": "/root/autodl-tmp/gemma-2-9b-it",
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def download_wikitext2():
    if os.path.exists(WIKITEXT_CACHE):
        with open(WIKITEXT_CACHE) as f:
            return f.read()
    import urllib.request
    url = "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    try:
        import pyarrow.parquet as pq
        opener = urllib.request.build_opener(urllib.request.HTTPHandler())
        resp = opener.open(url)
        with open("/tmp/wikitext2.parquet", "wb") as f:
            f.write(resp.read())
        text = "\n".join(pq.read_table("/tmp/wikitext2.parquet").column("text").to_pylist())
        with open(WIKITEXT_CACHE, "w") as f:
            f.write(text)
        return text
    except:
        return "The quick brown fox jumps over the lazy dog. " * 5000

def has_weights(path):
    if not os.path.exists(path): return False
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) > 1e9

def load_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    log(f"LOADED. VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok

# ===================== EVICTION STRATEGIES =====================
def sel_random(n, budget, seed=42):
    """Random eviction: randomly select budget tokens to keep"""
    rng = np.random.RandomState(seed)
    keep = max(1, int(n * budget))
    return sorted(rng.choice(n, keep, replace=False).tolist())

def sel_lru(n, budget):
    """LRU eviction: keep most recent tokens, evict oldest first"""
    keep = max(1, int(n * budget))
    return list(range(n - keep, n))

def sel_pdtrim(n, budget, fr=0.5):
    """PDTrim: keep first + last tokens based on first_ratio"""
    first_n = max(1, int(n * budget * fr))
    last_n = max(1, int(n * budget * (1 - fr)))
    return sorted(set(list(range(first_n)) + list(range(n - last_n, n))))

def sel_sws(n, budget, sink=16):
    """SWS: sliding window + attention sink with exponential decay"""
    keep = max(sink, int(n * budget))
    scores = np.zeros(n)
    scores[:sink] = 1e6
    decay = np.exp(-np.arange(n - sink) * 0.01)
    scores[sink:] = decay
    return sorted(np.argsort(scores)[-keep:].tolist())

def compute_ppl(model, tok, input_ids, selected_ids=None, position_ids=None):
    with torch.no_grad():
        if selected_ids is not None:
            inp = selected_ids.unsqueeze(0).to(model.device)
            pos = position_ids.unsqueeze(0).to(model.device) if position_ids is not None else None
            out = model(inp, position_ids=pos)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = selected_ids[1:].unsqueeze(0).to(model.device)
        else:
            inp = input_ids.unsqueeze(0).to(model.device)
            out = model(inp)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[1:].unsqueeze(0).to(model.device)
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    valid = len(losses)
    if valid == 0:
        return float('inf')
    return torch.exp(losses.mean()).item()

# ===================== P0-1: EVICTION COMPARISON =====================
def p0_1_eviction_compare(model, tok, name, text):
    log(f"P0-1: Eviction Strategy Comparison ({name})")
    seq_len = 2048
    
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
    n = len(input_ids)
    
    budgets = [0.1, 0.2, 0.3, 0.4, 0.5]
    strategies = [
        ("random", lambda n, b: sel_random(n, b, seed=42)),
        ("lru", lambda n, b: sel_lru(n, b)),
        ("pdtrim_fr50", lambda n, b: sel_pdtrim(n, b, fr=0.5)),
        ("sws_sink16", lambda n, b: sel_sws(n, b, sink=16)),
    ]
    
    # Multi-seed for random to get variance
    random_seeds = [42, 123, 456, 789, 1024]
    
    R = {"model": name, "seq_len": seq_len, "comparison": []}
    
    # Baseline
    baseline_ppl = compute_ppl(model, tok, input_ids)
    log(f"  Baseline PPL={baseline_ppl:.4f}")
    R["baseline_ppl"] = baseline_ppl
    
    for strat_name, strat_fn in strategies:
        for budget in budgets:
            ix = strat_fn(n, budget)
            selected_ids = input_ids[ix]
            pos_ids = torch.tensor(ix, dtype=torch.long)
            ppl = compute_ppl(model, tok, input_ids, selected_ids, pos_ids)
            delta = (ppl - baseline_ppl) / baseline_ppl * 100
            
            R["comparison"].append({
                "strategy": strat_name, "budget": budget,
                "ppl": round(ppl, 4), "delta_pct": round(delta, 2),
                "n_kept": len(ix), "n_total": n,
            })
            log(f"  {strat_name} b={budget}: PPL={ppl:.2f} ({delta:+.1f}%)")
            torch.cuda.empty_cache()
    
    # Random with multiple seeds for variance
    log(f"  Computing random eviction variance (5 seeds)...")
    for budget in budgets:
        ppls = []
        for seed in random_seeds:
            ix = sel_random(n, budget, seed=seed)
            selected_ids = input_ids[ix]
            pos_ids = torch.tensor(ix, dtype=torch.long)
            ppl = compute_ppl(model, tok, input_ids, selected_ids, pos_ids)
            ppls.append(ppl)
            torch.cuda.empty_cache()
        
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        delta = (mean_ppl - baseline_ppl) / baseline_ppl * 100
        
        R["comparison"].append({
            "strategy": "random_mean5", "budget": budget,
            "ppl": round(mean_ppl, 4), "delta_pct": round(delta, 2),
            "std": round(std_ppl, 4),
            "seeds": random_seeds, "individual_ppls": [round(p, 4) for p in ppls],
        })
        log(f"  random_mean5 b={budget}: PPL={mean_ppl:.2f}±{std_ppl:.2f} ({delta:+.1f}%)")
    
    return R

# ===================== P0-2: LONG CONTEXT PPL =====================
def p0_2_longctx_ppl(model, tok, name, text):
    log(f"P0-2: Long Context PPL Scaling ({name})")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    seq_lens = [2048, 4096]
    if vram_gb >= 45:
        seq_lens += [8192]
    
    budgets = [0.3, 0.5, 0.7]
    strategies = [
        ("random", lambda n, b: sel_random(n, b, seed=42)),
        ("lru", lambda n, b: sel_lru(n, b)),
        ("pdtrim_fr50", lambda n, b: sel_pdtrim(n, b, fr=0.5)),
        ("sws_sink16", lambda n, b: sel_sws(n, b, sink=16)),
    ]
    
    R = {"model": name, "vram_gb": round(vram_gb, 1), "results": []}
    
    skip_longer = False
    for seq_len in seq_lens:
        if skip_longer:
            break
        
        input_ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
        n = len(input_ids)
        
        # Baseline for this seq_len
        try:
            baseline_ppl = compute_ppl(model, tok, input_ids)
            log(f"  seq={seq_len} baseline PPL={baseline_ppl:.4f}")
        except torch.cuda.OutOfMemoryError:
            log(f"  OOM: baseline seq={seq_len}")
            torch.cuda.empty_cache()
            skip_longer = True
            continue
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                skip_longer = True
                continue
            raise
        
        R["results"].append({
            "strategy": "full", "budget": 1.0, "seq_len": seq_len,
            "ppl": round(baseline_ppl, 4), "delta_pct": 0.0,
        })
        
        for strat_name, strat_fn in strategies:
            for budget in budgets:
                try:
                    ix = strat_fn(n, budget)
                    selected_ids = input_ids[ix]
                    pos_ids = torch.tensor(ix, dtype=torch.long)
                    ppl = compute_ppl(model, tok, input_ids, selected_ids, pos_ids)
                    delta = (ppl - baseline_ppl) / baseline_ppl * 100
                    
                    R["results"].append({
                        "strategy": strat_name, "budget": budget, "seq_len": seq_len,
                        "ppl": round(ppl, 4), "delta_pct": round(delta, 2),
                    })
                    log(f"  seq={seq_len} {strat_name} b={budget}: PPL={ppl:.2f} ({delta:+.1f}%)")
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    log(f"  OOM: seq={seq_len} {strat_name} b={budget}")
                    torch.cuda.empty_cache()
                    skip_longer = True
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        skip_longer = True
                        break
                    log(f"  ERR: {e}")
    
    return R

# ===================== MAIN =====================
def main():
    log("="*60)
    log("P0 Experiments: Eviction Comparison + Long Context PPL")
    log("="*60)
    
    text = download_wikitext2()
    log(f"WikiText-2: {len(text)} chars")
    
    for name, path in MODELS.items():
        if not has_weights(path):
            log(f"SKIP {name}: no weights")
            continue
        
        # Check if already done
        f1 = os.path.join(OUTPUT_DIR, f"exp_eviction_compare_{name}.json")
        f2 = os.path.join(OUTPUT_DIR, f"exp_longctx_ppl_{name}.json")
        if os.path.exists(f1) and os.path.exists(f2):
            log(f"SKIP {name}: both P0 experiments already exist")
            continue
        
        log(f"\nProcessing: {name}")
        try:
            model, tok = load_model(path)
            
            # P0-1: Eviction Comparison
            if not os.path.exists(f1):
                r1 = p0_1_eviction_compare(model, tok, name, text)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(f1, "w") as f:
                    json.dump(r1, f, indent=2)
                log(f"SAVED {f1}")
            
            # P0-2: Long Context PPL
            if not os.path.exists(f2):
                r2 = p0_2_longctx_ppl(model, tok, name, text)
                with open(f2, "w") as f:
                    json.dump(r2, f, indent=2)
                log(f"SAVED {f2}")
            
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            log(f"UNLOADED. Model NOT deleted.")
        except Exception as e:
            log(f"ERROR for {name}: {e}")
            traceback.print_exc()
    
    log("\nALL DONE - No models deleted from disk.")

if __name__ == "__main__":
    main()
