#!/usr/bin/env python3
"""
Exp7: Fine-grained budget scan for smooth PPL curves
=====================================================
Budget ∈ [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Strategies: pdtrim(fr=0.5), sws(sink=16), sws(sink=0)
seq_len=2048, single-seed for speed

Combined with Exp8 latency benchmark.

Output: exp7_budget_scan_{model}.json, exp8_latency_{model}.json
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

def sel(n, strategy, budget, **kw):
    if strategy == "pdtrim":
        fr = kw.get("fr", 0.5)
        first_n = max(1, int(n * budget * fr))
        last_n = max(1, int(n * budget * (1 - fr)))
        return sorted(set(list(range(first_n)) + list(range(n - last_n, n))))
    elif strategy == "sws":
        sink = kw.get("sink", 16)
        keep = max(sink, int(n * budget))
        scores = np.zeros(n)
        scores[:sink] = 1e6
        decay = np.exp(-np.arange(n - sink) * 0.01)
        scores[sink:] = decay
        return sorted(np.argsort(scores)[-keep:].tolist())
    return list(range(n))

def compute_ppl(model, tok, input_ids, selected_ids=None, position_ids=None):
    """Compute perplexity on the selected tokens"""
    with torch.no_grad():
        if selected_ids is not None:
            inp = selected_ids.unsqueeze(0).to(model.device)
            pos = position_ids.unsqueeze(0).to(model.device) if position_ids is not None else None
            out = model(inp, position_ids=pos)
            # Shift for next-token prediction
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

# ===================== EXP7: BUDGET SCAN =====================
def exp7(model, tok, name, text):
    log(f"EXP7 Budget Scan ({name})")
    seq_len = 2048
    
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
    n = len(input_ids)
    
    budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    strategies = [
        ("pdtrim", {"fr": 0.5}),
        ("sws", {"sink": 16}),
        ("sws", {"sink": 0}),
    ]
    
    R = {"model": name, "seq_len": seq_len, "scan": []}
    
    # Baseline
    baseline_ppl = compute_ppl(model, tok, input_ids)
    log(f"  Baseline PPL={baseline_ppl:.4f}")
    R["baseline_ppl"] = baseline_ppl
    
    for strat_name, kwargs in strategies:
        for budget in budgets:
            if budget >= 1.0:
                ppl = baseline_ppl
                delta = 0.0
            else:
                ix = sel(n, strat_name, budget, **kwargs)
                selected_ids = input_ids[ix]
                pos_ids = torch.tensor(ix, dtype=torch.long)
                ppl = compute_ppl(model, tok, input_ids, selected_ids, pos_ids)
                delta = (ppl - baseline_ppl) / baseline_ppl * 100
            
            R["scan"].append({
                "strategy": strat_name, "budget": budget,
                "kwargs": str(kwargs), "ppl": round(ppl, 4),
                "delta_pct": round(delta, 2)
            })
            log(f"  {strat_name} b={budget} {kwargs}: PPL={ppl:.2f} ({delta:+.1f}%)")
            
            torch.cuda.empty_cache()
    
    return R

# ===================== EXP8: LATENCY BENCH =====================
def exp8(model, tok, name, text):
    log(f"EXP8 Latency Benchmark ({name})")
    
    has_chat = hasattr(tok, 'apply_chat_template') and tok.chat_template is not None
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    seq_lens = [2048, 4096]
    if vram_gb >= 45:
        seq_lens += [8192]
    
    strategies = [
        ("full", 1.0, {}),
        ("pdtrim", 0.3, {"fr": 0.5}),
        ("pdtrim", 0.5, {"fr": 0.5}),
        ("pdtrim", 0.7, {"fr": 0.5}),
        ("sws", 0.3, {"sink": 16}),
        ("sws", 0.5, {"sink": 16}),
        ("sws", 0.7, {"sink": 16}),
    ]
    
    R = {"model": name, "vram_gb": round(vram_gb, 1), "results": []}
    
    skip_longer = False
    for seq_len in seq_lens:
        if skip_longer:
            break
        input_ids_full = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
        n = len(input_ids_full)
        
        for strat_name, budget, kwargs in strategies:
            try:
                if strat_name == "full" or budget >= 1.0:
                    inp = input_ids_full.unsqueeze(0).to(model.device)
                    pos = None
                else:
                    ix = sel(n, strat_name, budget, **kwargs)
                    inp = input_ids_full[ix].unsqueeze(0).to(model.device)
                    pos = torch.tensor(ix, dtype=torch.long).unsqueeze(0).to(model.device)
                
                # TTFT (1-token generation)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model.generate(inp, position_ids=pos, max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                ttft = time.perf_counter() - t0
                
                # Full generation (32 tokens)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(inp, position_ids=pos, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
                torch.cuda.synchronize()
                total_time = time.perf_counter() - t0
                
                n_gen = out.shape[1] - inp.shape[1]
                decode_time = total_time - ttft
                tps = n_gen / max(decode_time, 1e-6)
                
                R["results"].append({
                    "seq_len": seq_len, "strategy": strat_name, "budget": budget,
                    "ttft_ms": round(ttft * 1000, 1),
                    "total_ms": round(total_time * 1000, 1),
                    "tps": round(tps, 1),
                    "tokens_generated": n_gen,
                })
                log(f"  seq={seq_len} {strat_name} b={budget}: TTFT={ttft*1000:.0f}ms TPS={tps:.1f}")
                
                del out
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
    log("Exp7 (budget scan) + Exp8 (latency) Combined Pipeline")
    log("="*60)
    
    text = download_wikitext2()
    log(f"WikiText-2: {len(text)} chars")
    
    for name, path in MODELS.items():
        if not has_weights(path):
            log(f"SKIP {name}: no weights")
            continue
        
        # Check if both already done
        exp7_file = os.path.join(OUTPUT_DIR, f"exp7_budget_scan_{name}.json")
        exp8_file = os.path.join(OUTPUT_DIR, f"exp8_latency_{name}.json")
        if os.path.exists(exp7_file) and os.path.exists(exp8_file):
            log(f"SKIP {name}: both exp7+exp8 already exist")
            continue
        
        log(f"\nProcessing: {name}")
        try:
            model, tok = load_model(path)
            
            # Exp7
            if not os.path.exists(exp7_file):
                r7 = exp7(model, tok, name, text)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(exp7_file, "w") as f:
                    json.dump(r7, f, indent=2)
                log(f"SAVED {exp7_file}")
            
            # Exp8
            if not os.path.exists(exp8_file):
                r8 = exp8(model, tok, name, text)
                with open(exp8_file, "w") as f:
                    json.dump(r8, f, indent=2)
                log(f"SAVED {exp8_file}")
            
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            log(f"UNLOADED. VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB. Model NOT deleted.")
        except Exception as e:
            log(f"ERROR for {name}: {e}")
            traceback.print_exc()
    
    log("\nALL DONE - No models deleted from disk.")

if __name__ == "__main__":
    main()
