#!/usr/bin/env python3
"""
NIAH Depth Scan: Qwen-7B at 4K and 8K
======================================
补齐Qwen模型的中长上下文数据。
"""
import json, os, sys, time, gc, traceback, ssl, string
import torch, numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = "/root/autodl-tmp/kvcache-lab/gpu-experiments/experiment_results_new"

MODELS = {
    "qwen7b": {"path": "/root/autodl-tmp/Qwen2.5-7B-Instruct", "seq_lens": [4096, 8192]},
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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

def build_haystack_with_needle(tok, target_tokens, needle_text, insert_fraction, max_haystack):
    passage = ("The development of computational methods has transformed many fields of research. "
               "Researchers analyze large datasets to identify patterns and trends. "
               "Modern algorithms process information efficiently across distributed systems. "
               "Statistical models provide frameworks for understanding complex phenomena. "
               "The integration of technology continues to reshape professional practices. ")
    passage_ids = tok(passage, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needle_ids = tok(needle_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    available = max_haystack - len(needle_ids) - 10
    if available < 100:
        log(f"  WARNING: Very short haystack ({available} tokens)")
    repeats = (available // len(passage_ids)) + 1
    full_passage = passage_ids.repeat(repeats)[:available]
    insert_pos = int(len(full_passage) * insert_fraction)
    haystack_with_needle = torch.cat([
        full_passage[:insert_pos],
        needle_ids,
        full_passage[insert_pos:]
    ])
    haystack_with_needle = haystack_with_needle[:max_haystack]
    return haystack_with_needle, insert_pos

def check_niah_retrieval(model, tok, input_ids, needle_text, strategy, budget, 
                         insert_fraction, model_name, **kwargs):
    n = len(input_ids)
    has_chat = hasattr(tok, 'apply_chat_template') and tok.chat_template is not None
    
    if strategy == "full" or budget >= 1.0:
        selected_ids = input_ids
        position_ids = None
        coverage = 1.0
    else:
        ix = sel(n, strategy, budget, **kwargs)
        selected_ids = input_ids[ix]
        position_ids = torch.tensor(ix, dtype=torch.long)
        insert_pos = int(n * insert_fraction)
        needle_len_est = max(5, int(n * 0.02))
        needle_range = set(range(max(0, insert_pos - 2), min(n, insert_pos + needle_len_est + 2)))
        kept_set = set(ix)
        overlap = needle_range & kept_set
        coverage = len(overlap) / max(len(needle_range), 1)
    
    needle_value = needle_text.strip()
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided text."},
        {"role": "user", "content": f"Read the following text carefully and answer the question.\n\n{tok.decode(selected_ids)}\n\nWhat is the confirmation number mentioned in the text? Reply with only the number."}
    ]
    
    if has_chat:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        max_input_len = min(len(input_ids) + 200, 16384)
        input_tensor = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_len).input_ids.to(model.device)
    else:
        input_tensor = selected_ids.unsqueeze(0).to(model.device)
    
    try:
        with torch.no_grad():
            out = model.generate(
                input_tensor, 
                max_new_tokens=30,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id if tok.eos_token_id else 0,
            )
        generated = out[0][input_tensor.shape[1]:]
        response = tok.decode(generated, skip_special_tokens=True).strip()
        
        needle_words = needle_value.lower().split()
        found = False
        for word in needle_words:
            cleaned = word.rstrip(string.punctuation)
            if len(cleaned) >= 4 and cleaned.isalnum() and not cleaned.startswith("the"):
                if cleaned in response.lower():
                    found = True
                    break
        needle_stripped = needle_value.lower().rstrip(string.punctuation)
        if needle_stripped in response.lower():
            found = True
        return found, response[:100], coverage
    except Exception as e:
        return False, f"ERROR: {e}", coverage

def main():
    log("="*60)
    log("NIAH Depth Scan: Qwen-7B 4K + 8K")
    log("="*60)
    
    needles = [
        "The confirmation number is 7294.",
        "The passcode value is 5831.",
        "The reference code is 4168.",
    ]
    depths = [0.1, 0.25, 0.5, 0.75, 0.9]
    strategies_configs = [
        ("full", 1.0, {}),
        ("pdtrim", 0.9, {"fr": 0.5}),
        ("pdtrim", 0.7, {"fr": 0.5}),
        ("pdtrim", 0.5, {"fr": 0.5}),
        ("sws", 0.9, {"sink": 16}),
        ("sws", 0.7, {"sink": 16}),
        ("sws", 0.5, {"sink": 16}),
    ]
    
    for name, cfg in MODELS.items():
        path = cfg["path"]
        if not has_weights(path):
            log(f"SKIP {name}: no weights at {path}")
            continue
        
        # Load model once, run both seq_lens
        log(f"\nLoading: {name}")
        try:
            model, tok = load_model(path)
            
            for seq_len in cfg["seq_lens"]:
                out_file = os.path.join(OUTPUT_DIR, f"niah_depth_scan_{seq_len}_{name}.json")
                if os.path.exists(out_file):
                    log(f"SKIP {name} seq_len={seq_len}: output already exists")
                    continue
                
                log(f"\nRunning: {name} (seq_len={seq_len})")
                R = {"model": name, "seq_len": seq_len, "context": "extended", "results": []}
                
                for ni, needle in enumerate(needles):
                    for depth in depths:
                        input_ids, insert_pos = build_haystack_with_needle(
                            tok, seq_len, needle, depth, seq_len
                        )
                        log(f"  [{ni+1}/{len(needles)}] Needle at {depth*100:.0f}% (pos={insert_pos}), n_tokens={len(input_ids)}")
                        
                        for strat, budget, kwargs in strategies_configs:
                            found, response, coverage = check_niah_retrieval(
                                model, tok, input_ids, needle, strat, budget,
                                depth, name, **kwargs
                            )
                            R["results"].append({
                                "needle": needle, "depth": depth, "insert_pos": insert_pos,
                                "n_tokens": len(input_ids), "strategy": strat, "budget": budget,
                                "kwargs": str(kwargs), "retrieved": found, "response": response,
                                "needle_coverage": round(coverage, 2),
                            })
                            status = "Y" if found else "N"
                            log(f"    {strat} b={budget}: {status} cov={coverage:.2f} resp='{response[:40]}'")
                            torch.cuda.empty_cache()
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(out_file, "w") as f:
                    json.dump(R, f, indent=2)
                log(f"SAVED {out_file}")
            
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            log(f"UNLOADED. Model NOT deleted from disk.")
        except Exception as e:
            log(f"ERROR for {name}: {e}")
            traceback.print_exc()
    
    log("\nALL DONE - No models deleted from disk.")

if __name__ == "__main__":
    main()
