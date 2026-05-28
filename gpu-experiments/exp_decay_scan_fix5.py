#!/usr/bin/env python3
"""
Fixed SWS Decay Rate Scan v5 - Final
=====================================
Problem: Qwen2.5 eager attention produces NaN PPL
Solution: Use default (flash) model for PPL, KV-norm for importance scores
Mistral/Gemma: eager attention works -> use attn scores
"""
import json, os, sys, time, gc, traceback, ssl
import torch
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = "/root/autodl-tmp/kvcache-lab/gpu-experiments/experiment_results_new"

MODELS = {
    "qwen7b": "/root/autodl-tmp/Qwen2.5-7B-Instruct",
    "mistral7b": "/root/autodl-tmp/Mistral-7B-Instruct-v0.3",
    "gemma9b": "/root/autodl-tmp/gemma-2-9b-it",
}

DECAY_RATES = [0.001, 0.005, 0.01, 0.02, 0.05]
BUDGETS = [0.3, 0.5, 0.7]
SEQ_LEN = 2048
SINK = 16

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_wikitext2():
    cache = "/root/autodl-tmp/wikitext2_test.txt"
    if os.path.exists(cache):
        with open(cache) as f:
            return f.read()
    import urllib.request
    url = "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    import pyarrow.parquet as pq
    opener = urllib.request.build_opener(urllib.request.HTTPHandler())
    resp = opener.open(url)
    with open("/tmp/wikitext2.parquet", "wb") as f:
        f.write(resp.read())
    text = "\n".join(pq.read_table("/tmp/wikitext2.parquet").column("text").to_pylist())
    with open(cache, "w") as f:
        f.write(text)
    return text

def get_kv_norm_scores(model, input_ids):
    """Use KV cache L2 norm as importance proxy. Works with any attention impl."""
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs.past_key_values
        norms = []
        for layer_kv in past_kv:
            if isinstance(layer_kv, (tuple, list)):
                k, v = layer_kv[0], layer_kv[1]
            elif hasattr(layer_kv, 'key'):
                k, v = layer_kv.key, layer_kv.value
            else:
                k, v = layer_kv[0], layer_kv[1]
            k_norm = k.norm(dim=-1).mean(dim=1).squeeze(0).cpu().numpy()
            v_norm = v.norm(dim=-1).mean(dim=1).squeeze(0).cpu().numpy()
            norms.append(k_norm + v_norm)
        avg_norms = np.mean(norms, axis=0)
        avg_norms = avg_norms / avg_norms.max()
    return avg_norms

def sel_sws_importance_decay(importance_scores, n, budget, sink, decay_rate):
    """Select KV using importance scores * exponential decay."""
    keep = max(sink, int(n * budget))
    decay = np.exp(-np.arange(n - sink) * decay_rate)
    combined = np.zeros(n)
    combined[:sink] = 1e6
    combined[sink:] = importance_scores[sink:] * decay
    ix = sorted(np.argsort(combined)[-keep:].tolist())
    return ix

def compute_ppl(model, input_ids, keep_indices):
    """Compute PPL on kept tokens only."""
    n = input_ids.shape[1]
    keep_set = set(keep_indices)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    loss_positions = [i for i in range(n - 1) if i in keep_set]

    if not loss_positions:
        return float('inf')

    sel_logits = shift_logits[:, loss_positions, :]
    sel_labels = shift_labels[:, loss_positions]

    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(sel_logits.reshape(-1, sel_logits.size(-1)),
                    sel_labels.reshape(-1))
    ppl = torch.exp(loss).item()
    return ppl

def run_decay_scan(model_name, model_path):
    log(f"Loading {model_name} (default attention for PPL)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use default attention (flash/sdpa) for PPL - guaranteed to work
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    text = load_wikitext2()
    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"][:, :SEQ_LEN].to(model.device)
    n = input_ids.shape[1]
    log(f"  Input: {n} tokens")

    # Baseline PPL
    log("  Computing baseline PPL...")
    baseline_ppl = compute_ppl(model, input_ids, list(range(n)))
    log(f"  Baseline PPL = {baseline_ppl:.4f}")

    if np.isnan(baseline_ppl) or np.isinf(baseline_ppl):
        log("  FATAL: Baseline PPL is NaN/Inf with default attention! Skipping model.")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Get KV-norm importance scores (works with any attention impl)
    log("  Extracting KV-norm importance scores...")
    importance_scores = get_kv_norm_scores(model, input_ids)
    log(f"  KV-norm scores: range=[{importance_scores.min():.4f}, {importance_scores.max():.4f}], mean={importance_scores.mean():.4f}")

    # Scan decay rates
    results = []
    for dr in DECAY_RATES:
        for budget in BUDGETS:
            keep = max(SINK, int(n * budget))
            ix = sel_sws_importance_decay(importance_scores, n, budget, SINK, dr)
            ppl = compute_ppl(model, input_ids, ix)
            delta_pct = (ppl - baseline_ppl) / baseline_ppl * 100

            # Overlap with pure positional decay
            decay_only = np.exp(-np.arange(n - SINK) * dr)
            combined_only = np.zeros(n)
            combined_only[:SINK] = 1e6
            combined_only[SINK:] = decay_only
            ix_pure = set(sorted(np.argsort(combined_only)[-keep:].tolist()))
            overlap = len(set(ix) & ix_pure) / keep * 100

            log(f"  decay={dr} b={budget}: PPL={ppl:.2f} ({delta_pct:+.1f}%) overlap={overlap:.0f}%")
            results.append({
                "decay_rate": dr,
                "budget": budget,
                "ppl": round(ppl, 4),
                "delta_pct": round(delta_pct, 2),
                "keep_tokens": keep,
                "sink": SINK,
                "method": "kv_norm_x_decay",
                "overlap_with_pure_decay_pct": round(overlap, 1)
            })

    out = {
        "model": model_name,
        "seq_len": n,
        "baseline_ppl": round(baseline_ppl, 4),
        "sink": SINK,
        "importance_method": "kv_norm",
        "method": "kv_norm_x_exponential_decay",
        "note": "Uses KV-cache L2 norm as importance proxy weighted by positional decay. Different decay rates produce different KV selections.",
        "scan": results
    }

    outpath = os.path.join(OUTPUT_DIR, f"exp_decay_scan_{model_name}.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    log(f"SAVED {outpath}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"UNLOADED {model_name}. Model NOT deleted from disk.")

def main():
    available = []
    for name, path in MODELS.items():
        if os.path.isdir(path):
            available.append(name)
        else:
            log(f"SKIP {name}: no weights at {path}")

    if not available:
        log("No models found! Exiting.")
        return

    log(f"Available models: {available}")

    target = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            target = arg.split("=")[1]

    if target:
        if target in available:
            run_decay_scan(target, MODELS[target])
        else:
            log(f"Model {target} not available")
    else:
        for name in available:
            run_decay_scan(name, MODELS[name])

    log("\nALL DONE - No models deleted from disk.")

if __name__ == "__main__":
    main()
