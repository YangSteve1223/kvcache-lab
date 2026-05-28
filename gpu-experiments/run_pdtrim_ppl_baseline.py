#!/usr/bin/env python3
"""
PDTrim-style PPL Baseline: "first_k + last_w" token retention strategy.

This script compares PDTrim's heuristic (first+last token sequences) with
our sink-aware SWS strategy on the same models and budgets.

Usage: python run_pdtrim_ppl_baseline.py --model <model_name> --seq_len 2048 --budgets 0.3 0.5

Requires GPU with sufficient VRAM for model loading.
"""

import argparse
import json
import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral7b": "Mistral/Mistral-7B-Instruct-v0.3",
    "gemma9b": "google/gemma-2-9b-it",
}

def load_model(model_name, hf_path):
    print(f"Loading {model_name} from {hf_path}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def get_wikitext2_data(tokenizer, seq_len=2048, num_samples=5):
    """Load WikiText-2 data, fallback to manual download if datasets not available."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        import urllib.request
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        zip_path = "/tmp/wikitext-2-raw.zip"
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open('wikitext-2-raw/wiki.test.raw') as f:
                text = f.read().decode('utf-8')
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    samples = []
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        if end <= len(input_ids):
            samples.append(input_ids[start:end])
    return samples

def compute_ppl_baseline(model, input_ids):
    """Full KV baseline PPL."""
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0).to(model.device), labels=input_ids.unsqueeze(0).to(model.device))
        return torch.exp(outputs.loss).item()

def compute_ppl_pdtrim(model, input_ids, budget, first_ratio=0.5):
    """
    PDTrim strategy: keep first_k + last_w tokens.
    first_ratio: fraction of budget allocated to first tokens (default 0.5 = equal split)
    """
    n = len(input_ids)
    budget_tokens = int(n * budget)
    first_k = int(budget_tokens * first_ratio)
    last_w = budget_tokens - first_k
    
    # Select positions: first_k + last_w
    first_indices = list(range(first_k))
    last_indices = list(range(n - last_w, n))
    selected_indices = sorted(set(first_indices + last_indices))
    
    # Build selected input with original position_ids
    selected_ids = input_ids[selected_indices].to(model.device)
    position_ids = torch.tensor(selected_indices, dtype=torch.long, device=model.device)
    
    with torch.no_grad():
        outputs = model(
            selected_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            labels=selected_ids.unsqueeze(0),
        )
        return torch.exp(outputs.loss).item()

def compute_ppl_sink_aware(model, input_ids, budget, sink_count=16):
    """
    Our sink-aware SWS strategy: keep first sink_count + last (budget*sink_count) tokens.
    """
    n = len(input_ids)
    budget_tokens = int(n * budget)
    window = budget_tokens - sink_count
    if window < 0:
        window = 0
        sink_count = budget_tokens
    
    # Select positions: first sink_count + last window
    sink_indices = list(range(sink_count))
    window_indices = list(range(n - window, n))
    selected_indices = sorted(set(sink_indices + window_indices))
    
    selected_ids = input_ids[selected_indices].to(model.device)
    position_ids = torch.tensor(selected_indices, dtype=torch.long, device=model.device)
    
    with torch.no_grad():
        outputs = model(
            selected_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            labels=selected_ids.unsqueeze(0),
        )
        return torch.exp(outputs.loss).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--sink_counts", type=int, nargs="+", default=[0, 4, 8, 16])
    parser.add_argument("--first_ratios", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output_dir", default="experiment_results_pdtrim")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    hf_path = MODELS[args.model]
    model, tokenizer = load_model(args.model, hf_path)
    samples = get_wikitext2_data(tokenizer, args.seq_len, args.num_samples)
    
    results = {
        "model": args.model,
        "hf_path": hf_path,
        "seq_len": args.seq_len,
        "num_samples": len(samples),
        "baseline_ppl": None,
        "pdtrim_results": [],
        "sink_aware_results": [],
    }
    
    # Baseline
    print("Computing baseline PPL...")
    baseline_ppls = []
    for i, sample in enumerate(samples):
        ppl = compute_ppl_baseline(model, sample)
        baseline_ppls.append(ppl)
        print(f"  Sample {i}: baseline PPL = {ppl:.4f}")
    baseline_ppl = np.mean(baseline_ppls)
    results["baseline_ppl"] = baseline_ppl
    print(f"Mean baseline PPL: {baseline_ppl:.4f}")
    
    # PDTrim baselines
    for budget in args.budgets:
        for first_ratio in args.first_ratios:
            print(f"\nPDTrim: budget={budget}, first_ratio={first_ratio}")
            ppls = []
            for i, sample in enumerate(samples):
                ppl = compute_ppl_pdtrim(model, sample, budget, first_ratio)
                delta = (ppl / baseline_ppl - 1) * 100
                ppls.append(ppl)
                print(f"  Sample {i}: PPL={ppl:.4f} (Δ={delta:+.2f}%)")
            mean_ppl = np.mean(ppls)
            mean_delta = (mean_ppl / baseline_ppl - 1) * 100
            results["pdtrim_results"].append({
                "budget": budget,
                "first_ratio": first_ratio,
                "strategy": f"first_{int(budget*first_ratio*100)}%+last_{int(budget*(1-first_ratio)*100)}%",
                "mean_ppl": mean_ppl,
                "mean_delta_pct": round(mean_delta, 2),
                "sample_ppls": ppls,
            })
    
    # Sink-aware SWS
    for budget in args.budgets:
        for sink_count in args.sink_counts:
            print(f"\nSink-aware: budget={budget}, sink={sink_count}")
            ppls = []
            for i, sample in enumerate(samples):
                ppl = compute_ppl_sink_aware(model, sample, budget, sink_count)
                delta = (ppl / baseline_ppl - 1) * 100
                ppls.append(ppl)
                print(f"  Sample {i}: PPL={ppl:.4f} (Δ={delta:+.2f}%)")
            mean_ppl = np.mean(ppls)
            mean_delta = (mean_ppl / baseline_ppl - 1) * 100
            results["sink_aware_results"].append({
                "budget": budget,
                "sink_count": sink_count,
                "mean_ppl": mean_ppl,
                "mean_delta_pct": round(mean_delta, 2),
                "sample_ppls": ppls,
            })
    
    # Save results
    output_path = os.path.join(args.output_dir, f"pdtrim_vs_sinkaware_{args.model}_seq{args.seq_len}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"COMPARISON TABLE: {args.model} (baseline PPL={baseline_ppl:.4f})")
    print("="*80)
    print(f"{'Strategy':<35} {'Budget':<8} {'Mean PPL':<10} {'Delta%':<10}")
    print("-"*65)
    for r in results["pdtrim_results"]:
        strategy = r["strategy"]
        print(f"PDTrim {strategy:<25} {r['budget']:<8.1f} {r['mean_ppl']:<10.4f} {r['mean_delta_pct']:+.2f}%")
    for r in results["sink_aware_results"]:
        print(f"SinkAware(sink={r['sink_count']}){' '*(20-len(str(r['sink_count'])))} {r['budget']:<8.1f} {r['mean_ppl']:<10.4f} {r['mean_delta_pct']:+.2f}%")

if __name__ == "__main__":
    main()
