#!/usr/bin/env python3
"""Mistral-7B Locality Characterization - Eager Mode with output_attentions=True
This gives REAL attention weights including RoPE decay + SWA mask effects.
"""
import json
import os
import gc
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_PATH = "/root/autodl-tmp/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "/root/autodl-tmp/experiment_results_multimodel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LENGTHS = [1024, 2048, 4096]
DEVICE = "cuda:0"

def compute_locality_metrics(attn_weights_all_layers, seq_len):
    """Compute locality metrics from real attention weights."""
    gini_coeffs = []
    active_set_ratios = []
    remote_ratios = []
    
    for layer_attn in attn_weights_all_layers:
        # layer_attn: (num_heads, seq_len, seq_len)
        # Average across heads
        avg_attn = layer_attn.mean(axis=0)  # (seq_len, seq_len)
        
        # For each query position, compute attention concentration
        for i in range(seq_len):
            row = avg_attn[i, :i+1]  # causal: only look at positions <= i
            if len(row) < 2:
                continue
            
            row = row / (row.sum() + 1e-10)  # normalize
            
            # Gini coefficient
            sorted_row = np.sort(row)
            n = len(sorted_row)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_row) / (n * np.sum(sorted_row) + 1e-10)) - (n + 1) / n
            gini_coeffs.append(gini)
            
            # Active set: fraction of tokens receiving 90% of attention
            cumsum = np.cumsum(sorted_row[::-1])
            threshold = 0.9 * sorted_row.sum()
            active_count = np.searchsorted(cumsum, threshold) + 1
            active_set_ratios.append(active_count / n)
            
            # Remote attention: attention to distant tokens (beyond 50% of context)
            mid = i // 2
            remote_attn = row[:mid].sum() if mid > 0 else 0
            remote_ratios.append(remote_attn)
    
    return {
        "gini_mean": float(np.mean(gini_coeffs)),
        "gini_std": float(np.std(gini_coeffs)),
        "active_set_ratio_mean": float(np.mean(active_set_ratios)),
        "active_set_ratio_std": float(np.std(active_set_ratios)),
        "remote_attention_mean": float(np.mean(remote_ratios)),
        "remote_attention_std": float(np.std(remote_ratios)),
        "n_observations": len(gini_coeffs)
    }

def main():
    print("=" * 60)
    print("Mistral-7B Eager Locality Characterization")
    print("=" * 60)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in EAGER mode
    print("Loading Mistral-7B in EAGER mode (no SDPA)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    
    results = {}
    
    for seq_len in SEQ_LENGTHS:
        print(f"\n--- seq_len={seq_len} ---")
        
        # Generate input
        prompt = "Write a detailed analysis of machine learning systems and their optimization. " * (seq_len // 15 + 1)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=seq_len, truncation=True).to(DEVICE)
        actual_len = inputs.input_ids.shape[1]
        print(f"Input length: {actual_len}")
        
        # Forward pass with attention outputs
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                use_cache=False
            )
        
        print(f"Got {len(outputs.attentions)} attention layers")
        
        # Extract attention weights
        attn_weights_all_layers = []
        for layer_idx, attn in enumerate(outputs.attentions):
            attn_np = attn[0].float().cpu().numpy()
            attn_weights_all_layers.append(attn_np)
            
            if layer_idx == 0:
                # Sanity check: verify SWA window
                row = attn_np[0, -1, :]
                nonzero_positions = np.where(row > 1e-6)[0]
                window_size = actual_len - nonzero_positions[0] if len(nonzero_positions) > 0 else actual_len
                print(f"  Layer 0, Head 0: effective window size ~{window_size}")
        
        # Compute metrics
        metrics = compute_locality_metrics(attn_weights_all_layers, actual_len)
        results[f"seq_{seq_len}"] = {
            "actual_length": actual_len,
            "num_layers": len(outputs.attentions),
            **metrics
        }
        print(f"  Gini: {metrics['gini_mean']:.3f} +/- {metrics['gini_std']:.3f}")
        print(f"  Active set: {metrics['active_set_ratio_mean']:.1%} +/- {metrics['active_set_ratio_std']:.1%}")
        print(f"  Remote attn: {metrics['remote_attention_mean']:.1%} +/- {metrics['remote_attention_std']:.1%}")
        
        # Free memory
        del outputs
        del attn_weights_all_layers
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    gini_vals = [results[f"seq_{s}"]["gini_mean"] for s in SEQ_LENGTHS]
    active_vals = [results[f"seq_{s}"]["active_set_ratio_mean"] for s in SEQ_LENGTHS]
    remote_vals = [results[f"seq_{s}"]["remote_attention_mean"] for s in SEQ_LENGTHS]
    
    overall = {
        "model": "Mistral-7B-Instruct-v0.3",
        "method": "eager_output_attentions",
        "gini_mean": float(np.mean(gini_vals)),
        "active_set_ratio_mean": float(np.mean(active_vals)),
        "remote_attention_mean": float(np.mean(remote_vals)),
        "per_sequence": results,
        "PASS": float(np.mean(gini_vals)) > 0.85
    }
    
    print(f"Overall Gini: {overall['gini_mean']:.3f}")
    print(f"Overall Active: {overall['active_set_ratio_mean']:.1%}")
    print(f"Overall Remote: {overall['remote_attention_mean']:.1%}")
    print(f"PASS: {overall['PASS']} (threshold: Gini > 0.85)")
    
    # Save results
    out_file = os.path.join(OUTPUT_DIR, "mistral_locality_eager.json")
    with open(out_file, 'w') as f:
        json.dump(overall, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()
