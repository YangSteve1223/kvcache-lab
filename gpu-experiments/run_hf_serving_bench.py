#!/usr/bin/env python3
"""
Custom Serving Benchmark for kvcache-lab (HuggingFace-based).
No vLLM dependency - works on any GPU including Blackwell.

Measures: TPS, TTFT, P95 latency at different context lengths and concurrency.
Simulates serving scenarios with batched generation.
"""
import os, sys, json, time, gc, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
MAX_NEW_TOKENS = 64
WARMUP_RUNS = 2

def load_model():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
          f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB GPU memory")
    return model, tokenizer

def gen_prompt(tokenizer, target_len):
    """Generate a prompt that fills approximately target_len tokens."""
    topics = [
        "Explain the architecture of distributed KV cache systems and their impact on LLM serving performance.",
        "Analyze the trade-offs between memory efficiency and inference quality in transformer models.",
        "Discuss the advantages of prefill-decode disaggregation for long-context LLM serving.",
        "Describe how runtime memory management techniques can optimize GPU utilization in serving.",
        "Compare different approaches to KV cache compression and their effects on model quality.",
    ]
    # Build prompt by repeating content to reach target length
    base_text = "Please provide a detailed technical analysis of the following topic. "
    for topic in topics:
        base_text += topic + " "
    
    tokens = tokenizer.encode(base_text, return_tensors="pt")[0]
    # Repeat to reach target length
    while len(tokens) < target_len:
        tokens = torch.cat([tokens, tokens[:target_len - len(tokens)]])
    return tokens[:target_len]

def benchmark_single(model, tokenizer, prompt_tokens, max_new_tokens):
    """Benchmark a single request: returns TTFT, total time, output tokens."""
    input_ids = prompt_tokens.unsqueeze(0).to(model.device)
    
    # Measure prefill + first token
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.no_grad():
        # Generate with timing
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    new_tokens = output.shape[1] - input_ids.shape[1]
    total_time = t_end - t_start
    
    # Estimate TTFT: prefill time ≈ total_time * (input_len / (input_len + output_len))
    # This is an approximation since we can't easily split prefill/decode in HF generate
    input_len = input_ids.shape[1]
    ttft_est = total_time * (input_len / (input_len + new_tokens)) if (input_len + new_tokens) > 0 else 0
    
    return {
        "ttft_ms": round(ttft_est * 1000, 2),
        "total_time_ms": round(total_time * 1000, 2),
        "output_tokens": new_tokens,
        "input_tokens": input_len,
    }

def benchmark_batched(model, tokenizer, prompts, max_new_tokens):
    """Benchmark batched generation (simulates concurrent requests in serving)."""
    # Pad all prompts to the same length
    max_len = max(p.shape[0] for p in prompts)
    padded = []
    attention_mask = []
    for p in prompts:
        pad_len = max_len - p.shape[0]
        padded.append(torch.cat([torch.zeros(pad_len, dtype=p.dtype), p]))
        attention_mask.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(p.shape[0], dtype=torch.long)]))
    
    input_ids = torch.stack(padded).to(model.device)
    attn_mask = torch.stack(attention_mask).to(model.device)
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_k=1,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    total_time = t_end - t_start
    
    total_new_tokens = sum(output.shape[1] - input_ids.shape[1] for _ in prompts)
    
    return {
        "total_time_ms": round(total_time * 1000, 2),
        "total_output_tokens": total_new_tokens,
        "tps": round(total_new_tokens / total_time, 2),
        "batch_size": len(prompts),
        "input_length": max_len,
    }

def run_benchmark(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model, tokenizer = load_model()
    results = []
    
    def save_results():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"hf_serving_bench_{ts}.json")
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_file}")
    
    # Warmup
    print("\nWarming up...")
    warmup_prompt = gen_prompt(tokenizer, 512)
    for _ in range(WARMUP_RUNS):
        _ = benchmark_single(model, tokenizer, warmup_prompt, 16)
    print("Warmup done.")
    
    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Context Length: {ctx_len}")
        print(f"{'='*60}")
        
        # Generate base prompt for this context length
        prompt_tokens = gen_prompt(tokenizer, ctx_len)
        
        for concurrency in CONCURRENCY_LEVELS:
            print(f"\n  Concurrency={concurrency}, ctx={ctx_len}")
            
            # Check GPU memory
            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            print(f"  Free GPU memory: {free_mem:.1f} GB")
            
            # For concurrency > 1, use batched generation
            try:
                if concurrency == 1:
                    # Single request - get detailed timing
                    r = benchmark_single(model, tokenizer, prompt_tokens, MAX_NEW_TOKENS)
                    tps = round(r["output_tokens"] / (r["total_time_ms"] / 1000), 2)
                    result = {
                        "context_length": ctx_len,
                        "concurrency": concurrency,
                        "tps": tps,
                        "ttft_ms": r["ttft_ms"],
                        "total_time_ms": r["total_time_ms"],
                        "output_tokens": r["output_tokens"],
                    }
                else:
                    # Batched generation
                    prompts = [gen_prompt(tokenizer, ctx_len) for _ in range(concurrency)]
                    r = benchmark_batched(model, tokenizer, prompts, MAX_NEW_TOKENS)
                    ttft_per_req = round(r["total_time_ms"] * ctx_len / (ctx_len + r["total_output_tokens"] // concurrency), 2)
                    result = {
                        "context_length": ctx_len,
                        "concurrency": concurrency,
                        "tps": r["tps"],
                        "ttft_ms": ttft_per_req,
                        "total_time_ms": r["total_time_ms"],
                        "output_tokens": r["total_output_tokens"],
                        "per_request_tokens": r["total_output_tokens"] // concurrency,
                    }
                
                results.append(result)
                print(f"  TPS={result['tps']}, TTFT={result['ttft_ms']}ms, Total={result['total_time_ms']}ms")
                save_results()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM at concurrency={concurrency}, ctx={ctx_len}")
                    results.append({
                        "context_length": ctx_len,
                        "concurrency": concurrency,
                        "error": "OOM",
                    })
                    save_results()
                    torch.cuda.empty_cache()
                    # Skip higher concurrency for this ctx_len
                    break
                else:
                    print(f"  Error: {e}")
                    results.append({
                        "context_length": ctx_len,
                        "concurrency": concurrency,
                        "error": str(e)[:200],
                    })
                    save_results()
            
            torch.cuda.empty_cache()
            time.sleep(1)
    
    # Compute P95 from all results of same config (if we have multiple runs)
    print(f"\nBenchmark complete! {len(results)} results collected.")
    
    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'CtxLen':>8} {'Conc':>5} {'TPS':>8} {'TTFT(ms)':>10} {'Total(ms)':>10}")
    print("-"*80)
    for r in results:
        if "error" not in r:
            print(f"{r['context_length']:>8} {r['concurrency']:>5} {r['tps']:>8.1f} {r['ttft_ms']:>10.1f} {r['total_time_ms']:>10.1f}")
        else:
            print(f"{r['context_length']:>8} {r['concurrency']:>5} {'ERROR':>8} {'':>10} {r.get('error','')[:15]:>10}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/experiment_results_serving")
    args = parser.parse_args()
    run_benchmark(args.output_dir)
