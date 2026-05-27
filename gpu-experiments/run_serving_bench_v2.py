#!/usr/bin/env python3
"""Simple vLLM Serving Benchmark for kvcache-lab.
Fixed: max_model_len capped at 32768 (Qwen2.5-7B max_position_embeddings).
"""
import os, json, time, argparse, gc
from datetime import datetime
from vllm import LLM, SamplingParams

MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
MAX_MODEL_LEN = 32768
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
MAX_TOKENS = 64

def gen_prompts(n, target_len):
    topics = [
        "Explain the architecture of distributed KV cache systems and their impact on LLM serving performance",
        "Analyze the trade-offs between memory efficiency and inference quality in transformer models",
        "Discuss the advantages of prefill-decode disaggregation for long-context LLM serving",
        "Describe how runtime memory management techniques can optimize GPU utilization in serving",
        "Compare different approaches to KV cache compression and their effects on model quality",
    ]
    base = "Please write a detailed analysis. " * 50
    prompts = []
    for i in range(n):
        t = topics[i % len(topics)]
        needed = max(0, target_len * 4 - len(t))
        padding = base[:needed]
        prompts.append(t + " " + padding)
    return prompts

def run_benchmark(output_dir):
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'='*60}")
        print(f"Context Length: {ctx_len}")
        print(f"{'='*60}")

        if ctx_len > MAX_MODEL_LEN:
            print(f"  Skipping: ctx_len={ctx_len} > max_model_len={MAX_MODEL_LEN}")
            continue

        try:
            llm = LLM(
                model=MODEL_PATH,
                max_model_len=ctx_len,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                disable_log_stats=True,
            )
        except Exception as e:
            print(f"  Failed to create LLM: {e}")
            continue

        for concurrency in CONCURRENCY_LEVELS:
            print(f"\n  Concurrency={concurrency}, ctx={ctx_len}")

            prompts = gen_prompts(concurrency, ctx_len)
            sampling = SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0)

            try:
                _ = llm.generate(prompts[:1], sampling)
            except Exception as e:
                print(f"  Warmup failed: {e}")
                continue

            try:
                t0 = time.time()
                outputs = llm.generate(prompts, sampling)
                elapsed = time.time() - t0

                total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                tps = total_tokens / elapsed
                ttft_list = []
                for o in outputs:
                    if o.metrics and hasattr(o.metrics, 'first_token_time') and hasattr(o.metrics, 'arrival_time'):
                        ttft_list.append(o.metrics.first_token_time - o.metrics.arrival_time)
                ttft_avg = sum(ttft_list) / len(ttft_list) * 1000 if ttft_list else 0
                ttft_sorted = sorted(ttft_list)
                ttft_p95 = ttft_sorted[int(len(ttft_sorted)*0.95)] * 1000 if len(ttft_sorted) >= 2 else ttft_avg

                result = {
                    "context_length": ctx_len,
                    "concurrency": concurrency,
                    "tps": round(tps, 2),
                    "ttft_avg_ms": round(ttft_avg, 2),
                    "ttft_p95_ms": round(ttft_p95, 2),
                    "total_tokens": total_tokens,
                    "elapsed_s": round(elapsed, 2),
                    "num_requests": concurrency,
                }
                results.append(result)
                print(f"  TPS={tps:.1f}, TTFT_avg={ttft_avg:.1f}ms, TTFT_P95={ttft_p95:.1f}ms")
            except Exception as e:
                print(f"  Benchmark failed: {e}")
                results.append({
                    "context_length": ctx_len,
                    "concurrency": concurrency,
                    "error": str(e),
                })

        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        time.sleep(2)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"serving_bench_{ts}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/root/autodl-tmp/experiment_results_serving")
    args = parser.parse_args()
    run_benchmark(args.output_dir)
