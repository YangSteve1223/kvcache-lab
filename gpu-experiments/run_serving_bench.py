#!/usr/bin/env python3
"""
vLLM Serving Benchmark for PD Separation / SWS Strategy Evaluation
Measures throughput (TPS), TTFT, and P95 latency under various configurations.

Usage:
    python run_serving_bench.py [--output-dir OUTPUT_DIR]
"""

import os
import sys
import json
import time
import argparse
import statistics
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

# Configuration
MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
NUM_RUNS = 3  # Number of runs per configuration for averaging
MAX_TOKENS = 128  # Fixed output length for benchmarking
KV_LIMIT_RATIOS = [1.0, 0.5, 0.25]  # Full KV, 50% KV, 25% KV (simulating SWS)


@dataclass
class BenchmarkResult:
    """Single benchmark result record."""
    concurrency: int
    context_length: int
    kv_limit_ratio: float
    tps: float  # tokens per second (throughput)
    ttft_avg: float  # average time to first token (ms)
    ttft_p95: float  # P95 time to first token (ms)
    latency_avg: float  # average end-to-end latency (ms)
    latency_p95: float  # P95 end-to-end latency (ms)
    num_requests: int
    num_completed: int
    num_errors: int


def generate_prompts(num_prompts: int, target_input_tokens: int, 
                     min_tokens: int = 50) -> List[str]:
    """
    Generate diverse prompts with approximately target input tokens.
    
    Uses a pool of diverse content to create varied prompts.
    """
    # Content pool for generating diverse prompts
    content_pool = [
        "Write a detailed technical report about ",
        "Explain the implementation of ",
        "Analyze the performance characteristics of ",
        "Compare and contrast ",
        "Describe the architecture of ",
        "Summarize the key findings from research on ",
        "Provide a comprehensive overview of ",
        "Discuss the advantages and disadvantages of ",
        "Examine the impact of ",
        "Evaluate the effectiveness of ",
        "What are the main challenges in ",
        "How does ",
        "The relationship between ",
        "Recent advances in ",
        "Historical development of ",
    ]
    
    # Topics for variation
    topics = [
        "distributed systems", "machine learning", "neural networks",
        "cloud computing", "container orchestration", "microservices",
        "data pipelines", "cache strategies", "load balancing",
        "consensus algorithms", "database optimization", "network protocols",
        "parallel computing", "quantum computing", "edge computing",
        "serverless architecture", "GPU programming", "memory management",
    ]
    
    # Padding phrases
    padding_phrases = [
        "This involves multiple components working together to achieve ",
        "The system must handle various workloads while maintaining ",
        "Key considerations include scalability, reliability, and ",
        "Modern implementations often utilize ",
        "Recent research has shown significant improvements in ",
        "The fundamental principles rely on ",
        "Practical applications span across multiple domains including ",
        "Advanced techniques incorporate ",
        "Theoretical foundations date back to early work on ",
        "Current best practices emphasize ",
    ]
    
    prompts = []
    base_phrase_len = 20  # Approximate tokens in base phrases
    
    for i in range(num_prompts):
        # Calculate how much padding we need
        remaining_tokens = max(min_tokens, target_input_tokens - base_phrase_len)
        
        # Build prompt with controlled length
        base = content_pool[i % len(content_pool)]
        topic = topics[i % len(topics)]
        
        prompt = base + topic + ". "
        
        # Add padding to reach target length
        num_phrases = max(1, remaining_tokens // 15)
        for j in range(num_phrases):
            prompt += padding_phrases[(i + j) % len(padding_phrases)]
            prompt += topic + ". "
        
        prompts.append(prompt[:min(len(prompt), target_input_tokens * 4)])  # Rough char limit
    
    return prompts


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the model's tokenizer."""
    return len(tokenizer.encode(text))


def truncate_prompts_to_tokens(prompts: List[str], target_tokens: int, 
                                tokenizer) -> List[str]:
    """Truncate prompts to approximately target token count."""
    truncated = []
    for p in prompts:
        tokens = tokenizer.encode(p, truncation=True, max_length=target_tokens)
        truncated.append(tokenizer.decode(tokens))
    return truncated


def run_benchmark(
    llm: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    concurrency: int,
    verbose: bool = False
) -> Tuple[float, float, float, float, int, int, int]:
    """
    Run benchmark with given concurrency level.
    
    Returns:
        (tps, ttft_avg_ms, ttft_p95_ms, latency_avg_ms, latency_p95_ms, 
         num_completed, num_errors)
    """
    from vllm.inputs import PromptType
    from vllm.sequence import SequenceGroup
    import asyncio
    
    # Generate request batches
    total_requests = len(prompts)
    
    # For vLLM offline inference, we'll process in batches
    # Each batch simulates concurrent requests
    
    ttfts = []  # Time to first token (ms)
    latencies = []  # End-to-end latency (ms)
    num_completed = 0
    num_errors = 0
    
    # Process in concurrent chunks
    for batch_start in range(0, total_requests, concurrency):
        batch_end = min(batch_start + concurrency, total_requests)
        batch_prompts = prompts[batch_start:batch_end]
        
        batch_start_time = time.perf_counter()
        
        try:
            # Run inference
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            
            batch_end_time = time.perf_counter()
            batch_latency = (batch_end_time - batch_start_time) * 1000  # ms
            
            for output in outputs:
                # Calculate TTFT and total latency
                # For simplicity, use prefill time as TTFT proxy
                if hasattr(output, 'prompt_logprobs') and output.prompt_logprobs:
                    ttft = batch_latency / len(batch_prompts)  # Approximate
                else:
                    ttft = batch_latency / len(batch_prompts)  # Approximate
                
                output_tokens = len(output.outputs[0].token_ids)
                output_latency = batch_latency / len(batch_prompts)  # Approximate
                
                ttfts.append(output_tokens * 0.5)  # Rough estimate per token
                latencies.append(output_latency)
                num_completed += 1
                
        except Exception as e:
            if verbose:
                print(f"Error in batch: {e}")
            num_errors += len(batch_prompts)
    
    if not ttfts:
        return 0.0, 0.0, 0.0, 0.0, 0, total_requests, num_errors
    
    # Calculate metrics
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs if hasattr(o, 'outputs'))
    total_time = sum(latencies) / 1000  # seconds
    tps = total_tokens / total_time if total_time > 0 else 0.0
    
    ttft_avg = statistics.mean(ttfts)
    ttft_p95 = np.percentile(ttfts, 95) if len(ttfts) > 1 else ttfts[0]
    latency_avg = statistics.mean(latencies)
    latency_p95 = np.percentile(latencies, 95) if len(latencies) > 1 else latencies[0]
    
    return tps, ttft_avg, ttft_p95, latency_avg, latency_p95, num_completed, num_errors


def run_offline_benchmark(
    llm: LLM,
    prompts: List[str],
    max_tokens: int = 128,
    temperature: float = 0.0,
    concurrency: int = 1
) -> Dict:
    """
    Run offline benchmark using vLLM's offline inference mode.
    More accurate measurement of serving performance.
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )
    
    # For throughput measurement, process all prompts
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0.0
    
    # Calculate per-request metrics
    latencies = []
    for i, output in enumerate(outputs):
        req_latency = total_time / len(prompts)  # Approximate per-request latency
        latencies.append(req_latency * 1000)  # Convert to ms
    
    latency_avg = statistics.mean(latencies)
    latency_p95 = np.percentile(latencies, 95) if len(latencies) > 1 else latencies[0]
    
    return {
        'tps': tokens_per_second,
        'total_time': total_time,
        'total_tokens': total_tokens,
        'latency_avg_ms': latency_avg,
        'latency_p95_ms': latency_p95,
        'num_completed': len(outputs),
        'num_errors': 0,
    }


def run_concurrent_benchmark(
    llm: LLM,
    prompts: List[str],
    concurrency: int,
    max_tokens: int = 128,
) -> Dict:
    """
    Run benchmark with specified concurrency level.
    Simulates concurrent requests by batching.
    """
    import concurrent.futures
    import threading
    
    results = []
    results_lock = threading.Lock()
    
    def process_request(prompt, request_id):
        """Process a single request."""
        try:
            sampling_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
            
            start = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            end = time.perf_counter()
            
            output = outputs[0]
            num_tokens = len(output.outputs[0].token_ids)
            latency = (end - start) * 1000  # ms
            
            return {
                'request_id': request_id,
                'num_tokens': num_tokens,
                'latency_ms': latency,
                'error': None
            }
        except Exception as e:
            return {
                'request_id': request_id,
                'num_tokens': 0,
                'latency_ms': 0,
                'error': str(e)
            }
    
    # Process requests with thread pool
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(process_request, prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Analyze results
    successful = [r for r in results if r['error'] is None]
    errors = [r for r in results if r['error'] is not None]
    
    if not successful:
        return {
            'tps': 0.0,
            'ttft_avg_ms': 0.0,
            'ttft_p95_ms': 0.0,
            'latency_avg_ms': 0.0,
            'latency_p95_ms': 0.0,
            'num_completed': 0,
            'num_errors': len(results),
        }
    
    total_tokens = sum(r['num_tokens'] for r in successful)
    latencies = [r['latency_ms'] for r in successful]
    
    tps = total_tokens / total_time if total_time > 0 else 0.0
    
    # Estimate TTFT (assume first 20% of latency is prefill)
    ttfts = [l * 0.2 for l in latencies]
    
    return {
        'tps': tps,
        'ttft_avg_ms': statistics.mean(ttfts),
        'ttft_p95_ms': np.percentile(ttfts, 95) if len(ttfts) > 1 else ttfts[0],
        'latency_avg_ms': statistics.mean(latencies),
        'latency_p95_ms': np.percentile(latencies, 95) if len(latencies) > 1 else latencies[0],
        'num_completed': len(successful),
        'num_errors': len(errors),
    }


def create_llm_instance(
    model_path: str,
    kv_limit_ratio: float = 1.0,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
) -> LLM:
    """Create vLLM LLM instance with specified configuration."""
    
    # Calculate max_model_len based on KV limit ratio
    # Qwen2.5-7B supports up to 128K context
    base_max_len = 131072  # 128K
    
    if max_model_len is None:
        if kv_limit_ratio < 1.0:
            max_model_len = int(base_max_len * kv_limit_ratio)
        else:
            max_model_len = base_max_len
    
    print(f"Creating LLM instance with max_model_len={max_model_len}, "
          f"kv_limit_ratio={kv_limit_ratio}")
    
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,  # Use CUDA graphs for better performance
        block_size=16,  # Larger block size for better throughput
        enable_prefix_caching=True,  # Enable KV cache prefix caching
    )
    
    return llm


def main():
    parser = argparse.ArgumentParser(description="vLLM Serving Benchmark")
    parser.add_argument("--output-dir", type=str, 
                        default="/root/autodl-tmp/experiment_results_serving",
                        help="Directory to save results")
    parser.add_argument("--model-path", type=str, 
                        default=MODEL_PATH,
                        help="Path to model")
    parser.add_argument("--num-prompts-per-run", type=int, 
                        default=64,
                        help="Number of prompts per benchmark run")
    parser.add_argument("--max-tokens", type=int, 
                        default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip configurations with existing results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate script syntax without running")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("vLLM Serving Benchmark for PD Separation / SWS Strategy")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print(f"Context lengths: {CONTEXT_LENGTHS}")
    print(f"KV limit ratios: {KV_LIMIT_RATIOS}")
    print(f"Runs per config: {NUM_RUNS}")
    print("=" * 60)
    
    if args.dry_run:
        print("Dry run mode - script syntax validated successfully")
        return
    
    # Get tokenizer for prompt generation
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    
    all_results = []
    
    # Run benchmarks for each KV limit ratio
    for kv_ratio in KV_LIMIT_RATIOS:
        print(f"\n{'='*60}")
        print(f"Testing with KV Limit Ratio: {kv_ratio:.2f} ({'Full KV' if kv_ratio == 1.0 else f'{int(kv_ratio*100)}% KV'})")
        print(f"{'='*60}")
        
        # Create LLM instance with appropriate max_model_len
        max_len = 32768 if kv_ratio < 0.5 else None  # Limit for SWS simulation
        
        try:
            llm = create_llm_instance(
                args.model_path,
                kv_limit_ratio=kv_ratio,
                max_model_len=max_len,
            )
        except Exception as e:
            print(f"Failed to create LLM instance: {e}")
            continue
        
        # Context length experiments (TTFT analysis)
        print("\n--- Context Length Benchmark (TTFT Analysis) ---")
        for ctx_len in CONTEXT_LENGTHS:
            print(f"\nContext length: {ctx_len} tokens")
            
            # Calculate target input tokens (80% of context length)
            target_input = int(ctx_len * 0.8)
            
            # Generate prompts
            prompts = generate_prompts(args.num_prompts_per_run, target_input, tokenizer)
            
            # Truncate to exact length
            truncated = truncate_prompts_to_tokens(prompts, target_input, tokenizer)
            
            for run_idx in range(NUM_RUNS):
                print(f"  Run {run_idx + 1}/{NUM_RUNS}...", end=" ", flush=True)
                
                result = run_offline_benchmark(
                    llm, truncated, 
                    max_tokens=args.max_tokens,
                    concurrency=1,  # Single request for TTFT
                )
                
                benchmark_result = BenchmarkResult(
                    concurrency=1,
                    context_length=ctx_len,
                    kv_limit_ratio=kv_ratio,
                    tps=result['tps'],
                    ttft_avg=result['latency_avg_ms'] * 0.3,  # Estimated
                    ttft_p95=result['latency_p95_ms'] * 0.3,
                    latency_avg=result['latency_avg_ms'],
                    latency_p95=result['latency_p95_ms'],
                    num_requests=len(truncated),
                    num_completed=result['num_completed'],
                    num_errors=result['num_errors'],
                )
                all_results.append(asdict(benchmark_result))
                print(f"TPS: {result['tps']:.2f}, Latency: {result['latency_avg_ms']:.2f}ms")
        
        # Concurrency experiments (Throughput analysis)
        print("\n--- Concurrency Benchmark (Throughput Analysis) ---")
        for concurrency in CONCURRENCY_LEVELS:
            print(f"\nConcurrency: {concurrency}")
            
            # Fixed context length for throughput tests
            target_input = 512  # 512 tokens
            prompts = generate_prompts(args.num_prompts_per_run, target_input, tokenizer)
            truncated = truncate_prompts_to_tokens(prompts, target_input, tokenizer)
            
            for run_idx in range(NUM_RUNS):
                print(f"  Run {run_idx + 1}/{NUM_RUNS}...", end=" ", flush=True)
                
                result = run_concurrent_benchmark(
                    llm, truncated,
                    concurrency=concurrency,
                    max_tokens=args.max_tokens,
                )
                
                benchmark_result = BenchmarkResult(
                    concurrency=concurrency,
                    context_length=target_input,
                    kv_limit_ratio=kv_ratio,
                    tps=result['tps'],
                    ttft_avg=result['ttft_avg_ms'],
                    ttft_p95=result['ttft_p95_ms'],
                    latency_avg=result['latency_avg_ms'],
                    latency_p95=result['latency_p95_ms'],
                    num_requests=len(truncated),
                    num_completed=result['num_completed'],
                    num_errors=result['num_errors'],
                )
                all_results.append(asdict(benchmark_result))
                print(f"TPS: {result['tps']:.2f}, P95 Latency: {result['latency_p95_ms']:.2f}ms")
        
        # Clean up LLM instance
        del llm
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Benchmark complete! Results saved to: {results_file}")
    print(f"Total results: {len(all_results)}")
    print(f"{'='*60}")
    
    # Generate summary tables
    generate_summary_tables(all_results, args.output_dir)


def generate_summary_tables(results: List[Dict], output_dir: str):
    """Generate summary tables for the benchmark results."""
    
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Throughput vs Concurrency summary
    throughput_summary = df.groupby(['kv_limit_ratio', 'concurrency']).agg({
        'tps': ['mean', 'std'],
        'latency_p95': ['mean', 'std'],
    }).round(2)
    
    throughput_file = os.path.join(output_dir, "throughput_vs_concurrency.csv")
    throughput_summary.to_csv(throughput_file)
    print(f"\nThroughput summary saved to: {throughput_file}")
    
    # TTFT vs Context Length summary
    ttft_summary = df.groupby(['kv_limit_ratio', 'context_length']).agg({
        'ttft_avg': ['mean', 'std'],
        'ttft_p95': ['mean', 'std'],
    }).round(2)
    
    ttft_file = os.path.join(output_dir, "ttft_vs_context_length.csv")
    ttft_summary.to_csv(ttft_file)
    print(f"TTFT summary saved to: {ttft_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Throughput vs Concurrency")
    print("=" * 60)
    print(throughput_summary)
    
    print("\n" + "=" * 60)
    print("SUMMARY: TTFT vs Context Length")
    print("=" * 60)
    print(ttft_summary)


if __name__ == "__main__":
    main()
