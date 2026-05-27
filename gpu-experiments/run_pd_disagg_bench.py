#!/usr/bin/env python3
"""
PD-Disaggregated Serving Benchmark with KV Tiering
===================================================
Measures end-to-end latency & throughput for PD separation with SWS tiering.

Dual-GPU mode (same machine):
  GPU 0: Prefill Worker (full model, full KV)
  GPU 1: Decode Worker  (SWS tiered KV, GPU active + CPU remote)

Single-GPU fallback (simulate KV transfer):
  Uses one GPU, simulates network hop via GPU→CPU→GPU round-trip.

Usage:
  # Dual-GPU
  python run_pd_disagg_bench.py --model_path /root/autodl-tmp/Qwen2.5-7B-Instruct

  # Single-GPU simulation
  python run_pd_disagg_bench.py --model_path /root/autodl-tmp/Qwen2.5-7B-Instruct --single_gpu

Outputs: pd_disagg_results.json
"""

import os
import sys
import json
import time
import gc
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import torch
import numpy as np


def set_hf_mirror():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ============================================================
# KV Cache Utilities
# ============================================================

def kv_to_device(kv, device):
    """Move KV cache to specified device (non-blocking)."""
    return tuple(
        (k.to(device, non_blocking=True), v.to(device, non_blocking=True))
        for k, v in kv
    )


def kv_size_mb(kv):
    """Calculate total KV cache size in MB."""
    total = 0
    for k, v in kv:
        total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
    return total / (1024 ** 2)


def kv_seq_len(kv):
    """Get sequence length from KV cache."""
    return kv[0][0].shape[2]


def clone_kv(kv):
    """Deep clone KV cache."""
    return tuple((k.clone(), v.clone()) for k, v in kv)


# ============================================================
# SWS Tiering Manager
# ============================================================

class SWSTieringManager:
    """
    Manages KV tiering: GPU (active window) + CPU (remote KV).
    
    In PD disaggregation:
      - Full KV is generated at Prefill GPU and transferred to Decode GPU
      - Decode GPU keeps only SWS window in HBM (active set)
      - Remaining KV is demoted to CPU DRAM (remote storage)
      - When remote KV is needed, it's fetched back to GPU (simulates network fetch)
    """

    def __init__(self, budget: float, gpu_device: str = "cuda:1"):
        self.budget = budget
        self.gpu_device = gpu_device
        self.gpu_kv = None       # Active set on GPU
        self.cpu_kv = None       # Remote KV on CPU
        self.window_size = 0
        self.full_seq_len = 0

    def tier(self, full_kv):
        """Split full KV into GPU active set + CPU remote storage."""
        self.full_seq_len = kv_seq_len(full_kv)
        self.window_size = max(1, int(self.full_seq_len * self.budget))

        gpu_kv = []
        cpu_kv = []

        for k, v in full_kv:
            if self.budget >= 1.0:
                gpu_kv.append((k, v))
                cpu_kv.append((None, None))
            else:
                k_gpu = k[:, :, -self.window_size:, :].clone()
                v_gpu = v[:, :, -self.window_size:, :].clone()
                k_cpu = k[:, :, :-self.window_size, :].clone().to("cpu")
                v_cpu = v[:, :, :-self.window_size, :].clone().to("cpu")
                gpu_kv.append((k_gpu, v_gpu))
                cpu_kv.append((k_cpu, v_cpu))

        self.gpu_kv = tuple(gpu_kv)
        self.cpu_kv = cpu_kv
        return self.gpu_kv

    def fetch_remote(self, n_tokens=64):
        """
        Fetch last n_tokens from remote (CPU) KV back to GPU.
        Simulates: remote storage → network → Decode GPU HBM.
        Returns combined KV (fetched + gpu active).
        """
        if self.budget >= 1.0 or self.cpu_kv[0][0] is None:
            return self.gpu_kv

        fetch_size = min(n_tokens, self.cpu_kv[0][0].shape[2])
        combined = []
        for (k_cpu, v_cpu), (k_gpu, v_gpu) in zip(self.cpu_kv, self.gpu_kv):
            k_fetch = k_cpu[:, :, -fetch_size:, :].to(self.gpu_device, non_blocking=True)
            v_fetch = v_cpu[:, :, -fetch_size:, :].to(self.gpu_device, non_blocking=True)
            k_comb = torch.cat([k_fetch, k_gpu], dim=2)
            v_comb = torch.cat([v_fetch, v_gpu], dim=2)
            combined.append((k_comb, v_comb))
        return tuple(combined)

    def update_after_decode(self, new_kv):
        """Update GPU KV after a decode step (slide window forward)."""
        updated = []
        for i, (k_new, v_new) in enumerate(new_kv):
            k_gpu, v_gpu = self.gpu_kv[i]
            k_upd = torch.cat([k_gpu, k_new[:, :, -1:, :]], dim=2)
            v_upd = torch.cat([v_gpu, v_new[:, :, -1:, :]], dim=2)
            if k_upd.shape[2] > self.window_size:
                k_upd = k_upd[:, :, -self.window_size:, :]
                v_upd = v_upd[:, :, -self.window_size:, :]
            updated.append((k_upd, v_upd))
        self.gpu_kv = tuple(updated)
        return self.gpu_kv

    def gpu_kv_size_mb(self):
        if self.gpu_kv is None:
            return 0
        return kv_size_mb(self.gpu_kv)

    def remote_kv_size_mb(self):
        if self.cpu_kv is None or self.cpu_kv[0][0] is None:
            return 0
        total = 0
        for k, v in self.cpu_kv:
            if k is not None:
                total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
        return total / (1024 ** 2)


# ============================================================
# PD-Disaggregated Benchmark
# ============================================================

class PDDisaggBenchmark:
    """
    End-to-end PD disaggregation benchmark.
    
    Experiments:
      1. KV Transfer Latency: GPU0 → CPU → GPU1
      2. Decode TPOT: local-only vs remote-fetch
      3. Max Concurrency: how many requests fit in decode GPU memory
      4. Concurrent Throughput: tokens/sec at various concurrency levels
    """

    def __init__(self, model_path, gpu_prefill=0, gpu_decode=1, single_gpu=False):
        self.model_path = model_path
        self.gpu_prefill = f"cuda:{gpu_prefill}"
        self.gpu_decode = f"cuda:{gpu_decode}"
        self.single_gpu = single_gpu
        self.tokenizer = None
        self.prefill_model = None
        self.decode_model = None

    def load_models(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[Init] Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.single_gpu:
            print(f"[Init] Single-GPU mode: loading model on {self.gpu_decode}...")
            self.decode_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.gpu_decode,
                trust_remote_code=True,
            )
            self.decode_model.eval()
            self.prefill_model = self.decode_model
            self.gpu_prefill = self.gpu_decode
        else:
            print(f"[Init] Dual-GPU mode: loading prefill model on {self.gpu_prefill}...")
            self.prefill_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.gpu_prefill,
                trust_remote_code=True,
            )
            self.prefill_model.eval()

            print(f"[Init] Loading decode model on {self.gpu_decode}...")
            self.decode_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.gpu_decode,
                trust_remote_code=True,
            )
            self.decode_model.eval()

        print("[Init] All models loaded ✓")

    # ----------------------------------------------------------
    # Experiment 1: KV Transfer Latency
    # ----------------------------------------------------------

    def measure_kv_transfer(self, seq_len, n_trials=3):
        """Measure KV transfer latency: GPU0 → CPU → GPU1."""
        device_from = self.gpu_prefill
        device_to = self.gpu_decode

        input_ids = torch.randint(100, 5000, (1, seq_len), device=device_from)

        # Prefill
        with torch.no_grad():
            outputs = self.prefill_model(input_ids=input_ids, use_cache=True)
            kv = outputs.past_key_values

        full_kv_mb = kv_size_mb(kv)

        # Transfer benchmark
        transfer_times = []
        for _ in range(n_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            kv_cpu = kv_to_device(kv, "cpu")
            torch.cuda.synchronize()

            kv_target = kv_to_device(kv_cpu, device_to)
            torch.cuda.synchronize(device_to)

            t1 = time.perf_counter()
            transfer_times.append(t1 - t0)
            del kv_cpu, kv_target

        del kv, outputs
        torch.cuda.empty_cache()

        return {
            "seq_len": seq_len,
            "full_kv_mb": round(full_kv_mb, 2),
            "transfer_avg_ms": round(np.mean(transfer_times) * 1000, 2),
            "transfer_min_ms": round(min(transfer_times) * 1000, 2),
            "transfer_max_ms": round(max(transfer_times) * 1000, 2),
        }

    # ----------------------------------------------------------
    # Experiment 2: Decode TPOT (local vs remote)
    # ----------------------------------------------------------

    def measure_decode_tpot(self, seq_len, budget, max_new_tokens=64, n_trials=1):
        """Measure decode TPOT with SWS tiering."""
        device = self.gpu_decode

        # Prefill + transfer
        input_ids = torch.randint(100, 5000, (1, seq_len), device=self.gpu_prefill)
        with torch.no_grad():
            outputs = self.prefill_model(input_ids=input_ids, use_cache=True)
            kv = outputs.past_key_values

        # Transfer KV
        if not self.single_gpu:
            kv = kv_to_device(kv_to_device(kv, "cpu"), device)
        prefill_kv_mb = kv_size_mb(kv)

        all_results = []

        for trial in range(n_trials):
            kv_clone = clone_kv(kv) if n_trials > 1 else kv

            # Apply SWS tiering
            tiering = SWSTieringManager(budget=budget, gpu_device=device)
            local_kv = tiering.tier(kv_clone)

            # Starting token
            next_token = torch.tensor([[100]], dtype=torch.long, device=device)

            # Warmup
            with torch.no_grad():
                _ = self.decode_model(
                    input_ids=next_token, past_key_values=local_kv, use_cache=True
                )
            torch.cuda.synchronize(device)

            # Decode loop
            tpot_local = []
            tpot_remote = []
            tpot_remote_fetch_only = []  # Just the CPU→GPU fetch time
            tokens = 0

            for step in range(max_new_tokens):
                test_remote = (step % 10 == 5) and (budget < 1.0)

                if test_remote:
                    # Measure: fetch + decode
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()

                    combined_kv = tiering.fetch_remote(n_tokens=64)
                    with torch.no_grad():
                        out = self.decode_model(
                            input_ids=next_token, past_key_values=combined_kv, use_cache=True
                        )
                    torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    tpot_remote.append((t1 - t0) * 1000)

                    # Also measure fetch-only overhead
                    torch.cuda.synchronize(device)
                    tf0 = time.perf_counter()
                    _ = tiering.fetch_remote(n_tokens=64)
                    torch.cuda.synchronize(device)
                    tf1 = time.perf_counter()
                    tpot_remote_fetch_only.append((tf1 - tf0) * 1000)

                    del combined_kv
                else:
                    # Local-only decode
                    torch.cuda.synchronize(device)
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = self.decode_model(
                            input_ids=next_token, past_key_values=local_kv, use_cache=True
                        )
                    torch.cuda.synchronize(device)
                    t1 = time.perf_counter()
                    tpot_local.append((t1 - t0) * 1000)

                # Update KV
                local_kv = tiering.update_after_decode(out.past_key_values)
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                tokens += 1

            # Compute metrics
            all_tpot = sorted(tpot_local + tpot_remote)

            result = {
                "seq_len": seq_len,
                "budget": budget,
                "prefill_kv_mb": round(prefill_kv_mb, 2),
                "sws_kv_mb": round(tiering.gpu_kv_size_mb(), 2),
                "remote_kv_mb": round(tiering.remote_kv_size_mb(), 2),
                "avg_tpot_local_ms": round(float(np.mean(tpot_local)), 2) if tpot_local else 0,
                "avg_tpot_remote_ms": round(float(np.mean(tpot_remote)), 2) if tpot_remote else 0,
                "remote_fetch_overhead_ms": round(float(np.mean(tpot_remote_fetch_only)), 2) if tpot_remote_fetch_only else 0,
                "p50_tpot_ms": round(float(np.percentile(all_tpot, 50)), 2) if all_tpot else 0,
                "p95_tpot_ms": round(float(np.percentile(all_tpot, 95)), 2) if all_tpot else 0,
                "p99_tpot_ms": round(float(np.percentile(all_tpot, 99)), 2) if all_tpot else 0,
                "max_tpot_ms": round(float(max(all_tpot)), 2) if all_tpot else 0,
                "tokens_generated": tokens,
            }
            all_results.append(result)

            del kv_clone, tiering
            torch.cuda.empty_cache(device)

        del kv, outputs
        torch.cuda.empty_cache()

        # Average across trials
        if n_trials > 1:
            avg_result = {"seq_len": seq_len, "budget": budget}
            for key in all_results[0]:
                if key not in ("seq_len", "budget"):
                    vals = [r[key] for r in all_results]
                    avg_result[key] = round(float(np.mean(vals)), 2)
            return avg_result
        return all_results[0]

    # ----------------------------------------------------------
    # Experiment 3: Max Concurrency Estimate
    # ----------------------------------------------------------

    def estimate_max_concurrency(self, seq_len, budget):
        """Estimate max concurrent requests that fit in decode GPU memory."""
        device = self.gpu_decode
        input_ids = torch.randint(100, 5000, (1, seq_len), device=self.gpu_prefill)

        with torch.no_grad():
            outputs = self.prefill_model(input_ids=input_ids, use_cache=True)
            kv = outputs.past_key_values

        if not self.single_gpu:
            kv = kv_to_device(kv_to_device(kv, "cpu"), device)

        tiering = SWSTieringManager(budget=budget, gpu_device=device)
        tiering.tier(kv)
        per_req_kv_mb = tiering.gpu_kv_size_mb()

        # Get available memory
        torch.cuda.empty_cache(device)
        total_mem_mb = torch.cuda.get_device_properties(device).total_mem / (1024 ** 2)
        used_mem_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        model_mem_mb = used_mem_mb - per_req_kv_mb  # Model weights
        available_mb = total_mem_mb - model_mem_mb - 512  # 512MB safety

        max_conc = int(available_mb / per_req_kv_mb) if per_req_kv_mb > 0 else 1
        baseline_kv_mb = kv_size_mb(kv)
        baseline_conc = int(available_mb / baseline_kv_mb) if baseline_kv_mb > 0 else 1

        del kv, outputs, tiering
        torch.cuda.empty_cache(device)

        return {
            "seq_len": seq_len,
            "budget": budget,
            "per_request_kv_mb": round(per_req_kv_mb, 2),
            "baseline_kv_mb": round(baseline_kv_mb, 2),
            "model_mem_mb": round(model_mem_mb, 0),
            "available_mb": round(available_mb, 0),
            "max_concurrent_sws": max(1, max_conc),
            "max_concurrent_baseline": max(1, baseline_conc),
            "concurrency_gain": round(max(1, max_conc) / max(1, baseline_conc), 1),
        }

    # ----------------------------------------------------------
    # Experiment 4: Concurrent Throughput
    # ----------------------------------------------------------

    def measure_concurrent_throughput(self, seq_len, budget, n_requests, max_new_tokens=32):
        """
        Measure actual concurrent decode throughput.
        Loads n_requests KV caches and decodes them in batch.
        """
        device = self.gpu_decode

        # Batch prefill
        input_ids = torch.randint(100, 5000, (n_requests, seq_len), device=self.gpu_prefill)
        with torch.no_grad():
            outputs = self.prefill_model(input_ids=input_ids, use_cache=True)
            full_kv = outputs.past_key_values

        full_kv_mb = kv_size_mb(full_kv)

        # Transfer
        if not self.single_gpu:
            full_kv = kv_to_device(kv_to_device(full_kv, "cpu"), device)

        # Apply SWS (trim all requests' KV to window)
        window = max(1, int(seq_len * budget))
        if budget < 1.0:
            kv = tuple(
                (k[:, :, -window:, :], v[:, :, -window:, :])
                for k, v in full_kv
            )
        else:
            kv = full_kv

        sws_kv_mb = kv_size_mb(kv)

        # Starting tokens for all requests
        next_tokens = torch.randint(100, 5000, (n_requests, 1), device=device)

        # Warmup
        with torch.no_grad():
            _ = self.decode_model(input_ids=next_tokens, past_key_values=kv, use_cache=True)
        torch.cuda.synchronize(device)

        # Decode loop
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        total_tokens = 0
        oom = False

        try:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    out = self.decode_model(
                        input_ids=next_tokens, past_key_values=kv, use_cache=True
                    )
                # Trim KV to maintain window size
                new_kv = out.past_key_values
                if budget < 1.0:
                    kv = tuple(
                        (k[:, :, -window:, :], v[:, :, -window:, :])
                        for k, v in new_kv
                    )
                else:
                    kv = new_kv

                next_tokens = out.logits[:, -1:, :].argmax(dim=-1)
                total_tokens += n_requests

            torch.cuda.synchronize(device)
            t_end = time.perf_counter()
            elapsed = t_end - t_start
        except torch.cuda.OutOfMemoryError:
            oom = True
            elapsed = 0
            total_tokens = 0

        del full_kv, kv, outputs
        if 'out' in dir():
            del out
        torch.cuda.empty_cache(device)

        return {
            "seq_len": seq_len,
            "budget": budget,
            "n_requests": n_requests,
            "full_kv_mb": round(full_kv_mb, 2),
            "sws_kv_mb": round(sws_kv_mb, 2),
            "total_tokens": total_tokens,
            "elapsed_s": round(elapsed, 4) if not oom else 0,
            "throughput_tps": round(total_tokens / elapsed, 2) if not oom and elapsed > 0 else 0,
            "per_request_tps": round(total_tokens / elapsed / n_requests, 2) if not oom and elapsed > 0 else 0,
            "oom": oom,
        }

    # ----------------------------------------------------------
    # Full Benchmark Runner
    # ----------------------------------------------------------

    def run_all(self, config: dict):
        """Run all experiments and return results."""
        seq_lengths = config.get("seq_lengths", [1024, 2048, 4096, 8192])
        budgets = config.get("budgets", [1.0, 0.5, 0.3])
        max_new_tokens = config.get("max_new_tokens", 64)
        concurrent_list = config.get("concurrent_list", [1, 2, 4, 8])

        results = {
            "kv_transfer": [],
            "decode_tpot": [],
            "max_concurrency": [],
            "concurrent_throughput": [],
        }

        # ---- Phase 1: KV Transfer Latency ----
        print("\n" + "=" * 70)
        print("PHASE 1: KV Transfer Latency")
        print("=" * 70)
        for seq_len in seq_lengths:
            r = self.measure_kv_transfer(seq_len)
            results["kv_transfer"].append(r)
            print(f"  seq={seq_len}: KV={r['full_kv_mb']:.0f}MB, "
                  f"transfer={r['transfer_avg_ms']:.1f}ms")

        # ---- Phase 2: Decode TPOT ----
        print("\n" + "=" * 70)
        print("PHASE 2: Decode TPOT (local vs remote)")
        print("=" * 70)
        for seq_len in seq_lengths:
            for budget in budgets:
                r = self.measure_decode_tpot(seq_len, budget, max_new_tokens)
                results["decode_tpot"].append(r)
                print(f"  seq={seq_len}, budget={budget}: "
                      f"local={r['avg_tpot_local_ms']:.1f}ms, "
                      f"remote={r['avg_tpot_remote_ms']:.1f}ms, "
                      f"p99={r['p99_tpot_ms']:.1f}ms")

        # ---- Phase 3: Max Concurrency ----
        print("\n" + "=" * 70)
        print("PHASE 3: Max Concurrency Estimate")
        print("=" * 70)
        for seq_len in seq_lengths:
            for budget in budgets:
                r = self.estimate_max_concurrency(seq_len, budget)
                results["max_concurrency"].append(r)
                print(f"  seq={seq_len}, budget={budget}: "
                      f"baseline={r['max_concurrent_baseline']}x → "
                      f"SWS={r['max_concurrent_sws']}x "
                      f"(gain={r['concurrency_gain']}x)")

        # ---- Phase 4: Concurrent Throughput ----
        print("\n" + "=" * 70)
        print("PHASE 4: Concurrent Throughput")
        print("=" * 70)
        for seq_len in seq_lengths:
            for budget in budgets:
                # Get max concurrency
                mc = self.estimate_max_concurrency(seq_len, budget)
                max_n = mc["max_concurrent_sws"]

                for n_req in concurrent_list:
                    if n_req > max_n:
                        print(f"  seq={seq_len}, budget={budget}, n={n_req}: SKIP (exceeds max {max_n})")
                        continue
                    r = self.measure_concurrent_throughput(seq_len, budget, n_req)
                    results["concurrent_throughput"].append(r)
                    status = "OOM" if r["oom"] else f"{r['throughput_tps']:.1f} tps"
                    print(f"  seq={seq_len}, budget={budget}, n={n_req}: {status}")

        return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PD-Disaggregated Serving Benchmark with KV Tiering"
    )
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/Qwen2.5-7B-Instruct",
                        help="Path to HF model")
    parser.add_argument("--output_file", type=str,
                        default="pd_disagg_results.json",
                        help="Output JSON file")
    parser.add_argument("--gpu_prefill", type=int, default=0,
                        help="GPU ID for prefill worker")
    parser.add_argument("--gpu_decode", type=int, default=1,
                        help="GPU ID for decode worker")
    parser.add_argument("--single_gpu", action="store_true",
                        help="Single-GPU mode (simulate KV transfer)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max decode tokens per request")
    parser.add_argument("--seq_lengths", type=str, default="1024,2048,4096,8192",
                        help="Comma-separated sequence lengths")
    parser.add_argument("--budgets", type=str, default="1.0,0.5,0.3",
                        help="Comma-separated SWS budgets (1.0=baseline)")
    parser.add_argument("--concurrent_list", type=str, default="1,2,4,8",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--skip_transfer", action="store_true",
                        help="Skip KV transfer latency test")
    parser.add_argument("--skip_throughput", action="store_true",
                        help="Skip concurrent throughput test")
    args = parser.parse_args()

    set_hf_mirror()

    config = {
        "seq_lengths": [int(x) for x in args.seq_lengths.split(",")],
        "budgets": [float(x) for x in args.budgets.split(",")],
        "max_new_tokens": args.max_new_tokens,
        "concurrent_list": [int(x) for x in args.concurrent_list.split(",")],
    }

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2 and not args.single_gpu:
        print(f"⚠️  Only {n_gpus} GPU(s) detected. Switching to single-GPU mode.")
        args.single_gpu = True

    if args.single_gpu:
        print("📋 Mode: Single-GPU (simulated KV transfer)")
    else:
        print(f"📋 Mode: Dual-GPU (GPU {args.gpu_prefill}=Prefill, GPU {args.gpu_decode}=Decode)")

    print(f"📋 Model: {args.model_path}")
    print(f"📋 Seq lengths: {config['seq_lengths']}")
    print(f"📋 Budgets: {config['budgets']}")
    print(f"📋 Max new tokens: {config['max_new_tokens']}")

    # Initialize benchmark
    bench = PDDisaggBenchmark(
        model_path=args.model_path,
        gpu_prefill=args.gpu_prefill,
        gpu_decode=args.gpu_decode,
        single_gpu=args.single_gpu,
    )
    bench.load_models()

    # Run experiments
    results = bench.run_all(config)

    # Add metadata
    output = {
        "metadata": {
            "model_path": args.model_path,
            "single_gpu": args.single_gpu,
            "gpu_prefill": args.gpu_prefill,
            "gpu_decode": args.gpu_decode,
            "n_gpus": n_gpus,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(n_gpus)],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "config": config,
        "results": results,
    }

    # Save
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {args.output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Decode TPOT")
    print("=" * 80)
    print(f"{'SeqLen':>6} {'Budget':>6} {'KV(MB)':>8} {'SWS(MB)':>8} "
          f"{'Local':>8} {'Remote':>8} {'p99':>8} {'Fetch':>8}")
    print("-" * 80)
    for r in results["decode_tpot"]:
        print(f"{r['seq_len']:>6} {r['budget']:>6.1f} {r['prefill_kv_mb']:>8.0f} "
              f"{r['sws_kv_mb']:>8.0f} {r['avg_tpot_local_ms']:>7.1f}ms "
              f"{r['avg_tpot_remote_ms']:>7.1f}ms {r['p99_tpot_ms']:>7.1f}ms "
              f"{r['remote_fetch_overhead_ms']:>7.1f}ms")

    print("\n" + "=" * 80)
    print("SUMMARY: Max Concurrency")
    print("=" * 80)
    print(f"{'SeqLen':>6} {'Budget':>6} {'Base':>5} {'SWS':>5} {'Gain':>5}")
    print("-" * 40)
    for r in results["max_concurrency"]:
        print(f"{r['seq_len']:>6} {r['budget']:>6.1f} "
              f"{r['max_concurrent_baseline']:>5} {r['max_concurrent_sws']:>5} "
              f"{r['concurrency_gain']:>4.1f}x")

    if results["concurrent_throughput"]:
        print("\n" + "=" * 80)
        print("SUMMARY: Concurrent Throughput")
        print("=" * 80)
        print(f"{'SeqLen':>6} {'Budget':>6} {'N':>4} {'TPS':>8} {'PerReq':>8} {'OOM':>4}")
        print("-" * 50)
        for r in results["concurrent_throughput"]:
            oom_str = "YES" if r["oom"] else ""
            print(f"{r['seq_len']:>6} {r['budget']:>6.1f} {r['n_requests']:>4} "
                  f"{r['throughput_tps']:>7.1f} {r['per_request_tps']:>7.1f} {oom_str:>4}")


if __name__ == "__main__":
    main()
