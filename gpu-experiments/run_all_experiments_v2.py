#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU实验一键运行脚本 v2

修复TAA注入: 使用register_forward_pre_hook在attention mask上注入TAA bias,
让模型在前向传播中真正使用修改后的注意力。

核心改动:
    - load_model() 添加 attn_implementation="eager"
    - run_with_taa() 使用 register_forward_pre_hook 注入 TAA bias
    - compute_perplexity_with_taa() 计算带TAA的PPL
    - G3d 真正对比 before/after PPL

环境要求:
    - CUDA 12.x+
    - Python 3.10+
    - transformers + torch

使用方法:
    python3 run_all_experiments.py --model /root/autodl-tmp/Qwen2.5-7B-Instruct --output-dir ./results
    python3 run_all_experiments.py --skip-g1 --skip-g2 --skip-g3  # 只跑G3d
===============================================================================
"""

import argparse
import json
import os
import sys
import time
import subprocess
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# 全局配置
# =============================================================================

DEFAULT_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
PD_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]
ALPHA_VALUES = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
BANDWIDTHS = {
    "100gbit": "100 Gbps (标准RDMA)",
    "25gbit": "25 Gbps (拥塞)",
    "12.5gbit": "12.5 Gbps (低配)",
    "unlimited": "不限速 (理想)",
}
GENERATION_LENGTH_SHORT = 128
GENERATION_LENGTH_LONG = 512
NUM_RUNS_LATENCY = 10
NUM_RUNS_OTHER = 3  # Reduced for faster iteration


# =============================================================================
# 工具函数
# =============================================================================

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def check_gpu() -> Dict:
    """检查GPU环境"""
    ret, out = subprocess.getstatusoutput(
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
    )
    info = {"raw": out.strip()}
    if ret == 0 and out.strip():
        parts = [p.strip() for p in out.strip().split(",")]
        if len(parts) >= 3:
            info["gpu_name"] = parts[0]
            info["gpu_memory"] = parts[1]
            info["driver_version"] = parts[2]
    return info


# =============================================================================
# TAA核心: Transmission-Aware Attention
# =============================================================================

class TAAInjector:
    """将TAA bias注入到模型的attention mask中"""
    
    def __init__(self, model, alpha: float, cost_vector: torch.Tensor,
                 start_layer: int = 0, end_layer: int = -1):
        """
        Args:
            model: HF causal LM model
            alpha: TAA强度参数
            cost_vector: [seq_len] 每个token的传输成本 (0=本地, 1=远端)
            start_layer: 从第几层开始应用TAA (inclusive)
            end_layer: 到第几层结束 (exclusive, -1=到最后一层)
        """
        self.model = model
        self.alpha = alpha
        self.cost_vector = cost_vector
        self.hooks = []
        self.modified_layers = 0
        
        # 计算TAA bias: b_i = -α × tanh((cost_i - μ) / σ)
        if alpha > 0 and cost_vector is not None:
            mu = cost_vector.mean()
            sigma = cost_vector.std()
            if sigma < 1e-8:
                sigma = torch.tensor(1.0, device=cost_vector.device)
            self.bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)  # [seq]
        else:
            self.bias_1d = None
        
        # 确定层范围
        total_layers = len(model.model.layers)
        self.start_layer = start_layer if start_layer >= 0 else total_layers + start_layer
        self.end_layer = end_layer if end_layer > 0 else total_layers
    
    def _make_pre_hook(self, bias_1d: torch.Tensor):
        """创建forward pre-hook, 将TAA bias添加到attention_mask"""
        def pre_hook(module, args, kwargs):
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                mask = kwargs['attention_mask']
                # mask: [batch, 1, seq_q, seq_k]
                # bias: [seq_k] → [1, 1, 1, seq_k]
                seq_k = mask.shape[-1]
                bias = bias_1d[:seq_k].view(1, 1, 1, -1)
                # 创建新mask, 不修改原始mask
                kwargs['attention_mask'] = mask + bias.to(mask.dtype)
            return args, kwargs
        return pre_hook
    
    def install(self):
        """安装hooks"""
        if self.bias_1d is None:
            return 0
        
        for layer_idx in range(self.start_layer, self.end_layer):
            attn = self.model.model.layers[layer_idx].self_attn
            hook = attn.register_forward_pre_hook(
                self._make_pre_hook(self.bias_1d), with_kwargs=True
            )
            self.hooks.append(hook)
            self.modified_layers += 1
        
        return self.modified_layers
    
    def remove(self):
        """移除hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.modified_layers = 0
    
    def measure_overhead_us(self, prompt_inputs) -> float:
        """测量TAA注入的额外开销"""
        # Baseline: 不带TAA的prefill
        self.remove()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = self.model(**prompt_inputs, use_cache=True)
        torch.cuda.synchronize()
        t_baseline = time.perf_counter() - t0
        
        # With TAA
        self.install()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            _ = self.model(**prompt_inputs, use_cache=True)
        torch.cuda.synchronize()
        t_taa = time.perf_counter() - t1
        
        # Overhead
        overhead_us = max(0, (t_taa - t_baseline)) * 1e6
        return overhead_us


# =============================================================================
# 实验基类
# =============================================================================

class Experiment:
    def __init__(self, name: str, output_dir: Path, model_name: str):
        self.name = name
        self.output_dir = output_dir / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.results = []
        self.tokenizer = None
        self.model = None

    def load_model(self, attn_implementation="eager"):
        """加载模型 - 默认使用eager attention以支持attention mask修改"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        log(f"加载模型: {self.model_name} (attn_impl={attn_implementation})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        mem = torch.cuda.memory_allocated() / 1e9
        log(f"模型加载完成, 显存: {mem:.1f} GB")

    def unload_model(self):
        """释放模型"""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        log("模型已释放")

    def generate_prompt(self, target_length: int) -> str:
        """生成指定长度的prompt"""
        base_text = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. """
        tokens = self.tokenizer.encode(base_text)
        while len(tokens) < target_length:
            tokens = tokens + tokens
        tokens = tokens[:target_length]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        text += "\n\nBased on the above text, summarize the key points about the history of artificial intelligence:"
        return text

    def compute_perplexity(self, text: str, max_length: int = 2048) -> Optional[float]:
        """计算perplexity (不带TAA)"""
        try:
            encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
            return torch.exp(neg_log_likelihood).item()
        except Exception as e:
            log(f"PPL计算失败: {e}", "WARN")
            return None

    def compute_perplexity_with_taa(self, text: str, taa_injector: TAAInjector,
                                     max_length: int = 2048) -> Optional[float]:
        """计算perplexity (带TAA)"""
        try:
            encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = encodings.input_ids.to(self.model.device)
            
            # 安装TAA hooks
            taa_injector.install()
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                neg_log_likelihood = outputs.loss
            # 移除hooks
            taa_injector.remove()
            
            return torch.exp(neg_log_likelihood).item()
        except Exception as e:
            taa_injector.remove()  # 确保清理
            log(f"TAA PPL计算失败: {e}", "WARN")
            return None

    def save_results(self):
        """保存实验结果"""
        result_file = self.output_dir / "results.json"
        with open(result_file, "w") as f:
            json.dump({
                "experiment": self.name,
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "results": self.results,
            }, f, indent=2, ensure_ascii=False)
        log(f"结果已保存: {result_file}")

    def run(self):
        raise NotImplementedError


# =============================================================================
# G1: 单卡基线推理
# =============================================================================

class G1Baseline(Experiment):
    def __init__(self, output_dir: Path, model_name: str):
        super().__init__("G1_baseline", output_dir, model_name)

    def run(self):
        log("=" * 60)
        log("G1: 单卡基线推理 - 开始")
        log("=" * 60)
        
        # G1不需要eager attention, 用sdpa更快
        self.load_model(attn_implementation="sdpa")
        
        for ctx_len in CONTEXT_LENGTHS:
            log(f"  上下文长度: {ctx_len}")
            prompt = self.generate_prompt(ctx_len)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            for run_idx in range(NUM_RUNS_LATENCY):
                # TTFT
                torch.cuda.synchronize()
                t_start = time.perf_counter()
                with torch.no_grad():
                    first_out = self.model(**inputs, max_new_tokens=1, use_cache=True)
                    first_token = first_out.logits[:, -1:, :].argmax(dim=-1)
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - t_start) * 1000
                
                # Full generation
                torch.cuda.synchronize()
                t_total_start = time.perf_counter()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=GENERATION_LENGTH_SHORT, do_sample=False
                    )
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - t_total_start) * 1000
                
                tpot = (total_time - ttft) / GENERATION_LENGTH_SHORT
                ppl = self.compute_perplexity(prompt)
                gpu_mem = torch.cuda.max_memory_allocated() / 1e9
                
                result = {
                    "ttft_ms": round(ttft, 2),
                    "tpot_ms": round(tpot, 2),
                    "total_time_ms": round(total_time, 2),
                    "generated_tokens": GENERATION_LENGTH_SHORT,
                    "perplexity": round(ppl, 4) if ppl else None,
                    "gpu_memory_gb": round(gpu_mem, 2),
                    "context_length": ctx_len,
                    "run_index": run_idx,
                }
                self.results.append(result)
                
                if run_idx == 0:
                    log(f"    TTFT={ttft:.1f}ms, TPOT={tpot:.1f}ms, PPL={ppl:.4f}, GPU={gpu_mem:.1f}GB")
                
                del first_out, outputs
                torch.cuda.empty_cache()
        
        self.unload_model()
        self.save_results()
        
        # 汇总
        summary = {}
        for ctx_len in CONTEXT_LENGTHS:
            ttfts = [r["ttft_ms"] for r in self.results if r["context_length"] == ctx_len]
            tpots = [r["tpot_ms"] for r in self.results if r["context_length"] == ctx_len]
            ppls = [r["perplexity"] for r in self.results if r["context_length"] == ctx_len and r["perplexity"]]
            if ttfts:
                summary[ctx_len] = {
                    "ttft_mean": round(np.mean(ttfts), 2),
                    "tpot_mean": round(np.mean(tpots), 2),
                    "ppl_mean": round(np.mean(ppls), 4) if ppls else None,
                }
        log(f"G1汇总: {json.dumps(summary, indent=2)}")
        return summary


# =============================================================================
# G2: PD分离基线
# =============================================================================

class G2PDSeparation(Experiment):
    def __init__(self, output_dir: Path, model_name: str):
        super().__init__("G2_pd_separation", output_dir, model_name)

    def run(self):
        log("=" * 60)
        log("G2: PD分离基线 - 开始")
        log("=" * 60)
        
        self.load_model(attn_implementation="sdpa")
        
        bandwidth_configs = [
            ("100gbit", "100 Gbps"),
            ("25gbit", "25 Gbps"),
            ("unlimited", "不限速"),
        ]
        
        for ctx_len in PD_CONTEXT_LENGTHS:
            prompt = self.generate_prompt(ctx_len)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            for bw_name, bw_desc in bandwidth_configs:
                log(f"  上下文={ctx_len}, 带宽={bw_desc}")
                
                for run_idx in range(NUM_RUNS_LATENCY):
                    # Step 1: Prefill (P节点)
                    torch.cuda.synchronize()
                    t_prefill_start = time.perf_counter()
                    with torch.no_grad():
                        prefill_outputs = self.model(
                            **inputs, use_cache=True, output_hidden_states=False,
                        )
                    torch.cuda.synchronize()
                    prefill_time = (time.perf_counter() - t_prefill_start) * 1000
                    
                    # Step 2: KV序列化 + 传输模拟
                    kv_cache = prefill_outputs.past_key_values
                    kv_size_bytes = sum(
                        t.numel() * t.element_size() 
                        for layer_kv in kv_cache 
                        for t in layer_kv
                        if t is not None
                    )
                    kv_size_mb = kv_size_bytes / (1024 * 1024)
                    
                    # 模拟传输延迟
                    if bw_name != "unlimited":
                        bw_bps = float(bw_name.replace("gbit", "")) * 1e9 / 8
                        transfer_time = (kv_size_bytes / bw_bps) * 1000
                    else:
                        transfer_time = 0.01
                    
                    # Step 3: Decode (D节点)
                    torch.cuda.synchronize()
                    t_decode_start = time.perf_counter()
                    next_token = prefill_outputs.logits[:, -1:, :].argmax(dim=-1)
                    
                    generated_tokens = []
                    for _ in range(GENERATION_LENGTH_SHORT):
                        with torch.no_grad():
                            decode_outputs = self.model(
                                input_ids=next_token,
                                past_key_values=kv_cache,
                                use_cache=True,
                            )
                        kv_cache = decode_outputs.past_key_values
                        next_token = decode_outputs.logits[:, -1:, :].argmax(dim=-1)
                        generated_tokens.append(next_token.item())
                    torch.cuda.synchronize()
                    decode_time = (time.perf_counter() - t_decode_start) * 1000
                    
                    result = {
                        "context_length": ctx_len,
                        "bandwidth": bw_desc,
                        "bandwidth_gbit": bw_name,
                        "prefill_time_ms": round(prefill_time, 2),
                        "kv_size_mb": round(kv_size_mb, 2),
                        "transfer_time_ms": round(transfer_time, 2),
                        "decode_time_ms": round(decode_time, 2),
                        "tpot_ms": round(decode_time / GENERATION_LENGTH_SHORT, 2),
                        "run_index": run_idx,
                    }
                    self.results.append(result)
                    
                    del prefill_outputs, decode_outputs, kv_cache
                    torch.cuda.empty_cache()
        
        self.unload_model()
        self.save_results()
        
        # 汇总
        summary = {}
        for ctx_len in PD_CONTEXT_LENGTHS:
            for bw_name, bw_desc in bandwidth_configs:
                key = f"{ctx_len}_{bw_name}"
                runs = [r for r in self.results 
                        if r["context_length"] == ctx_len and r["bandwidth_gbit"] == bw_name]
                if runs:
                    ttfts = [r["prefill_time_ms"] + r["transfer_time_ms"] for r in runs]
                    transfers = [r["transfer_time_ms"] for r in runs]
                    summary[key] = {
                        "ttft_mean": round(np.mean(ttfts), 2),
                        "transfer_mean": round(np.mean(transfers), 2),
                    }
        log(f"G2汇总: {json.dumps(summary, indent=2)}")
        return summary


# =============================================================================
# G3: TAA核心验证 (v2 - 正确注入TAA)
# =============================================================================

class G3TAA(Experiment):
    def __init__(self, output_dir: Path, model_name: str):
        super().__init__("G3_TAA", output_dir, model_name)

    def build_cost_vector(self, seq_len: int, remote_ratio: float = 0.7) -> torch.Tensor:
        """构建传输成本向量: 前70%为远端(cost=1), 后30%为本地(cost=0)"""
        cost_vector = torch.zeros(seq_len, device=self.model.device)
        remote_end = int(seq_len * remote_ratio)
        cost_vector[:remote_end] = 1.0  # 远端KV cost高
        cost_vector[remote_end:] = 0.0  # 本地KV cost低
        return cost_vector

    def run_with_taa(
        self,
        prompt: str,
        alpha: float,
        max_new_tokens: int = 128,
        layer_config: str = "last_1_3",
    ) -> Dict:
        """使用TAA进行生成 - 通过register_forward_pre_hook注入attention bias"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        seq_len = inputs.input_ids.shape[1]
        
        # 构建cost vector
        cost_vector = self.build_cost_vector(seq_len)
        
        # 确定TAA应用层范围
        total_layers = len(self.model.model.layers)
        if layer_config == "last_1_3":
            start_layer = total_layers * 2 // 3
        elif layer_config == "last_1_2":
            start_layer = total_layers // 2
        elif layer_config == "all_layers":
            start_layer = 0
        else:
            start_layer = total_layers * 2 // 3
        
        # 创建TAA注入器
        taa_injector = TAAInjector(
            self.model, alpha, cost_vector,
            start_layer=start_layer, end_layer=total_layers
        )
        
        # ===== Prefill with TAA =====
        # 先测量baseline prefill (不带TAA)
        torch.cuda.synchronize()
        t_prefill_baseline_start = time.perf_counter()
        with torch.no_grad():
            baseline_out = self.model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        t_prefill_baseline = (time.perf_counter() - t_prefill_baseline_start) * 1000
        
        # 清理
        del baseline_out
        torch.cuda.empty_cache()
        
        # 带TAA的prefill
        taa_injector.install()
        modified_layers = taa_injector.modified_layers
        
        torch.cuda.synchronize()
        t_taa_start = time.perf_counter()
        with torch.no_grad():
            prefill_out = self.model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        t_prefill_taa = (time.perf_counter() - t_taa_start) * 1000
        
        taa_overhead_us = max(0, (t_prefill_taa - t_prefill_baseline)) * 1000  # ms → us would be *1000, but overhead in ms
        taa_overhead_us = max(0, (t_prefill_taa - t_prefill_baseline)) * 1e6  # s → us
        
        # ===== Decode (不带TAA, 因为TAA优化的是prefill阶段的KV传输) =====
        taa_injector.remove()
        
        torch.cuda.synchronize()
        t_decode_start = time.perf_counter()
        kv_cache = prefill_out.past_key_values
        next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)
        
        generated_tokens = []
        for step in range(max_new_tokens):
            with torch.no_grad():
                decode_out = self.model(
                    input_ids=next_token,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache = decode_out.past_key_values
            next_token = decode_out.logits[:, -1:, :].argmax(dim=-1)
            generated_tokens.append(next_token.item())
        
        torch.cuda.synchronize()
        t_decode = (time.perf_counter() - t_decode_start) * 1000
        
        # ===== PPL: with TAA vs without =====
        ppl_no_taa = self.compute_perplexity(prompt)
        
        # 重建cost_vector用于PPL计算 (PPL截断到max_length)
        ppl_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        ppl_seq_len = ppl_inputs.input_ids.shape[1]
        cost_vector_ppl = self.build_cost_vector(ppl_seq_len)
        taa_injector_ppl = TAAInjector(
            self.model, alpha, cost_vector_ppl,
            start_layer=start_layer, end_layer=total_layers
        )
        ppl_with_taa = self.compute_perplexity_with_taa(prompt, taa_injector_ppl)
        
        # 生成文本样本
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        result = {
            "alpha": alpha,
            "prefill_baseline_ms": round(t_prefill_baseline, 2),
            "prefill_taa_ms": round(t_prefill_taa, 2),
            "taa_overhead_us": round(taa_overhead_us, 1),
            "modified_layers": modified_layers,
            "start_layer": start_layer,
            "total_layers": total_layers,
            "decode_time_ms": round(t_decode, 2),
            "tpot_ms": round(t_decode / max_new_tokens, 2),
            "perplexity_no_taa": round(ppl_no_taa, 4) if ppl_no_taa else None,
            "perplexity_with_taa": round(ppl_with_taa, 4) if ppl_with_taa else None,
            "ppl_delta": round(ppl_with_taa - ppl_no_taa, 4) if (ppl_with_taa and ppl_no_taa) else None,
            "generated_text_sample": generated_text[:200],
        }
        
        del kv_cache, prefill_out, decode_out
        torch.cuda.empty_cache()
        
        return result

    def run(self):
        log("=" * 60)
        log("G3: Transmission-Aware Attention 核心验证 v2 - 开始")
        log("=" * 60)
        
        self.load_model(attn_implementation="eager")
        
        # G3a: α参数扫描
        log("--- G3a: α参数扫描 ---")
        for ctx_len in [8192, 32768]:
            prompt = self.generate_prompt(ctx_len)
            log(f"  上下文={ctx_len}")
            
            for alpha in ALPHA_VALUES:
                log(f"    α={alpha}")
                for run_idx in range(NUM_RUNS_OTHER):
                    result = self.run_with_taa(prompt, alpha, max_new_tokens=GENERATION_LENGTH_SHORT)
                    result["context_length"] = ctx_len
                    result["run_index"] = run_idx
                    result["sub_experiment"] = "G3a_alpha_scan"
                    self.results.append(result)
                
                # 报告
                last_runs = [r for r in self.results 
                            if r["alpha"] == alpha and r["context_length"] == ctx_len 
                            and r["sub_experiment"] == "G3a_alpha_scan"]
                if last_runs:
                    r = last_runs[-1]
                    log(f"      PPL(no_TAA)={r['perplexity_no_taa']}, PPL(TAA)={r['perplexity_with_taa']}, "
                        f"ΔPPL={r['ppl_delta']}, overhead={r['taa_overhead_us']:.0f}μs")
        
        # G3b: 带宽×TAA矩阵
        log("--- G3b: 带宽×TAA矩阵 ---")
        for ctx_len in [8192, 32768]:
            prompt = self.generate_prompt(ctx_len)
            for bw_name, bw_desc in [("25gbit", "25 Gbps(拥塞)"), ("100gbit", "100 Gbps(标准)"), ("unlimited", "不限速")]:
                for alpha in [0.0, 0.1, 0.15]:
                    log(f"  上下文={ctx_len}, 带宽={bw_desc}, α={alpha}")
                    for run_idx in range(NUM_RUNS_OTHER):
                        result = self.run_with_taa(prompt, alpha, max_new_tokens=GENERATION_LENGTH_SHORT)
                        result["context_length"] = ctx_len
                        result["bandwidth"] = bw_desc
                        result["run_index"] = run_idx
                        result["sub_experiment"] = "G3b_bw_matrix"
                        self.results.append(result)
        
        # G3c: TAA质量影响 (长生成)
        log("--- G3c: TAA质量影响 (512 tokens) ---")
        ctx_len = 32768
        prompt = self.generate_prompt(ctx_len)
        for alpha in ALPHA_VALUES:
            log(f"  α={alpha}, 上下文=32K, 生成512tokens")
            for run_idx in range(3):
                result = self.run_with_taa(prompt, alpha, max_new_tokens=GENERATION_LENGTH_LONG)
                result["context_length"] = ctx_len
                result["run_index"] = run_idx
                result["sub_experiment"] = "G3c_quality"
                self.results.append(result)
        
        self.unload_model()
        self.save_results()
        
        # 汇总
        summary = {"G3c_ppl_vs_alpha": {}}
        for alpha in ALPHA_VALUES:
            ppls_no = [r["perplexity_no_taa"] for r in self.results 
                      if r["alpha"] == alpha and r.get("sub_experiment") == "G3c_quality" 
                      and r.get("perplexity_no_taa")]
            ppls_with = [r["perplexity_with_taa"] for r in self.results 
                        if r["alpha"] == alpha and r.get("sub_experiment") == "G3c_quality" 
                        and r.get("perplexity_with_taa")]
            if ppls_no and ppls_with:
                summary["G3c_ppl_vs_alpha"][str(alpha)] = {
                    "ppl_no_taa": round(np.mean(ppls_no), 4),
                    "ppl_with_taa": round(np.mean(ppls_with), 4),
                    "delta": round(np.mean(ppls_with) - np.mean(ppls_no), 4),
                }
        log(f"G3汇总(PPL vs α): {json.dumps(summary, indent=2)}")
        return summary


# =============================================================================
# G3d: Layer Sensitivity (v2 - 正确对比before/after)
# =============================================================================

class G3dLayerSensitivity(Experiment):
    def __init__(self, output_dir: Path, model_name: str):
        super().__init__("G3d_layer_sensitivity", output_dir, model_name)

    def run(self):
        log("=" * 60)
        log("G3d: TAA Layer Sensitivity v2 - 开始")
        log("=" * 60)
        
        self.load_model(attn_implementation="eager")
        
        ctx_len = 32768
        alpha = 0.1
        prompt = self.generate_prompt(ctx_len)
        
        layer_configs = {
            "all_layers": (0, -1),         # 所有层
            "last_1_3": (-1/3, -1),        # 后1/3 (default)
            "last_1_2": (-1/2, -1),        # 后1/2
        }
        
        # Baseline PPL (不带TAA)
        ppl_baseline = self.compute_perplexity(prompt)
        log(f"  Baseline PPL (no TAA): {ppl_baseline:.4f}")
        
        for config_name, (start_ratio, end_idx) in layer_configs.items():
            total_layers = len(self.model.model.layers)
            if isinstance(start_ratio, float) and start_ratio < 0:
                start_layer = total_layers + int(total_layers * start_ratio)
            else:
                start_layer = int(total_layers * start_ratio)
            end_layer = total_layers if end_idx == -1 else end_idx
            
            log(f"  配置: {config_name}, layers={start_layer}-{end_layer-1}")
            
            # 构建cost vector
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            seq_len = inputs.input_ids.shape[1]
            cost_vector = torch.zeros(seq_len, device=self.model.device)
            remote_end = int(seq_len * 0.7)
            cost_vector[:remote_end] = 1.0
            
            for run_idx in range(3):
                # PPL with TAA (via proper injection)
                taa_injector = TAAInjector(
                    self.model, alpha, cost_vector,
                    start_layer=start_layer, end_layer=end_layer
                )
                ppl_with_taa = self.compute_perplexity_with_taa(prompt, taa_injector)
                
                # 生成质量对比
                taa_injector.install()
                torch.cuda.synchronize()
                with torch.no_grad():
                    prefill_out = self.model(**inputs, use_cache=True)
                taa_injector.remove()
                
                # Decode without TAA
                kv_cache = prefill_out.past_key_values
                next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)
                gen_tokens = []
                for _ in range(128):
                    with torch.no_grad():
                        decode_out = self.model(
                            input_ids=next_token,
                            past_key_values=kv_cache,
                            use_cache=True,
                        )
                    kv_cache = decode_out.past_key_values
                    next_token = decode_out.logits[:, -1:, :].argmax(dim=-1)
                    gen_tokens.append(next_token.item())
                
                gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                result = {
                    "config": config_name,
                    "alpha": alpha,
                    "context_length": ctx_len,
                    "ppl_no_taa": round(ppl_baseline, 4) if ppl_baseline else None,
                    "ppl_with_taa": round(ppl_with_taa, 4) if ppl_with_taa else None,
                    "ppl_delta": round(ppl_with_taa - ppl_baseline, 4) if (ppl_with_taa and ppl_baseline) else None,
                    "start_layer": start_layer,
                    "total_layers": total_layers,
                    "modified_layers": end_layer - start_layer,
                    "generated_text_sample": gen_text[:200],
                    "run_index": run_idx,
                }
                self.results.append(result)
                log(f"    PPL: {ppl_baseline:.4f} → {ppl_with_taa:.4f} (Δ={ppl_with_taa-ppl_baseline:+.4f}), "
                    f"layers={start_layer}-{end_layer-1}")
                
                del prefill_out, decode_out, kv_cache
                torch.cuda.empty_cache()
        
        self.unload_model()
        self.save_results()
        
        log("G3d 完成 ✅")
        return self.results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="kvcache-lab GPU实验一键运行 v2")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称/路径")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/experiment_results", help="结果输出目录")
    parser.add_argument("--phase", default="A", choices=["A", "B", "all"], help="实验阶段")
    parser.add_argument("--skip-g1", action="store_true", help="跳过G1(已跑过)")
    parser.add_argument("--skip-g2", action="store_true", help="跳过G2(已跑过)")
    parser.add_argument("--skip-g3", action="store_true", help="跳过G3(已跑过)")
    parser.add_argument("--only-g3d", action="store_true", help="只跑G3d")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 环境检查
    log("=" * 60)
    log("环境检查")
    log("=" * 60)
    gpu_info = check_gpu()
    log(f"GPU: {gpu_info.get('gpu_name', 'Unknown')}, Memory: {gpu_info.get('gpu_memory', 'Unknown')}")
    log(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    
    all_results = {}
    
    try:
        if args.only_g3d:
            g3d = G3dLayerSensitivity(output_dir, args.model)
            all_results["G3d"] = g3d.run()
        elif args.phase in ["A", "all"]:
            if not args.skip_g1:
                g1 = G1Baseline(output_dir, args.model)
                all_results["G1"] = g1.run()
            
            if not args.skip_g2:
                g2 = G2PDSeparation(output_dir, args.model)
                all_results["G2"] = g2.run()
            
            if not args.skip_g3:
                g3 = G3TAA(output_dir, args.model)
                all_results["G3"] = g3.run()
            
            g3d = G3dLayerSensitivity(output_dir, args.model)
            all_results["G3d"] = g3d.run()
        
        if args.phase in ["B", "all"]:
            log("Phase B待G3验证通过后执行")
    
    except Exception as e:
        log(f"实验异常: {e}", "ERROR")
        traceback.print_exc()
    
    # 保存全局结果
    global_result_file = output_dir / "all_results.json"
    with open(global_result_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    
    log("=" * 60)
    log("全部实验完成!")
    log(f"结果目录: {output_dir}")
    log("=" * 60)
    print("EXPERIMENT_ALL_DONE")


if __name__ == "__main__":
    main()
