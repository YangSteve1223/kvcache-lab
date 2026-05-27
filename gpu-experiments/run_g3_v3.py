#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU实验脚本 v3 - TAA真实注入验证

策略:
    - PPL + Attention分析: eager attention + 2048 tokens (避免OOM)
    - 延迟测量: SDPA + 完整上下文 (TAA overhead单独测)
    - 核心验证: TAA是否真的将注意力从远端KV转移到本地KV

修复记录:
    v1: TAA未真正注入(post-hoc修改attention, 不影响模型输出)
    v2: eager attention OOM (32K序列注意力矩阵55GB)
    v3: 分离短序列PPL验证和长序列延迟测量
===============================================================================
"""

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# 全局配置
# =============================================================================

DEFAULT_MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
CONTEXT_LENGTHS = [1024, 4096, 8192, 16384, 32768]
PD_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768]
ALPHA_VALUES = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
PPL_SEQ_LENGTH = 2048  # PPL验证用短序列, 避免eager OOM
NUM_RUNS = 3


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


# =============================================================================
# TAA注入器: 通过register_forward_pre_hook修改attention_mask
# =============================================================================

class TAAInjector:
    """将TAA bias注入到模型的attention mask中
    
    TAA公式: b_i = -α × tanh((cost_i - μ) / σ)
    修改后的attention: score_i = QK^T_i + mask_i + b_i
    """
    
    def __init__(self, model, alpha: float, cost_vector: torch.Tensor,
                 start_layer: int = 0, end_layer: int = -1):
        self.model = model
        self.alpha = alpha
        self.hooks = []
        self.modified_layers = 0
        
        # 计算TAA bias
        if alpha > 0 and cost_vector is not None:
            mu = cost_vector.mean()
            sigma = cost_vector.std()
            if sigma < 1e-8:
                sigma = torch.tensor(1.0, device=cost_vector.device)
            self.bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        else:
            self.bias_1d = None
        
        total_layers = len(model.model.layers)
        self.start_layer = start_layer if start_layer >= 0 else total_layers + start_layer
        self.end_layer = end_layer if end_layer > 0 else total_layers
    
    def _make_pre_hook(self, bias_1d: torch.Tensor):
        """创建forward pre-hook, 将TAA bias添加到attention_mask"""
        def pre_hook(module, args, kwargs):
            if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
                mask = kwargs['attention_mask']
                seq_k = mask.shape[-1]
                bias = bias_1d[:seq_k]
                
                if mask.dim() == 4:
                    # [batch, 1, seq_q, seq_k]
                    bias_expanded = bias.view(1, 1, 1, -1).to(mask.dtype)
                elif mask.dim() == 2:
                    # [batch, seq_k]
                    bias_expanded = bias.view(1, -1).to(mask.dtype)
                else:
                    return args, kwargs
                
                kwargs['attention_mask'] = mask + bias_expanded
            return args, kwargs
        return pre_hook
    
    def install(self):
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
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.modified_layers = 0


# =============================================================================
# 实验G3_v3: TAA核心验证 (PPL + Attention分析 + 延迟)
# =============================================================================

def run_g3_taa_validation(model_path: str, output_dir: Path):
    """
    G3核心验证:
    1. PPL对比: 用eager attention + 短序列(2048), 真正注入TAA
    2. Attention分布分析: TAA是否把注意力从远端KV转移到本地KV
    3. 延迟对比: 用SDPA + 长序列, TAA overhead单独测
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    results = {"experiment": "G3_TAA_v3", "results": []}
    
    # =========================================================================
    # Part 1: PPL + Attention分析 (eager, 短序列)
    # =========================================================================
    log("=" * 60)
    log("Part 1: PPL + Attention分析 (eager, seq=2048)")
    log("=" * 60)
    
    log(f"加载模型: {model_path} (attn_impl=eager)")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    log(f"模型加载完成, 显存: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    total_layers = len(model.model.layers)
    default_start_layer = total_layers * 2 // 3  # 后1/3层
    
    # 生成短序列prompt
    base_text = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. """
    tokens = tokenizer.encode(base_text)
    while len(tokens) < PPL_SEQ_LENGTH:
        tokens = tokens + tokens
    tokens = tokens[:PPL_SEQ_LENGTH]
    prompt = tokenizer.decode(tokens, skip_special_tokens=True)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=PPL_SEQ_LENGTH).to(model.device)
    seq_len = inputs.input_ids.shape[1]
    log(f"Prompt长度: {seq_len} tokens")
    
    # 构建cost vector: 前70%远端(cost=1), 后30%本地(cost=0)
    cost_vector = torch.zeros(seq_len, device=model.device)
    remote_end = int(seq_len * 0.7)
    cost_vector[:remote_end] = 1.0
    cost_vector[remote_end:] = 0.0
    log(f"Cost vector: 远端tokens={remote_end}, 本地tokens={seq_len-remote_end}")
    
    # --- α参数扫描 ---
    ppl_results = []
    attn_results = []
    
    for alpha in ALPHA_VALUES:
        log(f"  α={alpha}")
        
        # 1) Baseline PPL (不带TAA)
        with torch.no_grad():
            baseline_out = model(**inputs, use_cache=True, output_attentions=True)
        
        shift_logits = baseline_out.logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        baseline_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        baseline_ppl = torch.exp(baseline_loss).item()
        
        # 2) Baseline attention分布 (后1/3层的平均)
        baseline_local_attn = 0.0
        baseline_remote_attn = 0.0
        attn_count = 0
        for layer_idx, attn in enumerate(baseline_out.attentions):
            if layer_idx >= default_start_layer:
                # attn: [batch, heads, seq_q, seq_k]
                # 取最后一个query token的attention分布
                last_token_attn = attn[0, :, -1, :]  # [heads, seq_k]
                avg_attn = last_token_attn.mean(dim=0)  # [seq_k]
                baseline_local_attn += avg_attn[remote_end:].sum().item()
                baseline_remote_attn += avg_attn[:remote_end].sum().item()
                attn_count += 1
        baseline_local_ratio = baseline_local_attn / (baseline_local_attn + baseline_remote_attn + 1e-10)
        
        del baseline_out
        torch.cuda.empty_cache()
        
        # 3) TAA PPL (带TAA注入)
        taa_injector = TAAInjector(
            model, alpha, cost_vector,
            start_layer=default_start_layer, end_layer=total_layers
        )
        taa_injector.install()
        modified_layers = taa_injector.modified_layers
        
        with torch.no_grad():
            taa_out = model(**inputs, use_cache=True, output_attentions=True)
        
        shift_logits_taa = taa_out.logits[:, :-1, :].contiguous()
        taa_loss = F.cross_entropy(
            shift_logits_taa.view(-1, shift_logits_taa.size(-1)),
            shift_labels.view(-1)
        )
        taa_ppl = torch.exp(taa_loss).item()
        
        # 4) TAA attention分布
        taa_local_attn = 0.0
        taa_remote_attn = 0.0
        for layer_idx, attn in enumerate(taa_out.attentions):
            if layer_idx >= default_start_layer:
                last_token_attn = attn[0, :, -1, :]
                avg_attn = last_token_attn.mean(dim=0)
                taa_local_attn += avg_attn[remote_end:].sum().item()
                taa_remote_attn += avg_attn[:remote_end].sum().item()
        taa_local_ratio = taa_local_attn / (taa_local_attn + taa_remote_attn + 1e-10)
        
        taa_injector.remove()
        del taa_out, shift_logits_taa
        torch.cuda.empty_cache()
        
        # 结果
        ppl_delta = taa_ppl - baseline_ppl
        attn_shift = taa_local_ratio - baseline_local_ratio
        
        result = {
            "alpha": alpha,
            "seq_len": seq_len,
            "modified_layers": modified_layers,
            "baseline_ppl": round(baseline_ppl, 4),
            "taa_ppl": round(taa_ppl, 4),
            "ppl_delta": round(ppl_delta, 4),
            "ppl_delta_pct": round(ppl_delta / baseline_ppl * 100, 2),
            "baseline_local_attn_ratio": round(baseline_local_ratio, 4),
            "taa_local_attn_ratio": round(taa_local_ratio, 4),
            "attn_shift": round(attn_shift, 4),
            "attn_shift_pct": round(attn_shift / baseline_local_ratio * 100, 2) if baseline_local_ratio > 0 else 0,
        }
        ppl_results.append(result)
        results["results"].append(result)
        
        log(f"    PPL: {baseline_ppl:.4f} → {taa_ppl:.4f} (Δ={ppl_delta:+.4f}, {ppl_delta/baseline_ppl*100:+.2f}%)")
        log(f"    Local attn: {baseline_local_ratio:.4f} → {taa_local_ratio:.4f} (shift={attn_shift:+.4f}, {attn_shift/baseline_local_ratio*100:+.1f}%)")
    
    # 释放eager模型
    del model, tokenizer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    log("Part 1完成, 释放eager模型")
    
    # =========================================================================
    # Part 2: Layer Sensitivity (eager, 短序列)
    # =========================================================================
    log("=" * 60)
    log("Part 2: Layer Sensitivity (eager, seq=2048)")
    log("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    
    alpha = 0.1  # 固定α
    layer_configs = {
        "all_layers": (0, -1),
        "last_1_2": (-1/2, -1),
        "last_1_3": (-1/3, -1),   # default
        "last_1_4": (-1/4, -1),
    }
    
    # Baseline PPL
    with torch.no_grad():
        baseline_logits = model(**inputs).logits
    shift_logits = baseline_logits[:, :-1, :].contiguous()
    shift_labels = inputs.input_ids[:, 1:].contiguous()
    baseline_ppl = torch.exp(F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )).item()
    del baseline_logits
    torch.cuda.empty_cache()
    log(f"  Baseline PPL: {baseline_ppl:.4f}")
    
    layer_results = []
    for config_name, (start_ratio, end_idx) in layer_configs.items():
        total_layers_l = len(model.model.layers)
        if isinstance(start_ratio, float) and start_ratio < 0:
            start_layer = total_layers_l + int(total_layers_l * start_ratio)
        else:
            start_layer = int(total_layers_l * start_ratio)
        end_layer = total_layers_l if end_idx == -1 else end_idx
        
        taa_injector = TAAInjector(
            model, alpha, cost_vector,
            start_layer=start_layer, end_layer=end_layer
        )
        taa_injector.install()
        
        with torch.no_grad():
            taa_logits = model(**inputs).logits
        taa_ppl = torch.exp(F.cross_entropy(
            taa_logits[:, :-1, :].contiguous().view(-1, taa_logits.size(-1)),
            shift_labels.view(-1)
        )).item()
        
        taa_injector.remove()
        del taa_logits
        torch.cuda.empty_cache()
        
        ppl_delta = taa_ppl - baseline_ppl
        result = {
            "config": config_name,
            "alpha": alpha,
            "start_layer": start_layer,
            "end_layer": end_layer - 1,
            "modified_layers": end_layer - start_layer,
            "baseline_ppl": round(baseline_ppl, 4),
            "taa_ppl": round(taa_ppl, 4),
            "ppl_delta": round(ppl_delta, 4),
            "ppl_delta_pct": round(ppl_delta / baseline_ppl * 100, 2),
        }
        layer_results.append(result)
        results["results"].append(result)
        log(f"    {config_name}: PPL {baseline_ppl:.4f} → {taa_ppl:.4f} (Δ={ppl_delta:+.4f}), layers={start_layer}-{end_layer-1}")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    log("Part 2完成")
    
    # =========================================================================
    # Part 3: 延迟 + TAA overhead (SDPA, 长序列)
    # =========================================================================
    log("=" * 60)
    log("Part 3: 延迟 + TAA overhead (SDPA, 长序列)")
    log("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    log(f"SDPA模型加载完成, 显存: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    
    latency_results = []
    
    for ctx_len in [8192, 32768]:
        # 生成长序列prompt
        base_text_long = """The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. """
        tokens_long = tokenizer.encode(base_text_long)
        while len(tokens_long) < ctx_len:
            tokens_long = tokens_long + tokens_long
        tokens_long = tokens_long[:ctx_len]
        prompt_long = tokenizer.decode(tokens_long, skip_special_tokens=True)
        inputs_long = tokenizer(prompt_long, return_tensors="pt").to(model.device)
        actual_len = inputs_long.input_ids.shape[1]
        
        log(f"  上下文={actual_len} tokens")
        
        # Baseline prefill (SDPA, 不带TAA)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**inputs_long, use_cache=True)
        torch.cuda.synchronize()
        baseline_ttft = (time.perf_counter() - t0) * 1000
        
        # TAA overhead (只测bias计算)
        cost_vec_long = torch.zeros(actual_len, device=model.device)
        cost_vec_long[:int(actual_len * 0.7)] = 1.0
        alpha_test = 0.1
        mu = cost_vec_long.mean()
        sigma = cost_vec_long.std()
        bias_1d = -alpha_test * torch.tanh((cost_vec_long - mu) / sigma)
        
        torch.cuda.synchronize()
        t_hook = time.perf_counter()
        for _ in range(default_start_layer):  # 后1/3层
            dummy = torch.zeros(1, 1, 1, actual_len, dtype=torch.float16, device=model.device)
            _ = dummy + bias_1d.view(1, 1, 1, -1).to(torch.float16)
        torch.cuda.synchronize()
        taa_overhead_us = (time.perf_counter() - t_hook) * 1e6
        
        result = {
            "sub_experiment": "latency",
            "context_length": actual_len,
            "baseline_ttft_ms": round(baseline_ttft, 2),
            "taa_overhead_us": round(taa_overhead_us, 1),
            "overhead_vs_ttft_pct": round(taa_overhead_us / (baseline_ttft * 1000) * 100, 4),
        }
        latency_results.append(result)
        results["results"].append(result)
        log(f"    Baseline TTFT={baseline_ttft:.1f}ms, TAA overhead={taa_overhead_us:.0f}μs "
            f"({taa_overhead_us/(baseline_ttft*1000)*100:.3f}%)")
        
        del inputs_long
        torch.cuda.empty_cache()
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "G3_TAA_v3_results.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"结果已保存: {result_file}")
    
    # 汇总
    log("=" * 60)
    log("G3 v3 汇总")
    log("=" * 60)
    log("Part 1 - PPL vs α:")
    for r in ppl_results:
        log(f"  α={r['alpha']}: PPL Δ={r['ppl_delta']:+.4f} ({r['ppl_delta_pct']:+.2f}%), "
            f"Local attn shift={r['attn_shift']:+.4f} ({r['attn_shift_pct']:+.1f}%)")
    log("Part 2 - Layer Sensitivity:")
    for r in layer_results:
        log(f"  {r['config']}: PPL Δ={r['ppl_delta']:+.4f} ({r['ppl_delta_pct']:+.2f}%)")
    log("Part 3 - Latency:")
    for r in latency_results:
        log(f"  ctx={r['context_length']}: TTFT={r['baseline_ttft_ms']:.1f}ms, "
            f"overhead={r['taa_overhead_us']:.0f}μs ({r['overhead_vs_ttft_pct']:.3f}%)")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="kvcache-lab G3 TAA v3验证")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/experiment_results_v3")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 环境检查
    log("=" * 60)
    log("环境检查")
    log("=" * 60)
    log(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        run_g3_taa_validation(args.model, output_dir)
    except Exception as e:
        log(f"实验异常: {e}", "ERROR")
        traceback.print_exc()
    
    log("EXPERIMENT_ALL_DONE")


if __name__ == "__main__":
    main()
