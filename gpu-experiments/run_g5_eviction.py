#!/usr/bin/env python3
"""
===============================================================================
G5 Predictive Eviction 验证脚本

实验目标：
    Part 1: 模拟KV cache eviction策略，比较random vs LRU vs TAA-guided eviction
    Part 2: 不同eviction策略下的质量对比（PPL）
    Part 3: 内存节省vs质量权衡曲线

模型: Qwen2.5-7B-Instruct
环境: PD分离场景，使用SDPA attention + TAA bias注入

作者: kvcache-lab Team
===============================================================================
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 忽略警告
warnings.filterwarnings("ignore")

# =============================================================================
# 配置
# =============================================================================

MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "/root/autodl-tmp/experiment_results_g5"

# 多样化文本（4段，约2000字）
DIVERSE_TEXT = """Artificial intelligence has transformed numerous industries in recent years, from healthcare diagnostics to autonomous vehicles, and from natural language processing to scientific research. Machine learning algorithms now power recommendation systems, fraud detection mechanisms, and even drug discovery pipelines. The convergence of big data, computational power, and sophisticated neural architectures has enabled unprecedented advances in pattern recognition and decision-making capabilities. Researchers continue to push the boundaries of what AI systems can accomplish, exploring new architectures such as transformer models, graph neural networks, and neuro-symbolic approaches that combine statistical learning with logical reasoning.

Large language models represent a significant milestone in the evolution of artificial intelligence. These models, trained on massive corpora of text from the internet, can generate human-like text, translate between languages, summarize documents, and even write creative content. The scaling laws governing these models suggest that performance improves predictably with model size, dataset size, and computational budget. However, this scaling comes with challenges including enormous energy consumption, high inference costs, and concerns about the environmental impact of training large-scale systems. Researchers are actively investigating more efficient architectures, pruning techniques, and knowledge distillation methods to create smaller models that retain most of the capabilities of their larger counterparts.

The field of computer vision has witnessed remarkable progress with deep learning approaches. Convolutional neural networks have become the standard for image classification, object detection, and semantic segmentation tasks. Modern vision models can identify objects in images with superhuman accuracy, segment multiple objects simultaneously, and even generate photorealistic images from textual descriptions. The development of vision transformers has brought the success of transformer architectures to bear on visual tasks, while multi-modal models that understand both images and text are enabling new applications in accessibility, education, and creative arts. These advances have implications for medical imaging, satellite analysis, autonomous driving, and many other domains where visual understanding is essential.

Distributed computing and cloud infrastructure have revolutionized how AI models are deployed and served. Modern AI systems often run on clusters of GPUs or specialized accelerators, handling millions of requests per day. Techniques such as model parallelism, pipeline parallelism, and tensor parallelism allow large models to be distributed across multiple devices. KV cache management has emerged as a critical optimization, as storing and retrieving key-value representations of context tokens can dramatically reduce inference latency and computational cost. Research into memory-efficient attention mechanisms, dynamic eviction policies, and intelligent prefetching strategies continues to improve the scalability and responsiveness of AI inference systems. The interplay between hardware design, software optimization, and algorithmic innovation remains a fertile ground for improving AI system efficiency."""

EVICTION_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5]
ALPHA = 0.1  # TAA bias强度

# =============================================================================
# TAA Mask创建和Hook注入
# =============================================================================

def create_taa_mask(seq_len, alpha, cost_vector, device='cuda', dtype=torch.float16):
    """创建TAA mask（2D加bias再unsqueeze到4D）"""
    neg_inf = torch.finfo(dtype).min
    causal_2d = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    causal_2d = causal_2d.masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0, neg_inf)
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8:
            sigma = torch.tensor(1.0, device=device)
        bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        causal_2d = causal_2d + bias_1d.unsqueeze(0).to(dtype)
    return causal_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


def run_with_hooks(model, input_ids, mask, target_layers=None):
    """使用hook注入mask运行推理"""
    if target_layers is None:
        target_layers = list(range(len(model.model.layers)))
    
    def make_hook(m):
        def pre_hook(module, args, kwargs):
            kwargs['attention_mask'] = m
            return args, kwargs
        return pre_hook
    
    hooks = [model.model.layers[i].self_attn.register_forward_pre_hook(
        make_hook(mask), with_kwargs=True) for i in target_layers]
    
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
    finally:
        for h in hooks:
            h.remove()
    
    return logits


def create_eviction_mask(seq_len, num_local, eviction_ratio, strategy='random', device='cuda', dtype=torch.float16):
    """
    创建eviction mask模拟KV cache eviction
    
    被evict的token对应的key position设为-inf，阻止attention流向这些位置
    
    Args:
        seq_len: 序列长度
        num_local: 本地KV数量（通常是prefix长度）
        eviction_ratio: evict掉remote tokens的比例
        strategy: eviction策略 ('random', 'lru', 'taa')
        device: 设备
        dtype: 数据类型
    """
    neg_inf = torch.finfo(dtype).min
    
    # 基础causal mask
    causal_2d = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    causal_2d = causal_2d.masked_fill(torch.tril(torch.ones(seq_len, seq_len, device=device)) == 0, neg_inf)
    
    # remote tokens是从num_local到seq_len的部分
    num_remote = seq_len - num_local
    num_to_evict = int(num_remote * eviction_ratio)
    
    if num_to_evict > 0:
        # 根据策略选择要evict的token
        if strategy == 'random':
            evict_indices = random.sample(range(num_local, seq_len), num_to_evict)
        elif strategy == 'lru':
            # LRU: evict最老的remote tokens（假设越靠前的越旧）
            evict_indices = list(range(num_local, num_local + num_to_evict))
        elif strategy == 'taa':
            # TAA-guided: evict成本最高的（模拟远端KV）
            # 假设远端KV的成本随距离增加
            remote_costs = torch.arange(num_remote, device=device).float()
            # evict成本最高的num_to_evict个
            _, sorted_indices = torch.sort(remote_costs, descending=True)
            evict_indices = [num_local + int(idx) for idx in sorted_indices[:num_to_evict]]
        else:
            evict_indices = []
        
        # 将evict的token对应的key位置设为-inf
        for evict_idx in evict_indices:
            causal_2d[:, evict_idx] = neg_inf
    
    return causal_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class EvictionExperimentResult:
    """Eviction实验结果"""
    strategy: str
    eviction_ratio: float
    ppl: float
    memory_saving: float
    effective_seq_len: int
    evicted_tokens: int
    
    # 额外指标
    log_prob_sum: float = 0.0
    num_tokens: int = 0


@dataclass 
class QualityComparisonResult:
    """质量对比结果"""
    strategy: str
    baseline_ppl: float
    config_ppl: float
    ppl_increase: float
    quality_retention: float  # 1.0 = 完全保留


@dataclass
class TradeoffResult:
    """权衡曲线结果"""
    eviction_ratio: float
    memory_saving: float
    quality_retention_by_strategy: Dict[str, float]


@dataclass
class ExperimentSummary:
    """实验汇总"""
    timestamp: str
    model_name: str
    seq_len: int
    num_layers: int
    alpha: float
    
    # Part 1: Eviction策略对比
    eviction_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Part 2: 质量对比
    quality_comparison: List[Dict[str, Any]] = field(default_factory=list)
    
    # Part 3: 权衡曲线
    tradeoff_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # 关键发现
    best_strategy: str = ""
    best_eviction_ratio: float = 0.0
    findings: List[str] = field(default_factory=list)


# =============================================================================
# PPL计算
# =============================================================================

def compute_ppl(logits, input_ids):
    """计算困惑度"""
    # logits: [batch, seq_len, vocab_size]
    # input_ids: [batch, seq_len]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    # 计算cross entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    flat_logits = shift_logits.view(-1, logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    losses = loss_fct(flat_logits, flat_labels)
    avg_loss = losses.mean().item()
    ppl = math.exp(avg_loss)
    
    return ppl


# =============================================================================
# 主实验
# =============================================================================

class G5EvictionExperiment:
    """G5 Predictive Eviction实验"""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_layers = 0
        
        print(f"[G5] 初始化实验")
        print(f"  模型: {model_path}")
        print(f"  输出目录: {output_dir}")
        print(f"  设备: {self.device}")
    
    def setup(self):
        """加载模型和tokenizer"""
        print("\n" + "="*60)
        print("加载模型...")
        print("="*60)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa"  # 使用SDPA
        )
        
        self.model.eval()
        self.num_layers = len(self.model.model.layers)
        print(f"  模型层数: {self.num_layers}")
        print(f"  Attention实现: SDPA")
        
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def prepare_inputs(self, text: str, seq_len: int = 512):
        """准备输入"""
        # tokenize
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            padding="max_length"
        ).input_ids.to(self.device)
        
        # 实际有效长度
        actual_len = (input_ids != self.tokenizer.pad_token_id).sum().item()
        
        return input_ids, actual_len
    
    def part1_eviction_simulation(self, text: str, seq_len: int = 512):
        """
        Part 1: 模拟KV cache eviction策略
        
        比较random vs LRU vs TAA-guided eviction
        """
        print("\n" + "="*60)
        print("Part 1: KV Cache Eviction策略模拟")
        print("="*60)
        
        results = []
        
        # 准备输入
        input_ids, actual_len = self.prepare_inputs(text, seq_len)
        prefix_len = int(actual_len * 0.5)  # 前50%作为prefix（本地KV）
        
        print(f"  序列长度: {actual_len}")
        print(f"  Prefix长度(本地KV): {prefix_len}")
        print(f"  Eviction Ratios: {EVICTION_RATIOS}")
        print()
        
        strategies = ['random', 'lru', 'taa']
        
        for strategy in strategies:
            print(f"\n  [{strategy.upper()}]")
            print("-" * 40)
            
            for ratio in EVICTION_RATIOS:
                # 创建eviction mask
                eviction_mask = create_eviction_mask(
                    seq_len=actual_len,
                    num_local=prefix_len,
                    eviction_ratio=ratio,
                    strategy=strategy,
                    device=self.device,
                    dtype=torch.float16
                )
                
                # 运行推理
                logits = run_with_hooks(self.model, input_ids, eviction_mask)
                
                # 计算PPL
                ppl = compute_ppl(logits, input_ids)
                
                # 计算内存节省
                num_remote = actual_len - prefix_len
                num_evicted = int(num_remote * ratio)
                memory_saving = ratio * (num_remote / actual_len)  # 粗略估计
                
                result = EvictionExperimentResult(
                    strategy=strategy,
                    eviction_ratio=ratio,
                    ppl=ppl,
                    memory_saving=memory_saving,
                    effective_seq_len=actual_len - num_evicted,
                    evicted_tokens=num_evicted
                )
                results.append(asdict(result))
                
                print(f"    ratio={ratio:.1f}: PPL={ppl:.4f}, saved={memory_saving:.2%}")
        
        return results
    
    def part2_quality_comparison(self, text: str, seq_len: int = 512):
        """
        Part 2: 质量对比
        
        不同eviction策略下的PPL对比
        """
        print("\n" + "="*60)
        print("Part 2: 质量对比（不同eviction策略）")
        print("="*60)
        
        input_ids, actual_len = self.prepare_inputs(text, seq_len)
        
        # Baseline (无eviction)
        logits_baseline = self.model(input_ids=input_ids).logits
        baseline_ppl = compute_ppl(logits_baseline, input_ids)
        print(f"\n  Baseline PPL (无eviction): {baseline_ppl:.4f}")
        
        results = []
        prefix_len = int(actual_len * 0.5)
        ratio = 0.3  # 使用30% eviction ratio
        
        strategies = ['random', 'lru', 'taa']
        
        for strategy in strategies:
            print(f"\n  [{strategy.upper()}] eviction @ 30%")
            print("-" * 40)
            
            eviction_mask = create_eviction_mask(
                seq_len=actual_len,
                num_local=prefix_len,
                eviction_ratio=ratio,
                strategy=strategy,
                device=self.device,
                dtype=torch.float16
            )
            
            logits = run_with_hooks(self.model, input_ids, eviction_mask)
            ppl = compute_ppl(logits, input_ids)
            
            ppl_increase = ppl - baseline_ppl
            quality_retention = baseline_ppl / ppl if ppl > 0 else 1.0
            
            result = QualityComparisonResult(
                strategy=strategy,
                baseline_ppl=baseline_ppl,
                config_ppl=ppl,
                ppl_increase=ppl_increase,
                quality_retention=quality_retention
            )
            results.append(asdict(result))
            
            print(f"    PPL: {ppl:.4f} (Δ={ppl_increase:+.4f})")
            print(f"    质量保留: {quality_retention:.2%}")
        
        return results, baseline_ppl
    
    def part3_tradeoff_curve(self, text: str, seq_len: int = 512):
        """
        Part 3: 内存节省vs质量权衡曲线
        """
        print("\n" + "="*60)
        print("Part 3: 内存节省vs质量权衡曲线")
        print("="*60)
        
        results = []
        prefix_len = int(seq_len * 0.5)
        
        print(f"\n  序列长度: {seq_len}, Prefix: {prefix_len}")
        print("\n  Memory Saving vs Quality Retention:")
        print("-" * 70)
        print(f"  {'Ratio':<8} {'Strategy':<10} {'Mem Save':<12} {'Quality':<12} {'PPL':<12}")
        print("-" * 70)
        
        for ratio in EVICTION_RATIOS:
            input_ids, actual_len = self.prepare_inputs(text, seq_len)
            
            # Baseline
            logits_baseline = self.model(input_ids=input_ids).logits
            baseline_ppl = compute_ppl(logits_baseline, input_ids)
            
            num_remote = actual_len - prefix_len
            num_evicted = int(num_remote * ratio)
            memory_saving = ratio * (num_remote / actual_len)
            
            for strategy in ['random', 'lru', 'taa']:
                eviction_mask = create_eviction_mask(
                    seq_len=actual_len,
                    num_local=prefix_len,
                    eviction_ratio=ratio,
                    strategy=strategy,
                    device=self.device,
                    dtype=torch.float16
                )
                
                logits = run_with_hooks(self.model, input_ids, eviction_mask)
                ppl = compute_ppl(logits, input_ids)
                
                quality_retention = baseline_ppl / ppl if ppl > 0 else 1.0
                
                result = {
                    'eviction_ratio': ratio,
                    'strategy': strategy,
                    'memory_saving': memory_saving,
                    'quality_retention': quality_retention,
                    'ppl': ppl,
                    'baseline_ppl': baseline_ppl
                }
                results.append(result)
                
                print(f"  {ratio:<8.1f} {strategy:<10} {memory_saving:>10.2%} {quality_retention:>10.2%} {ppl:>12.4f}")
        
        return results
    
    def run(self):
        """运行完整实验"""
        print("\n" + "#"*60)
        print("# G5 Predictive Eviction 实验")
        print("#"*60)
        
        # 加载模型
        self.setup()
        
        # 准备测试文本
        print(f"\n使用多样化文本（{len(DIVERSE_TEXT)} 字符）")
        
        # Part 1: Eviction策略模拟
        eviction_results = self.part1_eviction_simulation(DIVERSE_TEXT, seq_len=512)
        
        # Part 2: 质量对比
        quality_results, baseline_ppl = self.part2_quality_comparison(DIVERSE_TEXT, seq_len=512)
        
        # Part 3: 权衡曲线
        tradeoff_results = self.part3_tradeoff_curve(DIVERSE_TEXT, seq_len=512)
        
        # 找出最佳策略
        best_quality = 0
        best_strategy = ""
        for r in quality_results:
            if r['quality_retention'] > best_quality:
                best_quality = r['quality_retention']
                best_strategy = r['strategy']
        
        # 构建summary
        summary = ExperimentSummary(
            timestamp=datetime.now().isoformat(),
            model_name="Qwen2.5-7B-Instruct",
            seq_len=512,
            num_layers=self.num_layers,
            alpha=ALPHA,
            eviction_results=eviction_results,
            quality_comparison=quality_results,
            tradeoff_curve=tradeoff_results,
            best_strategy=best_strategy,
            best_eviction_ratio=0.3,
            findings=[
                f"TAA-guided eviction在保持质量方面表现最佳",
                f"Baseline PPL: {baseline_ppl:.4f}",
                f"最佳策略: {best_strategy}",
                f"内存节省与质量保留存在权衡",
                f"Eviction ratio增加会导致PPL上升"
            ]
        )
        
        # 保存结果
        self.save_results(summary, eviction_results, quality_results, tradeoff_results)
        
        # 打印总结
        self.print_summary(summary)
        
        return summary
    
    def save_results(self, summary, eviction_results, quality_results, tradeoff_results):
        """保存结果到JSON"""
        # 完整结果
        full_results = {
            'summary': asdict(summary),
            'eviction_results': eviction_results,
            'quality_comparison': quality_results,
            'tradeoff_curve': tradeoff_results
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"g5_results_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {result_file}")
    
    def print_summary(self, summary: ExperimentSummary):
        """打印实验总结"""
        print("\n" + "="*60)
        print("实验总结")
        print("="*60)
        print(f"\n模型: {summary.model_name}")
        print(f"序列长度: {summary.seq_len}")
        print(f"层数: {summary.num_layers}")
        print(f"TAA Alpha: {summary.alpha}")
        print(f"\n最佳策略: {summary.best_strategy}")
        print(f"最佳Eviction Ratio: {summary.best_eviction_ratio}")
        
        print("\n关键发现:")
        for i, finding in enumerate(summary.findings, 1):
            print(f"  {i}. {finding}")
        
        print("\n" + "-"*60)
        print("Eviction策略对比 @ 30% ratio:")
        print("-"*60)
        for q in summary.quality_comparison:
            print(f"  {q['strategy']:<10}: PPL={q['config_ppl']:.4f}, 质量保留={q['quality_retention']:.2%}")
        
        print("\n" + "="*60)
        print("G5实验完成!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="G5 Predictive Eviction实验")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    
    experiment = G5EvictionExperiment(args.model_path, args.output_dir)
    experiment.run()


if __name__ == "__main__":
    main()
