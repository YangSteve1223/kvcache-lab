#!/usr/bin/env python3
"""
===============================================================================
KV Access Locality Characterization Experiment

核心目标：
1. 验证KV access呈现长尾/Zipf分布
2. 证明runtime KV tiering的必要性
3. 量化remote attention开销

实验内容：
- Exp1: KV Access Frequency Distribution (CDF曲线)
- Exp2: Remote vs Local Attention Ratio
- Exp3: Active KV Set Size (working set analysis)

作者：kvcache-lab Team
===============================================================================
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import math

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# 配置区域
# =============================================================================

DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen2.5-7B-Instruct"

# 测试文本类型
TEXT_WORKLOADS = {
    "narrative": """The old clock tower stood at the center of the village, its brass bells 
rung every hour on the hour. Generations of villagers had grown up hearing its steady 
tick-tock rhythm, a heartbeat for the community that never missed a beat. Children 
would play in the cobblestone square below, their laughter echoing off the ancient 
stone walls. The watchmaker, Mr. Pemberton, claimed the clock had been running 
continuously for over three hundred years, though no one knew for certain who had 
first wound its intricate mechanisms. Every evening at sunset, the clock would chime 
eight times, signaling the end of another day in the peaceful countryside.""",
    
    "code_python": '''import numpy as np
from typing import List, Dict, Tuple, Optional
import heapq

class BinaryTreeNode:
    def __init__(self, value: int):
        self.value = value
        self.left: Optional[BinaryTreeNode] = None
        self.right: Optional[BinaryTreeNode] = None
    
    def insert(self, value: int) -> None:
        if value < self.value:
            if self.left is None:
                self.left = BinaryTreeNode(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTreeNode(value)
            else:
                self.right.insert(value)
    
    def inorder_traversal(self, result: List[int]) -> None:
        if self.left:
            self.left.inorder_traversal(result)
        result.append(self.value)
        if self.right:
            self.right.inorder_traversal(result)

def binary_search(arr: List[int], target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def merge_sorted_arrays(arrays: List[List[int]]) -> List[int]:
    heap = [(arr[0], i, 0) for i, arr in enumerate(arrays) if arr]
    heapq.heapify(heap)
    result = []
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    return result''',
    
    "qa_format": """Question: What are the main benefits of using attention mechanisms in neural networks?

Answer: Attention mechanisms provide several key benefits in neural networks. First, they enable 
models to selectively focus on relevant parts of the input when processing each output element, 
allowing for better handling of long-range dependencies. Second, they provide interpretability by 
showing which input elements the model considers most important for each prediction. Third, they 
allow parallel computation during training, making them more efficient than purely sequential 
architectures. Fourth, attention can dynamically adapt to different inputs, providing flexibility 
across varied tasks.

Question: How does transformer architecture handle variable-length sequences?

Answer: The transformer architecture handles variable-length sequences through the use of positional 
encodings and self-attention mechanisms. Positional encodings are added to input embeddings to provide 
information about the position of each token, allowing the model to distinguish between tokens at 
different positions. The self-attention mechanism naturally handles variable lengths because it 
computes relationships between all pairs of positions, regardless of sequence length. Additionally, 
padding masks are used to prevent attention to padding tokens, ensuring consistent processing.""",
    
    "summarization": """The conference on artificial intelligence ethics brought together researchers, 
philosophers, and industry leaders from around the world. The three-day event featured keynote 
speeches, panel discussions, and workshop sessions covering topics ranging from algorithmic bias 
to the societal implications of autonomous systems. Notable speakers included Dr. Sarah Chen, 
who presented her research on fairness in machine learning models, and Professor Michael Okonkwo, 
who discussed the philosophical foundations of AI decision-making. The conference concluded 
with a panel on regulatory frameworks, where participants debated the balance between innovation 
and safety in AI development. Attendees expressed optimism about the progress being made in 
responsible AI research while acknowledging the challenges that remain ahead.""",
    
    "repetitive": """The quick brown fox jumps over the lazy dog. A quick brown dog watches the fox 
with interest. The lazy fox observes the quick brown environment. Brown foxes are known for 
their quick movements. The dog barks at the quick shadow. Jump jump jump goes the fox. Quick 
quick quick is the pace. Brown spots mark the fox fur. Dog runs after the brown ball. Jump 
over the fence says the dog. The fox jumps higher. Quick movements confuse the dog. Brown 
eyes focus on the target. Dog lies in the grass. Fox finds a spot to rest. Jump quickly 
says the instinct. Brown feathers fall nearby. Dog sniffs the air. Fox ears perk up quickly."""
}

CONTEXT_LENGTHS = [512, 1024, 2048, 4096]


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class LocalityStats:
    """单次实验的locality统计数据"""
    timestamp: str
    workload_type: str
    context_length: int
    prompt_tokens: int
    
    # Attention mass distribution per layer
    layer_attention_mass: List[np.ndarray]  # [num_layers, seq_len]
    
    # Aggregate statistics
    cdf_x: np.ndarray  # KV rank (0 to 1)
    cdf_y: np.ndarray  # Cumulative attention mass (0 to 1)
    
    # Zipf analysis
    zipf_alpha: float  # Power-law exponent
    zipf_r_squared: float  # Goodness of fit
    
    # Locality metrics
    gini_coefficient: float
    top_20_pct_mass: float  # Mass in top 20% of KV tokens
    top_10_pct_mass: float
    top_5_pct_mass: float
    
    # Remote attention analysis (假设70% remote)
    remote_attention_pct_70: float
    remote_attention_pct_80: float
    remote_attention_pct_90: float
    
    # Active set size (80% coverage)
    active_set_size_80: int
    active_set_ratio_80: float


@dataclass
class ExperimentResult:
    """完整实验结果"""
    timestamp: str
    model_name: str
    num_layers: int
    num_heads: int
    head_dim: int
    
    workloads_tested: List[str]
    context_lengths_tested: List[int]
    
    results: List[LocalityStats]
    
    # Summary statistics
    avg_gini: float
    avg_top20_mass: float
    avg_active_ratio_80: float
    avg_remote_attention_70: float


# =============================================================================
# Attention统计工具
# =============================================================================

class AttentionStatisticsCollector:
    """在每层attention上注册hook，收集attention分布统计"""
    
    def __init__(self, num_layers: int, max_seq_len: int, device: str):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device
        
        # 存储每层的attention mass: [num_layers, seq_len]
        self.attention_mass = torch.zeros(num_layers, max_seq_len, dtype=torch.float32)
        
        # 临时存储Q和K（每个layer一个）
        self.q_storage = [None] * num_layers
        self.k_storage = [None] * num_layers
        self.seq_len_storage = [None] * num_layers
        
        # Hook handles
        self.hooks = []
        
        # 模型配置
        self.num_heads = None
        self.num_kv_heads = None
        self.head_dim = None
        
    def _create_q_hook(self, layer_idx: int):
        """创建Q投影的hook"""
        def hook(module, input, output):
            with torch.no_grad():
                self.q_storage[layer_idx] = output.detach().float()
                self.seq_len_storage[layer_idx] = output.shape[1]
        return hook
    
    def _create_k_hook(self, layer_idx: int):
        """创建K投影的hook"""
        def hook(module, input, output):
            with torch.no_grad():
                self.k_storage[layer_idx] = output.detach().float()
        return hook
    
    def _create_layer_hook(self, layer_idx: int):
        """创建层的forward hook，在Q和K计算完后计算attention"""
        def hook(module, input, output):
            q = self.q_storage[layer_idx]
            k = self.k_storage[layer_idx]
            seq_len = self.seq_len_storage[layer_idx]
            
            if q is None or k is None:
                return
            
            with torch.no_grad():
                batch_size = q.shape[0]
                
                # Reshape for multi-head attention
                # Q: [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
                q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                # GQA: Expand K to match Q heads
                if self.num_kv_heads < self.num_heads:
                    n_rep = self.num_heads // self.num_kv_heads
                    k = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1).reshape(
                        batch_size, self.num_heads, seq_len, self.head_dim
                    )
                
                # 计算attention scores: Q @ K^T / sqrt(d)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                
                # 应用causal mask
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                    diagonal=1
                )
                scores.masked_fill_(causal_mask, float('-inf'))
                
                # Softmax得到attention weights
                attn_weights = F.softmax(scores, dim=-1)
                
                # 计算每个key position的attention mass
                # 对batch, heads, query positions求和
                mass_per_key = attn_weights.sum(dim=(0, 1, 2))  # [key_seq]
                
                # 累加到统计中
                actual_seq_len = min(seq_len, self.max_seq_len)
                self.attention_mass[layer_idx, :actual_seq_len] += mass_per_key[:actual_seq_len].cpu()
                
                # 清空临时存储
                self.q_storage[layer_idx] = None
                self.k_storage[layer_idx] = None
                
        return hook
    
    def register_hooks(self, model):
        """在模型所有attention层注册hooks"""
        config = model.config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        print(f"  注意力配置: {self.num_heads} heads, {self.num_kv_heads} KV heads, head_dim={self.head_dim}")
        
        # 为每一层注册hooks
        for layer_idx in range(self.num_layers):
            layer = model.model.layers[layer_idx]
            self_attn = layer.self_attn
            
            # 注册Q hook
            q_hook = self._create_q_hook(layer_idx)
            handle_q = self_attn.q_proj.register_forward_hook(q_hook)
            self.hooks.append(handle_q)
            
            # 注册K hook
            k_hook = self._create_k_hook(layer_idx)
            handle_k = self_attn.k_proj.register_forward_hook(k_hook)
            self.hooks.append(handle_k)
            
            # 注册layer hook (在self_attn的forward之后)
            layer_hook = self._create_layer_hook(layer_idx)
            handle_layer = self_attn.register_forward_hook(layer_hook)
            self.hooks.append(handle_layer)
        
        print(f"  已注册 {len(self.hooks)} 个hooks")
        
    def remove_hooks(self):
        """移除所有hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print("  已移除所有hooks")
    
    def get_aggregate_attention(self) -> np.ndarray:
        """获取所有层的aggregate attention mass"""
        return self.attention_mass.sum(dim=0).numpy()
    
    def get_layer_attention(self, layer_idx: int) -> np.ndarray:
        """获取特定层的attention mass"""
        return self.attention_mass[layer_idx].numpy()


# =============================================================================
# 分析函数
# =============================================================================

def compute_cdf(attention_mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算attention mass的CDF
    
    Returns:
        cdf_x: KV rank (0 to 1, normalized position)
        cdf_y: cumulative attention mass (0 to 1)
    """
    # 按attention mass降序排序
    sorted_mass = np.sort(attention_mass)[::-1]
    total_mass = sorted_mass.sum()
    
    if total_mass == 0:
        return np.array([0.0]), np.array([0.0])
    
    # 计算CDF
    cumulative_mass = np.cumsum(sorted_mass) / total_mass
    n = len(sorted_mass)
    
    # x轴是归一化的rank (0 to 1)
    cdf_x = np.arange(1, n + 1) / n
    cdf_y = cumulative_mass
    
    return cdf_x, cdf_y


def compute_gini(attention_mass: np.ndarray) -> float:
    """计算Gini系数（衡量locality强度）"""
    sorted_mass = np.sort(attention_mass)
    n = len(sorted_mass)
    if n == 0:
        return 0.0
    
    cumsum = np.cumsum(sorted_mass)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_mass))) / (n * cumsum[-1]) - (n + 1) / n
    return max(0.0, min(1.0, gini))


def compute_zipf_fit(attention_mass: np.ndarray) -> Tuple[float, float]:
    """拟合Zipf分布，返回alpha和R²"""
    # 排序后的mass
    sorted_mass = np.sort(attention_mass)[::-1]
    n = len(sorted_mass)
    
    # 避免0值
    sorted_mass = sorted_mass[sorted_mass > 0]
    if len(sorted_mass) < 10:
        return 0.0, 0.0
    
    # Zipf: mass ~ 1/rank^alpha
    # log(mass) = -alpha * log(rank) + const
    ranks = np.arange(1, len(sorted_mass) + 1).astype(float)
    
    log_ranks = np.log(ranks)
    log_mass = np.log(sorted_mass)
    
    # 线性回归
    valid_idx = np.isfinite(log_mass) & np.isfinite(log_ranks)
    if valid_idx.sum() < 10:
        return 0.0, 0.0
    
    log_mass = log_mass[valid_idx]
    log_ranks = log_ranks[valid_idx]
    
    # 最小二乘法
    n = len(log_ranks)
    mean_x = np.mean(log_ranks)
    mean_y = np.mean(log_mass)
    
    numerator = np.sum((log_ranks - mean_x) * (log_mass - mean_y))
    denominator = np.sum((log_ranks - mean_x) ** 2)
    
    if denominator == 0:
        return 0.0, 0.0
    
    alpha = -numerator / denominator  # Zipf alpha (negative slope)
    
    # 计算R²
    y_pred = mean_y - alpha * log_ranks
    ss_res = np.sum((log_mass - y_pred) ** 2)
    ss_tot = np.sum((log_mass - mean_y) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return alpha, r_squared


def compute_top_mass(attention_mass: np.ndarray, top_pct: float) -> float:
    """计算top p%的KV tokens包含的attention mass"""
    sorted_mass = np.sort(attention_mass)[::-1]
    total_mass = sorted_mass.sum()
    
    if total_mass == 0:
        return 0.0
    
    n = len(sorted_mass)
    k = max(1, int(n * top_pct))
    
    top_mass = sorted_mass[:k].sum()
    return top_mass / total_mass


def compute_remote_attention(attention_mass: np.ndarray, remote_pct: float) -> float:
    """计算remote attention比例（假设前(1-remote_pct)%是local）"""
    sorted_mass = np.sort(attention_mass)[::-1]
    total_mass = sorted_mass.sum()
    
    if total_mass == 0:
        return 0.0
    
    n = len(sorted_mass)
    local_count = int(n * (1 - remote_pct))
    local_count = max(1, min(local_count, n - 1))
    
    # 假设local是hot（排在前面的）
    local_mass = sorted_mass[:local_count].sum()
    
    return 1.0 - (local_mass / total_mass)


def compute_active_set_size(attention_mass: np.ndarray, coverage_pct: float = 0.80) -> Tuple[int, float]:
    """计算覆盖指定attention mass需要的最少KV tokens数量"""
    sorted_mass = np.sort(attention_mass)[::-1]
    total_mass = sorted_mass.sum()
    
    if total_mass == 0:
        return 0, 0.0
    
    cumulative_mass = np.cumsum(sorted_mass)
    target_mass = total_mass * coverage_pct
    
    # 找到覆盖target_mass需要的最少tokens
    active_count = np.searchsorted(cumulative_mass, target_mass) + 1
    active_count = min(active_count, n := len(sorted_mass))
    
    return active_count, active_count / len(sorted_mass)


# =============================================================================
# 实验函数
# =============================================================================

def setup_device(gpu_id: int = 0) -> str:
    """设置计算设备"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA不可用")
        sys.exit(1)
    
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    props = torch.cuda.get_device_properties(gpu_id)
    print(f"使用GPU {gpu_id}: {props.name}")
    print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
    
    return device


def load_model(model_path: str, device: str):
    """加载模型和分词器"""
    print(f"\n加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 使用SDPA以获得良好性能
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",  # 使用SDPA
        trust_remote_code=True
    )
    
    model.eval()
    
    config = model.config
    print(f"  模型层数: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  Head dim: {config.hidden_size // config.num_attention_heads}")
    print(f"  使用SDPA attention")
    
    return model, tokenizer


def run_locality_experiment(
    model,
    tokenizer,
    device: str,
    text: str,
    workload_type: str,
    context_length: int,
    max_new_tokens: int = 32
) -> LocalityStats:
    """对单个文本运行locality实验"""
    print(f"\n  运行实验: {workload_type}, context_length={context_length}")
    
    # 获取模型配置
    config = model.config
    num_layers = config.num_hidden_layers
    
    # 截断文本到指定长度
    tokens = tokenizer.encode(text, return_tensors='pt')
    prompt_tokens = min(tokens.shape[1], context_length - max_new_tokens)
    
    if prompt_tokens < 100:
        # 文本太短，用padding填充
        input_ids = tokens[:, :max(1, prompt_tokens)]
        prompt_tokens = input_ids.shape[1]
    else:
        input_ids = tokens[:, :prompt_tokens]
    
    # 创建attention collector
    collector = AttentionStatisticsCollector(
        num_layers=num_layers,
        max_seq_len=context_length,
        device=device
    )
    
    # 注册hooks
    collector.register_hooks(model)
    
    # 清空CUDA缓存
    torch.cuda.empty_cache()
    
    try:
        # 运行forward pass（prefill阶段）
        with torch.no_grad():
            # 将input_ids移到device
            input_ids_device = input_ids.to(device)
            
            # 前向传播（捕获attention patterns）
            outputs = model(
                input_ids=input_ids_device,
                use_cache=False,
                return_dict=True
            )
            
            # 生成几个token以测试decode阶段
            generated_ids = input_ids_device
            for _ in range(min(max_new_tokens, 8)):  # 限制decode token数量
                outputs = model(
                    input_ids=generated_ids[:, -context_length:],
                    use_cache=True,
                    return_dict=True
                )
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    finally:
        # 移除hooks
        collector.remove_hooks()
    
    # 获取aggregate attention mass
    attention_mass = collector.get_aggregate_attention()
    print(f"    统计完成，有效token数: {len(attention_mass[attention_mass > 0])}")
    
    # 计算各种统计量
    cdf_x, cdf_y = compute_cdf(attention_mass)
    gini = compute_gini(attention_mass)
    alpha, r_sq = compute_zipf_fit(attention_mass)
    
    top_20 = compute_top_mass(attention_mass, 0.20)
    top_10 = compute_top_mass(attention_mass, 0.10)
    top_5 = compute_top_mass(attention_mass, 0.05)
    
    remote_70 = compute_remote_attention(attention_mass, 0.70)
    remote_80 = compute_remote_attention(attention_mass, 0.80)
    remote_90 = compute_remote_attention(attention_mass, 0.90)
    
    active_80, active_ratio_80 = compute_active_set_size(attention_mass, 0.80)
    
    stats = LocalityStats(
        timestamp=datetime.now().isoformat(),
        workload_type=workload_type,
        context_length=context_length,
        prompt_tokens=prompt_tokens,
        layer_attention_mass=[collector.get_layer_attention(i) for i in range(num_layers)],
        cdf_x=cdf_x.tolist(),
        cdf_y=cdf_y.tolist(),
        zipf_alpha=alpha,
        zipf_r_squared=r_sq,
        gini_coefficient=gini,
        top_20_pct_mass=top_20,
        top_10_pct_mass=top_10,
        top_5_pct_mass=top_5,
        remote_attention_pct_70=remote_70,
        remote_attention_pct_80=remote_80,
        remote_attention_pct_90=remote_90,
        active_set_size_80=active_80,
        active_set_ratio_80=active_ratio_80
    )
    
    print(f"    Gini: {gini:.3f}, Top20%: {top_20:.2%}, ActiveSet80: {active_ratio_80:.2%}")
    print(f"    Zipf alpha: {alpha:.2f} (R²={r_sq:.3f})")
    
    return stats


def run_full_experiment(
    model,
    tokenizer,
    device: str,
    context_lengths: List[int],
    workloads: Dict[str, str],
    output_dir: str
) -> ExperimentResult:
    """运行完整实验"""
    config = model.config
    
    all_results = []
    
    print("\n" + "="*60)
    print("开始KV Locality Characterization实验")
    print("="*60)
    
    # 对每种workload和context length组合运行实验
    for workload_type, text in workloads.items():
        for ctx_len in context_lengths:
            stats = run_locality_experiment(
                model=model,
                tokenizer=tokenizer,
                device=device,
                text=text,
                workload_type=workload_type,
                context_length=ctx_len,
                max_new_tokens=32
            )
            all_results.append(stats)
            
            # 保存中间结果
            save_result(stats, output_dir)
    
    # 计算汇总统计
    avg_gini = np.mean([r.gini_coefficient for r in all_results])
    avg_top20 = np.mean([r.top_20_pct_mass for r in all_results])
    avg_active = np.mean([r.active_set_ratio_80 for r in all_results])
    avg_remote = np.mean([r.remote_attention_pct_70 for r in all_results])
    
    experiment_result = ExperimentResult(
        timestamp=datetime.now().isoformat(),
        model_name="Qwen2.5-7B-Instruct",
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        workloads_tested=list(workloads.keys()),
        context_lengths_tested=context_lengths,
        results=all_results,
        avg_gini=avg_gini,
        avg_top20_mass=avg_top20,
        avg_active_ratio_80=avg_active,
        avg_remote_attention_70=avg_remote
    )
    
    print("\n" + "="*60)
    print("实验完成 - 汇总统计")
    print("="*60)
    print(f"  平均 Gini系数: {avg_gini:.3f}")
    print(f"  平均 Top20% Mass: {avg_top20:.2%}")
    print(f"  平均 Active Set Ratio (80%): {avg_active:.2%}")
    print(f"  平均 Remote Attention (70%配置): {avg_remote:.2%}")
    
    return experiment_result


def save_result(stats: LocalityStats, output_dir: str):
    """保存单次实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为可序列化格式
    result_dict = {
        'timestamp': stats.timestamp,
        'workload_type': stats.workload_type,
        'context_length': stats.context_length,
        'prompt_tokens': stats.prompt_tokens,
        'cdf_x': [float(x) for x in stats.cdf_x],
        'cdf_y': [float(y) for y in stats.cdf_y],
        'zipf_alpha': float(stats.zipf_alpha),
        'zipf_r_squared': float(stats.zipf_r_squared),
        'gini_coefficient': float(stats.gini_coefficient),
        'top_20_pct_mass': float(stats.top_20_pct_mass),
        'top_10_pct_mass': float(stats.top_10_pct_mass),
        'top_5_pct_mass': float(stats.top_5_pct_mass),
        'remote_attention_pct_70': float(stats.remote_attention_pct_70),
        'remote_attention_pct_80': float(stats.remote_attention_pct_80),
        'remote_attention_pct_90': float(stats.remote_attention_pct_90),
        'active_set_size_80': int(stats.active_set_size_80),
        'active_set_ratio_80': float(stats.active_set_ratio_80)
    }
    
    filename = f"{output_dir}/{stats.workload_type}_ctx{stats.context_length}.json"
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"    已保存: {filename}")


def save_experiment_summary(result: ExperimentResult, output_dir: str):
    """保存实验汇总"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_dict = {
        'timestamp': result.timestamp,
        'model_name': result.model_name,
        'num_layers': result.num_layers,
        'num_heads': result.num_heads,
        'head_dim': result.head_dim,
        'workloads_tested': result.workloads_tested,
        'context_lengths_tested': result.context_lengths_tested,
        'summary': {
            'avg_gini_coefficient': float(result.avg_gini),
            'avg_top20_pct_mass': float(result.avg_top20_mass),
            'avg_active_set_ratio_80': float(result.avg_active_ratio_80),
            'avg_remote_attention_70': float(result.avg_remote_attention_70)
        }
    }
    
    filename = f"{output_dir}/experiment_summary.json"
    with open(filename, 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    print(f"\n汇总已保存: {filename}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KV Access Locality Characterization")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=str, default="/root/autodl-tmp/experiment_results_locality")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=CONTEXT_LENGTHS)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--all-workloads", action="store_true", help="测试所有workload类型")
    parser.add_argument("--workload", type=str, default=None, help="测试单个workload类型")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KV Access Locality Characterization Experiment")
    print("="*60)
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"上下文长度: {args.context_lengths}")
    print(f"GPU ID: {args.gpu_id}")
    
    # 设置设备
    device = setup_device(args.gpu_id)
    
    # 加载模型
    model, tokenizer = load_model(args.model_path, device)
    
    # 选择要测试的workloads
    if args.workload:
        workloads = {args.workload: TEXT_WORKLOADS.get(args.workload)}
        if workloads[args.workload] is None:
            print(f"ERROR: Unknown workload type: {args.workload}")
            sys.exit(1)
    elif args.all_workloads:
        workloads = TEXT_WORKLOADS
    else:
        # 默认测试所有（完整实验）
        workloads = TEXT_WORKLOADS
    
    print(f"\n测试 workloads: {list(workloads.keys())}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    result = run_full_experiment(
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_lengths=args.context_lengths,
        workloads=workloads,
        output_dir=args.output_dir
    )
    
    # 保存汇总
    save_experiment_summary(result, args.output_dir)
    
    # 保存完整结果
    full_result_path = f"{args.output_dir}/full_results.json"
    with open(full_result_path, 'w') as f:
        json.dump({
            'timestamp': result.timestamp,
            'model_name': result.model_name,
            'num_layers': result.num_layers,
            'num_heads': result.num_heads,
            'head_dim': result.head_dim,
            'workloads_tested': result.workloads_tested,
            'context_lengths_tested': result.context_lengths_tested,
            'avg_gini': float(result.avg_gini),
            'avg_top20_mass': float(result.avg_top20_mass),
            'avg_active_ratio_80': float(result.avg_active_ratio_80),
            'avg_remote_attention_70': float(result.avg_remote_attention_70),
            'num_experiments': len(result.results)
        }, f, indent=2)
    
    print(f"\n完整结果已保存: {full_result_path}")
    print("\n实验完成！")


if __name__ == "__main__":
    main()
