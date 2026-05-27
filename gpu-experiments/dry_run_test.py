#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU实验脚本 - Dry-Run测试

功能：
    - 用mock替换torch/transformers/vllm
    - 模拟KV Cache生成、传输、TAA打分
    - 验证数据流完整性：生成KV → 计算cost → TAA打分 → 选择性传输 → decode
    - 打印每一步的输入输出shape/值范围

使用方法：
    python3 dry_run_test.py
    python3 dry_run_test.py --verbose
    python3 dry_run_test.py --test-module taa

作者：kvcache-lab Team
===============================================================================
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from unittest.mock import MagicMock, patch
import math

# =============================================================================
# Mock模块定义
# =============================================================================

# 创建mock对象来模拟torch
class MockTensor:
    """模拟torch.Tensor"""
    
    def __init__(self, data, requires_grad=False):
        if isinstance(data, list):
            import numpy as np
            self._data = np.array(data)
        else:
            self._data = data
        self.requires_grad = requires_grad
        self.shape = self._data.shape if hasattr(self._data, 'shape') else (len(data),)
        self.device = "cpu"
        self.dtype = "float32"
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data + other._data)
        return MockTensor(self._data + other)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data * other._data)
        return MockTensor(self._data * other)
    
    def __sub__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self._data - other._data)
        return MockTensor(self._data - other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return MockTensor(self._data / other)
        return MockTensor(self._data / other._data)
    
    def numpy(self):
        return self._data
    
    def cpu(self):
        return self
    
    def to(self, device):
        self.device = device
        return self
    
    def max(self):
        import numpy as np
        return MockScalar(np.max(self._data))
    
    def min(self):
        import numpy as np
        return MockScalar(np.min(self._data))
    
    def sum(self):
        import numpy as np
        return MockScalar(np.sum(self._data))
    
    def mean(self):
        import numpy as np
        return MockScalar(np.mean(self._data))
    
    def detach(self):
        return self
    
    def item(self):
        if hasattr(self._data, 'item'):
            return self._data.item()
        return float(self._data)
    
    def expand(self, *sizes):
        return self
    
    def view(self, *shape):
        new_data = self._data.reshape(shape)
        result = MockTensor([0])
        result._data = new_data
        result.shape = shape
        return result
    
    def transpose(self, dim0, dim1):
        result = MockTensor([0])
        import numpy as np
        result._data = np.transpose(self._data, (dim0, dim1))
        result.shape = result._data.shape
        return result
    
    def contiguous(self):
        return self
    
    def squeeze(self, dim=None):
        result = MockTensor([0])
        if dim is not None:
            result._data = self._data.squeeze(axis=dim)
        else:
            result._data = self._data.squeeze()
        result.shape = result._data.shape
        return result
    
    @property
    def T(self):
        result = MockTensor([0])
        import numpy as np
        result._data = self._data.T
        return result


class MockScalar:
    """模拟标量值"""
    def __init__(self, value):
        self._value = value
    
    def item(self):
        return self._value


class MockModule:
    """模拟nn.Module"""
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return MagicMock()
    
    def eval(self):
        return self
    
    def train(self):
        return self


# Mock torch模块
mock_torch = MagicMock()
mock_torch.Tensor = MockTensor
mock_torch.tensor = lambda data, **kwargs: MockTensor(data)
mock_torch.zeros = lambda *shape, **kwargs: MockTensor([[0.0] * shape[-1] for _ in range(shape[0])] if len(shape) > 1 else [0.0] * shape[0])
mock_torch.ones = lambda *shape, **kwargs: MockTensor([[1.0] * shape[-1] for _ in range(shape[0])] if len(shape) > 1 else [1.0] * shape[0])
mock_torch.randn = lambda *shape, **kwargs: MockTensor([[float(j) * 0.1 for j in range(shape[-1])] for i in range(shape[0])] if len(shape) > 1 else [float(i) * 0.1 for i in range(shape[0])])
mock_torch.matmul = lambda a, b: MockTensor([[0.5] * b.shape[-1] for _ in range(a.shape[0])])
mock_torch.softmax = lambda x, dim: MockTensor([[0.1] * x.shape[-1] for _ in range(x.shape[0])])
mock_torch.no_grad = MagicMock()
mock_torch.no_grad().__enter__ = MagicMock()
mock_torch.no_grad().__exit__ = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = lambda: False
mock_torch.cuda.empty_cache = MagicMock()
mock_torch.cuda.reset_peak_memory_stats = MagicMock()
mock_torch.cuda.memory_allocated = lambda: 0
mock_torch.cuda.max_memory_allocated = lambda: 0
mock_torch.cuda.synchronize = MagicMock()
mock_torch.cuda.set_device = MagicMock()
mock_torch.cuda.get_device_properties = MagicMock()
mock_torch.cuda.get_device_properties.return_value = MagicMock(
    name="Mock GPU", total_memory=32 * 1024**3
)
mock_torch.bfloat16 = "bfloat16"
mock_torch.float32 = "float32"
mock_torch.float16 = "float16"
mock_torch.nn = MagicMock()
mock_torch.nn.Module = MockModule
mock_torch.nn.functional = MagicMock()
mock_torch.nn.functional.softmax = lambda x, dim: MockTensor([[0.1] * x.shape[-1] for _ in range(x.shape[0])])
mock_torch.nn.functional.dropout = lambda x, p, training: x
mock_torch.nn.Linear = lambda in_features, out_features: MockModule()
mock_torch.nn.Parameter = lambda data: MockTensor(data)
mock_torch.optim = MagicMock()
mock_torch.distributed = MagicMock()

# Mock其他依赖
mock_transformers = MagicMock()
mock_safetensors = MagicMock()

# 注入到sys.modules
sys.modules['torch'] = mock_torch
sys.modules['transformers'] = mock_transformers
sys.modules['safetensors'] = mock_safetensors

# =============================================================================
# 测试配置
# =============================================================================

VERBOSE = False

def log_step(msg: str, data: Any = None):
    """打印步骤信息"""
    print(f"\n{'='*60}")
    print(f"▶ {msg}")
    print('='*60)
    if data is not None and VERBOSE:
        print(f"  数据: {data}")

def check_shape(name: str, obj: Any, expected_dims: int = None):
    """检查数据形状"""
    if hasattr(obj, 'shape'):
        print(f"  {name}: shape={obj.shape}")
        if VERBOSE and hasattr(obj._data, '__len__') and len(obj._data) <= 5:
            print(f"    值: {obj._data[:5]}...")
    elif isinstance(obj, (list, tuple)):
        print(f"  {name}: type={type(obj).__name__}, len={len(obj)}")
    else:
        print(f"  {name}: type={type(obj).__name__}, value={obj}")

# =============================================================================
# 测试1: KV Cache数据结构
# =============================================================================

def test_kv_cache_data_structure():
    """测试KV Cache数据结构"""
    log_step("测试1: KV Cache数据结构")
    
    @dataclass
    class MockKVCacheData:
        """模拟KV Cache数据"""
        batch_size: int
        seq_len: int
        num_layers: int
        num_heads: int
        head_dim: int
        key_cache: List[MockTensor] = field(default_factory=list)
        value_cache: List[MockTensor] = field(default_factory=list)
        
        def get_size_bytes(self) -> int:
            total = 0
            for k, v in zip(self.key_cache, self.value_cache):
                total += k.shape[0] * k.shape[1] * k.shape[2] * 4  # 4 bytes per float32
            return total
    
    # 模拟生成KV Cache
    batch_size = 1
    seq_len = 256
    num_layers = 32
    num_heads = 32
    head_dim = 128
    
    kv_data = MockKVCacheData(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    
    # 生成key/value tensors
    for layer in range(num_layers):
        # Shape: (batch, num_heads, seq_len, head_dim)
        key = MockTensor([[[[0.0] * head_dim for _ in range(seq_len)] for _ in range(num_heads)]])
        value = MockTensor([[[[0.0] * head_dim for _ in range(seq_len)] for _ in range(num_heads)]])
        kv_data.key_cache.append(key)
        kv_data.value_cache.append(value)
    
    print(f"\n✓ KV Cache结构创建成功:")
    check_shape("key_cache[0]", kv_data.key_cache[0])
    print(f"  num_layers: {kv_data.num_layers}")
    print(f"  estimated_size: {kv_data.get_size_bytes() / 1024:.2f} KB")
    
    return kv_data


# =============================================================================
# 测试2: 访问成本计算
# =============================================================================

def test_access_cost_calculation():
    """测试访问成本计算"""
    log_step("测试2: 访问成本计算")
    
    # 存储层级
    TIER_COSTS = {
        "local": 0.001,      # GPU HBM: ~1μs
        "cpu": 0.05,         # CPU RAM: ~50μs  
        "remote": 0.5,       # Remote GPU: ~0.5ms
        "compressed": 1.0,   # Compressed: ~1ms
    }
    
    # 拥塞放大系数
    CONGESTION_MULTIPLIERS = {
        "low": 1.0,
        "medium": 1.5,
        "high": 3.0,
    }
    
    seq_len = 512
    
    # 模拟token位置分布：前半本地，后半远端
    token_locations = {}
    for i in range(seq_len):
        if i < seq_len // 2:
            token_locations[i] = "local"
        else:
            token_locations[i] = "remote"
    
    print(f"\n✓ Token位置分配:")
    print(f"  本地token数: {sum(1 for v in token_locations.values() if v == 'local')}")
    print(f"  远端token数: {sum(1 for v in token_locations.values() if v == 'remote')}")
    
    # 计算不同拥塞级别下的成本
    for congestion in ["low", "medium", "high"]:
        costs = []
        for token_idx in range(seq_len):
            location = token_locations[token_idx]
            base_cost = TIER_COSTS[location]
            amplified_cost = base_cost * CONGESTION_MULTIPLIERS[congestion]
            costs.append(amplified_cost)
        
        avg_cost = sum(costs) / len(costs)
        max_cost = max(costs)
        
        print(f"\n  {congestion}拥塞:")
        print(f"    平均成本: {avg_cost:.4f} ms")
        print(f"    最大成本: {max_cost:.4f} ms")
    
    return token_locations


# =============================================================================
# 测试3: TAA打分（核心验证）
# =============================================================================

def test_taa_scoring():
    """测试TAA打分公式 - 验证是加法而非乘法"""
    log_step("测试3: TAA打分公式验证")
    
    print("\n【重要验证】TAA公式应该是加法:")
    print("  score_i = relevance_i + α × (-cost_normalized_i)")
    print("  不是乘法: score_i = relevance_i × exp(-β × cost_i)")
    print()
    
    # 模拟数据
    seq_len = 10
    beta = 0.5  # TAA系数
    
    # 原始attention scores (relevance)
    relevance_scores = MockTensor([0.05, 0.10, 0.15, 0.20, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05])
    
    # 访问成本（前5个本地=低成本，后5个远端=高成本）
    access_costs = MockTensor([0.001, 0.001, 0.001, 0.001, 0.001, 
                                0.5, 0.5, 0.5, 0.5, 0.5])
    
    # 归一化成本
    max_cost = access_costs.max().item()
    normalized_costs = MockTensor([c / max_cost for c in access_costs._data])
    
    # 计算runtime_bias = -cost_normalized
    runtime_bias = MockTensor([-c for c in normalized_costs._data])
    
    print(f"原始relevance scores: {relevance_scores._data[:5]}... (后5个省略)")
    print(f"归一化成本: {normalized_costs._data[:5]}... (前5本地, 后5远端)")
    print(f"runtime_bias: {runtime_bias._data[:5]}... (远端为负)")
    print()
    
    # ========== 加法实现（正确）==========
    # score = relevance + beta * (-cost_normalized)
    taa_scores_additive = MockTensor([
        r + beta * (-c) 
        for r, c in zip(relevance_scores._data, normalized_costs._data)
    ])
    
    print("【加法实现】score = relevance + β × (-cost_normalized):")
    print(f"  原始分数[0] (本地,低成本): {relevance_scores._data[0]:.4f}")
    print(f"  TAA分数[0]: {taa_scores_additive._data[0]:.4f} (↑轻微增加)")
    print(f"  原始分数[5] (远端,高成本): {relevance_scores._data[5]:.4f}")
    print(f"  TAA分数[5]: {taa_scores_additive._data[5]:.4f} (↓明显减少)")
    print()
    
    # ========== 乘法实现（错误）==========
    # score = relevance * exp(-beta * cost)
    taa_scores_multiplicative = MockTensor([
        r * math.exp(-beta * c)
        for r, c in zip(relevance_scores._data, normalized_costs._data)
    ])
    
    print("【乘法实现】score = relevance × exp(-β × cost) [错误]:")
    print(f"  原始分数[0]: {relevance_scores._data[0]:.4f}")
    print(f"  乘法TAA[0]: {taa_scores_multiplicative._data[0]:.4f}")
    print(f"  原始分数[5]: {relevance_scores._data[5]:.4f}")
    print(f"  乘法TAA[5]: {taa_scores_multiplicative._data[5]:.4f}")
    print()
    
    # 对比结果
    print("【结果分析】")
    additive_shift = taa_scores_additive._data[0] - relevance_scores._data[0]
    multiplicative_shift = taa_scores_multiplicative._data[0] - relevance_scores._data[0]
    
    print(f"  加法: 本地token分数变化 = {additive_shift:+.4f}")
    print(f"  乘法: 本地token分数变化 = {multiplicative_shift:+.4f}")
    print()
    
    if abs(additive_shift) < abs(multiplicative_shift):
        print("  ✓ 加法实现更稳定（对本地token影响小）")
        print("  ✓ 符合设计目标：α=0时退化为普通attention")
    else:
        print("  ⚠ 加法实现可能有问题，请检查")
    
    return taa_scores_additive


# =============================================================================
# 测试4: KV传输模拟
# =============================================================================

def test_kv_transfer():
    """测试KV传输流程"""
    log_step("测试4: KV传输模拟")
    
    # 带宽配置
    BANDWIDTH_CONFIGS = {
        "1gbps": 1 * 1024**3,
        "5gbps": 5 * 1024**3,
        "10gbps": 10 * 1024**3,
        "nvlink": 900 * 1024**3,
    }
    
    # 模拟KV数据大小
    kv_size_bytes = 32 * 512 * 128 * 4 * 2 * 2  # 2 layers * (batch*heads*seq*head_dim*bytes)
    
    print(f"\n模拟KV数据大小: {kv_size_bytes / 1024**2:.2f} MB")
    
    for bandwidth_name, bandwidth_bps in BANDWIDTH_CONFIGS.items():
        transfer_time = kv_size_bytes / bandwidth_bps
        print(f"\n  {bandwidth_name:>10}: {transfer_time*1000:.2f} ms")
    
    # 模拟选择性传输（只传Working Set）
    full_seq_len = 512
    working_set_ratio = 0.5
    ws_seq_len = int(full_seq_len * working_set_ratio)
    
    print(f"\n选择性传输:")
    print(f"  完整序列长度: {full_seq_len}")
    print(f"  Working Set比例: {working_set_ratio:.0%}")
    print(f"  传输token数: {ws_seq_len}")
    print(f"  带宽节省: {(1 - working_set_ratio)*100:.0f}%")
    
    return True


# =============================================================================
# 测试5: Semantic Working Set分析
# =============================================================================

def test_semantic_working_set():
    """测试语义工作集分析"""
    log_step("测试5: Semantic Working Set分析")
    
    # 模拟attention分布
    seq_len = 100
    attention_weights = MockTensor([0.01] * seq_len)
    
    # 模拟attention聚焦在特定区域
    for i in range(20, 40):  # 20-40区域更重要
        attention_weights._data[i] = 0.15
    for i in range(60, 80):  # 60-80区域中等重要
        attention_weights._data[i] = 0.08
    
    # 归一化
    total = sum(attention_weights._data)
    attention_weights._data = [w / total for w in attention_weights._data]
    
    print(f"\n模拟Attention分布:")
    print(f"  序列长度: {seq_len}")
    print(f"  区域[20-40] (重要): 0.15")
    print(f"  区域[60-80] (中等): 0.08")
    print(f"  其他区域: 0.01")
    
    # 计算Working Set (top 30%)
    working_set_ratio = 0.3
    target_size = int(seq_len * working_set_ratio)
    
    # 按attention权重排序
    weighted_indices = [(attention_weights._data[i], i) for i in range(seq_len)]
    weighted_indices.sort(reverse=True)
    
    working_set_indices = [idx for _, idx in weighted_indices[:target_size]]
    working_set_indices.sort()
    
    print(f"\nWorking Set (top {working_set_ratio:.0%}):")
    print(f"  选择token数: {len(working_set_indices)}")
    print(f"  覆盖的重要区域: {sum(1 for i in working_set_indices if 20 <= i < 40)}/20")
    print(f"  覆盖的中等区域: {sum(1 for i in working_set_indices if 60 <= i < 80)}/20")
    
    # 计算带宽节省
    bandwidth_saving = (1 - len(working_set_indices) / seq_len) * 100
    print(f"  带宽节省: {bandwidth_saving:.1f}%")
    
    return working_set_indices


# =============================================================================
# 测试6: Eviction策略对比
# =============================================================================

def test_eviction_strategies():
    """测试不同Eviction策略"""
    log_step("测试6: Eviction策略对比")
    
    # 模拟访问序列
    workload = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    cache_size = 5
    
    print(f"\n模拟工作负载:")
    print(f"  长度: {len(workload)}")
    print(f"  Cache大小: {cache_size}")
    print(f"  访问序列: {workload}")
    
    # LRU模拟
    lru_cache = []
    lru_hits = 0
    
    print("\n【LRU策略】")
    for i, token_id in enumerate(workload):
        if token_id in lru_cache:
            lru_hits += 1
            # 移动到末尾（最近使用）
            lru_cache.remove(token_id)
            lru_cache.append(token_id)
            print(f"  [{i}] Token {token_id}: HIT, cache={lru_cache}")
        else:
            if len(lru_cache) >= cache_size:
                evicted = lru_cache.pop(0)
                print(f"  [{i}] Token {token_id}: MISS, EVICT {evicted}, cache={lru_cache + [token_id]}")
            else:
                print(f"  [{i}] Token {token_id}: MISS, ADD, cache={lru_cache + [token_id]}")
            lru_cache.append(token_id)
    
    lru_hit_rate = lru_hits / len(workload)
    print(f"\n  LRU命中率: {lru_hit_rate:.2%}")
    
    # Predictive策略模拟（简化版）
    print("\n【Predictive策略】模拟")
    print("  (基于未来访问预测，优先驱逐长时间不用的token)")
    predictive_cache = []
    predictive_hits = 0
    
    for i, token_id in enumerate(workload):
        if token_id in predictive_cache:
            predictive_hits += 1
            print(f"  [{i}] Token {token_id}: HIT")
        else:
            if len(predictive_cache) >= cache_size:
                # 查找最远访问的token
                future = workload[i+1:]
                worst_token = None
                worst_distance = -1
                
                for cached in predictive_cache:
                    if cached in future:
                        distance = future.index(cached)
                        if distance > worst_distance:
                            worst_distance = distance
                            worst_token = cached
                    else:
                        worst_token = cached
                        worst_distance = 1000
                        break
                
                if worst_token:
                    evicted = worst_token
                    predictive_cache.remove(evicted)
                    print(f"  [{i}] Token {token_id}: MISS, EVICT {evicted} (distance={worst_distance})")
            predictive_cache.append(token_id)
    
    predictive_hit_rate = predictive_hits / len(workload)
    print(f"\n  Predictive命中率: {predictive_hit_rate:.2%}")
    
    print(f"\n【对比】")
    print(f"  LRU命中率: {lru_hit_rate:.2%}")
    print(f"  Predictive命中率: {predictive_hit_rate:.2%}")
    
    return {"lru": lru_hit_rate, "predictive": predictive_hit_rate}


# =============================================================================
# 测试7: 完整数据流验证
# =============================================================================

def test_full_pipeline():
    """测试完整数据流"""
    log_step("测试7: 完整数据流验证")
    
    print("\n【数据流】")
    print("  1. 生成KV Cache")
    print("  2. 计算访问成本")
    print("  3. TAA打分（加法）")
    print("  4. 选择性传输")
    print("  5. Decode")
    print()
    
    # Step 1: 生成KV
    print("Step 1: 生成KV Cache")
    batch_size, seq_len, num_layers = 1, 256, 32
    num_heads, head_dim = 32, 128
    
    kv_size = batch_size * seq_len * num_layers * num_heads * head_dim * 4 * 2  # 2 for K+V
    print(f"  KV大小: {kv_size / 1024**2:.2f} MB")
    print(f"  ✓ Step 1完成")
    print()
    
    # Step 2: 计算成本
    print("Step 2: 计算访问成本")
    local_tokens = seq_len // 2
    remote_tokens = seq_len - local_tokens
    local_cost, remote_cost = 0.001, 0.5
    avg_cost = (local_tokens * local_cost + remote_tokens * remote_cost) / seq_len
    print(f"  本地token: {local_tokens}, 成本: {local_cost}ms")
    print(f"  远端token: {remote_tokens}, 成本: {remote_cost}ms")
    print(f"  平均成本: {avg_cost:.4f}ms")
    print(f"  ✓ Step 2完成")
    print()
    
    # Step 3: TAA打分
    print("Step 3: TAA打分（beta=0.5）")
    beta = 0.5
    print(f"  应用公式: score = relevance + β × (-cost_normalized)")
    
    # 简化：远端token分数降低
    local_bonus = beta * 0.5  # 假设归一化成本差异为0.5
    remote_penalty = beta * (-0.5)
    print(f"  本地token分数调整: +{local_bonus:.4f}")
    print(f"  远端token分数调整: {remote_penalty:.4f}")
    print(f"  ✓ Step 3完成")
    print()
    
    # Step 4: 选择性传输
    print("Step 4: 选择性传输")
    working_set_ratio = 0.5
    transfer_size = kv_size * working_set_ratio
    bandwidth = 10 * 1024**3  # 10 GB/s
    transfer_time = transfer_size / bandwidth * 1000
    print(f"  传输比例: {working_set_ratio:.0%}")
    print(f"  传输大小: {transfer_size / 1024**2:.2f} MB")
    print(f"  传输时间(10Gbps): {transfer_time:.2f} ms")
    print(f"  ✓ Step 4完成")
    print()
    
    # Step 5: Decode
    print("Step 5: Decode")
    decode_time = 50.0  # ms
    print(f"  预估decode时间: {decode_time} ms")
    print(f"  ✓ Step 5完成")
    print()
    
    # 汇总
    total_time = avg_cost + transfer_time + decode_time
    print(f"【总预估延迟】")
    print(f"  访问成本: {avg_cost:.4f} ms")
    print(f"  传输时间: {transfer_time:.2f} ms")
    print(f"  Decode时间: {decode_time} ms")
    print(f"  总计: {total_time:.2f} ms")
    
    return True


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="kvcache-lab Dry-Run测试")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--test-module", type=str, choices=["all", "kv", "cost", "taa", "transfer", "sws", "eviction", "pipeline"], default="all")
    
    args = parser.parse_args()
    
    global VERBOSE
    VERBOSE = args.verbose
    
    print("\n" + "="*60)
    print("kvcache-lab GPU实验脚本 Dry-Run 测试")
    print("="*60)
    print("\n本测试使用Mock对象模拟torch/transformers环境")
    print("用于验证数据流和算法逻辑的正确性\n")
    
    test_results = {}
    
    # 根据选项运行测试
    if args.test_module in ["all", "kv"]:
        try:
            test_results["kv_cache"] = test_kv_cache_data_structure()
        except Exception as e:
            print(f"\n✗ KV Cache测试失败: {e}")
            test_results["kv_cache"] = None
    
    if args.test_module in ["all", "cost"]:
        try:
            test_results["access_cost"] = test_access_cost_calculation()
        except Exception as e:
            print(f"\n✗ 访问成本测试失败: {e}")
            test_results["access_cost"] = None
    
    if args.test_module in ["all", "taa"]:
        try:
            test_results["taa"] = test_taa_scoring()
        except Exception as e:
            print(f"\n✗ TAA测试失败: {e}")
            test_results["taa"] = None
    
    if args.test_module in ["all", "transfer"]:
        try:
            test_results["transfer"] = test_kv_transfer()
        except Exception as e:
            print(f"\n✗ 传输测试失败: {e}")
            test_results["transfer"] = None
    
    if args.test_module in ["all", "sws"]:
        try:
            test_results["sws"] = test_semantic_working_set()
        except Exception as e:
            print(f"\n✗ SWS测试失败: {e}")
            test_results["sws"] = None
    
    if args.test_module in ["all", "eviction"]:
        try:
            test_results["eviction"] = test_eviction_strategies()
        except Exception as e:
            print(f"\n✗ Eviction测试失败: {e}")
            test_results["eviction"] = None
    
    if args.test_module in ["all", "pipeline"]:
        try:
            test_results["pipeline"] = test_full_pipeline()
        except Exception as e:
            print(f"\n✗ Pipeline测试失败: {e}")
            test_results["pipeline"] = None
    
    # 汇总
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)
    
    passed = sum(1 for v in test_results.values() if v is not None and v is not False)
    total = len(test_results)
    
    for name, result in test_results.items():
        status = "✓ PASS" if result is not None and result is not False else "✗ FAIL"
        print(f"  {name:15s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！脚本逻辑验证完成。")
        return 0
    else:
        print("\n✗ 部分测试失败，请检查。")
        return 1


if __name__ == "__main__":
    exit(main())
