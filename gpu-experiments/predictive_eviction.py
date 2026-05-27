#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab Predictive Eviction 验证脚本

核心验证目标：
    - 模拟有限GPU内存场景（限制KV Cache大小）
    - 对比LRU/LFU/Predictive Eviction策略
    - 测量命中率 + 驱逐后重加载延迟

原理（来自kvcache-lab/src/agents/PlacementAgent.ts）:
    Predictive Eviction基于：
    1. Reuse Prediction: 预估token的重用距离和概率
    2. Memory Pressure: 根据GPU内存压力决定降级策略
    3. 层级感知: GPU HBM > CPU RAM > Remote > Compressed
    
    决策逻辑：
    - 热token (reuseDistance<=3) + 高reuse概率 → GPU HBM
    - 温token (reuseDistance<=10) + 中reuse概率 → CPU RAM
    - 冷token (reuseDistance<=50) + 低reuse概率 → Remote/Compressed

使用方法：
    python3 predictive_eviction.py --eviction-strategies lru,lfu,predictive
    python3 predictive_eviction.py --kv-cache-size-gb 40 --workload-size 1000

作者：kvcache-lab Team
===============================================================================
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

# =============================================================================
# 常量定义
# =============================================================================

# 默认EViction策略
DEFAULT_STRATEGIES = ["lru", "lfu", "predictive"]

# 存储层级
class StorageTier(Enum):
    GPU_HBM = "gpu_hbm"       # GPU显存
    CPU_RAM = "cpu_ram"        # CPU内存
    REMOTE = "remote"          # 远端存储
    COMPRESSED = "compressed"  # 压缩存储

# 访问成本 (ms)
ACCESS_COSTS = {
    StorageTier.GPU_HBM: 0.001,     # ~1μs
    StorageTier.CPU_RAM: 0.05,       # ~50μs
    StorageTier.REMOTE: 0.5,         # ~0.5ms
    StorageTier.COMPRESSED: 1.0,     # ~1ms
}

# 存储层级优先级
TIER_PRIORITY = {
    StorageTier.GPU_HBM: 4,
    StorageTier.CPU_RAM: 3,
    StorageTier.REMOTE: 2,
    StorageTier.COMPRESSED: 1,
}


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class TokenAccess:
    """Token访问记录"""
    token_id: int
    access_time: float       # 访问时间戳
    access_count: int = 1    # 累计访问次数
    reuse_distance: int = 0  # 预估重用距离
    reuse_probability: float = 0.0  # 重用概率
    
    # 当前位置
    current_tier: StorageTier = StorageTier.GPU_HBM
    tier_enter_time: float = 0.0


@dataclass
class EvictionResult:
    """驱逐决策结果"""
    evicted_token_id: int
    evicted_from: StorageTier
    evicted_to: StorageTier
    decision_reason: str
    confidence: float  # 决策置信度
    
    # 性能影响
    access_time_saved_ms: float = 0.0  # 提前驱逐节省的时间
    access_time_cost_ms: float = 0.0  # 未来可能增加的时间


@dataclass
class EvictionStats:
    """Eviction统计"""
    strategy: str
    
    # 命中率
    hit_count: int = 0
    miss_count: int = 0
    hit_rate: float = 0.0
    
    # 驱逐统计
    eviction_count: int = 0
    premature_evictions: int = 0  # 过早驱逐（被驱逐后又被访问）
    late_evictions: int = 0       # 过晚驱逐（应该驱逐但没驱逐）
    
    # 延迟
    avg_access_time_ms: float = 0.0
    avg_eviction_overhead_ms: float = 0.0
    
    # 内存使用
    peak_gpu_usage_bytes: int = 0
    avg_gpu_usage_bytes: int = 0
    
    # 分层统计
    tier_access_counts: Dict[str, int] = field(default_factory=dict)
    tier_hit_rates: Dict[str, float] = field(default_factory=dict)
    
    # 时间戳
    simulation_time_ms: float = 0.0
    
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """完整实验结果"""
    strategy: str
    stats: EvictionStats
    
    # 工作负载信息
    workload_size: int
    kv_cache_size_tokens: int
    
    # 质量指标（如果适用）
    quality_impact: float = 0.0
    
    # 时间线
    timestamp: str
    duration_ms: float


@dataclass
class ExperimentSummary:
    """实验汇总"""
    timestamp: str
    strategies_compared: List[str]
    workload_size: int
    cache_size: int
    
    results: List[ExperimentResult]
    
    # 排名
    best_hit_rate_strategy: Optional[str] = None
    best_access_time_strategy: Optional[str] = None
    best_overall_strategy: Optional[str] = None
    
    findings: List[str] = field(default_factory=list)


# =============================================================================
# Eviction策略基类
# =============================================================================

class EvictionPolicy(ABC):
    """Eviction策略基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.tokens: Dict[int, TokenAccess] = {}
        
    @abstractmethod
    def select_eviction_candidate(
        self,
        current_tier: StorageTier,
        candidates: List[int],
        current_time: float,
    ) -> Optional[int]:
        """选择要驱逐的token"""
        pass
    
    def record_access(
        self,
        token_id: int,
        access_time: float,
        current_tier: StorageTier,
    ):
        """记录token访问"""
        if token_id in self.tokens:
            self.tokens[token_id].access_count += 1
            self.tokens[token_id].access_time = access_time
            self.tokens[token_id].current_tier = current_tier
        else:
            self.tokens[token_id] = TokenAccess(
                token_id=token_id,
                access_time=access_time,
                current_tier=current_tier,
            )
    
    def predict_reuse(
        self,
        token_id: int,
        future_accesses: List[int],
        current_idx: int,
    ) -> Tuple[int, float]:
        """
        预测token的重用
        
        Returns:
            reuse_distance: 下次访问的距离
            reuse_probability: 重用概率
        """
        if token_id not in self.tokens:
            return 1000, 0.0  # 从未访问过，认为不会重用
        
        # 查找下次访问
        remaining = future_accesses[current_idx + 1:]
        if token_id in remaining:
            reuse_distance = remaining.index(token_id)
            # 距离越近，概率越高
            reuse_probability = 1.0 / (1.0 + reuse_distance * 0.1)
            return reuse_distance, reuse_probability
        
        return 1000, 0.0


# =============================================================================
# LRU策略
# =============================================================================

class LRUPolicy(EvictionPolicy):
    """Least Recently Used - 最近最少使用"""
    
    def __init__(self):
        super().__init__("LRU")
        self.last_access_order: OrderedDict[int, float] = OrderedDict()
    
    def record_access(self, token_id: int, access_time: float, current_tier: StorageTier):
        super().record_access(token_id, access_time, current_tier)
        self.last_access_order[token_id] = access_time
    
    def select_eviction_candidate(
        self,
        current_tier: StorageTier,
        candidates: List[int],
        current_time: float,
    ) -> Optional[int]:
        if not candidates:
            return None
        
        # 选择最久未访问的
        min_access_time = float('inf')
        victim = None
        
        for token_id in candidates:
            if token_id in self.last_access_order:
                access_time = self.last_access_order[token_id]
                if access_time < min_access_time:
                    min_access_time = access_time
                    victim = token_id
            else:
                # 未记录的token，优先驱逐
                return token_id
        
        return victim if victim is not None else candidates[0]


# =============================================================================
# LFU策略
# =============================================================================

class LFUPolicy(EvictionPolicy):
    """Least Frequently Used - 最不经常使用"""
    
    def __init__(self):
        super().__init__("LFU")
        self.access_counts: Counter = Counter()
    
    def record_access(self, token_id: int, access_time: float, current_tier: StorageTier):
        super().record_access(token_id, access_time, current_tier)
        self.access_counts[token_id] += 1
    
    def select_eviction_candidate(
        self,
        current_tier: StorageTier,
        candidates: List[int],
        current_time: float,
    ) -> Optional[int]:
        if not candidates:
            return None
        
        # 选择访问次数最少的
        min_count = float('inf')
        victim = None
        
        for token_id in candidates:
            count = self.access_counts.get(token_id, 0)
            if count < min_count:
                min_count = count
                victim = token_id
        
        # 如果次数相同，选择较早的
        if victim is None:
            victim = candidates[0]
        
        return victim


# =============================================================================
# Predictive Eviction策略
# =============================================================================

class PredictiveEvictionPolicy(EvictionPolicy):
    """
    Predictive Eviction - 预测性驱逐
    
    基于以下因素综合评分：
    1. Reuse Distance: 重用距离越远，越应该驱逐
    2. Reuse Probability: 重用概率越低，越应该驱逐
    3. Memory Pressure: 内存压力越高，驱逐越激进
    4. Access Cost: 访问成本越低，可以延迟驱逐
    """
    
    def __init__(
        self,
        reuse_threshold: int = 50,
        probability_threshold: float = 0.3,
    ):
        super().__init__("Predictive")
        self.reuse_threshold = reuse_threshold
        self.probability_threshold = probability_threshold
        
        # 预测缓存
        self.reuse_predictions: Dict[int, Tuple[int, float]] = {}
        
    def update_predictions(
        self,
        token_id: int,
        reuse_distance: int,
        reuse_probability: float,
    ):
        """更新预测"""
        self.reuse_predictions[token_id] = (reuse_distance, reuse_probability)
    
    def score_token(
        self,
        token_id: int,
        memory_pressure: float,
        current_tier: StorageTier,
    ) -> float:
        """
        计算驱逐优先级分数
        
        分数越高，越应该被驱逐
        """
        score = 0.0
        
        # 1. Reuse Distance因素（距离越远，分数越高）
        if token_id in self.reuse_predictions:
            reuse_dist, reuse_prob = self.reuse_predictions[token_id]
            
            # 距离分数
            distance_score = min(1.0, reuse_dist / 100.0)
            score += distance_score * 0.4
            
            # 概率分数（概率越低，分数越高）
            prob_score = 1.0 - reuse_prob
            score += prob_score * 0.4
        else:
            score += 0.5  # 无预测信息，中等分数
        
        # 2. Memory Pressure因素（压力越大，越激进）
        score += memory_pressure * 0.2
        
        return score
    
    def select_eviction_candidate(
        self,
        current_tier: StorageTier,
        candidates: List[int],
        current_time: float,
    ) -> Optional[int]:
        if not candidates:
            return None
        
        # 假设内存压力为0.5（中等）
        memory_pressure = 0.5
        
        max_score = -float('inf')
        victim = None
        
        for token_id in candidates:
            score = self.score_token(token_id, memory_pressure, current_tier)
            if score > max_score:
                max_score = score
                victim = token_id
        
        return victim if victim is not None else candidates[0]
    
    def predict_reuse_future(
        self,
        token_id: int,
        workload: List[int],
        current_idx: int,
    ) -> Tuple[int, float]:
        """利用未来信息进行预测（用于离线评估）"""
        remaining = workload[current_idx + 1:]
        
        if token_id not in remaining:
            return 1000, 0.0
        
        next_access = remaining.index(token_id)
        reuse_probability = 1.0 / (1.0 + next_access * 0.1)
        
        return next_access, reuse_probability


# =============================================================================
# 模拟器
# =============================================================================

class EvictionSimulator:
    """Eviction策略模拟器"""
    
    def __init__(
        self,
        cache_size_tokens: int,
        token_size_bytes: int = 1024,
        bandwidth_gbps: float = 10.0,
    ):
        self.cache_size_tokens = cache_size_tokens
        self.cache_size_bytes = cache_size_tokens * token_size_bytes
        self.token_size_bytes = token_size_bytes
        self.bandwidth_bps = bandwidth_gbps * 1024**3
        
        # 当前缓存状态
        self.current_tokens: Dict[int, TokenAccess] = {}
        self.tier_distribution: Dict[StorageTier, List[int]] = {
            tier: [] for tier in StorageTier
        }
        
        # 统计
        self.stats = EvictionStats(strategy="unknown")
        
    def reset(self, strategy: str):
        """重置模拟器"""
        self.current_tokens = {}
        self.tier_distribution = {tier: [] for tier in StorageTier}
        
        self.stats = EvictionStats(
            strategy=strategy,
            config={
                "cache_size_tokens": self.cache_size_tokens,
                "cache_size_bytes": self.cache_size_bytes,
                "bandwidth_gbps": self.bandwidth_bps / 1024**3,
            }
        )
    
    def simulate(
        self,
        policy: EvictionPolicy,
        workload: List[int],
        trace_ground_truth: bool = True,
    ) -> EvictionStats:
        """
        模拟Eviction策略
        
        Args:
            policy: Eviction策略
            workload: 访问序列
            trace_ground_truth: 是否追踪真实重用信息（用于Predictive策略）
        """
        self.reset(policy.name)
        start_time = time.perf_counter()
        
        total_access_time = 0.0
        eviction_overhead_total = 0.0
        
        for i, token_id in enumerate(workload):
            current_time = i * 0.001  # 假设每1ms一次访问
            
            # 更新预测（如果有ground truth）
            if trace_ground_truth and isinstance(policy, PredictiveEvictionPolicy):
                reuse_dist, reuse_prob = policy.predict_reuse_future(
                    token_id, workload, i
                )
                policy.update_predictions(token_id, reuse_dist, reuse_prob)
            
            # 记录访问
            policy.record_access(token_id, current_time, StorageTier.GPU_HBM)
            
            # 查找token
            if token_id in self.current_tokens:
                # Cache Hit
                self.stats.hit_count += 1
                
                token = self.current_tokens[token_id]
                access_cost = ACCESS_COSTS[token.current_tier]
                total_access_time += access_cost
                
                # 统计分层命中率
                tier_name = token.current_tier.value
                if tier_name not in self.stats.tier_access_counts:
                    self.stats.tier_access_counts[tier_name] = 0
                self.stats.tier_access_counts[tier_name] += 1
                
                # 如果不在GPU，需要加载
                if token.current_tier != StorageTier.GPU_HBM:
                    # 模拟加载延迟
                    load_time = self.token_size_bytes / self.bandwidth_bps * 1000
                    total_access_time += load_time
                    eviction_overhead_total += load_time
                    
                    # 移动到GPU
                    self.tier_distribution[token.current_tier].remove(token_id)
                    self.tier_distribution[StorageTier.GPU_HBM].append(token_id)
                    token.current_tier = StorageTier.GPU_HBM
                
            else:
                # Cache Miss
                self.stats.miss_count += 1
                total_access_time += ACCESS_COSTS[StorageTier.COMPRESSED]
                
                # 如果缓存满了，需要驱逐
                if len(self.current_tokens) >= self.cache_size_tokens:
                    # 选择驱逐候选
                    candidates = list(self.current_tokens.keys())
                    victim_id = policy.select_eviction_candidate(
                        StorageTier.GPU_HBM, candidates, current_time
                    )
                    
                    if victim_id is not None:
                        # 检查是否驱逐正确
                        if trace_ground_truth:
                            future = workload[i + 1:]
                            if victim_id in future:
                                future_dist = future.index(victim_id)
                                if future_dist < 20:
                                    # 过早驱逐
                                    self.stats.premature_evictions += 1
                            else:
                                # 正确驱逐
                                pass
                        
                        # 驱逐
                        evicted_token = self.current_tokens.pop(victim_id)
                        self.tier_distribution[StorageTier.GPU_HBM].remove(victim_id)
                        
                        # 估算驱逐开销
                        evict_time = self.token_size_bytes / self.bandwidth_bps * 1000
                        eviction_overhead_total += evict_time
                        self.stats.eviction_count += 1
                
                # 添加新token
                self.current_tokens[token_id] = TokenAccess(
                    token_id=token_id,
                    access_time=current_time,
                    current_tier=StorageTier.GPU_HBM,
                )
                self.tier_distribution[StorageTier.GPU_HBM].append(token_id)
            
            # 更新峰值内存
            current_usage = len(self.current_tokens) * self.token_size_bytes
            self.stats.peak_gpu_usage_bytes = max(
                self.stats.peak_gpu_usage_bytes,
                current_usage
            )
        
        # 计算统计
        total_requests = self.stats.hit_count + self.stats.miss_count
        self.stats.hit_rate = self.stats.hit_count / total_requests if total_requests > 0 else 0
        self.stats.avg_access_time_ms = total_access_time / total_requests if total_requests > 0 else 0
        self.stats.avg_eviction_overhead_ms = (
            eviction_overhead_total / self.stats.eviction_count
            if self.stats.eviction_count > 0 else 0
        )
        self.stats.simulation_time_ms = (time.perf_counter() - start_time) * 1000
        
        return self.stats


# =============================================================================
# 工作负载生成器
# =============================================================================

class WorkloadGenerator:
    """工作负载生成器"""
    
    @staticmethod
    def generate_hot_cold(
        num_tokens: int,
        num_requests: int,
        hot_ratio: float = 0.2,
        reuse_distance: int = 5,
    ) -> List[int]:
        """
        生成热点工作负载
        
        特点：
        - 一小部分token被频繁访问（热点）
        - 大部分token很少被访问（冷门）
        """
        # 确定热点token数量
        num_hot = int(num_tokens * hot_ratio)
        hot_tokens = list(range(num_hot))
        cold_tokens = list(range(num_hot, num_tokens))
        
        workload = []
        hot_idx = 0
        
        for _ in range(num_requests):
            if random.random() < 0.8:  # 80%概率访问热点
                token = hot_tokens[hot_idx % len(hot_tokens)]
                hot_idx += 1
                
                # 模拟重用：短时间内重复访问
                if hot_idx % reuse_distance == 0:
                    workload.append(token)
            else:  # 20%概率访问冷门
                token = random.choice(cold_tokens)
            
            workload.append(token)
        
        return workload
    
    @staticmethod
    def generate_linear_scan(
        num_tokens: int,
        num_requests: int,
        scan_size: int = 100,
    ) -> List[int]:
        """生成线性扫描工作负载"""
        workload = []
        
        for _ in range(num_requests):
            start = random.randint(0, num_tokens - scan_size)
            # 顺序扫描
            workload.extend(range(start, start + scan_size))
        
        return workload
    
    @staticmethod
    def generate_random(
        num_tokens: int,
        num_requests: int,
    ) -> List[int]:
        """生成随机工作负载"""
        return [random.randint(0, num_tokens - 1) for _ in range(num_requests)]
    
    @staticmethod
    def generate_zipfian(
        num_tokens: int,
        num_requests: int,
        alpha: float = 1.2,
    ) -> List[int]:
        """生成Zipfian分布工作负载"""
        weights = np.power(np.arange(1, num_tokens + 1), -alpha)
        weights /= weights.sum()
        
        return np.random.choice(num_tokens, size=num_requests, p=weights).tolist()


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="kvcache-lab Predictive Eviction验证脚本",
    )
    
    # 模拟配置
    parser.add_argument(
        "--eviction-strategies",
        type=str,
        default="lru,lfu,predictive",
        help="要比较的驱逐策略 (默认: lru,lfu,predictive)"
    )
    parser.add_argument(
        "--kv-cache-size-gb",
        type=float,
        default=40.0,
        help="KV Cache大小 (GB, 默认: 40)"
    )
    parser.add_argument(
        "--workload-size",
        type=int,
        default=1000,
        help="工作负载大小 (请求数, 默认: 1000)"
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=10000,
        help="总token数量 (默认: 10000)"
    )
    parser.add_argument(
        "--workload-type",
        type=str,
        choices=["hot_cold", "linear_scan", "random", "zipfian"],
        default="hot_cold",
        help="工作负载类型 (默认: hot_cold)"
    )
    parser.add_argument(
        "--token-size-bytes",
        type=int,
        default=1024,
        help="每个token的KV大小 (bytes, 默认: 1024)"
    )
    parser.add_argument(
        "--bandwidth-gbps",
        type=float,
        default=10.0,
        help="带宽 (GB/s, 默认: 10)"
    )
    
    # 输出
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    # 解析策略
    strategies = args.eviction_strategies.split(",")
    
    # 计算cache大小（token数）
    cache_size_tokens = int(args.kv_cache_size_gb * 1024**3 / args.token_size_bytes)
    
    print(f"\n{'='*60}")
    print("kvcache-lab Predictive Eviction 验证实验")
    print(f"{'='*60}")
    print(f"驱逐策略: {strategies}")
    print(f"KV Cache大小: {args.kv_cache_size_gb} GB ({cache_size_tokens} tokens)")
    print(f"工作负载: {args.workload_type} ({args.workload_size} 请求)")
    print(f"Token数量: {args.num_tokens}")
    print(f"{'='*60}\n")
    
    # 生成工作负载
    print("生成工作负载...")
    if args.workload_type == "hot_cold":
        workload = WorkloadGenerator.generate_hot_cold(
            args.num_tokens, args.workload_size, hot_ratio=0.2, reuse_distance=5
        )
    elif args.workload_type == "linear_scan":
        workload = WorkloadGenerator.generate_linear_scan(
            args.num_tokens, args.workload_size
        )
    elif args.workload_type == "random":
        workload = WorkloadGenerator.generate_random(
            args.num_tokens, args.workload_size
        )
    elif args.workload_type == "zipfian":
        workload = WorkloadGenerator.generate_zipfian(
            args.num_tokens, args.workload_size
        )
    
    print(f"工作负载生成完成: {len(workload)} 次访问")
    
    # 创建模拟器
    simulator = EvictionSimulator(
        cache_size_tokens=cache_size_tokens,
        token_size_bytes=args.token_size_bytes,
        bandwidth_gbps=args.bandwidth_gbps,
    )
    
    # 创建策略
    policy_map = {
        "lru": LRUPolicy(),
        "lfu": LFUPolicy(),
        "predictive": PredictiveEvictionPolicy(),
    }
    
    # 运行实验
    all_results = []
    
    for strategy_name in strategies:
        print(f"\n{'='*40}")
        print(f"模拟策略: {strategy_name}")
        print(f"{'='*40}")
        
        policy = policy_map.get(strategy_name)
        if policy is None:
            print(f"未知策略: {strategy_name}")
            continue
        
        # 模拟
        stats = simulator.simulate(
            policy=policy,
            workload=workload,
            trace_ground_truth=True,
        )
        
        # 打印结果
        print(f"\n结果:")
        print(f"  命中率: {stats.hit_rate:.2%}")
        print(f"  访问次数: {stats.hit_count + stats.miss_count}")
        print(f"  平均访问时间: {stats.avg_access_time_ms:.4f} ms")
        print(f"  驱逐次数: {stats.eviction_count}")
        print(f"  过早驱逐: {stats.premature_evictions}")
        print(f"  驱逐开销: {stats.avg_eviction_overhead_ms:.4f} ms")
        
        result = ExperimentResult(
            strategy=strategy_name,
            stats=stats,
            workload_size=len(workload),
            kv_cache_size_tokens=cache_size_tokens,
            timestamp=datetime.now().isoformat(),
            duration_ms=stats.simulation_time_ms,
        )
        all_results.append(result)
    
    # 汇总分析
    print(f"\n{'='*60}")
    print("实验汇总")
    print(f"{'='*60}")
    
    # 按命中率排序
    by_hit_rate = sorted(all_results, key=lambda x: x.stats.hit_rate, reverse=True)
    print(f"\n命中率排名:")
    for r in by_hit_rate:
        print(f"  {r.strategy}: {r.stats.hit_rate:.2%}")
    
    # 按访问时间排序
    by_access_time = sorted(all_results, key=lambda x: x.stats.avg_access_time_ms)
    print(f"\n平均访问时间排名:")
    for r in by_access_time:
        print(f"  {r.strategy}: {r.stats.avg_access_time_ms:.4f} ms")
    
    # 综合最佳
    if len(all_results) >= 2:
        best = min(all_results, key=lambda x: (
            -x.stats.hit_rate,
            x.stats.avg_access_time_ms,
            x.stats.premature_evictions
        ))
        print(f"\n综合最佳: {best.strategy}")
        print(f"  - 命中率: {best.stats.hit_rate:.2%}")
        print(f"  - 访问时间: {best.stats.avg_access_time_ms:.4f} ms")
        print(f"  - 过早驱逐: {best.stats.premature_evictions}")
    
    # 保存结果
    if args.save_results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"eviction_results_{args.workload_type}_{timestamp}.json"
        
        output_data = {
            "config": {
                "strategies": strategies,
                "cache_size_gb": args.kv_cache_size_gb,
                "cache_size_tokens": cache_size_tokens,
                "workload_size": args.workload_size,
                "workload_type": args.workload_type,
                "num_tokens": args.num_tokens,
                "token_size_bytes": args.token_size_bytes,
                "bandwidth_gbps": args.bandwidth_gbps,
            },
            "results": [
                {
                    "strategy": r.strategy,
                    "hit_rate": r.stats.hit_rate,
                    "miss_count": r.stats.miss_count,
                    "hit_count": r.stats.hit_count,
                    "eviction_count": r.stats.eviction_count,
                    "premature_evictions": r.stats.premature_evictions,
                    "avg_access_time_ms": r.stats.avg_access_time_ms,
                    "avg_eviction_overhead_ms": r.stats.avg_eviction_overhead_ms,
                    "tier_access_counts": r.stats.tier_access_counts,
                    "simulation_time_ms": r.stats.simulation_time_ms,
                }
                for r in all_results
            ],
            "ranking": {
                "by_hit_rate": [r.strategy for r in by_hit_rate],
                "by_access_time": [r.strategy for r in by_access_time],
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n结果已保存: {filepath}")
    
    print("\nEviction验证实验完成!")


if __name__ == "__main__":
    main()
