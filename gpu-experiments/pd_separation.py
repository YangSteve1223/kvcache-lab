#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab PD分离基线脚本

功能：
    - P节点(prefill): 生成KV Cache → 序列化保存
    - KV传输: 模拟不同带宽 (1GB/s, 5GB/s, NVLink)
    - D节点(decode): 加载KV → decode生成
    - 测量各阶段延迟 + 传输数据量 + 生成质量

架构：
    [P-Node] --(KV Transfer)--> [D-Node]
      Prefill                      Decode

使用方法：
    # 单机模拟PD分离
    python3 pd_separation.py --mode simulate
    
    # 两节点模式（需要配置网络）
    python3 pd_separation.py --mode p-node --host localhost --port 50051
    python3 pd_separation.py --mode d-node --host localhost --port 50052

作者：kvcache-lab Team
===============================================================================
"""

import argparse
import json
import os
import sys
import time
import socket
import struct
import pickle
import threading
import warnings
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import shutil

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

# 忽略警告
warnings.filterwarnings("ignore")

# =============================================================================
# 配置区域
# =============================================================================

DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
SHARED_DIR = "./shared"  # P-D节点共享存储目录

# 带宽配置 (bytes/s)
BANDWIDTH_CONFIGS = {
    "1gbps": 1 * 1024**3,      # 1 GB/s (模拟普通网络)
    "5gbps": 5 * 1024**3,      # 5 GB/s (模拟较好网络)
    "10gbps": 10 * 1024**3,    # 10 GB/s (万兆网)
    "nvlink": 900 * 1024**3,   # 900 GB/s (NVLink理论带宽)
}

# 测试Prompt
TEST_PROMPT = """Calculate the following step by step:
1. 234 + 567 = ?
2. 1234 * 56 = ?
3. sqrt(4096) = ?
4. log2(1024) = ?
Show your work for each calculation."""


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class KVCacheData:
    """KV Cache数据结构"""
    # 形状信息
    batch_size: int
    seq_len: int
    num_layers: int
    num_heads: int
    head_dim: int
    
    # 实际KV数据
    # key_cache: List[Tensor], value_cache: List[Tensor]
    key_cache: List[torch.Tensor] = field(default_factory=list)
    value_cache: List[torch.Tensor] = field(default_factory=list)
    
    # 元数据
    token_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    
    def get_size_bytes(self) -> int:
        """计算序列化后的大小"""
        total_bytes = 0
        for k, v in zip(self.key_cache, self.value_cache):
            total_bytes += k.element_size() * k.nelement()
            total_bytes += v.element_size() * v.nelement()
        return total_bytes
    
    def serialize(self) -> bytes:
        """序列化为字节"""
        return pickle.dumps({
            'batch_size': self.batch_size,
            'seq_len': self.seq_len,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'key_cache': [k.cpu() for k in self.key_cache],
            'value_cache': [v.cpu() for v in self.value_cache],
            'token_ids': self.token_ids.cpu() if self.token_ids is not None else None,
            'attention_mask': self.attention_mask.cpu() if self.attention_mask is not None else None,
        })
    
    @staticmethod
    def deserialize(data: bytes) -> 'KVCacheData':
        """从字节反序列化"""
        obj = pickle.loads(data)
        return KVCacheData(
            batch_size=obj['batch_size'],
            seq_len=obj['seq_len'],
            num_layers=obj['num_layers'],
            num_heads=obj['num_heads'],
            head_dim=obj['head_dim'],
            key_cache=[k.to('cuda' if torch.cuda.is_available() else 'cpu') for k in obj['key_cache']],
            value_cache=[v.to('cuda' if torch.cuda.is_available() else 'cpu') for v in obj['value_cache']],
            token_ids=obj['token_ids'].to('cuda' if torch.cuda.is_available() else 'cpu') if obj['token_ids'] is not None else None,
            attention_mask=obj['attention_mask'].to('cuda' if torch.cuda.is_available() else 'cpu') if obj['attention_mask'] is not None else None,
        )


@dataclass
class TransferStats:
    """传输统计"""
    bandwidth_bps: int
    data_size_bytes: int
    transfer_time_ms: float
    effective_bandwidth_bps: float
    overhead_ms: float  # 序列化/反序列化开销
    
    # 拥塞指标
    congestion_level: str = "low"
    queue_length: int = 0


@dataclass
class PDStageResult:
    """PD分离阶段结果"""
    stage: str  # "prefill" or "decode"
    
    # 延迟分解
    compute_time_ms: float
    kv_save_time_ms: float = 0.0
    kv_load_time_ms: float = 0.0
    
    # KV传输
    kv_transfer: Optional[TransferStats] = None
    
    # 输出
    output_tokens: int = 0
    perplexity: float = 0.0
    
    # 内存
    peak_memory_mb: float = 0.0


@dataclass
class PDSeparationResult:
    """PD分离完整结果"""
    timestamp: str
    model_name: str
    prompt: str
    prompt_tokens: int
    
    # 阶段结果
    prefill_result: PDStageResult
    transfer_result: Optional[TransferStats] = None
    decode_result: Optional[PDStageResult] = None
    
    # 端到端指标
    ttft_ms: float = 0.0  # Time To First Token (D节点首token时间)
    total_decode_time_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    
    # 质量
    final_perplexity: float = 0.0
    output_text: str = ""
    
    # 带宽配置
    bandwidth_config: str = "unknown"
    
    config: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# KV Cache提取工具
# =============================================================================

class KVCacheExtractor:
    """从模型中提取KV Cache"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = None
        self.hooks = []
        
    def _register_hooks(self):
        """注册Hook来捕获KV Cache"""
        self.kv_cache = {'k': [], 'v': []}
        
        def hook_k(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                # 尝试获取attention output
                attn_output = output[0]
                self.kv_cache['k'].append(attn_output.detach().clone())
                self.kv_cache['v'].append(attn_output.detach().clone())
        
        def hook_v(module, input, output):
            pass
        
        # 遍历所有attention层
        for name, module in self.model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                if hasattr(module, 'o'):
                    # HuggingFace风格
                    self.hooks.append(module.o.register_forward_hook(hook_k))
                else:
                    self.hooks.append(module.register_forward_hook(hook_k))
    
    def _remove_hooks(self):
        """移除Hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> KVCacheData:
        """提取KV Cache"""
        
        self._register_hooks()
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
        finally:
            self._remove_hooks()
        
        # 获取配置
        config = self.model.config
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads
        
        # 创建KVCacheData
        seq_len = input_ids.shape[1]
        
        kv_data = KVCacheData(
            batch_size=1,
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            key_cache=[],
            value_cache=[],
            token_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # 简化：从outputs中提取past_key_values
        if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
            pkv = outputs.past_key_values
            if isinstance(pkv, tuple):
                for layer_idx in range(len(pkv)):
                    if isinstance(pkv[layer_idx], tuple) and len(pkv[layer_idx]) >= 2:
                        k, v = pkv[layer_idx]
                        kv_data.key_cache.append(k.squeeze(0).contiguous())
                        kv_data.value_cache.append(v.squeeze(0).contiguous())
                    else:
                        # 某些模型格式
                        kv_data.key_cache.append(pkv[layer_idx][0].squeeze(0).contiguous())
                        kv_data.value_cache.append(pkv[layer_idx][1].squeeze(0).contiguous())
        
        return kv_data


# =============================================================================
# 带宽限制器
# =============================================================================

class BandwidthThrottler:
    """带宽限制器 - 模拟不同网络条件"""
    
    def __init__(self, bandwidth_bps: int):
        self.bandwidth_bps = bandwidth_bps
        self.chunk_size = 1024 * 1024  # 1MB chunks
        
    def apply(self, data: bytes) -> float:
        """应用带宽限制，返回传输时间(秒)"""
        data_size = len(data)
        
        if self.bandwidth_bps <= 0:
            # 无限制
            return 0.0
        
        # 计算理论传输时间
        transfer_time = data_size / self.bandwidth_bps
        
        # 由于是模拟，返回理论时间
        # 实际环境中可使用 tc (traffic control) 命令
        return transfer_time
    
    def get_effective_bandwidth(self, data_size: int, transfer_time: float) -> float:
        """计算有效带宽"""
        if transfer_time > 0:
            return data_size / transfer_time
        return float('inf')


# =============================================================================
# P节点服务
# =============================================================================

class PNode:
    """Prefill节点 - 生成KV Cache"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        shared_dir: str = SHARED_DIR,
    ):
        self.device = device
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"加载P节点模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.kv_extractor = KVCacheExtractor(self.model, self.tokenizer)
        
        print(f"P节点初始化完成 (设备: {device})")
    
    def prefill(
        self,
        prompt: str,
        request_id: str,
        save_to_file: bool = True,
    ) -> Tuple[PDStageResult, Optional[str]]:
        """执行prefill阶段，返回KV Cache和统计"""
        
        # 清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        prompt_tokens = input_ids.shape[1]
        
        # 计时开始
        start_time = time.perf_counter()
        
        # Prefill计算
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
        
        compute_time = (time.perf_counter() - start_time) * 1000
        
        # 提取KV Cache
        kv_save_start = time.perf_counter()
        kv_data = self.kv_extractor.extract(input_ids, attention_mask)
        
        # 保存到共享存储
        kv_path = None
        if save_to_file:
            kv_path = self.shared_dir / f"kv_{request_id}.safetensors"
            self._save_kv_cache(kv_data, kv_path)
        
        kv_save_time = (time.perf_counter() - kv_save_start) * 1000
        
        # 获取峰值内存
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        result = PDStageResult(
            stage="prefill",
            compute_time_ms=compute_time,
            kv_save_time_ms=kv_save_time,
            output_tokens=0,  # Prefill不生成token
            peak_memory_mb=peak_memory,
        )
        
        return result, kv_path, kv_data
    
    def _save_kv_cache(self, kv_data: KVCacheData, path: Path):
        """保存KV Cache到文件"""
        tensors = {}
        for i, (k, v) in enumerate(zip(kv_data.key_cache, kv_data.value_cache)):
            tensors[f'key_layer_{i}'] = k
            tensors[f'value_layer_{i}'] = v
        
        # 保存为safetensors
        save_file(tensors, str(path))
        
        # 保存元数据
        metadata = {
            'token_ids': kv_data.token_ids,
            'attention_mask': kv_data.attention_mask,
            'seq_len': kv_data.seq_len,
            'num_layers': kv_data.num_layers,
        }
        metadata_path = path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'seq_len': metadata['seq_len'],
                'num_layers': metadata['num_layers'],
            }, f)


# =============================================================================
# D节点服务
# =============================================================================

class DNode:
    """Decode节点 - 使用KV Cache进行decode"""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        shared_dir: str = SHARED_DIR,
    ):
        self.device = device
        self.shared_dir = Path(shared_dir)
        
        print(f"加载D节点模型: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"D节点初始化完成 (设备: {device})")
    
    def load_kv_cache(self, kv_path: Path) -> Tuple[KVCacheData, float]:
        """加载KV Cache，返回数据和加载时间"""
        
        load_start = time.perf_counter()
        
        # 加载safetensors
        tensors = load_file(str(kv_path))
        
        # 加载元数据
        metadata_path = kv_path.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 重建KVCacheData
        kv_data = KVCacheData(
            batch_size=1,
            seq_len=metadata['seq_len'],
            num_layers=metadata['num_layers'],
            num_heads=self.model.config.num_attention_heads,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            key_cache=[],
            value_cache=[],
        )
        
        for i in range(metadata['num_layers']):
            kv_data.key_cache.append(tensors[f'key_layer_{i}'].to(self.device))
            kv_data.value_cache.append(tensors[f'value_layer_{i}'].to(self.device))
        
        load_time = (time.perf_counter() - load_start) * 1000
        
        return kv_data, load_time
    
    def decode(
        self,
        prompt: str,
        kv_data: Optional[KVCacheData] = None,
        max_new_tokens: int = 256,
    ) -> PDStageResult:
        """执行decode阶段"""
        
        # 清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Tokenize prompt
        encodings = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        input_ids = encodings["input_ids"].to(self.device)
        
        # 如果有KV Cache，直接使用
        if kv_data is not None and kv_data.token_ids is not None:
            # 使用KV Cache中的token ids
            input_ids = kv_data.token_ids
        
        # 计时开始
        start_time = time.perf_counter()
        
        # Decode生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        compute_time = (time.perf_counter() - start_time) * 1000
        output_tokens = outputs.sequences.shape[1] - input_ids.shape[1]
        
        # 解码输出
        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # 获取峰值内存
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        result = PDStageResult(
            stage="decode",
            compute_time_ms=compute_time,
            output_tokens=output_tokens,
            peak_memory_mb=peak_memory,
        )
        
        return result, output_text


# =============================================================================
# 传输管理器
# =============================================================================

class TransferManager:
    """KV传输管理器 - 模拟网络传输"""
    
    def __init__(self, bandwidth_bps: int):
        self.bandwidth = bandwidth_bps
        self.throttler = BandwidthThrottler(bandwidth_bps)
    
    def transfer_file(self, src_path: Path, dst_path: Path) -> TransferStats:
        """传输文件，返回统计信息"""
        
        if not src_path.exists():
            raise FileNotFoundError(f"源文件不存在: {src_path}")
        
        # 读取数据
        with open(src_path, 'rb') as f:
            data = f.read()
        
        data_size = len(data)
        
        # 序列化开销
        overhead_start = time.perf_counter()
        # 模拟序列化（实际已经序列化好了）
        serialized = data
        overhead_ms = (time.perf_counter() - overhead_start) * 1000
        
        # 传输时间
        transfer_start = time.perf_counter()
        transfer_time = self.throttler.apply(serialized)
        time.sleep(transfer_time)  # 模拟实际延迟
        actual_transfer_time = (time.perf_counter() - transfer_start) * 1000
        
        # 写入目标
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, 'wb') as f:
            f.write(serialized)
        
        # 计算有效带宽
        effective_bw = self.throttler.get_effective_bandwidth(
            data_size, actual_transfer_time / 1000
        )
        
        return TransferStats(
            bandwidth_bps=self.bandwidth,
            data_size_bytes=data_size,
            transfer_time_ms=actual_transfer_time,
            effective_bandwidth_bps=effective_bw,
            overhead_ms=overhead_ms,
        )
    
    def transfer_direct(self, kv_data: KVCacheData) -> TransferStats:
        """直接传输KV数据（不落盘）"""
        
        # 序列化
        serialize_start = time.perf_counter()
        serialized = kv_data.serialize()
        serialize_time = (time.perf_counter() - serialize_start) * 1000
        
        data_size = len(serialized)
        
        # 传输
        transfer_start = time.perf_counter()
        transfer_time = self.throttler.apply(serialized)
        time.sleep(transfer_time)
        actual_transfer_time = (time.perf_counter() - transfer_start) * 1000
        
        effective_bw = self.throttler.get_effective_bandwidth(
            data_size, actual_transfer_time / 1000
        )
        
        return TransferStats(
            bandwidth_bps=self.bandwidth,
            data_size_bytes=data_size,
            transfer_time_ms=actual_transfer_time,
            effective_bandwidth_bps=effective_bw,
            overhead_ms=serialize_time,
        )


# =============================================================================
# PD分离模拟器
# =============================================================================

class PDSeparationSimulator:
    """PD分离模拟器 - 完整的P→D流程"""
    
    def __init__(
        self,
        model_path: str,
        p_device: str = "cuda:0",
        d_device: str = "cuda:1",
        bandwidth_bps: int = BANDWIDTH_CONFIGS["10gbps"],
    ):
        self.p_device = p_device
        self.d_device = d_device
        self.bandwidth_bps = bandwidth_bps
        
        # 创建节点
        print("\n初始化P节点...")
        self.p_node = PNode(model_path, p_device)
        
        print("\n初始化D节点...")
        self.d_node = DNode(model_path, d_device)
        
        # 创建传输管理器
        self.transfer_manager = TransferManager(bandwidth_bps)
        
        print(f"\nPD分离模拟器初始化完成")
        print(f"  P节点: {p_device}")
        print(f"  D节点: {d_device}")
        print(f"  带宽: {bandwidth_bps / 1024**3:.1f} GB/s")
    
    def run(
        self,
        prompt: str,
        request_id: str,
        max_new_tokens: int = 256,
        use_file_transfer: bool = False,
    ) -> PDSeparationResult:
        """运行完整的PD分离流程"""
        
        print(f"\n{'='*60}")
        print(f"PD分离流程 - Request: {request_id}")
        print(f"{'='*60}")
        
        # Step 1: P节点Prefill
        print("\n[1/4] P节点Prefill...")
        prefill_result, kv_path, kv_data = self.p_node.prefill(
            prompt, request_id, save_to_file=use_file_transfer
        )
        print(f"  计算时间: {prefill_result.compute_time_ms:.2f} ms")
        print(f"  KV保存时间: {prefill_result.kv_save_time_ms:.2f} ms")
        print(f"  峰值内存: {prefill_result.peak_memory_mb:.2f} MB")
        
        # Step 2: KV传输
        print("\n[2/4] KV传输...")
        transfer_result = None
        
        if use_file_transfer and kv_path:
            # 文件传输
            dst_path = self.d_node.shared_dir / f"kv_{request_id}.safetensors"
            transfer_result = self.transfer_manager.transfer_file(kv_path, dst_path)
            # D节点加载
            kv_data, load_time = self.d_node.load_kv_cache(dst_path)
            transfer_result.kv_load_time_ms = load_time if hasattr(transfer_result, 'kv_load_time_ms') else load_time
        else:
            # 直接传输
            transfer_result = self.transfer_manager.transfer_direct(kv_data)
        
        print(f"  传输数据量: {transfer_result.data_size_bytes / 1024**2:.2f} MB")
        print(f"  传输时间: {transfer_result.transfer_time_ms:.2f} ms")
        print(f"  有效带宽: {transfer_result.effective_bandwidth_bps / 1024**3:.2f} GB/s")
        
        # Step 3: D节点Decode
        print("\n[3/4] D节点Decode...")
        decode_result, output_text = self.d_node.decode(
            prompt, kv_data, max_new_tokens
        )
        print(f"  计算时间: {decode_result.compute_time_ms:.2f} ms")
        print(f"  输出Token数: {decode_result.output_tokens}")
        
        # Step 4: 汇总
        print("\n[4/4] 汇总结果...")
        
        # TTFT = Prefill时间 + 传输时间 + D节点处理时间
        ttft = prefill_result.compute_time_ms + transfer_result.transfer_time_ms
        
        # E2E延迟 = Prefill时间 + 传输时间 + Decode时间
        e2e_latency = (
            prefill_result.compute_time_ms +
            transfer_result.transfer_time_ms +
            decode_result.compute_time_ms
        )
        
        result = PDSeparationResult(
            timestamp=datetime.now().isoformat(),
            model_name="Llama-2-7B",
            prompt=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            prompt_tokens=len(self.p_node.tokenizer.encode(prompt)),
            prefill_result=prefill_result,
            transfer_result=transfer_result,
            decode_result=decode_result,
            ttft_ms=ttft,
            total_decode_time_ms=decode_result.compute_time_ms,
            e2e_latency_ms=e2e_latency,
            output_text=output_text,
            bandwidth_config=f"{bandwidth_bps / 1024**3:.1f}GB/s",
        )
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"结果汇总:")
        print(f"  TTFT: {ttft:.2f} ms")
        print(f"  Decode时间: {decode_result.compute_time_ms:.2f} ms")
        print(f"  E2E延迟: {e2e_latency:.2f} ms")
        print(f"  输出Token数: {decode_result.output_tokens}")
        print(f"{'='*60}")
        
        return result


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="kvcache-lab PD分离基线脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 运行模式
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulate", "p-node", "d-node"],
        default="simulate",
        help="运行模式: simulate(单机模拟), p-node(P节点), d-node(D节点)"
    )
    
    # 节点配置
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    
    # 模型配置
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--p-device", type=str, default="cuda:0")
    parser.add_argument("--d-device", type=str, default="cuda:1")
    
    # 带宽配置
    parser.add_argument(
        "--bandwidth",
        type=str,
        choices=list(BANDWIDTH_CONFIGS.keys()),
        default="10gbps",
        help="带宽配置"
    )
    
    # 实验配置
    parser.add_argument("--prompt", type=str, default=TEST_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=3)
    
    # 输出
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./results")
    
    args = parser.parse_args()
    
    bandwidth_bps = BANDWIDTH_CONFIGS[args.bandwidth]
    
    if args.mode == "simulate":
        # 单机模拟模式
        print(f"\n{'='*60}")
        print("PD分离模拟器 - 单机模式")
        print(f"{'='*60}")
        print(f"模型路径: {args.model_path}")
        print(f"P节点设备: {args.p_device}")
        print(f"D节点设备: {args.d_device}")
        print(f"带宽: {args.bandwidth} ({bandwidth_bps / 1024**3:.1f} GB/s)")
        
        # 检查GPU数量
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count < 2:
                print(f"\n警告: 只有 {gpu_count} 个GPU，将在同一GPU上运行")
                args.d_device = args.p_device
        
        # 创建模拟器
        simulator = PDSeparationSimulator(
            model_path=args.model_path,
            p_device=args.p_device,
            d_device=args.d_device,
            bandwidth_bps=bandwidth_bps,
        )
        
        # 运行实验
        results = []
        for run_idx in range(args.num_runs):
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_idx}"
            
            result = simulator.run(
                prompt=args.prompt,
                request_id=request_id,
                max_new_tokens=args.max_new_tokens,
                use_file_transfer=True,
            )
            
            results.append(result)
            
            # 清理KV文件
            for f in Path(SHARED_DIR).glob(f"kv_{request_id}*"):
                f.unlink()
        
        # 保存结果
        if args.save_results:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"pd_separation_{args.bandwidth}_{timestamp}.json"
            
            output_data = {
                "bandwidth": args.bandwidth,
                "bandwidth_bps": bandwidth_bps,
                "num_runs": args.num_runs,
                "results": [
                    {
                        "ttft_ms": r.ttft_ms,
                        "e2e_latency_ms": r.e2e_latency_ms,
                        "prefill_time_ms": r.prefill_result.compute_time_ms,
                        "transfer_time_ms": r.transfer_result.transfer_time_ms if r.transfer_result else 0,
                        "decode_time_ms": r.decode_result.compute_time_ms if r.decode_result else 0,
                        "data_size_mb": r.transfer_result.data_size_bytes / 1024**2 if r.transfer_result else 0,
                    }
                    for r in results
                ],
            }
            
            with open(filepath, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\n结果已保存: {filepath}")
        
        print("\nPD分离实验完成!")
        
    elif args.mode == "p-node":
        print(f"P节点模式 - 监听 {args.host}:{args.port}")
        # TODO: 实现P节点网络服务模式
        print("TODO: 尚未实现")
        
    elif args.mode == "d-node":
        print(f"D节点模式 - 连接 {args.host}:{args.port}")
        # TODO: 实现D节点网络服务模式
        print("TODO: 尚未实现")


if __name__ == "__main__":
    main()
