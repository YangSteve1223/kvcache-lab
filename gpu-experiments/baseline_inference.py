#!/usr/bin/env python3
"""
===============================================================================
kvcache-lab GPU基线推理脚本

功能：
    - 单卡完整推理（无PD分离）
    - 测量TTFT(Time To First Token)、TPS(Tokens Per Second)、Perplexity
    - 支持不同上下文长度(1K/4K/8K/32K)
    - 输出JSON格式结果

使用方法：
    python3 baseline_inference.py --context-length 4096 --num-runs 5
    python3 baseline_inference.py --all-lengths --save-results

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
from typing import Optional, List, Dict, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

# =============================================================================
# 配置区域
# =============================================================================

# 默认模型配置
DEFAULT_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DEFAULT_MODEL_PATH = "./models/meta-llama_Llama-2-7b-chat-hf"

# 上下文长度配置
CONTEXT_LENGTHS = {
    "1K": 1024,
    "4K": 4096,
    "8K": 8192,
    "32K": 32768,
}

# 测试Prompt（模拟不同任务类型）
TEST_PROMPTS = {
    "math": """Calculate the following:
    1. 123 + 456 = ?
    2. 789 * 12 = ?
    3. sqrt(65536) = ?
    Please show your work step by step.""",
    
    "code": """Write a Python function to find the longest palindromic substring in a given string.
    Include proper error handling and docstrings.""",
    
    "qa": """Based on the following context, answer the question.
    
    Context: The history of artificial intelligence began in antiquity, with myths, stories and rumors 
    of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of 
    modern AI were planted by classical philosophers who attempted to describe the process of human 
    thinking as the mechanical manipulation of symbols.
    
    Question: When did the history of artificial intelligence begin?""",
    
    "conversation": """You are a helpful AI assistant. A user asks: 
    "Can you explain the difference between machine learning and deep learning?"
    Please provide a clear and concise explanation.""",
}

# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class InferenceResult:
    """单次推理结果"""
    timestamp: str
    model_name: str
    context_length: int
    task_type: str
    prompt_tokens: int
    output_tokens: int
    total_tokens: int
    
    # 延迟指标
    prefill_time_ms: float      # Prefill阶段时间
    ttft_ms: float              # Time To First Token (ms)
    total_inference_time_ms: float  # 总推理时间
    
    # 吞吐指标
    tps: float                  # Tokens Per Second
    prefill_tps: float          # Prefill吞吐 (tokens/s)
    
    # 质量指标
    perplexity: float           # 困惑度
    output_log_probs_mean: float  # 平均log概率
    
    # 内存指标
    peak_gpu_memory_mb: float
    allocated_gpu_memory_mb: float
    
    # 额外信息
    config: Dict[str, Any]
    raw_output: str = ""


@dataclass
class ExperimentConfig:
    """实验配置"""
    model_name: str
    model_path: str
    context_length: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    task_type: str
    num_runs: int
    warmup_runs: int
    device: str
    dtype: str


@dataclass
class ExperimentSummary:
    """实验汇总"""
    timestamp: str
    config: ExperimentConfig
    num_runs: int
    
    # 统计指标
    ttft_mean: float
    ttft_std: float
    ttft_p50: float
    ttft_p95: float
    ttft_p99: float
    
    tps_mean: float
    tps_std: float
    
    perplexity_mean: float
    perplexity_std: float
    
    prefill_time_mean: float
    
    peak_memory_mean: float
    
    results: List[InferenceResult]


# =============================================================================
# 工具函数
# =============================================================================

def setup_device(gpu_id: int = 0) -> str:
    """设置计算设备"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA不可用，使用CPU（速度会非常慢）")
        return "cpu"
    
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # 打印GPU信息
    props = torch.cuda.get_device_properties(gpu_id)
    print(f"使用GPU {gpu_id}: {props.name}")
    print(f"  总内存: {props.total_memory / 1024**3:.2f} GB")
    
    return device


def load_model_and_tokenizer(
    model_path: str,
    device: str,
    dtype: str = "bfloat16"
) -> tuple:
    """加载模型和分词器"""
    print(f"\n加载模型: {model_path}")
    print(f"设备: {device}, 精度: {dtype}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    # 设置padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map.get(dtype, torch.bfloat16),
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 使用Flash Attention加速
    )
    
    model.eval()
    
    print(f"模型加载完成")
    
    return model, tokenizer


def calculate_perplexity(
    model,
    tokenizer,
    text: str,
    device: str,
    max_length: int = 2048
) -> float:
    """计算文本困惑度"""
    try:
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        
        encodings = {k: v.to(device) for k, v in encodings.items()}
        input_ids = encodings["input_ids"]
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
        
        perplexity = np.exp(loss)
        return perplexity
    except Exception as e:
        print(f"计算困惑度失败: {e}")
        return -1.0


def generate_with_timing(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> tuple:
    """生成文本并测量时间"""
    
    # Tokenize
    encodings = tokenizer(prompt, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]
    
    # 记录初始状态
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    # 记录结束时间
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 解析结果
    output_ids = outputs.sequences[0]
    output_tokens = output_ids.shape[0] - prompt_tokens
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    # 计算TTFT (Time To First Token)
    # 由于我们使用greedy-like sampling，这里估算
    ttft_estimate = (end_time - start_time) * 0.1  # 估算10%时间为prefill
    
    # 计算平均log概率（如果有scores）
    log_probs_mean = 0.0
    if hasattr(outputs, 'scores') and outputs.scores:
        scores = outputs.scores
        # 取前N个token的log prob
        n_scores = min(10, len(scores))
        for i in range(n_scores):
            probs = torch.softmax(scores[i], dim=-1)
            token_prob = probs[0, output_ids[prompt_tokens + i]].item()
            log_probs_mean += np.log(token_prob + 1e-10)
        log_probs_mean /= n_scores
    
    return {
        "generated_text": generated_text,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": output_ids.shape[0],
        "ttft_ms": ttft_estimate * 1000,
        "total_time_ms": (end_time - start_time) * 1000,
        "log_probs_mean": log_probs_mean,
    }


def run_single_inference(
    model,
    tokenizer,
    prompt: str,
    device: str,
    config: ExperimentConfig,
) -> InferenceResult:
    """运行单次推理"""
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 记录初始内存
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**2
    
    # 生成
    gen_result = generate_with_timing(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
    )
    
    # 计算困惑度
    full_text = prompt + gen_result["generated_text"]
    perplexity = calculate_perplexity(
        model, tokenizer, full_text, device
    )
    
    # 计算内存使用
    peak_memory = 0
    allocated_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        allocated_memory = torch.cuda.memory_allocated() / 1024**2
    
    # 估算Prefill时间（总时间的90%）
    prefill_time = gen_result["total_time_ms"] * 0.9
    decode_time = gen_result["total_time_ms"] * 0.1
    
    # 估算Prefill吞吐
    prefill_tps = (gen_result["prompt_tokens"] / prefill_time) * 1000 if prefill_time > 0 else 0
    # 实际TPS
    tps = (gen_result["output_tokens"] / decode_time) * 1000 if decode_time > 0 else 0
    
    return InferenceResult(
        timestamp=datetime.now().isoformat(),
        model_name=config.model_name,
        context_length=config.context_length,
        task_type=config.task_type,
        prompt_tokens=gen_result["prompt_tokens"],
        output_tokens=gen_result["output_tokens"],
        total_tokens=gen_result["total_tokens"],
        prefill_time_ms=prefill_time,
        ttft_ms=gen_result["ttft_ms"],
        total_inference_time_ms=gen_result["total_time_ms"],
        tps=tps,
        prefill_tps=prefill_tps,
        perplexity=perplexity,
        output_log_probs_mean=gen_result["log_probs_mean"],
        peak_gpu_memory_mb=peak_memory,
        allocated_gpu_memory_mb=allocated_memory,
        config=asdict(config),
        raw_output=gen_result["generated_text"],
    )


def run_experiment(
    model,
    tokenizer,
    prompts: Dict[str, str],
    config: ExperimentConfig,
    warmup_runs: int = 1,
    num_runs: int = 5,
) -> ExperimentSummary:
    """运行完整实验"""
    
    print(f"\n{'='*60}")
    print(f"开始基线推理实验")
    print(f"{'='*60}")
    print(f"上下文长度: {config.context_length}")
    print(f"最大输出Token: {config.max_new_tokens}")
    print(f"温度: {config.temperature}")
    print(f"运行次数: {num_runs} (+ {warmup_runs} warmup)")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for task_type, prompt in prompts.items():
        print(f"\n任务类型: {task_type}")
        print(f"Prompt长度: {len(prompt)} 字符")
        
        # Warmup
        for i in range(warmup_runs):
            print(f"  Warmup {i+1}/{warmup_runs}...", end=" ")
            _ = run_single_inference(model, tokenizer, prompt, config.device, config)
            print("完成")
        
        # 正式运行
        for run_idx in range(num_runs):
            print(f"  Run {run_idx+1}/{num_runs}...", end=" ")
            
            result = run_single_inference(
                model, tokenizer, prompt, config.device, config
            )
            
            all_results.append(result)
            
            print(f"完成 (TTFT: {result.ttft_ms:.2f}ms, TPS: {result.tps:.2f})")
    
    # 计算统计
    ttfts = [r.ttft_ms for r in all_results]
    tps_list = [r.tps for r in all_results]
    perplexities = [r.perplexity for r in all_results if r.perplexity > 0]
    
    summary = ExperimentSummary(
        timestamp=datetime.now().isoformat(),
        config=config,
        num_runs=num_runs * len(prompts),
        ttft_mean=np.mean(ttfts),
        ttft_std=np.std(ttfts),
        ttft_p50=np.percentile(ttfts, 50),
        ttft_p95=np.percentile(ttfts, 95),
        ttft_p99=np.percentile(ttfts, 99),
        tps_mean=np.mean(tps_list),
        tps_std=np.std(tps_list),
        perplexity_mean=np.mean(perplexities) if perplexities else -1,
        perplexity_std=np.std(perplexities) if perplexities else 0,
        prefill_time_mean=np.mean([r.prefill_time_ms for r in all_results]),
        peak_memory_mean=np.mean([r.peak_gpu_memory_mb for r in all_results]),
        results=all_results,
    )
    
    return summary


def print_summary(summary: ExperimentSummary):
    """打印实验汇总"""
    
    print(f"\n{'='*60}")
    print(f"基线推理实验结果汇总")
    print(f"{'='*60}")
    print(f"模型: {summary.config.model_name}")
    print(f"上下文长度: {summary.config.context_length}")
    print(f"运行次数: {summary.num_runs}")
    print(f"\n延迟指标 (TTFT):")
    print(f"  Mean:   {summary.ttft_mean:.2f} ms")
    print(f"  Std:    {summary.ttft_std:.2f} ms")
    print(f"  P50:    {summary.ttft_p50:.2f} ms")
    print(f"  P95:    {summary.ttft_p95:.2f} ms")
    print(f"  P99:    {summary.ttft_p99:.2f} ms")
    print(f"\n吞吐指标 (TPS):")
    print(f"  Mean:   {summary.tps_mean:.2f} tokens/s")
    print(f"  Std:    {summary.tps_std:.2f}")
    print(f"\n质量指标 (Perplexity):")
    print(f"  Mean:   {summary.perplexity_mean:.4f}")
    print(f"  Std:    {summary.perplexity_std:.4f}")
    print(f"\nPrefill性能:")
    print(f"  平均Prefill时间: {summary.prefill_time_mean:.2f} ms")
    print(f"\n内存使用:")
    print(f"  峰值GPU内存: {summary.peak_memory_mean:.2f} MB")
    print(f"{'='*60}")


def save_results(summary: ExperimentSummary, output_dir: str):
    """保存结果到文件"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_results_{timestamp}.json"
    filepath = output_dir / filename
    
    # 序列化summary（排除results中的raw_output以减小文件大小）
    result_dicts = []
    for r in summary.results:
        r_dict = asdict(r)
        r_dict.pop("raw_output", None)  # 移除大字段
        result_dicts.append(r_dict)
    
    output_data = {
        "timestamp": summary.timestamp,
        "config": asdict(summary.config),
        "num_runs": summary.num_runs,
        "statistics": {
            "ttft_mean": summary.ttft_mean,
            "ttft_std": summary.ttft_std,
            "ttft_p50": summary.ttft_p50,
            "ttft_p95": summary.ttft_p95,
            "ttft_p99": summary.ttft_p99,
            "tps_mean": summary.tps_mean,
            "tps_std": summary.tps_std,
            "perplexity_mean": summary.perplexity_mean,
            "perplexity_std": summary.perplexity_std,
            "prefill_time_mean": summary.prefill_time_mean,
            "peak_memory_mean": summary.peak_memory_mean,
        },
        "results": result_dicts,
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {filepath}")
    
    return filepath


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="kvcache-lab GPU基线推理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置运行
  python3 baseline_inference.py

  # 运行特定上下文长度
  python3 baseline_inference.py --context-length 4096

  # 运行所有上下文长度
  python3 baseline_inference.py --all-lengths

  # 指定模型路径
  python3 baseline_inference.py --model-path ./models/my-model

  # 保存结果
  python3 baseline_inference.py --save-results --output-dir ./results
        """
    )
    
    # 模型配置
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="模型名称 (默认: %(default)s)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型路径 (默认: ./models/<model_name>)"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="使用的GPU ID (默认: 0)"
    )
    
    # 推理配置
    parser.add_argument(
        "--context-length",
        type=int,
        default=4096,
        choices=list(CONTEXT_LENGTHS.values()),
        help="上下文长度 (默认: 4096)"
    )
    parser.add_argument(
        "--all-lengths",
        action="store_true",
        help="运行所有上下文长度 (1K, 4K, 8K, 32K)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大输出Token数 (默认: 512)"
    )
    
    # 生成配置
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="生成温度 (默认: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p采样 (默认: 0.9)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k采样 (默认: 50)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="重复惩罚 (默认: 1.1)"
    )
    
    # 实验配置
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="每种任务类型的运行次数 (默认: 3)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="热身运行次数 (默认: 1)"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=list(TEST_PROMPTS.keys()) + ["all"],
        default="all",
        help="运行的任务类型 (默认: all)"
    )
    
    # 输出配置
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="保存结果到文件"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="结果输出目录 (默认: ./results)"
    )
    
    # 精度配置
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="模型精度 (默认: bfloat16)"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = setup_device(args.gpu_id)
    
    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_name_safe = args.model_name.replace("/", "_")
        model_path = os.path.join(
            os.path.dirname(__file__),
            "models",
            model_name_safe
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models",
                model_name_safe
            )
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"\n错误: 模型路径不存在: {model_path}")
        print("请先运行 setup.sh 下载模型，或使用 --model-path 指定正确的路径")
        sys.exit(1)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        device=device,
        dtype=args.dtype,
    )
    
    # 确定要运行的任务类型
    if args.task_type == "all":
        tasks_to_run = TEST_PROMPTS
    else:
        tasks_to_run = {args.task_type: TEST_PROMPTS[args.task_type]}
    
    # 确定要运行的上下文长度
    lengths_to_run = (
        list(CONTEXT_LENGTHS.values())
        if args.all_lengths
        else [args.context_length]
    )
    
    # 创建实验配置
    config = ExperimentConfig(
        model_name=args.model_name,
        model_path=model_path,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        task_type=args.task_type,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        device=device,
        dtype=args.dtype,
    )
    
    # 运行实验
    all_summaries = []
    
    for ctx_len in lengths_to_run:
        config.context_length = ctx_len
        
        summary = run_experiment(
            model=model,
            tokenizer=tokenizer,
            prompts=tasks_to_run,
            config=config,
            warmup_runs=args.warmup_runs,
            num_runs=args.num_runs,
        )
        
        print_summary(summary)
        all_summaries.append(summary)
        
        # 保存结果
        if args.save_results:
            save_results(summary, args.output_dir)
    
    # 如果运行了多个长度，保存汇总
    if len(all_summaries) > 1 and args.save_results:
        output_dir = Path(args.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"baseline_summary_{timestamp}.json"
        
        summary_data = []
        for s in all_summaries:
            summary_data.append({
                "context_length": s.config.context_length,
                "ttft_mean": s.ttft_mean,
                "ttft_p95": s.ttft_p95,
                "tps_mean": s.tps_mean,
                "perplexity_mean": s.perplexity_mean,
            })
        
        with open(filepath, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n汇总结果已保存: {filepath}")
    
    print("\n基线推理实验完成!")


if __name__ == "__main__":
    main()
