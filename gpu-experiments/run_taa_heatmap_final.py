#!/usr/bin/env python3
"""
===============================================================================
TAA Attention Heatmap 实验脚本

目标：可视化TAA如何改变Local vs Remote Attention Ratio

Exp1: Per-Layer Local vs Remote Attention Ratio
    - 对每一层，计算TAA前后local KV和remote KV分别获得了多少attention mass

Exp2: Attention Heatmap
    - 选取关键层（中间层和最后一层），画完整的attention weight heatmap
    - 对比: baseline vs TAA

Exp3: Local Attention Ratio across Layers
    - 折线图展示各层local attention ratio变化

===============================================================================
"""

import json
import warnings
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# 配置
# =============================================================================

MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"
OUTPUT_DIR = Path("/root/autodl-tmp/experiment_results_heatmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PD分离配置：前70% tokens是remote KV，后30%是local KV
REMOTE_RATIO = 0.7  # 前70%是remote

# 测试配置
SEQ_LENGTHS = [512, 1024]
ALPHA_VALUES = [0.0, 0.1, 0.2]  # 0.0=baseline, 0.1/0.2=TAA

# 测试文本
TEST_TEXT = """Artificial intelligence has transformed numerous industries in recent years. 
From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping 
how we approach complex problems. The development of large language models represents a 
particularly significant advancement, enabling natural language understanding and generation 
at unprecedented scale. These models, trained on vast corpora of text data, learn to predict 
the next token in a sequence, capturing intricate patterns of language, reasoning, and even 
world knowledge embedded in their parameters. The transformer architecture, with its attention 
mechanism allowing each token to attend to all other tokens, forms the backbone of most 
modern language models.

In the realm of computer architecture, the shift toward specialized accelerators has been 
driven by the computational demands of deep learning. Graphics processing units, originally 
designed for rendering graphics, have become the workhorses of AI training and inference. 
The emergence of tensor processing units and neural network processors further illustrates 
this trend toward domain-specific hardware. Memory bandwidth and capacity constraints often 
determine the achievable throughput for large model inference.

The concept of disaggregated computing has gained traction in data center architecture. 
By separating compute resources into specialized pools such as prefill and decode instances, 
systems can achieve better resource utilization and lower latency. This approach is particularly 
relevant for large language model serving, where the computational characteristics of prefill 
and decode phases differ significantly. Prefill involves processing the entire prompt and 
computing key-value caches, while decode generates tokens autoregressively.

Memory management in distributed systems presents unique challenges. Key-value caches, which 
store intermediate attention states, must be efficiently transferred between compute nodes 
while minimizing latency. The design of cache eviction policies and compression strategies 
directly impacts both throughput and quality of service. Transmission-aware attention mechanisms 
can help prioritize local key-value pairs to reduce transfer overhead."""


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}", flush=True)


# =============================================================================
# 工具函数
# =============================================================================

def make_input(tokenizer, text: str, target_len: int, device) -> torch.Tensor:
    """生成固定长度的input_ids"""
    tokens = tokenizer.encode(text)
    while len(tokens) < target_len + 100:
        tokens = tokens + tokens
    prompt = tokenizer.decode(tokens[:target_len], skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_len).to(device)
    ids = inputs.input_ids
    if ids.shape[1] < target_len:
        pad = torch.full((1, target_len - ids.shape[1]), tokenizer.eos_token_id or 0, dtype=ids.dtype, device=device)
        ids = torch.cat([ids, pad], dim=1)
    return ids


# =============================================================================
# Attention收集器 (使用hook)
# =============================================================================

class AttentionCollector:
    """
    使用hook收集每层的Q、K和attention weights
    支持GQA (Grouped Query Attention)
    """
    
    def __init__(self, model, seq_len: int):
        self.model = model
        self.seq_len = seq_len
        self.num_layers = len(model.model.layers)
        self.config = model.config
        
        # Qwen2.5-7B使用GQA
        self.num_q_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(self.config, 'num_key_value_heads', self.num_q_heads)
        self.head_dim = self.config.hidden_size // self.num_q_heads
        self.num_kv_groups = self.num_q_heads // self.num_kv_heads
        
        log(f"AttentionCollector: {self.num_q_heads} Q heads, {self.num_kv_heads} KV heads, head_dim={self.head_dim}")
        
        # 存储每层的数据
        self.layer_data = defaultdict(dict)
        self.hooks = []
    
    def setup_hooks(self, alpha: float, cost_vector: torch.Tensor, capture_full_attn: bool = True):
        """注册hooks收集Q、K、V和计算attention"""
        
        self.alpha = alpha
        self.cost_vector = cost_vector
        self.capture_full_attn = capture_full_attn
        self.hooks = []
        self.hidden_cache = {}  # 存储每层的hidden_states
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            # Pre-hook: 在forward之前获取hidden_states
            pre_handle = layer.self_attn.register_forward_pre_hook(
                self._make_pre_hook(layer_idx)
            )
            
            # Forward hook: 在forward之后计算attention
            handle = layer.self_attn.register_forward_hook(
                self._make_attention_hook(layer_idx)
            )
            
            self.hooks.extend([pre_handle, handle])
    
    def _make_pre_hook(self, layer_idx: int):
        """Pre-hook: 在forward之前存储hidden_states"""
        def pre_hook(module, args):
            # args格式: (hidden_states,) 或直接是hidden_states
            if isinstance(args, tuple) and len(args) > 0:
                self.hidden_cache[layer_idx] = args[0]
            elif isinstance(args, torch.Tensor):
                self.hidden_cache[layer_idx] = args
            return args
        return pre_hook
    
    def _get_current_hidden_states(self, layer_idx: int):
        """获取当前层的hidden_states"""
        return self.hidden_cache.get(layer_idx, None)
    
    def _make_attention_hook(self, layer_idx: int):
        """创建attention hook"""
        def hook(module, input, output):
            # 从缓存获取hidden_states
            hidden_states = self._get_current_hidden_states(layer_idx)
            
            # 如果缓存中没有，使用output（SDPA输出）
            if hidden_states is None:
                if isinstance(output, tuple) and len(output) > 0:
                    hidden_states = output[0]
                else:
                    return output
            
            # 如果hidden_states不可用，使用output
            if hidden_states is None or (isinstance(hidden_states, torch.Tensor) and hidden_states.numel() == 0):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
            
            # 计算Q、K、V
            q = module.q_proj(hidden_states)
            k = module.k_proj(hidden_states)
            v = module.v_proj(hidden_states)
            
            # Reshape: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
            bsz, seq_len, _ = q.shape
            q = q.view(bsz, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            
            # GQA: expand K、V to Q heads
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            
            # 存储Q、K、V
            self.layer_data[layer_idx]['q'] = q.detach().cpu()
            self.layer_data[layer_idx]['k'] = k.detach().cpu()
            self.layer_data[layer_idx]['v'] = v.detach().cpu()
            
            # 计算attention weights
            head_dim = q.shape[-1]
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(head_dim)
            
            # Causal mask
            seq_len_curr = scores.shape[2]
            neg_inf = torch.finfo(scores.dtype).min
            causal_mask = torch.triu(
                torch.ones(seq_len_curr, seq_len_curr, device=scores.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), neg_inf)
            
            # TAA bias
            if self.alpha > 0:
                mu = self.cost_vector.mean()
                sigma = self.cost_vector.std()
                if sigma < 1e-8:
                    sigma = torch.tensor(1.0, device=self.cost_vector.device)
                bias = -self.alpha * torch.tanh((self.cost_vector[:seq_len_curr] - mu) / sigma)
                scores = scores + bias.view(1, 1, 1, -1)
            
            # Softmax
            attn_weights = F.softmax(scores.float(), dim=-1)
            
            # 只对关键层存储完整attention matrix
            if self.capture_full_attn and layer_idx in {self.num_layers // 2, self.num_layers - 1}:
                self.layer_data[layer_idx]['attn_full'] = attn_weights.detach().cpu()
            
            # 对所有层存储平均的attention
            self.layer_data[layer_idx]['attn_avg'] = attn_weights.mean(dim=1).detach().cpu()
            
            return output
        
        return hook
    
    def cleanup(self):
        """移除所有hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def compute_layer_stats(self) -> Dict[int, Dict]:
        """分析每层的local vs remote attention ratio"""
        seq_len = self.seq_len
        remote_end = int(seq_len * REMOTE_RATIO)
        
        stats = {}
        
        for layer_idx in range(self.num_layers):
            if layer_idx not in self.layer_data or 'attn_avg' not in self.layer_data[layer_idx]:
                continue
            
            attn = self.layer_data[layer_idx]['attn_avg'][0]
            
            # Local: 后30% keys, Remote: 前70% keys
            local_mass = attn[:, remote_end:].sum(dim=-1)
            remote_mass = attn[:, :remote_end].sum(dim=-1)
            
            stats[layer_idx] = {
                'alpha': self.alpha,
                'local_ratio_mean': float(local_mass.mean().item()),
                'local_ratio_std': float(local_mass.std().item()),
                'remote_ratio_mean': float(remote_mass.mean().item()),
                'remote_ratio_std': float(remote_mass.std().item()),
                'local_mass_per_query': local_mass.cpu().numpy().tolist(),
            }
        
        return stats
    
    def get_heatmap_data(self, layer_idx: int) -> Optional[np.ndarray]:
        """获取指定层的完整attention matrix"""
        if layer_idx in self.layer_data and 'attn_full' in self.layer_data[layer_idx]:
            return self.layer_data[layer_idx]['attn_full'].mean(dim=1)[0].numpy()
        return None


# =============================================================================
# 实验运行
# =============================================================================

def run_single_experiment(
    model, tokenizer, device, seq_len: int, alpha: float, text: str, cost_vector: torch.Tensor
) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray], torch.Tensor]:
    """运行单次实验并返回layer stats和heatmap数据"""
    
    log(f"  Running seq_len={seq_len}, alpha={alpha}...")
    
    input_ids = make_input(tokenizer, text, seq_len, device)
    collector = AttentionCollector(model, seq_len)
    collector.setup_hooks(alpha, cost_vector)
    
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits
    finally:
        collector.cleanup()
    
    layer_stats = collector.compute_layer_stats()
    
    heatmap_data = {}
    key_layers = {len(model.model.layers) // 2, len(model.model.layers) - 1}
    for layer_idx in key_layers:
        hm = collector.get_heatmap_data(layer_idx)
        if hm is not None:
            heatmap_data[layer_idx] = hm
    
    return layer_stats, heatmap_data, logits


def run_all_experiments():
    """运行所有实验配置"""
    log("=" * 60)
    log("TAA Attention Heatmap Experiment")
    log("=" * 60)
    
    log(f"Loading model from {MODEL}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = model.device
    num_layers = len(model.model.layers)
    log(f"Model loaded: {num_layers} layers on {device}")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'num_layers': num_layers,
        'seq_lengths': SEQ_LENGTHS,
        'alpha_values': ALPHA_VALUES,
        'remote_ratio': REMOTE_RATIO,
        'experiments': []
    }
    
    all_heatmaps = {}
    
    for seq_len in SEQ_LENGTHS:
        log(f"\n{'='*40}")
        log(f"Processing seq_len={seq_len}")
        log(f"{'='*40}")
        
        cost_vector = torch.zeros(seq_len, device=device)
        remote_end = int(seq_len * REMOTE_RATIO)
        cost_vector[:remote_end] = 1.0
        
        seq_results = {
            'seq_len': seq_len,
            'remote_boundary': remote_end,
            'alpha_results': {}
        }
        
        for alpha in ALPHA_VALUES:
            layer_stats, heatmap_data, _ = run_single_experiment(
                model, tokenizer, device, seq_len, alpha, TEST_TEXT, cost_vector
            )
            
            seq_results['alpha_results'][f'alpha_{alpha}'] = {
                'alpha': alpha,
                'layer_stats': layer_stats,
            }
            
            for layer_idx, heatmap in heatmap_data.items():
                key = f'seq{seq_len}_alpha{alpha}_layer{layer_idx}'
                all_heatmaps[key] = {
                    'data': heatmap.tolist(),
                    'seq_len': seq_len,
                    'alpha': alpha,
                    'layer_idx': layer_idx,
                    'remote_boundary': remote_end,
                }
        
        all_results['experiments'].append(seq_results)
        torch.cuda.empty_cache()
    
    # 生成Exp3数据
    layer_ratio_data = {
        'seq_len': SEQ_LENGTHS[0],
        'num_layers': num_layers,
        'layers': list(range(num_layers)),
        'baseline': [],
        'taa_01': [],
        'taa_02': [],
    }
    
    for exp in all_results['experiments']:
        if exp['seq_len'] == layer_ratio_data['seq_len']:
            for alpha_key, alpha_data in exp['alpha_results'].items():
                layer_stats = alpha_data['layer_stats']
                for layer_idx in range(num_layers):
                    if layer_idx in layer_stats:
                        local_ratio = layer_stats[layer_idx]['local_ratio_mean']
                        if 'alpha_0.0' in alpha_key:
                            layer_ratio_data['baseline'].append(local_ratio)
                        elif 'alpha_0.1' in alpha_key:
                            layer_ratio_data['taa_01'].append(local_ratio)
                        elif 'alpha_0.2' in alpha_key:
                            layer_ratio_data['taa_02'].append(local_ratio)
    
    # 保存结果
    log("\nSaving results...")
    
    results_path = OUTPUT_DIR / "taa_heatmap_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"  Saved: {results_path}")
    
    heatmap_path = OUTPUT_DIR / "heatmap_data.json"
    with open(heatmap_path, 'w') as f:
        json.dump(all_heatmaps, f)
    log(f"  Saved: {heatmap_path}")
    
    layer_ratio_path = OUTPUT_DIR / "layer_ratio_data.json"
    with open(layer_ratio_path, 'w') as f:
        json.dump(layer_ratio_data, f, indent=2)
    log(f"  Saved: {layer_ratio_path}")
    
    viz_script = generate_visualization_script(layer_ratio_data, all_heatmaps)
    viz_path = OUTPUT_DIR / "visualize_results.py"
    with open(viz_path, 'w') as f:
        f.write(viz_script)
    log(f"  Saved: {viz_path}")
    
    # 打印摘要
    log("\n" + "=" * 60)
    log("EXPERIMENT SUMMARY")
    log("=" * 60)
    
    for exp in all_results['experiments']:
        log(f"\nSeq Len: {exp['seq_len']}, Remote boundary: {exp['remote_boundary']}")
        for alpha_key, alpha_data in exp['alpha_results'].items():
            layer_stats = alpha_data['layer_stats']
            local_ratios = [s['local_ratio_mean'] for s in layer_stats.values()]
            avg_local = np.mean(local_ratios) if local_ratios else 0
            avg_remote = np.mean([s['remote_ratio_mean'] for s in layer_stats.values()]) if layer_stats else 0
            log(f"  {alpha_key}: avg_local={avg_local:.4f}, avg_remote={avg_remote:.4f}")
    
    log(f"\nResults saved to: {OUTPUT_DIR}")
    log("Run 'python visualize_results.py' to generate plots")
    
    return all_results, layer_ratio_data, all_heatmaps


def generate_visualization_script(layer_ratio_data: Dict, heatmap_data: Dict) -> str:
    """生成可视化脚本"""
    
    viz_script = '''#!/usr/bin/env python3
"""
TAA Attention Heatmap 可视化脚本
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path("/root/autodl-tmp/experiment_results_heatmap")

with open(OUTPUT_DIR / "layer_ratio_data.json") as f:
    layer_data = json.load(f)
with open(OUTPUT_DIR / "heatmap_data.json") as f:
    all_heatmaps = json.load(f)
with open(OUTPUT_DIR / "taa_heatmap_results.json") as f:
    results = json.load(f)


def plot_layer_ratios():
    fig, ax = plt.subplots(figsize=(14, 7))
    
    layers = layer_data['layers']
    baseline = layer_data['baseline']
    taa_01 = layer_data['taa_01']
    taa_02 = layer_data['taa_02']
    
    ax.plot(layers, baseline, 'b-o', label='Baseline (alpha=0)', linewidth=2, markersize=5, alpha=0.8)
    ax.plot(layers, taa_01, 'g-s', label='TAA (alpha=0.1)', linewidth=2, markersize=5, alpha=0.8)
    ax.plot(layers, taa_02, 'r-^', label='TAA (alpha=0.2)', linewidth=2, markersize=5, alpha=0.8)
    
    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('Local Attention Mass Ratio', fontsize=13)
    ax.set_title('TAA Effect on Local vs Remote Attention per Layer (Qwen2.5-7B, Seq Len: %d)' % layer_data['seq_len'], fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, len(layers))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "layer_ratio_comparison.png", dpi=150)
    print("Saved: layer_ratio_comparison.png")
    plt.close()


def plot_heatmaps():
    plot_configs = [
        {'key': 'seq512_alpha0.0_layer13', 'title': 'Middle Layer - Baseline', 'color': 'Blues'},
        {'key': 'seq512_alpha0.2_layer13', 'title': 'Middle Layer - TAA', 'color': 'Reds'},
        {'key': 'seq512_alpha0.0_layer27', 'title': 'Last Layer - Baseline', 'color': 'Blues'},
        {'key': 'seq512_alpha0.2_layer27', 'title': 'Last Layer - TAA', 'color': 'Reds'},
    ]
    
    for cfg in plot_configs:
        key = cfg['key']
        if key not in all_heatmaps:
            continue
        
        data = all_heatmaps[key]
        attn_matrix = np.array(data['data'])
        seq_len = data['seq_len']
        alpha = data['alpha']
        layer = data['layer_idx']
        boundary = data['remote_boundary']
        
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(attn_matrix, cmap=cfg['color'], aspect='auto', vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Weight')
        
        ax.axvline(x=boundary - 0.5, color='lime', linestyle='--', linewidth=2.5)
        ax.text(boundary // 2, seq_len - 20, 'Remote (First 70%)', ha='center', va='top', color='white', fontsize=11, fontweight='bold')
        ax.text(boundary + (seq_len - boundary) // 2, seq_len - 20, 'Local (Last 30%)', ha='center', va='top', color='white', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)
        ax.set_title('Attention Heatmap - Layer %d, alpha=%.1f' % (layer, alpha), fontsize=13)
        
        plt.tight_layout()
        safe_key = key.replace('.', '_').replace('-', '_')
        plt.savefig(OUTPUT_DIR / ("heatmap_" + safe_key + ".png"), dpi=150)
        print("Saved: heatmap_" + safe_key + ".png")
        plt.close()
    
    # 对比图
    for layer in [13, 27]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for idx, (alpha_val, cmap, title) in enumerate([(0.0, 'Blues', 'Baseline'), (0.2, 'Reds', 'TAA (alpha=0.2)')]):
            key = 'seq512_alpha%.1f_layer%d' % (alpha_val, layer)
            if key not in all_heatmaps:
                continue
            data = all_heatmaps[key]
            attn = np.array(data['data'])
            ax = axes[idx]
            im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=0)
            ax.axvline(x=data['remote_boundary'] - 0.5, color='lime', linestyle='--', linewidth=2)
            ax.set_xlabel('Key Position', fontsize=11)
            ax.set_ylabel('Query Position', fontsize=11)
            ax.set_title('%s (Layer %d)' % (title, layer), fontsize=12)
            plt.colorbar(im, ax=ax, shrink=0.8)
        plt.suptitle('Attention Comparison - Layer %d' % layer, fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / ("heatmap_comparison_layer%d.png" % layer), dpi=150)
        print("Saved: heatmap_comparison_layer%d.png" % layer)
        plt.close()


def plot_exp1_summary():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, exp in enumerate(results['experiments']):
        ax = axes[idx]
        seq_len = exp['seq_len']
        boundary = exp['remote_boundary']
        
        alphas, local_ratios, remote_ratios = [], [], []
        for alpha_key, alpha_data in exp['alpha_results'].items():
            layer_stats = alpha_data['layer_stats']
            alphas.append(float(alpha_key.split('_')[1]))
            local_ratios.append(np.mean([s['local_ratio_mean'] for s in layer_stats.values()]))
            remote_ratios.append(np.mean([s['remote_ratio_mean'] for s in layer_stats.values()]))
        
        x = np.arange(len(alphas))
        width = 0.35
        ax.bar(x - width/2, local_ratios, width, label='Local (Last 30%)', color='steelblue')
        ax.bar(x + width/2, remote_ratios, width, label='Remote (First 70%)', color='coral')
        ax.set_xlabel('Alpha (TAA Coefficient)', fontsize=12)
        ax.set_ylabel('Attention Mass Ratio', fontsize=12)
        ax.set_title('Local vs Remote (Seq Len: %d)' % seq_len, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(['alpha=%.1f' % a for a in alphas])
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
    
    plt.suptitle('TAA Effect on Attention Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp1_summary.png", dpi=150)
    print("Saved: exp1_summary.png")
    plt.close()


def main():
    print("Generating visualizations...")
    plot_layer_ratios()
    plot_heatmaps()
    plot_exp1_summary()
    print("Done! Results in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
'''
    
    return viz_script


if __name__ == "__main__":
    run_all_experiments()
