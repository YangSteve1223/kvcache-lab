#!/usr/bin/env python3
"""
===============================================================================
Multi-Model Generalization Experiments for kvcache-lab
验证decode locality在不同模型架构上的普遍性

支持模型: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct
核心实验:
  1. Locality vs Context Length (256~8192, 50% budget ΔPPL)
  2. Memory-Quality Pareto (2K/4K/8K, narrative/code/QA)

显存需求: 所有模型8K以内 < 20GB, 5090-32GB完全够用
===============================================================================
"""
import json, warnings, time, os, gc
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np

warnings.filterwarnings("ignore")

# ========================= Tsinghua Mirror Support =========================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========================= 配置 =========================
MODEL_CONFIGS = {
    "Llama-3.1-8B-Instruct": {
        "path": "/root/autodl-tmp/Llama-3.1-8B-Instruct",
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
    },
    "Mistral-7B-Instruct-v0.3": {
        "path": "/root/autodl-tmp/Mistral-7B-Instruct-v0.3",
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
    },
    "Qwen2.5-7B-Instruct": {
        "path": "/root/autodl-tmp/Qwen2.5-7B-Instruct",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": True,
    },
}

OUTPUT_DIR = "/root/autodl-tmp/experiment_results_multimodel"

SEQ_LENGTHS_LOCALITY = [256, 512, 1024, 2048, 4096, 8192]
SEQ_LENGTHS_PARETO = [2048, 4096, 8192]
BUDGETS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TAA_ALPHA = 0.1
REMOTE_RATIO = 0.7

# ========================= 文本素材 =========================
TEXT = """Artificial intelligence has transformed numerous industries in recent years. From healthcare diagnostics to autonomous vehicles, machine learning algorithms are reshaping how we approach complex problems. The development of large language models represents a particularly significant advancement, enabling natural language understanding and generation at unprecedented scale.

In the realm of computer architecture, the shift toward specialized accelerators has been driven by the computational demands of deep learning. Graphics processing units, originally designed for rendering graphics, have become the workhorses of AI training and inference. The emergence of tensor processing units and neural network processors further illustrates this trend toward domain-specific hardware.

The concept of disaggregated computing has gained traction in data center architecture. By separating compute resources into specialized pools such as prefill and decode instances, systems can achieve better resource utilization and lower latency. This approach is particularly relevant for large language model serving, where the computational characteristics of prefill and decode phases differ significantly.

Memory management in distributed systems presents unique challenges. Key-value caches, which store intermediate attention states, must be efficiently transferred between compute nodes while minimizing latency. The design of cache eviction policies and compression strategies directly impacts both throughput and quality of service.

Network topology plays a crucial role in determining the performance of distributed training and inference systems. The bandwidth and latency characteristics of interconnects such as NVLink, PCIe, and Ethernet can become bottlenecks if not carefully managed. Recent advances in optical interconnects promise to alleviate some of these constraints.

The study of attention mechanisms has revealed interesting properties about how language models process information. Self-attention allows each token to attend to all other tokens, creating a rich contextual representation. However, this quadratic complexity in sequence length has motivated research into sparse attention patterns and efficient approximations.

Reinforcement learning from human feedback has emerged as a powerful technique for aligning language models with human preferences. By training reward models on human comparisons and using them to fine-tune policy models, researchers have achieved significant improvements in helpfulness and safety. The interplay between reward hacking and genuine capability improvement remains an active area of investigation.

The philosophy of consciousness remains one of the deepest unsolved problems. David Chalmers hard problem asks why and how physical processes give rise to subjective experience. Functionalist approaches suggest that consciousness arises from information processing patterns rather than specific substrate, while biological naturalism argues that consciousness requires biological implementation.

Quantum computing represents a fundamentally different paradigm for information processing. Unlike classical bits that exist in definite states, quantum bits or qubits can exist in superposition of states, enabling parallel computation of certain problems. Shor's algorithm for factoring and Grover's search algorithm demonstrate potential quantum advantages, though practical quantum computers face significant challenges in error correction and coherence maintenance.

The intersection of neuroscience and artificial intelligence has proven remarkably fruitful. Deep neural networks were inspired by biological neural circuits, and in turn, AI models have become tools for understanding brain function. Convolutional networks mirror the hierarchical processing in visual cortex, while transformer attention mechanisms share computational principles with prefrontal cortex working memory systems.

Sustainable computing has emerged as a critical concern as AI workloads grow exponentially. The energy consumption of training large language models can equal the lifetime carbon emissions of several automobiles. Techniques such as model distillation, pruning, and quantization aim to reduce computational requirements while preserving capability. Data center efficiency improvements through advanced cooling systems and renewable energy sourcing are equally important for responsible AI development."""

CODE_TEXT = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class KVCacheManager:
    def __init__(self, max_size_mb=1024):
        self.max_size = max_size_mb * 1024 * 1024
        self.cache = {}
        self.access_count = {}
        self.lru_order = []

    def get(self, key):
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            self.lru_order.remove(key)
            self.lru_order.append(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return
        while self._current_size() + len(value) > self.max_size:
            self._evict()
        self.cache[key] = value
        self.access_count[key] = 1
        self.lru_order.append(key)

    def _current_size(self):
        return sum(len(v) for v in self.cache.values())

    def _evict(self):
        if not self.lru_order:
            return
        evict_key = self.lru_order.pop(0)
        del self.cache[evict_key]
        del self.access_count[evict_key]

class PDDisaggregatedServer:
    def __init__(self, prefill_gpus, decode_gpus):
        self.prefill_instances = [GPUNode(g) for g in prefill_gpus]
        self.decode_instances = [GPUNode(g) for g in decode_gpus]
        self.kv_cache = KVCacheManager(max_size_mb=8192)
        self.request_queue = []

    def process_request(self, request):
        prefill_node = self._select_prefill(request)
        kv_tensors = prefill_node.prefill(request.prompt)
        self.kv_cache.put(request.id, kv_tensors)
        decode_node = self._select_decode()
        decode_node.start_decoding(request.id, kv_tensors)

    def _select_prefill(self, request):
        return min(self.prefill_instances, key=lambda n: n.load)

    def _select_decode(self):
        return min(self.decode_instances, key=lambda n: n.memory_used)

def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

class DistributedKVStore:
    def __init__(self, nodes):
        self.nodes = nodes
        self.replication_factor = 3
        self.consistent_hash = ConsistentHash(nodes)

    def get(self, key):
        node_ids = self.consistent_hash.get_nodes(key, self.replication_factor)
        for nid in node_ids:
            try:
                return self.nodes[nid].get(key)
            except NodeError:
                continue
        raise KeyNotFoundError(key)

    def put(self, key, value):
        node_ids = self.consistent_hash.get_nodes(key, self.replication_factor)
        for nid in node_ids:
            self.nodes[nid].put(key, value)"""

QA_TEXT = """Question: What is the capital of France?
Answer: The capital of France is Paris. Paris is located in the north-central part of France along the Seine River and has been the country's capital since the 10th century.

Question: How does a transformer model work?
Answer: A transformer model uses self-attention mechanisms to process input sequences in parallel. Each token attends to all other tokens through query, key, and value projections, computing weighted sums to capture contextual relationships. Multi-head attention allows the model to attend to different representation subspaces simultaneously.

Question: What is PD disaggregation in LLM serving?
Answer: PD disaggregation separates the prefill and decode phases of LLM inference onto different GPU instances. Prefill is compute-intensive and benefits from high FLOPS GPUs, while decode is memory-bandwidth-bound and benefits from GPUs with high memory capacity. This separation eliminates prefill-decode interference and allows independent resource scaling.

Question: Why is KV cache memory a bottleneck?
Answer: KV cache grows linearly with sequence length and must reside in GPU memory during decode. For a 7B parameter model with 32K context, KV cache requires approximately 1.8GB per request. This limits the number of concurrent requests a decode instance can serve, creating a memory bottleneck.

Question: What is locality in the context of KV cache access?
Answer: Locality refers to the observation that during the decode phase, a large portion of attention weight is concentrated on a small subset of recently generated tokens. This means many older remote KV entries contribute minimally to the decode output, enabling tiered storage strategies.

Question: How does sliding window attention differ from SWS?
Answer: Sliding window attention is an architectural modification that permanently restricts the attention receptive field. SWS is a runtime memory management policy that keeps only recent KV locally but retains remote KV in a separate tier that can be re-fetched on demand. Remote KV is demoted, not deleted.

Question: What is the relationship between working set and KV cache?
Answer: The working set in KV cache refers to the subset of KV entries that receive the majority of attention during decode. Similar to OS page working sets, the KV working set is typically much smaller than the total KV cache, enabling efficient tiered memory management.

Question: How does TAA guide KV cache management?
Answer: Task-Aware Attention introduces a cost vector that encodes the access cost of each KV entry (local vs remote). By adding a bias term to attention scores based on this cost, TAA encourages the model to concentrate attention on locally available KV entries, providing locality-aware guidance for memory tiering decisions."""

WORKLOADS = {"narrative": TEXT, "code": CODE_TEXT, "qa": QA_TEXT}

# ========================= 工具函数 =========================
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}", flush=True)

def compute_ppl(logits, input_ids):
    sl = logits[:, :-1, :].contiguous()
    lab = input_ids[:, 1:].contiguous()
    return torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))).item()

def make_input(tokenizer, text, target_len, device):
    tokens = tokenizer.encode(text)
    while len(tokens) < target_len + 200:
        tokens = tokens + tokens
    prompt = tokenizer.decode(tokens[:target_len], skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_len).to(device)
    ids = inputs.input_ids
    if ids.shape[1] < target_len:
        pad = torch.full((1, target_len - ids.shape[1]), tokenizer.eos_token_id or 0, dtype=ids.dtype, device=device)
        ids = torch.cat([ids, pad], dim=1)
    return ids

def make_sws_mask(seq_len, ws, alpha=0.0, cost_vector=None, device='cuda', dtype=torch.bfloat16):
    neg_inf = torch.finfo(dtype).min
    m = torch.full((seq_len, seq_len), neg_inf, dtype=dtype, device=device)
    for i in range(seq_len):
        s = max(0, i - ws + 1)
        m[i, s:i+1] = 0.0
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8:
            sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        m = m + b1d.unsqueeze(0).to(dtype)
    return m.unsqueeze(0).unsqueeze(0)

def make_cost_vector(seq_len, remote_ratio=0.7, device='cuda'):
    cv = torch.zeros(seq_len, device=device)
    cv[:int(seq_len * remote_ratio)] = 1.0
    return cv

def run_hooked(model, input_ids, mask):
    def mk(m):
        def ph(_, args, kwargs):
            kwargs['attention_mask'] = m
            return args, kwargs
        return ph
    hooks = [l.self_attn.register_forward_pre_hook(mk(mask), with_kwargs=True) for l in model.model.layers]
    try:
        with torch.no_grad():
            return model(input_ids=input_ids).logits
    finally:
        for h in hooks:
            h.remove()

def get_model_info(model):
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = getattr(config, 'hidden_size', 4096) // config.num_attention_heads
    kv_bytes = 2 * n_kv_heads * head_dim * n_layers * 2  # bf16
    return {
        "layers": n_layers,
        "kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "kv_bytes_per_token": kv_bytes,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
    }

# ========================= Locality Characterization Functions =========================
def compute_gini(values):
    """计算Gini系数，values是非负数组"""
    values = np.sort(np.array(values, dtype=np.float64))
    n = len(values)
    if n == 0 or np.sum(values) == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

def compute_active_set(attention_dist, threshold=0.8):
    """计算覆盖threshold比例attention所需的最小token子集占比"""
    sorted_attn = np.sort(attention_dist)[::-1]  # 降序
    cumsum = np.cumsum(sorted_attn)
    total = cumsum[-1]
    if total == 0:
        return 0.0
    k = np.searchsorted(cumsum, threshold * total) + 1
    return k / len(attention_dist)

def compute_remote_attention(attention_weights, remote_ratio=0.7):
    """
    计算remote attention占比
    attention_weights: shape [seq_len, seq_len] 的注意力矩阵
    remote_ratio: 前70%作为remote tier
    """
    seq_len = attention_weights.shape[-1]
    remote_start = int(seq_len * (1 - remote_ratio))
    # 对每个query position，计算remote tokens的attention总和
    if remote_start <= 0:
        return 0.0
    remote_attn = attention_weights[:, remote_start:].sum()
    total_attn = attention_weights.sum()
    if total_attn == 0:
        return 0.0
    return remote_attn / total_attn

def compute_top_k_coverage(attention_weights, k_percentiles=[1, 5, 10, 20]):
    """
    计算Top-k% tokens覆盖的attention比例
    attention_weights: shape [seq_len, seq_len] 的注意力矩阵
    """
    seq_len = attention_weights.shape[-1]
    # 沿key维度求平均，得到每个key position的attention权重
    avg_attn_per_key = attention_weights.mean(dim=0).cpu().numpy()
    
    sorted_attn = np.sort(avg_attn_per_key)[::-1]  # 降序
    cumsum = np.cumsum(sorted_attn)
    total = cumsum[-1]
    
    coverage = {}
    for k_pct in k_percentiles:
        k = max(1, int(seq_len * k_pct / 100))
        coverage[f"top_{k_pct}pct_coverage"] = cumsum[k-1] / total if total > 0 else 0.0
    
    return coverage

def is_long_tail(remote_attention_pct, threshold=0.10):
    """判断是否为long-tail attention pattern"""
    return remote_attention_pct < threshold

def extract_attention_and_compute_locality(model, input_ids, last_n_layers=None):
    """
    提取attention weights并计算locality指标
    
    Args:
        model: 语言模型
        input_ids: 输入token IDs
        last_n_layers: 只取最后n层，如果为None则取所有层
    
    Returns:
        locality_metrics: 包含所有locality指标的字典
    """
    seq_len = input_ids.shape[1]
    all_attentions = []
    
    def capture_attention(module, input, output):
        # output[1] 是 attention weights
        if len(output) > 1 and output[1] is not None:
            all_attentions.append(output[1].detach())
    
    # 注册hook捕获attention
    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(capture_attention))
    
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, output_attentions=True)
    except Exception as e:
        # fallback: 尝试不使用eager attention
        log(f"  Warning: output_attentions=True failed ({e}), trying without...")
        with torch.no_grad():
            _ = model(input_ids=input_ids)
        return None
    finally:
        for h in hooks:
            h.remove()
    
    if not all_attentions:
        log(f"  Warning: No attention weights captured")
        return None
    
    # 选择层（默认取后4层）
    if last_n_layers is None:
        last_n_layers = min(4, len(all_attentions))
    
    selected_attentions = all_attentions[-last_n_layers:]
    
    # 平均所有层和所有head的attention
    # attention shape: [batch, heads, seq_len, seq_len]
    avg_attention = torch.stack(selected_attentions).mean(dim=0).mean(dim=0)  # [seq_len, seq_len]
    
    # 归一化
    row_sums = avg_attention.sum(dim=-1, keepdim=True)
    row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
    avg_attention = avg_attention / row_sums
    
    # 计算各项指标
    # 1. 每个key position的平均attention
    key_attention = avg_attention.mean(dim=0).cpu().numpy()  # [seq_len]
    
    # 2. Gini系数
    gini = compute_gini(key_attention)
    
    # 3. Active set (覆盖80% attention)
    active_set_pct = compute_active_set(key_attention, threshold=0.8)
    
    # 4. Remote attention (前70% tokens)
    remote_attention_pct = float(compute_remote_attention(avg_attention.cpu().numpy(), remote_ratio=0.7))
    
    # 5. Top-k coverage
    top_k_coverage = compute_top_k_coverage(avg_attention, k_percentiles=[1, 5, 10, 20])
    
    # 6. Long-tail判断
    long_tail = is_long_tail(remote_attention_pct, threshold=0.10)
    
    locality_metrics = {
        "gini": round(gini, 3),
        "active_set_pct": round(active_set_pct, 4),
        "remote_attention_pct": round(remote_attention_pct, 4),
        "top_1pct_coverage": round(top_k_coverage["top_1pct_coverage"], 4),
        "top_5pct_coverage": round(top_k_coverage["top_5pct_coverage"], 4),
        "top_10pct_coverage": round(top_k_coverage["top_10pct_coverage"], 4),
        "top_20pct_coverage": round(top_k_coverage["top_20pct_coverage"], 4),
        "long_tail": long_tail,
    }
    
    return locality_metrics

# ========================= 主实验 =========================
def run_model_experiments(model_name, model_config, output_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"\n{'='*70}")
    log(f"Model: {model_name}")
    log(f"{'='*70}")

    model_path = model_config["path"]
    trust_rc = model_config.get("trust_remote_code", False)

    if not os.path.exists(model_path):
        log(f"Local path not found: {model_path}")
        log(f"Downloading from HuggingFace: {model_config['hf_id']}")
        model_path = model_config["hf_id"]

    log(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_rc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading model from {model_path} (bfloat16, sdpa)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_rc,
        attn_implementation="sdpa",
    )
    model.eval()

    info = get_model_info(model)
    vram_gb = torch.cuda.memory_allocated() / 1e9
    log(f"Model: {info['layers']}L, {info['kv_heads']} KV heads, head_dim={info['head_dim']}, "
        f"KV={info['kv_bytes_per_token']} bytes/token, VRAM={vram_gb:.1f}GB")

    mask_dtype = torch.bfloat16

    all_results = {
        "model_name": model_name,
        "model_path": model_config['hf_id'],
        "model_info": info,
        "mask_dtype": str(mask_dtype),
        "vram_gb": round(vram_gb, 1),
    }

    # =========================================================
    # Exp0: Locality Characterization (narrative, 4K & 8K only)
    # =========================================================
    log(f"\n--- Exp0: Locality Characterization ---")
    
    locality_chars = {}
    for seq_target in [4096, 8192]:
        log(f"  seq={seq_target}")
        try:
            ids = make_input(tokenizer, TEXT, seq_target, model.device)
            sl = ids.shape[1]
            log(f"  Actual seq_len={sl}, extracting attention weights...")
            
            locality_metrics = extract_attention_and_compute_locality(model, ids, last_n_layers=4)
            
            if locality_metrics is not None:
                locality_chars[f"seq_{seq_target}"] = locality_metrics
                log(f"  Gini={locality_metrics['gini']:.3f}, ActiveSet={locality_metrics['active_set_pct']:.2%}, "
                    f"RemoteAttn={locality_metrics['remote_attention_pct']:.2%}, Long-tail={locality_metrics['long_tail']}")
            else:
                log(f"  Failed to extract attention weights")
                locality_chars[f"seq_{seq_target}"] = None
            
            del ids; torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            log(f"  OOM at seq={seq_target}")
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"  ERROR at seq={seq_target}: {e}")
            torch.cuda.empty_cache()
    
    all_results["locality_characterization"] = locality_chars
    
    # PASS condition check
    if locality_chars.get(f"seq_4096"):
        m = locality_chars["seq_4096"]
        gini_pass = m["gini"] > 0.85
        active_pass = m["active_set_pct"] < 0.15
        remote_pass = m["remote_attention_pct"] < 0.10
        longtail_pass = m["long_tail"]
        
        log(f"\n  PASS Condition Check (4K narrative):")
        log(f"    Gini > 0.85:        {m['gini']:.3f} -> {'✓ PASS' if gini_pass else '✗ FAIL'}")
        log(f"    Active set < 15%:   {m['active_set_pct']:.2%} -> {'✓ PASS' if active_pass else '✗ FAIL'}")
        log(f"    Remote attn < 10%:  {m['remote_attention_pct']:.2%} -> {'✓ PASS' if remote_pass else '✗ FAIL'}")
        log(f"    Long-tail observable: {m['long_tail']} -> {'✓ PASS' if longtail_pass else '✗ FAIL'}")

    # =========================================================
    # Exp1: Locality vs Context Length (narrative, 50% budget)
    # =========================================================
    log(f"\n--- Exp1: Locality vs Context Length ---")

    locality_data = []
    for seq_target in SEQ_LENGTHS_LOCALITY:
        log(f"  seq={seq_target}")
        try:
            ids = make_input(tokenizer, TEXT, seq_target, model.device)
            sl = ids.shape[1]
            cv = make_cost_vector(sl, REMOTE_RATIO, model.device)

            with torch.no_grad():
                bl = model(input_ids=ids).logits
            bp = compute_ppl(bl, ids)
            log(f"  Baseline PPL: {bp:.4f}")

            ws = max(int(sl * 0.5), 32)

            # SWS only
            mask = make_sws_mask(sl, ws, device=model.device, dtype=mask_dtype)
            logits = run_hooked(model, ids, mask)
            sws_ppl = compute_ppl(logits, ids)
            sws_delta = (sws_ppl / bp - 1) * 100
            del mask; torch.cuda.empty_cache()

            # SWS + TAA
            mask_t = make_sws_mask(sl, ws, alpha=TAA_ALPHA, cost_vector=cv,
                                   device=model.device, dtype=mask_dtype)
            logits_t = run_hooked(model, ids, mask_t)
            taa_ppl = compute_ppl(logits_t, ids)
            taa_delta = (taa_ppl / bp - 1) * 100
            del mask_t; torch.cuda.empty_cache()

            locality_data.append({
                "seq_len": sl,
                "baseline_ppl": round(bp, 4),
                "budget_50_sws_ppl": round(sws_ppl, 4),
                "budget_50_sws_delta_pct": round(sws_delta, 2),
                "budget_50_sws_taa_ppl": round(taa_ppl, 4),
                "budget_50_sws_taa_delta_pct": round(taa_delta, 2),
            })
            log(f"  50%budget: SWS={sws_delta:+.2f}% SWS+TAA={taa_delta:+.2f}%")
            del ids; torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            log(f"  OOM at seq={seq_target}, stopping")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            log(f"  ERROR at seq={seq_target}: {e}")
            torch.cuda.empty_cache()

    all_results["locality_vs_length"] = locality_data

    # =========================================================
    # Exp2: Memory-Quality Pareto (3 workloads × 3 lengths)
    # =========================================================
    log(f"\n--- Exp2: Memory-Quality Pareto ---")

    pareto_results = {}
    for wname, wtext in WORKLOADS.items():
        log(f"  Workload: {wname}")
        workload_data = []

        for seq_target in SEQ_LENGTHS_PARETO:
            log(f"    seq={seq_target}")
            try:
                ids = make_input(tokenizer, wtext, seq_target, model.device)
                sl = ids.shape[1]
                cv = make_cost_vector(sl, REMOTE_RATIO, model.device)

                with torch.no_grad():
                    bl = model(input_ids=ids).logits
                bp = compute_ppl(bl, ids)
                log(f"    Baseline PPL: {bp:.4f}")

                seq_data = {"seq_len": sl, "baseline_ppl": round(bp, 4), "budgets": []}

                for budget in BUDGETS:
                    ws = max(int(sl * budget), 32)
                    if ws >= sl:
                        seq_data["budgets"].append({
                            "budget": budget, "ws": ws,
                            "sws_ppl": round(bp, 4), "sws_delta_pct": 0.0,
                            "sws_taa_ppl": round(bp, 4), "sws_taa_delta_pct": 0.0,
                        })
                        continue

                    # SWS only
                    mask = make_sws_mask(sl, ws, device=model.device, dtype=mask_dtype)
                    try:
                        logits = run_hooked(model, ids, mask)
                        ppl = compute_ppl(logits, ids)
                        delta = (ppl / bp - 1) * 100
                    except Exception as e:
                        log(f"      budget={budget:.0%} SWS ERROR: {e}")
                        ppl, delta = float('inf'), float('inf')
                    del mask; torch.cuda.empty_cache()

                    # SWS + TAA
                    mask_t = make_sws_mask(sl, ws, alpha=TAA_ALPHA, cost_vector=cv,
                                           device=model.device, dtype=mask_dtype)
                    try:
                        logits_t = run_hooked(model, ids, mask_t)
                        ppl_t = compute_ppl(logits_t, ids)
                        delta_t = (ppl_t / bp - 1) * 100
                    except Exception as e:
                        log(f"      budget={budget:.0%} SWS+TAA ERROR: {e}")
                        ppl_t, delta_t = float('inf'), float('inf')
                    del mask_t; torch.cuda.empty_cache()

                    r = {
                        "budget": budget, "ws": ws,
                        "sws_ppl": round(ppl, 4) if ppl != float('inf') else None,
                        "sws_delta_pct": round(delta, 2) if delta != float('inf') else None,
                        "sws_taa_ppl": round(ppl_t, 4) if ppl_t != float('inf') else None,
                        "sws_taa_delta_pct": round(delta_t, 2) if delta_t != float('inf') else None,
                    }
                    seq_data["budgets"].append(r)

                    sws_s = f"PPL={ppl:.4f}({delta:+.2f}%)" if ppl != float('inf') else "ERR"
                    taa_s = f"PPL={ppl_t:.4f}({delta_t:+.2f}%)" if ppl_t != float('inf') else "ERR"
                    log(f"      budget={budget:.0%} ws={ws}: SWS={sws_s} SWS+TAA={taa_s}")

                workload_data.append(seq_data)
                del ids; torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                log(f"    OOM at seq={seq_target}")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                log(f"    ERROR at seq={seq_target}: {e}")
                torch.cuda.empty_cache()

        pareto_results[wname] = workload_data

    all_results["pareto"] = pareto_results

    # 保存
    out_file = Path(output_dir) / f"{model_name.replace('/', '_')}_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"Results saved to {out_file}")

    # 摘要
    log(f"\n{'='*70}")
    log(f"SUMMARY: {model_name}")
    log(f"{'='*70}")
    log(f"Locality vs Context Length (50% budget):")
    for d in locality_data:
        log(f"  seq={d['seq_len']:5d}: SWS ΔPPL={d['budget_50_sws_delta_pct']:+.2f}% "
            f"SWS+TAA ΔPPL={d['budget_50_sws_taa_delta_pct']:+.2f}%")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to run. Default: all")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--skip-qwen", action="store_true",
                        help="Skip Qwen2.5-7B (already have data)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.models:
        models_to_run = {k: v for k, v in MODEL_CONFIGS.items() if k in args.models}
    else:
        models_to_run = MODEL_CONFIGS.copy()

    if args.skip_qwen and "Qwen2.5-7B-Instruct" in models_to_run:
        del models_to_run["Qwen2.5-7B-Instruct"]

    log(f"Models to run: {list(models_to_run.keys())}")
    log(f"Output: {args.output_dir}")

    all_model_results = {}
    for model_name, model_config in models_to_run.items():
        try:
            results = run_model_experiments(model_name, model_config, args.output_dir)
            all_model_results[model_name] = results
        except Exception as e:
            log(f"FATAL ERROR for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 保存汇总
    summary_file = Path(args.output_dir) / "all_models_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_model_results, f, indent=2)
    log(f"\nAll models summary saved to {summary_file}")

    # 对比表
    log(f"\n{'='*70}")
    log(f"CROSS-MODEL COMPARISON")
    log(f"{'='*70}")
    
    # Table 1: Locality Characterization
    log(f"\n--- Locality Characterization (narrative, 4K) ---")
    log(f"{'Model':<30s} {'Gini':>8s} {'Active%':>8s} {'Remote%':>9s} {'Long-tail':>12s} {'4K SWS Δ%':>10s}")
    log("-" * 80)
    for name, res in all_model_results.items():
        loc = res.get("locality_characterization", {}).get("seq_4096")
        loc_4k = next((d for d in res.get("locality_vs_length", []) if d["seq_len"] >= 4096), None)
        
        if loc:
            gini = f"{loc['gini']:.3f}"
            active = f"{loc['active_set_pct']:.0%}"
            remote = f"{loc['remote_attention_pct']:.0%}"
            longtail = "YES" if loc['long_tail'] else "NO"
        else:
            gini = active = remote = longtail = "???"
        
        sws_4k = f"{loc_4k['budget_50_sws_delta_pct']:+.2f}%" if loc_4k else "N/A"
        
        log(f"{name:<30s} {gini:>8s} {active:>8s} {remote:>9s} {longtail:>12s} {sws_4k:>10s}")
    
    # Table 2: Original locality vs length comparison
    log(f"\n--- Locality vs Context Length (50% budget) ---")
    log(f"{'Model':<30s} {'4K 50%SWS':>12s} {'4K 50%SWS+TAA':>15s} {'8K 50%SWS':>12s}")
    log("-" * 70)
    for name, res in all_model_results.items():
        loc_4k = next((d for d in res.get("locality_vs_length", []) if d["seq_len"] >= 4096), None)
        loc_8k = next((d for d in res.get("locality_vs_length", []) if d["seq_len"] >= 8192), None)
        s4k = f"{loc_4k['budget_50_sws_delta_pct']:+.2f}%" if loc_4k else "N/A"
        t4k = f"{loc_4k['budget_50_sws_taa_delta_pct']:+.2f}%" if loc_4k else "N/A"
        s8k = f"{loc_8k['budget_50_sws_delta_pct']:+.2f}%" if loc_8k else "N/A"
        log(f"{name:<30s} {s4k:>12s} {t4k:>15s} {s8k:>12s}")
    
    # PASS Summary
    log(f"\n{'='*70}")
    log(f"PASS CONDITION SUMMARY")
    log(f"{'='*70}")
    log(f"| {'Model':<28s} | {'Gini>0.85':>10s} | {'Active<15%':>10s} | {'Remote<10%':>11s} | {'Long-tail':>10s} |")
    log("-" * 80)
    for name, res in all_model_results.items():
        loc = res.get("locality_characterization", {}).get("seq_4096")
        if loc:
            gini_pass = "✓" if loc['gini'] > 0.85 else "✗"
            active_pass = "✓" if loc['active_set_pct'] < 0.15 else "✗"
            remote_pass = "✓" if loc['remote_attention_pct'] < 0.10 else "✗"
            longtail_pass = "✓" if loc['long_tail'] else "✗"
        else:
            gini_pass = active_pass = remote_pass = longtail_pass = "?"
        log(f"| {name:<28s} | {gini_pass:>10s} | {active_pass:>10s} | {remote_pass:>11s} | {longtail_pass:>10s} |")


if __name__ == "__main__":
    main()
