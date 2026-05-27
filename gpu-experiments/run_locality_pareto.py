#!/usr/bin/env python3
"""
Long Context Pareto Frontier - 核心论文实验
证明decode locality随context增长而增强
"""
import json, warnings, time
from datetime import datetime
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np

warnings.filterwarnings("ignore")
MODEL = "/root/autodl-tmp/Qwen2.5-7B-Instruct"

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

def log(msg): print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [INFO] {msg}", flush=True)
def compute_ppl(logits, input_ids):
    sl = logits[:, :-1, :].contiguous(); lab = input_ids[:, 1:].contiguous()
    return torch.exp(F.cross_entropy(sl.view(-1, sl.size(-1)), lab.view(-1))).item()

def make_input(tokenizer, text, target_len, device):
    tokens = tokenizer.encode(text)
    while len(tokens) < target_len + 200: tokens = tokens + tokens
    prompt = tokenizer.decode(tokens[:target_len], skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_len).to(device)
    ids = inputs.input_ids
    if ids.shape[1] < target_len:
        pad = torch.full((1, target_len - ids.shape[1]), tokenizer.eos_token_id or 0, dtype=ids.dtype, device=device)
        ids = torch.cat([ids, pad], dim=1)
    return ids

def make_sws_mask(seq_len, ws, alpha=0.0, cost_vector=None, device='cuda'):
    neg_inf = torch.finfo(torch.float16).min
    m = torch.full((seq_len, seq_len), neg_inf, dtype=torch.float16, device=device)
    for i in range(seq_len):
        s = max(0, i - ws + 1)
        m[i, s:i+1] = 0.0
    if alpha > 0.0 and cost_vector is not None:
        mu, sigma = cost_vector.mean(), cost_vector.std()
        if sigma < 1e-8: sigma = torch.tensor(1.0, device=device)
        b1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
        m = m + b1d.unsqueeze(0).to(torch.float16)
    return m.unsqueeze(0).unsqueeze(0)

def make_cost_vector(seq_len, remote_ratio=0.7, device='cuda'):
    cv = torch.zeros(seq_len, device=device)
    cv[:int(seq_len * remote_ratio)] = 1.0
    return cv

def run_hooked(model, input_ids, mask):
    def mk(m):
        def ph(_, args, kwargs):
            kwargs['attention_mask'] = m; return args, kwargs
        return ph
    hooks = [l.self_attn.register_forward_pre_hook(mk(mask), with_kwargs=True) for l in model.model.layers]
    try:
        with torch.no_grad(): return model(input_ids=input_ids).logits
    finally:
        for h in hooks: h.remove()

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = Path("/root/autodl-tmp/experiment_results_locality")
    out.mkdir(parents=True, exist_ok=True)
    
    log("=" * 60)
    log("Long Context Pareto + Workload Locality")
    log("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto",
        trust_remote_code=True, attn_implementation="sdpa")
    model.eval()
    log(f"Model loaded, VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    all_results = {}
    
    # ===================================================================
    # Part A: Long Context Pareto (1K, 2K, 4K)
    # ===================================================================
    log("\n=== Part A: Long Context Pareto ===")
    
    workloads = {
        "narrative": TEXT,
        "code": CODE_TEXT,
        "qa": QA_TEXT,
    }
    
    seq_targets = [1024, 2048, 4096]
    budgets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for wname, wtext in workloads.items():
        log(f"\n--- Workload: {wname} ---")
        workload_data = []
        
        for seq_target in seq_targets:
            log(f"\n  seq={seq_target}")
            try:
                ids = make_input(tokenizer, wtext, seq_target, model.device)
                sl = ids.shape[1]
                cv = make_cost_vector(sl, 0.7, model.device)
                
                # Baseline
                with torch.no_grad():
                    bl = model(input_ids=ids).logits
                bp = compute_ppl(bl, ids)
                log(f"  Baseline PPL: {bp:.4f}")
                
                seq_data = {"seq_len": sl, "baseline_ppl": round(bp, 4), "budgets": []}
                
                for budget in budgets:
                    ws = max(int(sl * budget), 32)
                    if ws >= sl:
                        seq_data["budgets"].append({"budget": budget, "ws": ws, "ppl": round(bp, 4), "delta_pct": 0.0})
                        continue
                    
                    # SWS only
                    mask = make_sws_mask(sl, ws, device=model.device)
                    try:
                        logits = run_hooked(model, ids, mask)
                        ppl = compute_ppl(logits, ids)
                        delta = (ppl/bp - 1)*100
                    except Exception as e:
                        log(f"    budget={budget:.0%} ERROR: {e}")
                        ppl, delta = float('inf'), float('inf')
                    del mask; torch.cuda.empty_cache()
                    
                    # SWS + TAA
                    mask_t = make_sws_mask(sl, ws, alpha=0.1, cost_vector=cv, device=model.device)
                    try:
                        logits_t = run_hooked(model, ids, mask_t)
                        ppl_t = compute_ppl(logits_t, ids)
                        delta_t = (ppl_t/bp - 1)*100
                    except:
                        ppl_t, delta_t = float('inf'), float('inf')
                    del mask_t; torch.cuda.empty_cache()
                    
                    r = {"budget": budget, "ws": ws,
                         "sws_ppl": round(ppl, 4) if ppl != float('inf') else None,
                         "sws_delta_pct": round(delta, 2) if delta != float('inf') else None,
                         "sws_taa_ppl": round(ppl_t, 4) if ppl_t != float('inf') else None,
                         "sws_taa_delta_pct": round(delta_t, 2) if delta_t != float('inf') else None}
                    seq_data["budgets"].append(r)
                    log(f"    budget={budget:.0%} ws={ws} SWS:PPL={ppl:.4f}({delta:+.1f}%) SWS+TAA:PPL={ppl_t:.4f}({delta_t:+.1f}%)")
                
                workload_data.append(seq_data)
                del ids; torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                log(f"  OOM at seq={seq_target}, stopping")
                torch.cuda.empty_cache()
                break
        
        all_results[f"pareto_{wname}"] = workload_data
    
    # ===================================================================
    # Part B: Locality Strength vs Context Length
    # ===================================================================
    log("\n=== Part B: Locality Strength (PPL at 50% budget across lengths) ===")
    
    locality_data = []
    for seq_target in [256, 512, 1024, 2048, 4096]:
        try:
            ids = make_input(tokenizer, TEXT, seq_target, model.device)
            sl = ids.shape[1]
            cv = make_cost_vector(sl, 0.7, model.device)
            
            with torch.no_grad():
                bl = model(input_ids=ids).logits
            bp = compute_ppl(bl, ids)
            
            ws = max(int(sl * 0.5), 32)
            mask = make_sws_mask(sl, ws, device=model.device)
            logits = run_hooked(model, ids, mask)
            sws_ppl = compute_ppl(logits, ids)
            sws_delta = (sws_ppl/bp - 1)*100
            
            mask_t = make_sws_mask(sl, ws, alpha=0.1, cost_vector=cv, device=model.device)
            logits_t = run_hooked(model, ids, mask_t)
            taa_ppl = compute_ppl(logits_t, ids)
            taa_delta = (taa_ppl/bp - 1)*100
            
            locality_data.append({
                "seq_len": sl, "baseline_ppl": round(bp, 4),
                "budget_50_sws_delta": round(sws_delta, 2),
                "budget_50_sws_taa_delta": round(taa_delta, 2),
            })
            log(f"  seq={sl}: 50%budget SWS={sws_delta:+.2f}% SWS+TAA={taa_delta:+.2f}%")
            del ids, mask, mask_t; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            log(f"  OOM at seq={seq_target}")
            torch.cuda.empty_cache()
            break
    
    all_results["locality_vs_length"] = locality_data
    
    # Save
    with open(out / "locality_all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log(f"\nALL LOCALITY EXPERIMENTS COMPLETE")
    log(f"Results: {out}/locality_all_results.json")

if __name__ == "__main__":
    main()
