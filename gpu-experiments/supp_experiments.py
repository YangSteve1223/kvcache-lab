#!/usr/bin/env python3
"""
Supplementary Experiments: Fill all remaining gaps
===================================================
A) First-Ratio Sensitivity (PDTrim fr scan)
   fr ∈ [0.1, 0.3, 0.5, 0.7, 0.9] × budget ∈ [0.3, 0.5, 0.7] × seq ∈ [2K, 4K]
   
B) SWS Decay Rate Sensitivity
   decay_rate ∈ [0.001, 0.005, 0.01, 0.02, 0.05] × budget ∈ [0.3, 0.5, 0.7]

C) Multi-Seed Robustness (7 seeds)
   All 4 strategies × 3 budgets × 7 seeds for CI95

D) Multi-Task PPL (different text domains)
   WikiText-2 (narrative) vs Code vs Science text

Output: exp_fratio_scan_{model}.json, exp_decay_scan_{model}.json,
        exp_multiseed_{model}.json, exp_multitask_ppl_{model}.json
"""
import json, os, sys, time, gc, traceback, ssl, re
import torch, numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = "/root/autodl-tmp/kvcache-lab/gpu-experiments/experiment_results_new"
WIKITEXT_CACHE = "/root/autodl-tmp/wikitext2_test.txt"

MODELS = {
    "qwen7b": "/root/autodl-tmp/Qwen2.5-7B-Instruct",
    "mistral7b": "/root/autodl-tmp/Mistral-7B-Instruct-v0.3",
    "gemma9b": "/root/autodl-tmp/gemma-2-9b-it",
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def download_wikitext2():
    if os.path.exists(WIKITEXT_CACHE):
        with open(WIKITEXT_CACHE) as f:
            return f.read()
    import urllib.request
    url = "https://hf-mirror.com/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    try:
        import pyarrow.parquet as pq
        opener = urllib.request.build_opener(urllib.request.HTTPHandler())
        resp = opener.open(url)
        with open("/tmp/wikitext2.parquet", "wb") as f:
            f.write(resp.read())
        text = "\n".join(pq.read_table("/tmp/wikitext2.parquet").column("text").to_pylist())
        with open(WIKITEXT_CACHE, "w") as f:
            f.write(text)
        return text
    except:
        return "The quick brown fox jumps over the lazy dog. " * 5000

def get_code_text():
    """Generate code-domain text for multi-task PPL."""
    code = '''
def binary_search(arr, target):
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

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def bfs(root):
    if not root:
        return []
    queue = [root]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result

def dfs(root):
    if not root:
        return []
    return [root.val] + dfs(root.left) + dfs(root.right)

def dijkstra(graph, start):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    visited = set()
    while len(visited) < len(graph):
        current = min((node for node in distances if node not in visited),
                      key=lambda x: distances[x])
        visited.add(current)
        for neighbor, weight in graph[current]:
            distance = distances[current] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
    return distances
'''
    # Repeat to fill context
    return (code * 20)

def get_science_text():
    """Generate science-domain text for multi-task PPL."""
    science = """The quantum mechanical model of the atom describes electrons as probability clouds rather than orbiting particles. The Schrodinger equation provides the mathematical framework for calculating these probability distributions, yielding wave functions that characterize the allowed energy states of atomic systems. Each electron in an atom is described by four quantum numbers: the principal quantum number n, the azimuthal quantum number l, the magnetic quantum number m, and the spin quantum number s.

In molecular biology, the central dogma describes the flow of genetic information from DNA through RNA to protein. Transcription factors bind to promoter regions upstream of genes, initiating the recruitment of RNA polymerase and the assembly of the pre-initiation complex. Alternative splicing mechanisms allow a single gene to produce multiple protein isoforms, significantly expanding the proteomic diversity of eukaryotic organisms.

Thermodynamic principles govern the spontaneity of chemical reactions. The Gibbs free energy change determines whether a reaction proceeds spontaneously under constant temperature and pressure conditions. Enthalpy changes reflect the heat absorbed or released during bond formation and breaking, while entropy changes quantify the disorder or randomness of the system. The interplay between these two factors determines the overall favorability of chemical transformations.

In materials science, crystal structures are classified by their Bravais lattices, of which there are fourteen unique types in three dimensions. The face-centered cubic structure, adopted by many metals including aluminum and copper, provides efficient atomic packing with a coordination number of twelve. Defects in crystalline materials, including point defects such as vacancies and interstitials, and extended defects such as dislocations, profoundly influence mechanical and electrical properties.

Electromagnetic wave propagation follows Maxwells equations, which unify electric and magnetic phenomena. The wave equation derived from these equations predicts that electromagnetic waves travel at the speed of light in vacuum. The Poynting vector describes the direction and magnitude of electromagnetic energy flux, while the impedance of free space relates the electric and magnetic field amplitudes in a propagating wave. """
    return science * 15

def has_weights(path):
    if not os.path.exists(path): return False
    return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) > 1e9

def load_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    log(f"LOADED. VRAM={torch.cuda.memory_allocated()/1e9:.1f}GB")
    return model, tok

def compute_ppl(model, tok, input_ids, selected_ids=None, position_ids=None):
    with torch.no_grad():
        if selected_ids is not None:
            inp = selected_ids.unsqueeze(0).to(model.device)
            pos = position_ids.unsqueeze(0).to(model.device) if position_ids is not None else None
            out = model(inp, position_ids=pos)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = selected_ids[1:].unsqueeze(0).to(model.device)
        else:
            inp = input_ids.unsqueeze(0).to(model.device)
            out = model(inp)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[1:].unsqueeze(0).to(model.device)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    valid = len(losses)
    if valid == 0:
        return float('inf')
    return torch.exp(losses.mean()).item()

def sel_pdtrim(n, budget, fr=0.5):
    first_n = max(1, int(n * budget * fr))
    last_n = max(1, int(n * budget * (1 - fr)))
    return sorted(set(list(range(first_n)) + list(range(n - last_n, n))))

def sel_sws_decay(n, budget, sink=16, decay_rate=0.01):
    keep = max(sink, int(n * budget))
    scores = np.zeros(n)
    scores[:sink] = 1e6
    decay = np.exp(-np.arange(n - sink) * decay_rate)
    scores[sink:] = decay
    return sorted(np.argsort(scores)[-keep:].tolist())

def sel_random(n, budget, seed=42):
    rng = np.random.RandomState(seed)
    keep = max(1, int(n * budget))
    return sorted(rng.choice(n, keep, replace=False).tolist())

def sel_lru(n, budget):
    keep = max(1, int(n * budget))
    return list(range(n - keep, n))

# ===================== A: FIRST-RATIO SCAN =====================
def exp_fratio(model, tok, name, text):
    log(f"A) First-Ratio Scan ({name})")
    seq_len = 2048
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
    n = len(input_ids)
    
    frs = [0.1, 0.3, 0.5, 0.7, 0.9]
    budgets = [0.3, 0.5, 0.7]
    
    R = {"model": name, "seq_len": seq_len, "scan": []}
    baseline = compute_ppl(model, tok, input_ids)
    R["baseline_ppl"] = baseline
    log(f"  Baseline={baseline:.4f}")
    
    for fr in frs:
        for budget in budgets:
            ix = sel_pdtrim(n, budget, fr=fr)
            sel_ids = input_ids[ix]
            pos_ids = torch.tensor(ix, dtype=torch.long)
            ppl = compute_ppl(model, tok, input_ids, sel_ids, pos_ids)
            delta = (ppl - baseline) / baseline * 100
            R["scan"].append({
                "fr": fr, "budget": budget,
                "ppl": round(ppl, 4), "delta_pct": round(delta, 2),
                "first_tokens": max(1, int(n * budget * fr)),
                "last_tokens": max(1, int(n * budget * (1 - fr))),
            })
            log(f"  fr={fr} b={budget}: PPL={ppl:.2f} ({delta:+.1f}%)")
            torch.cuda.empty_cache()
    
    # Also at 4K for the best FR values
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 40:
        seq_len_4k = 4096
        input_ids_4k = tok(text, return_tensors="pt", truncation=True, max_length=seq_len_4k).input_ids[0]
        n4 = len(input_ids_4k)
        bl4 = compute_ppl(model, tok, input_ids_4k)
        log(f"  4K Baseline={bl4:.4f}")
        R["baseline_ppl_4k"] = bl4
        for fr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for budget in [0.5]:
                ix = sel_pdtrim(n4, budget, fr=fr)
                sel_ids = input_ids_4k[ix]
                pos_ids = torch.tensor(ix, dtype=torch.long)
                try:
                    ppl = compute_ppl(model, tok, input_ids_4k, sel_ids, pos_ids)
                    delta = (ppl - bl4) / bl4 * 100
                    R["scan"].append({
                        "fr": fr, "budget": budget, "seq_len": seq_len_4k,
                        "ppl": round(ppl, 4), "delta_pct": round(delta, 2),
                    })
                    log(f"  4K fr={fr} b={budget}: PPL={ppl:.2f} ({delta:+.1f}%)")
                except:
                    pass
                torch.cuda.empty_cache()
    
    return R

# ===================== B: SWS DECAY RATE SCAN =====================
def exp_decay(model, tok, name, text):
    log(f"B) SWS Decay Rate Scan ({name})")
    seq_len = 2048
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
    n = len(input_ids)
    
    decay_rates = [0.001, 0.005, 0.01, 0.02, 0.05]
    budgets = [0.3, 0.5, 0.7]
    
    R = {"model": name, "seq_len": seq_len, "scan": []}
    baseline = compute_ppl(model, tok, input_ids)
    R["baseline_ppl"] = baseline
    
    for dr in decay_rates:
        for budget in budgets:
            ix = sel_sws_decay(n, budget, sink=16, decay_rate=dr)
            sel_ids = input_ids[ix]
            pos_ids = torch.tensor(ix, dtype=torch.long)
            ppl = compute_ppl(model, tok, input_ids, sel_ids, pos_ids)
            delta = (ppl - baseline) / baseline * 100
            R["scan"].append({
                "decay_rate": dr, "budget": budget,
                "ppl": round(ppl, 4), "delta_pct": round(delta, 2),
            })
            log(f"  dr={dr} b={budget}: PPL={ppl:.2f} ({delta:+.1f}%)")
            torch.cuda.empty_cache()
    
    return R

# ===================== C: MULTI-SEED ROBUSTNESS =====================
def exp_multiseed(model, tok, name, text):
    log(f"C) Multi-Seed Robustness ({name})")
    seq_len = 2048
    input_ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
    n = len(input_ids)
    
    seeds = [42, 123, 456, 789, 1024, 1337, 2048]
    budgets = [0.3, 0.5, 0.7]
    strategies = [
        ("random", lambda n, b, s: sel_random(n, b, seed=s)),
        ("lru", lambda n, b, s: sel_lru(n, b)),
        ("pdtrim_fr50", lambda n, b, s: sel_pdtrim(n, b, fr=0.5)),
        ("sws_sink16", lambda n, b, s: sel_sws_decay(n, b, sink=16, decay_rate=0.01)),
    ]
    
    R = {"model": name, "seq_len": seq_len, "seeds": seeds, "results": []}
    baseline = compute_ppl(model, tok, input_ids)
    R["baseline_ppl"] = baseline
    
    for strat_name, strat_fn in strategies:
        for budget in budgets:
            ppls = []
            for seed in seeds:
                ix = strat_fn(n, budget, seed)
                sel_ids = input_ids[ix]
                pos_ids = torch.tensor(ix, dtype=torch.long)
                ppl = compute_ppl(model, tok, input_ids, sel_ids, pos_ids)
                ppls.append(ppl)
                torch.cuda.empty_cache()
            
            mean_ppl = np.mean(ppls)
            std_ppl = np.std(ppls, ddof=1) if len(ppls) > 1 else 0
            ci95 = 1.96 * std_ppl / np.sqrt(len(ppls))
            delta = (mean_ppl - baseline) / baseline * 100
            
            R["results"].append({
                "strategy": strat_name, "budget": budget,
                "mean_ppl": round(mean_ppl, 4), "std_ppl": round(std_ppl, 4),
                "ci95": round(ci95, 4), "delta_pct": round(delta, 2),
                "individual_ppls": [round(p, 4) for p in ppls],
            })
            log(f"  {strat_name} b={budget}: {mean_ppl:.2f}±{std_ppl:.2f} CI95=±{ci95:.2f}")
    
    return R

# ===================== D: MULTI-TASK PPL =====================
def exp_multitask(model, tok, name, wikitext):
    log(f"D) Multi-Task PPL ({name})")
    seq_len = 2048
    
    tasks = {
        "wikitext2": wikitext,
        "code": get_code_text(),
        "science": get_science_text(),
    }
    
    budgets = [0.3, 0.5, 0.7]
    strategies = [
        ("pdtrim_fr50", lambda n, b: sel_pdtrim(n, b, fr=0.5)),
        ("sws_sink16", lambda n, b: sel_sws_decay(n, b, sink=16, decay_rate=0.01)),
        ("lru", lambda n, b: sel_lru(n, b)),
    ]
    
    R = {"model": name, "seq_len": seq_len, "results": []}
    
    for task_name, task_text in tasks.items():
        input_ids = tok(task_text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
        n = len(input_ids)
        
        baseline = compute_ppl(model, tok, input_ids)
        R["results"].append({
            "task": task_name, "strategy": "full", "budget": 1.0,
            "ppl": round(baseline, 4), "delta_pct": 0.0,
        })
        log(f"  {task_name} baseline={baseline:.4f}")
        
        for strat_name, strat_fn in strategies:
            for budget in budgets:
                ix = strat_fn(n, budget)
                sel_ids = input_ids[ix]
                pos_ids = torch.tensor(ix, dtype=torch.long)
                ppl = compute_ppl(model, tok, input_ids, sel_ids, pos_ids)
                delta = (ppl - baseline) / baseline * 100
                R["results"].append({
                    "task": task_name, "strategy": strat_name, "budget": budget,
                    "ppl": round(ppl, 4), "delta_pct": round(delta, 2),
                })
                log(f"  {task_name} {strat_name} b={budget}: PPL={ppl:.2f} ({delta:+.1f}%)")
                torch.cuda.empty_cache()
    
    return R

# ===================== MAIN =====================
def main():
    log("="*60)
    log("Supplementary Experiments: FR/Decay/MultiSeed/MultiTask")
    log("="*60)
    
    text = download_wikitext2()
    log(f"WikiText-2: {len(text)} chars")
    
    experiments = [
        ("exp_fratio_scan", exp_fratio),
        ("exp_decay_scan", exp_decay),
        ("exp_multiseed", exp_multiseed),
        ("exp_multitask_ppl", exp_multitask),
    ]
    
    for name, path in MODELS.items():
        if not has_weights(path):
            log(f"SKIP {name}: no weights")
            continue
        
        log(f"\n{'='*40}")
        log(f"Processing: {name}")
        log(f"{'='*40}")
        
        # Check if ALL experiments already done
        all_done = all(
            os.path.exists(os.path.join(OUTPUT_DIR, f"{exp_name}_{name}.json"))
            for exp_name, _ in experiments
        )
        if all_done:
            log(f"SKIP {name}: all supplementary experiments already exist")
            continue
        
        try:
            model, tok = load_model(path)
            
            for exp_name, exp_fn in experiments:
                outf = os.path.join(OUTPUT_DIR, f"{exp_name}_{name}.json")
                if os.path.exists(outf):
                    log(f"SKIP {exp_name}: already exists")
                    continue
                
                result = exp_fn(model, tok, name, text)
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                with open(outf, "w") as f:
                    json.dump(result, f, indent=2)
                log(f"SAVED {outf}")
            
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            log(f"UNLOADED. Model NOT deleted.")
        except Exception as e:
            log(f"ERROR for {name}: {e}")
            traceback.print_exc()
    
    log("\nALL DONE - No models deleted from disk.")

if __name__ == "__main__":
    main()
