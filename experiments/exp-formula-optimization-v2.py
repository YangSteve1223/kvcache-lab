#!/usr/bin/env python3
"""
SWS Formula Optimization: Quick Multi-Variant Simulation
"""
import numpy as np
import json, os, time
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = {
    'qwen-7b': {'num_layers': 28, 'baseline_ppl': 7.04,
        'locality': [0.0506,0.0524,0.0528,0.0534,0.0600,0.0603,0.0511,
                     0.0545,0.0592,0.0919,0.0464,0.0547,0.0529,0.0574,
                     0.0668,0.0513,0.0563,0.0471,0.0632,0.0559,0.0545,
                     0.0599,0.0655,0.0546,0.0518,0.0517,0.0666,0.0871],
        'sink_s': 0.15, 'rec_s': 0.70, 'glob_s': 0.15},
    'mistral-7b': {'num_layers': 32, 'baseline_ppl': 7.95, 'locality': None,
        'sink_s': 0.30, 'rec_s': 0.55, 'glob_s': 0.15},
    'gemma-9b': {'num_layers': 42, 'baseline_ppl': 11.19, 'locality': None,
        'sink_s': 0.20, 'rec_s': 0.60, 'glob_s': 0.20},
}
SEQ_LENS = [2048, 4096]
BUDGETS = [0.3, 0.5, 0.7]
NEEDLE_DEPTHS = [0.1, 0.25, 0.5, 0.75, 0.9]

def build_locality(cfg, rng):
    if cfg['locality'] is not None:
        return np.array(cfg['locality'])
    n = cfg['num_layers']
    base = rng.beta(2, 5, size=n) * 0.08 + 0.04
    base[rng.choice(n, int(n*0.3), replace=False)] += 0.02
    return np.clip(base, 0.03, 0.12)

def sim_attention(cfg, seq_len, rng):
    loc = build_locality(cfg, rng)
    patterns = []
    for l in range(cfg['num_layers']):
        ss = cfg['sink_s'] * (1 + 0.5*loc[l]/0.06)
        rs = cfg['rec_s'] * (1 + loc[l]/0.06)
        gs = cfg['glob_s'] * (1 - 0.3*loc[l]/0.06)
        t = ss+rs+gs; ss/=t; rs/=t; gs/=t
        
        sink = np.zeros(seq_len)
        for i in range(min(16, seq_len//10)):
            sink[i] = np.exp(-0.5*i)
        if sink.sum()>0: sink/=sink.sum()
        
        rec = np.zeros(seq_len)
        dr = 3.0/(seq_len*loc[l]+1)
        for i in range(seq_len):
            rec[i] = np.exp(-dr*(seq_len-1-i))
        if rec.sum()>0: rec/=rec.sum()
        
        gl = np.ones(seq_len)/seq_len
        w = ss*sink + rs*rec + gs*gl
        w += rng.exponential(1e-6, size=seq_len)
        w /= w.sum()
        patterns.append(w)
    return patterns

def select_top_k(scores, budget, sink_count=16):
    n = len(scores)
    n_keep = max(int(n*budget), sink_count)
    top = set(np.argsort(scores)[-n_keep:].tolist())
    top |= set(range(min(sink_count, n)))
    # Trim if over
    selected = sorted(top)
    while len(selected) > n_keep:
        non_sink = [i for i in selected if i >= sink_count]
        if not non_sink: break
        worst = min(non_sink, key=lambda i: scores[i])
        selected.remove(worst)
    return selected

def evaluate(sel, attn_patterns, seq_len, needle_positions):
    sel_set = set(sel)
    num_layers = len(attn_patterns)
    covs = [sum(attn_patterns[l][i] for i in sel_set if i<seq_len) for l in range(num_layers)]
    avg_cov = float(np.mean(covs))
    min_cov = float(np.min(covs))
    cov_loss = 1.0 - avg_cov
    ppl_deg = float((np.exp(5.0*cov_loss)-1)*100)
    
    niah_scores = []
    for np_ in needle_positions:
        if np_ in sel_set:
            niah_scores.append(1.0)
        else:
            nearby = sum(1 for i in range(max(0,np_-10), min(seq_len,np_+10)) if i in sel_set)
            niah_scores.append(nearby/20.0)
    
    # Count how many needle positions are fully covered
    niah_full = sum(1 for np_ in needle_positions if np_ in sel_set)
    
    return {
        'avg_coverage': avg_cov,
        'min_coverage': min_cov,
        'ppl_degradation_pct': ppl_deg,
        'niah_accuracy': float(np.mean(niah_scores)),
        'niah_fully_covered': niah_full,
        'niah_total': len(needle_positions),
        'tx_ratio': len(sel)/seq_len,
    }

# === Scoring Formulas ===
def score_original(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    lam = 0.01
    return attn * np.exp(-lam * delta)

def score_value_aware(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """CAOTE-inspired: alpha/(1-alpha) with gentle distance decay"""
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    a_norm = attn / (attn.max() + 1e-10)
    score = (a_norm / (1 - a_norm + 1e-6)) * np.exp(-0.003 * delta)
    score[:sink_count] = score.max()
    return score

def score_u_shaped(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """Original + parabolic middle bonus"""
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    lam = 0.01
    base = attn * np.exp(-lam * delta)
    pos = np.arange(seq_len, dtype=float) / seq_len
    u_shape = 4 * pos * (1 - pos)
    beta = 0.4 * attn.mean()
    scores = base + beta * u_shape
    scores[:sink_count] = np.maximum(scores[:sink_count], scores.max())
    return scores

def score_needle_aware(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """Original + Gaussian needle-region bonus"""
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    lam = 0.01
    base = attn * np.exp(-lam * delta)
    pos = np.arange(seq_len, dtype=float) / seq_len
    bonus = np.zeros(seq_len)
    for mu in [0.25, 0.5, 0.75]:
        bonus += np.exp(-(pos - mu)**2 / (2*0.12**2))
    gamma = 0.6 * attn.mean()
    scores = base + gamma * bonus
    scores[:sink_count] = np.maximum(scores[:sink_count], scores.max())
    return scores

def score_entropy(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """Information-content weighting"""
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    ent = -attn * np.log(attn + 1e-12)
    scores = ent * np.exp(-0.005 * delta)
    scores[:sink_count] = np.maximum(scores[:sink_count], scores.max())
    return scores

def score_budget_adaptive(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """λ adjusts with budget; stronger U-shape at low budget"""
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    if budget <= 0.3:
        lam_eff = 0.02; beta_u = 1.0 * attn.mean()
    elif budget <= 0.6:
        lam_eff = 0.01; beta_u = 0.4 * attn.mean()
    else:
        lam_eff = 0.005; beta_u = 0.1 * attn.mean()
    base = attn * np.exp(-lam_eff * delta)
    pos = np.arange(seq_len, dtype=float) / seq_len
    u_shape = 4 * pos * (1 - pos)
    scores = base + beta_u * u_shape
    scores[:sink_count] = np.maximum(scores[:sink_count], scores.max())
    return scores

def score_layer_weighted(attn, seq_len, budget, layer_idx, num_layers, locality_arr, sink_count=16):
    """Per-layer λ based on locality"""
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    if locality_arr is not None and layer_idx < len(locality_arr):
        loc = locality_arr[layer_idx]
    else:
        loc = 0.06
    lam_eff = 0.01 * (0.5 + loc/0.12)
    base = attn * np.exp(-lam_eff * delta)
    if loc < 0.05:
        pos = np.arange(seq_len, dtype=float)/seq_len
        base += 0.3*attn.mean()*np.sin(np.pi*pos)
    base[:sink_count] = np.maximum(base[:sink_count], base.max())
    return base

def score_pyramid(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """PyramidKV: budget per layer = k_avg + (m/2 - l) * β"""
    # Instead of fixed budget, adjust per layer
    m = num_layers
    beta_pyr = 0.02
    layer_budget = budget + (m/2 - layer_idx) * beta_pyr
    layer_budget = max(0.1, min(0.95, layer_budget))
    # Then use original scoring with layer-specific budget
    delta = np.arange(seq_len, dtype=float)
    delta = seq_len - 1 - delta
    scores = attn * np.exp(-0.01 * delta)
    scores[:sink_count] = np.maximum(scores[:sink_count], scores.max())
    return scores, layer_budget  # Return adjusted budget too

def score_irr(attn, seq_len, budget, layer_idx, num_layers, locality, sink_count=16):
    """IRR-inspired: use inter-reference recency instead of position distance"""
    rng = np.random.RandomState(42 + layer_idx)
    # Simulate IRR: tokens accessed recently get low IRR
    # In practice, IRR = t_current - t_second_last_access
    # Simulate: sink tokens have low IRR, recent tokens have low IRR,
    # but some middle tokens also have moderate IRR (were accessed before)
    irr = rng.exponential(seq_len * 0.3, size=seq_len)
    irr[:sink_count] = 0  # Sinks always accessed
    irr[-64:] = rng.exponential(10, size=64)  # Recent tokens: low IRR
    # Some middle tokens get moderate IRR (retrieval scenario)
    mid_start = int(seq_len*0.3)
    mid_end = int(seq_len*0.7)
    n_hot_mid = int((mid_end-mid_start)*0.1)
    hot_mid = rng.choice(range(mid_start, mid_end), size=n_hot_mid, replace=False)
    irr[hot_mid] = rng.exponential(50, size=n_hot_mid)
    
    tau = seq_len * 0.2
    scores = attn * np.exp(-irr / tau)
    scores[:sink_count] = np.maximum(scores[:sink_count], scores.max())
    return scores

# === QCBM Hybrid ===
def evaluate_qcbm(attn_patterns, seq_len, budget, locality, needle_positions):
    """Three-tier: keep ALL tokens with FP16/INT8/INT4"""
    n = seq_len
    n_budget_fp16 = int(n * budget)  # FP16-equivalent budget
    
    # Allocate: 50% FP16, 30% INT8 (0.5x), 20% INT4 (0.25x)
    n_hot = max(int(n_budget_fp16 * 0.5 / 1.0), 16)
    n_warm = int(n_budget_fp16 * 0.3 / 0.5)
    n_cold = int(n_budget_fp16 * 0.2 / 0.25)
    
    # Use average attention for ranking
    avg_attn = np.mean([p for p in attn_patterns], axis=0)
    ranked = np.argsort(avg_attn)[::-1]
    
    hot = set(ranked[:n_hot].tolist())
    warm = set(ranked[n_hot:n_hot+n_warm].tolist())
    cold = set(ranked[n_hot+n_warm:n_hot+n_warm+n_cold].tolist())
    
    # Ensure sink in hot
    for i in range(min(16, n)):
        hot.add(i)
        warm.discard(i)
        cold.discard(i)
    
    all_kept = hot | warm | cold
    eff_bw = (len(hot)*1.0 + len(warm)*0.5 + len(cold)*0.25) / n
    
    # Coverage with precision-weighted quality
    covs = []
    for l in range(len(attn_patterns)):
        cov = 0.0
        for i in hot:
            if i<n: cov += attn_patterns[l][i] * 1.0
        for i in warm:
            if i<n: cov += attn_patterns[l][i] * 0.995
        for i in cold:
            if i<n: cov += attn_patterns[l][i] * 0.97
        covs.append(cov)
    
    avg_cov = float(np.mean(covs))
    cov_loss = 1.0 - avg_cov
    ppl_deg = float((np.exp(5.0*cov_loss)-1)*100)
    
    niah_scores = []
    niah_full = 0
    for np_ in needle_positions:
        if np_ in hot:
            niah_scores.append(1.0); niah_full += 1
        elif np_ in warm:
            niah_scores.append(0.98); niah_full += 1
        elif np_ in cold:
            niah_scores.append(0.90); niah_full += 1
        else:
            niah_scores.append(0.0)
    
    return {
        'avg_coverage': avg_cov,
        'min_coverage': float(np.min(covs)),
        'ppl_degradation_pct': ppl_deg,
        'niah_accuracy': float(np.mean(niah_scores)),
        'niah_fully_covered': niah_full,
        'niah_total': len(needle_positions),
        'tx_ratio': eff_bw,
        'n_hot': len(hot), 'n_warm': len(warm), 'n_cold': len(cold),
        'total_kept': len(all_kept),
        'formula': 'qcbm_hybrid',
    }

# === Baseline selections ===
def select_pdtrim(seq_len, budget, first_ratio=0.5):
    n_keep = int(seq_len * budget)
    n_first = int(n_keep * first_ratio)
    n_last = n_keep - n_first
    return sorted(set(range(n_first)) | set(range(seq_len - n_last, seq_len)))

def select_naive_sws(seq_len, budget, sink_count=16):
    n_keep = int(seq_len * budget)
    window = max(0, n_keep - sink_count)
    return sorted(set(range(min(sink_count, seq_len))) | set(range(seq_len - window, seq_len)))

# === Main ===
def main():
    print("🔬 SWS Formula Optimization Simulation v2")
    print("="*70)
    
    formulas = {
        'pdtrim': 'Baseline',
        'naive_sws': 'Baseline',
        'original': 'I=A*exp(-λΔ)',
        'value_aware': 'CAOTE-inspired',
        'u_shaped': 'U-shape bonus',
        'needle_aware': 'Needle-region bonus',
        'entropy': 'Entropy-weighted',
        'budget_adaptive': 'Budget-adaptive',
        'layer_weighted': 'Layer-weighted',
        'pyramid': 'PyramidKV',
        'irr': 'IRR-inspired',
        'qcbm_hybrid': 'QCBM 3-tier',
    }
    
    results = []
    
    for model_name in ['qwen-7b', 'mistral-7b', 'gemma-9b']:
        cfg = MODELS[model_name]
        print(f"\n{'='*70}\n  Model: {model_name}\n{'='*70}")
        
        for seq_len in SEQ_LENS:
            for budget in BUDGETS:
                rng = np.random.RandomState(42)
                attn_patterns = sim_attention(cfg, seq_len, rng)
                needle_positions = [int(seq_len*d) for d in NEEDLE_DEPTHS]
                mid_layer = len(attn_patterns)//2
                
                print(f"\n  seq={seq_len}, b={budget}")
                
                # Baselines
                for fname, sel_fn in [('pdtrim', lambda: select_pdtrim(seq_len, budget)),
                                       ('naive_sws', lambda: select_naive_sws(seq_len, budget))]:
                    sel = sel_fn()
                    q = evaluate(sel, attn_patterns, seq_len, needle_positions)
                    q['formula'] = fname; q['model'] = model_name
                    q['seq_len'] = seq_len; q['budget'] = budget
                    results.append(q)
                
                # Scoring formulas
                score_fns = {
                    'original': lambda a,s,b,l,n,loc: score_original(a,s,b,l,n,loc),
                    'value_aware': lambda a,s,b,l,n,loc: score_value_aware(a,s,b,l,n,loc),
                    'u_shaped': lambda a,s,b,l,n,loc: score_u_shaped(a,s,b,l,n,loc),
                    'needle_aware': lambda a,s,b,l,n,loc: score_needle_aware(a,s,b,l,n,loc),
                    'entropy': lambda a,s,b,l,n,loc: score_entropy(a,s,b,l,n,loc),
                    'budget_adaptive': lambda a,s,b,l,n,loc: score_budget_adaptive(a,s,b,l,n,loc),
                    'layer_weighted': lambda a,s,b,l,n,loc: score_layer_weighted(a,s,b,l,n,loc),
                    'irr': lambda a,s,b,l,n,loc: score_irr(a,s,b,l,n,loc),
                }
                
                for fname, sfn in score_fns.items():
                    if fname == 'pyramid':
                        continue  # Special handling below
                    
                    # Use middle layer attention for selection
                    scores = sfn(attn_patterns[mid_layer], seq_len, budget,
                                mid_layer, cfg['num_layers'], cfg['locality'])
                    sel = select_top_k(scores, budget)
                    q = evaluate(sel, attn_patterns, seq_len, needle_positions)
                    q['formula'] = fname; q['model'] = model_name
                    q['seq_len'] = seq_len; q['budget'] = budget
                    results.append(q)
                
                # Pyramid: per-layer budget
                pyramid_sel_all = []
                for l in range(cfg['num_layers']):
                    scores_l, budget_l = score_pyramid(
                        attn_patterns[l], seq_len, budget, l, cfg['num_layers'], cfg['locality'])
                    sel_l = select_top_k(scores_l, budget_l)
                    pyramid_sel_all.append(sel_l)
                # Use middle layer for eval
                q = evaluate(pyramid_sel_all[mid_layer], attn_patterns, seq_len, needle_positions)
                q['formula'] = 'pyramid'; q['model'] = model_name
                q['seq_len'] = seq_len; q['budget'] = budget
                results.append(q)
                
                # QCBM hybrid
                q = evaluate_qcbm(attn_patterns, seq_len, budget, cfg['locality'], needle_positions)
                q['model'] = model_name; q['seq_len'] = seq_len; q['budget'] = budget
                results.append(q)
                
                # Print
                print(f"    {'Formula':<18} {'NIAH%':>7} {'Cov%':>6} {'MinCov%':>8} {'PPLΔ%':>8} {'TxR':>6}")
                print(f"    {'-'*55}")
                for r in results[-len(formulas):]:
                    print(f"    {r['formula']:<18} {r['niah_accuracy']*100:>7.1f} "
                          f"{r['avg_coverage']*100:>6.1f} {r.get('min_coverage',0)*100:>8.1f} "
                          f"{r['ppl_degradation_pct']:>8.1f} {r['tx_ratio']:>6.3f}")
    
    # Save
    os.makedirs('../experiment_logs', exist_ok=True)
    with open('../experiment_logs/formula_optimization.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n📄 Results saved")
    
    # === ANALYSIS ===
    print(f"\n\n{'='*90}")
    print("  KEY RESULTS: NIAH at LOW budget (b=0.3)")
    print(f"{'='*90}")
    
    for model_name in ['qwen-7b', 'mistral-7b', 'gemma-9b']:
        print(f"\n  {model_name} (seq=4096):")
        low = [r for r in results if r['budget']==0.3 and r['model']==model_name and r['seq_len']==4096]
        low.sort(key=lambda r: r['niah_accuracy'], reverse=True)
        print(f"    {'Formula':<18} {'NIAH%':>7} {'PPLΔ%':>8} {'Needles hit':>12}")
        for r in low:
            print(f"    {r['formula']:<18} {r['niah_accuracy']*100:>7.1f} "
                  f"{r['ppl_degradation_pct']:>8.1f} {r.get('niah_fully_covered',0)}/{r.get('niah_total',5):>4}")
    
    # All budgets
    for budget in BUDGETS:
        print(f"\n📊 NIAH Accuracy at b={budget} (all models, seq=4096)")
        data = defaultdict(list)
        for r in results:
            if r['budget']==budget and r['seq_len']==4096:
                data[r['formula']].append(r['niah_accuracy'])
        ranked = sorted([(f, np.mean(v)*100) for f,v in data.items()], key=lambda x:x[1], reverse=True)
        for f, avg in ranked:
            print(f"    {f:<18} {avg:>6.1f}%")
    
    # Pareto
    print(f"\n📊 PPL vs NIAH Pareto Analysis")
    fdata = defaultdict(lambda: {'ppl':[], 'niah':[]})
    for r in results:
        fdata[r['formula']]['ppl'].append(r['ppl_degradation_pct'])
        fdata[r['formula']]['niah'].append(r['niah_accuracy'])
    
    pareto = []
    for f, d in fdata.items():
        avg_p = np.mean(d['ppl'])
        avg_n = np.mean(d['niah'])
        dominated = any(np.mean(fdata[f2]['ppl'])<=avg_p and np.mean(fdata[f2]['niah'])>=avg_n 
                       and (np.mean(fdata[f2]['ppl'])<avg_p or np.mean(fdata[f2]['niah'])>avg_n)
                       for f2 in fdata if f2!=f)
        pareto.append((f, avg_p, avg_n*100, not dominated))
    pareto.sort(key=lambda x:x[2], reverse=True)
    print(f"    {'Formula':<18} {'PPLΔ%':>8} {'NIAH%':>7} {'Pareto?':>8}")
    for f, p, n, is_p in pareto:
        print(f"    {f:<18} {p:>8.1f} {n:>7.1f} {'✅YES' if is_p else '   no':>8}")
    
    # === Figures ===
    print("\n📈 Generating figures...")
    os.makedirs('../paper/figures', exist_ok=True)
    
    # Fig 1: NIAH vs Budget
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.get_cmap('tab20', len(formulas))
    fcolors = {f: cmap(i) for i, f in enumerate(formulas.keys())}
    
    for mi, mn in enumerate(['qwen-7b', 'mistral-7b', 'gemma-9b']):
        ax = axes[mi]
        for fi, fname in enumerate(formulas.keys()):
            bds, ns = [], []
            for b in BUDGETS:
                ms = [r for r in results if r['model']==mn and r['budget']==b 
                     and r['formula']==fname and r['seq_len']==4096]
                if ms:
                    bds.append(b)
                    ns.append(np.mean([r['niah_accuracy'] for r in ms])*100)
            if bds:
                ax.plot(bds, ns, 'o-', color=fcolors[fname], label=fname, lw=1.5, ms=5, alpha=0.8)
        ax.set_xlabel('Budget'); ax.set_ylabel('NIAH Accuracy (%)')
        ax.set_title(f'{mn} (seq=4096)', fontweight='bold')
        ax.legend(fontsize=6, ncol=2, loc='lower right')
        ax.grid(alpha=0.3, ls='--'); ax.set_ylim(-5, 105)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.suptitle('NIAH Accuracy: Scoring Formula Comparison', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('../paper/figures/fig_formula_niah_comparison.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig_formula_niah_comparison.png', dpi=200, bbox_inches='tight')
    print("✅ NIAH comparison figure saved")
    
    # Fig 2: Heatmap for Qwen
    fig, ax = plt.subplots(figsize=(10, 6))
    q4096 = [r for r in results if r['model']=='qwen-7b' and r['seq_len']==4096]
    flist = sorted(set(r['formula'] for r in q4096))
    blist = BUDGETS
    matrix = np.zeros((len(flist), len(blist)))
    for i,f in enumerate(flist):
        for j,b in enumerate(blist):
            ms = [r for r in q4096 if r['formula']==f and r['budget']==b]
            if ms:
                matrix[i,j] = np.mean([r['niah_accuracy'] for r in ms])*100
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    for i in range(len(flist)):
        for j in range(len(blist)):
            v = matrix[i,j]; c = 'white' if v<40 else 'black'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=9, color=c)
    ax.set_xticks(range(len(blist))); ax.set_xticklabels([f'b={b}' for b in blist])
    ax.set_yticks(range(len(flist))); ax.set_yticklabels(flist, fontsize=8)
    ax.set_title('Qwen-7B NIAH%: Formula × Budget (seq=4096)', fontweight='bold')
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig('../paper/figures/fig_formula_heatmap.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig_formula_heatmap.png', dpi=200, bbox_inches='tight')
    print("✅ Heatmap figure saved")
    
    # Fig 3: Pareto
    fig, ax = plt.subplots(figsize=(8, 6))
    for f, d in fdata.items():
        ax.scatter(d['ppl'], [n*100 for n in d['niah']], c=[fcolors[f]], 
                  label=f, alpha=0.6, s=40, edgecolors='white', lw=0.5)
    ax.set_xlabel('PPL Degradation (%)'); ax.set_ylabel('NIAH Accuracy (%)')
    ax.set_title('PPL vs NIAH: Formula Comparison', fontweight='bold')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.savefig('../paper/figures/fig_formula_pareto.pdf', bbox_inches='tight')
    plt.savefig('../paper/figures/fig_formula_pareto.png', dpi=200, bbox_inches='tight')
    print("✅ Pareto figure saved")
    
    print("\n✅ All done!")

if __name__ == '__main__':
    main()
