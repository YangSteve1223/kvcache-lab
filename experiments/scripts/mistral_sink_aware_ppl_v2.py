#!/usr/bin/env python3
"""Mistral-7B Sink-Aware Sliding Window PPL Experiment v2
Key: pass correct position_ids so RoPE encodings are right.
Sink tokens at positions [0..sink_size-1], window tokens at [win_start..pos].
"""
import json, os, gc, time
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_PATH = "/root/autodl-tmp/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "/root/autodl-tmp/experiment_results_multimodel"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda:0"
SEQ_LENGTH = 2048
CHUNK_SIZE = 32
SINK_SIZES = [0, 1, 4, 8, 16]
BUDGETS = [0.3, 0.5, 0.7]

def compute_ppl_sink_aware(model, input_ids, actual_len, budget, sink_size):
    keep_total = max(int(actual_len * budget), sink_size + 1)
    window_size = keep_total - sink_size
    total_log_prob = 0.0
    total_tokens = 0

    with torch.no_grad():
        for chunk_start in range(0, actual_len - 1, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, actual_len - 1)

            if chunk_start < keep_total or sink_size == 0:
                # No eviction: full prefix
                token_ids = input_ids[0:chunk_end + 1]
                pos_ids = torch.arange(chunk_end + 1, device=DEVICE)
                prefix_len = chunk_start  # logits before this are prefix
            else:
                # Sink + window + chunk
                sink_ids = input_ids[:sink_size]
                win_start = max(sink_size, chunk_start - window_size)
                window_and_chunk = input_ids[win_start:chunk_end + 1]
                token_ids = torch.cat([sink_ids, window_and_chunk])

                # Correct position_ids for RoPE
                sink_pos = torch.arange(sink_size, device=DEVICE)
                win_pos = torch.arange(win_start, chunk_end + 1, device=DEVICE)
                pos_ids = torch.cat([sink_pos, win_pos])

                prefix_len = len(sink_ids) + (chunk_start - win_start)

            # Forward with position_ids
            out = model(
                input_ids=token_ids.unsqueeze(0),
                position_ids=pos_ids.unsqueeze(0),
                use_cache=False,
                output_attentions=False,
            )
            logits = out.logits[0]  # (seq_len, vocab)

            # Extract log probs for the chunk tokens
            actual_chunk_len = chunk_end - chunk_start + 1
            for j in range(actual_chunk_len - 1):
                logit_idx = prefix_len + j
                if logit_idx < 0 or logit_idx >= len(logits) - 1:
                    continue
                target = input_ids[chunk_start + j + 1]
                lp = torch.log_softmax(logits[logit_idx].float(), dim=-1)[target].item()
                total_log_prob += lp
                total_tokens += 1

            del out
            if chunk_start % (CHUNK_SIZE * 8) == 0:
                torch.cuda.empty_cache()

    if total_tokens == 0:
        return float('inf')
    return np.exp(-total_log_prob / total_tokens)

def main():
    print("=" * 60)
    print("Mistral-7B Sink-Aware SWS PPL v2 (correct position_ids)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map=DEVICE, attn_implementation="eager",
    )
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Prepare input
    prompt = "Write a detailed analysis of machine learning systems and their optimization. " * (SEQ_LENGTH // 15 + 1)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=SEQ_LENGTH, truncation=True).to(DEVICE)
    input_ids = inputs.input_ids[0]
    actual_len = input_ids.shape[0]
    print(f"Input length: {actual_len}")

    # Baseline
    print("\n--- Baseline PPL ---")
    with torch.no_grad():
        out = model(inputs.input_ids, use_cache=False)
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss(reduction='mean')(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        baseline_ppl = torch.exp(loss).item()
        del out
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    results = {
        "model": "Mistral-7B-Instruct-v0.3",
        "method": "sink_aware_position_ids",
        "seq_len": SEQ_LENGTH, "actual_len": actual_len,
        "baseline_ppl": baseline_ppl, "experiments": {}
    }

    configs = [
        (0.5, 0), (0.5, 1), (0.5, 4), (0.5, 8), (0.5, 16),
        (0.3, 4), (0.3, 8),
        (0.7, 4), (0.7, 8),
    ]

    for budget, sink_size in configs:
        keep_total = max(int(actual_len * budget), sink_size + 1)
        window_size = keep_total - sink_size
        if sink_size >= keep_total:
            continue

        t0 = time.time()
        print(f"  budget={budget}, sink={sink_size}, win={window_size}...", end=" ", flush=True)
        try:
            ppl = compute_ppl_sink_aware(model, input_ids, actual_len, budget, sink_size)
            change = ((ppl - baseline_ppl) / baseline_ppl) * 100
            elapsed = time.time() - t0
            key = f"budget{int(budget*100)}_sink{sink_size}"
            results["experiments"][key] = {
                "budget": budget, "sink_size": sink_size,
                "window_size": window_size, "keep_total": keep_total,
                "ppl": ppl, "ppl_change_pct": change, "time_sec": elapsed
            }
            tag = "OK" if abs(change) < 10 else ("WARN" if abs(change) < 100 else "BAD")
            print(f"PPL={ppl:.2f} ({change:+.1f}%) [{tag}] {elapsed:.0f}s")
        except Exception as e:
            print(f"ERR: {e}")
            import traceback; traceback.print_exc()

        torch.cuda.empty_cache(); gc.collect()

    out_file = os.path.join(OUTPUT_DIR, "mistral_sink_aware_ppl.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Baseline PPL: {baseline_ppl:.4f}")
    print(f"{'Budget':>8} {'Sink':>6} {'Window':>8} {'PPL':>10} {'Change':>10}")
    print("-" * 48)
    for k, e in sorted(results["experiments"].items()):
        if "error" not in e:
            print(f"{e['budget']:>7.0%} {e['sink_size']:>6d} {e['window_size']:>8d} {e['ppl']:>10.2f} {e['ppl_change_pct']:>+9.1f}%")

if __name__ == "__main__":
    main()
