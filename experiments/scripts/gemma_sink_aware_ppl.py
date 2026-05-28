import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/gemma-2-9b-it"
RESULTS_DIR = "/root/autodl-tmp/experiment_results_multimodel"
SEQ_LEN = 2048

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("Gemma-2-9b-it Sink-Aware SWS PPL Experiment")
print("=" * 60)

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
print(f"Model loaded on {model.device}")

# Use a local text file for evaluation - try multiple sources
eval_text = None

# Option 1: Check for existing wikitext data
local_paths = [
    "/root/autodl-tmp/wikitext_test.txt",
    "/root/autodl-tmp/eval_text.txt",
]
for p in local_paths:
    if os.path.exists(p):
        with open(p) as f:
            eval_text = f.read()
        print(f"Loaded eval text from {p}")
        break

# Option 2: Download wikitext-2 test set directly
if eval_text is None:
    try:
        import urllib.request
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        import zipfile, io
        print("Downloading WikiText-2...")
        resp = urllib.request.urlopen(url, timeout=30)
        with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
            with z.open("wikitext-2-raw/wiki.test.raw") as f:
                eval_text = f.read().decode("utf-8")
        # Save for reuse
        with open("/root/autodl-tmp/wikitext_test.txt", "w") as f:
            f.write(eval_text)
        print(f"Downloaded and cached WikiText-2 ({len(eval_text)} chars)")
    except Exception as e:
        print(f"Download failed: {e}")

# Option 3: Generate a long text from model's tokenizer
if eval_text is None:
    print("Using tokenizer-generated text as fallback")
    eval_text = "The history of computing is a fascinating subject. " * 5000

# Encode
encodings = tokenizer(eval_text, return_tensors="pt")
input_ids = encodings.input_ids[0]

if len(input_ids) > SEQ_LEN:
    input_ids = input_ids[:SEQ_LEN]

print(f"Input: {len(input_ids)} tokens")

# Baseline PPL
print("\nComputing baseline PPL...")
with torch.no_grad():
    outputs = model(input_ids.unsqueeze(0).to(model.device))
    logits = outputs.logits[0, :-1, :]
    labels = input_ids[1:]
    loss = torch.nn.functional.cross_entropy(logits, labels.to(logits.device))
    baseline_ppl = torch.exp(loss).item()

print(f"Baseline PPL: {baseline_ppl:.4f}")

# Sink-aware SWS PPL
budgets = [0.3, 0.5, 0.7]
sink_counts = [0, 1, 4, 8, 16]
results = {"model": "gemma-2-9b-it", "baseline_ppl": baseline_ppl, "seq_len": SEQ_LEN, "experiments": []}

print("\nRunning sink-aware SWS experiments...")
print("-" * 60)

for budget_frac in budgets:
    budget = int(SEQ_LEN * budget_frac)
    for n_sink in sink_counts:
        if n_sink >= budget:
            continue

        n_window = budget - n_sink

        sink_ids = input_ids[:n_sink]
        sink_pos = torch.arange(n_sink)

        window_ids = input_ids[SEQ_LEN - n_window:SEQ_LEN]
        window_pos = torch.arange(SEQ_LEN - n_window, SEQ_LEN)

        kept_ids = torch.cat([sink_ids, window_ids])
        kept_pos = torch.cat([sink_pos, window_pos])

        with torch.no_grad():
            outputs = model(
                kept_ids.unsqueeze(0).to(model.device),
                position_ids=kept_pos.unsqueeze(0).to(model.device),
            )
            logits = outputs.logits[0, :-1, :]
            labels_kept = kept_ids[1:]

            window_start_idx = n_sink
            if window_start_idx > 0 and len(logits) > window_start_idx:
                window_logits = logits[window_start_idx - 1:]
                window_labels = labels_kept[window_start_idx - 1:]
                loss = torch.nn.functional.cross_entropy(window_logits, window_labels.to(window_logits.device))
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels_kept.to(logits.device))

            ppl = torch.exp(loss).item()

        pct_change = (ppl - baseline_ppl) / baseline_ppl * 100
        result = {
            "budget_frac": budget_frac,
            "budget_tokens": budget,
            "n_sink": n_sink,
            "n_window": n_window,
            "ppl": round(ppl, 4),
            "pct_change": round(pct_change, 2),
        }
        results["experiments"].append(result)
        status = "OK" if abs(pct_change) < 5 else "WARN" if abs(pct_change) < 20 else "BAD"
        print(f"  [{status}] Budget={budget_frac:.0%}, Sink={n_sink:2d}, Window={n_window:4d}: PPL={ppl:.4f} ({pct_change:+.2f}%)")

output_path = os.path.join(RESULTS_DIR, "gemma_sink_aware_ppl.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: Gemma-2-9b-it | Baseline PPL: {baseline_ppl:.4f}")
print("-" * 60)
for exp in results["experiments"]:
    s = "PASS" if abs(exp["pct_change"]) < 5 else "WARN" if abs(exp["pct_change"]) < 20 else "FAIL"
    print(f"  {s} | Budget={exp['budget_frac']:.0%} | Sink={exp['n_sink']:2d} | Window={exp['n_window']:4d} | PPL={exp['ppl']:.4f} ({exp['pct_change']:+.2f}%)")
print(f"\nResults saved to {output_path}")
