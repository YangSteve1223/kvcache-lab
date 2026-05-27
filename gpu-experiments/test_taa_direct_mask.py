import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

model_name = '/root/autodl-tmp/Qwen2.5-7B-Instruct'
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map='auto', trust_remote_code=True, attn_implementation='sdpa')
model.eval()

prompt = 'The history of artificial intelligence began in antiquity, with myths and stories of artificial beings.'
inputs = tok(prompt, return_tensors='pt').to(model.device)
seq_len = inputs.input_ids.shape[1]
print(f'Seq len: {seq_len}')

# Baseline PPL
with torch.no_grad():
    out_base = model(**inputs, use_cache=True)
shift_labels = inputs.input_ids[:, 1:].contiguous()
loss_base = F.cross_entropy(out_base.logits[:, :-1, :].contiguous().view(-1, out_base.logits.size(-1)), shift_labels.view(-1))
ppl_base = torch.exp(loss_base).item()
print(f'Baseline PPL: {ppl_base:.4f}')

# Build TAA bias
cost_vector = torch.zeros(seq_len, device=model.device)
remote_end = int(seq_len * 0.7)
cost_vector[:remote_end] = 1.0

# Test with alpha=0.1 and alpha=5.0
for alpha in [0.1, 5.0]:
    mu = cost_vector.mean()
    sigma = cost_vector.std()
    bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)
    
    # Create a 4D causal mask WITH TAA bias
    # For SDPA, we need to create a [1, 1, seq, seq] float mask
    # Standard causal mask: 0 for attend, -inf for masked
    # With TAA: 0 + bias_i for attend, -inf + bias_i for masked (bias doesn't matter for masked positions)
    
    # Create causal mask
    causal_mask = torch.zeros(seq_len, seq_len, dtype=torch.float16, device=model.device)
    # Upper triangle = -inf (can't attend to future)
    causal_mask = causal_mask.masked_fill(
        torch.tril(torch.ones(seq_len, seq_len, device=model.device)) == 0,
        float('-inf')
    )
    # Add TAA bias to key dimension (last dim)
    # bias_1d: [seq], expand to [1, 1, 1, seq]
    taa_bias = bias_1d.view(1, 1, 1, -1).to(torch.float16)
    mask_with_taa = causal_mask.unsqueeze(0).unsqueeze(0) + taa_bias  # [1, 1, seq, seq]
    
    # Forward with TAA mask
    with torch.no_grad():
        out_taa = model(input_ids=inputs.input_ids, attention_mask=mask_with_taa, use_cache=True)
    
    has_nan = torch.isnan(out_taa.logits).any().item()
    if has_nan:
        # Check where NaN appears
        nan_count = torch.isnan(out_taa.logits).sum().item()
        total = out_taa.logits.numel()
        print(f'alpha={alpha}: NaN detected ({nan_count}/{total})')
        # Try with fp32 mask instead
        mask_fp32 = mask_with_taa.to(torch.float32)
        with torch.no_grad():
            out_taa2 = model(input_ids=inputs.input_ids, attention_mask=mask_fp32, use_cache=True)
        has_nan2 = torch.isnan(out_taa2.logits).any().item()
        print(f'alpha={alpha} (fp32 mask): NaN={has_nan2}')
        if not has_nan2:
            loss_taa = F.cross_entropy(out_taa2.logits[:, :-1, :].contiguous().view(-1, out_taa2.logits.size(-1)), shift_labels.view(-1))
            ppl_taa = torch.exp(loss_taa).item()
            print(f'alpha={alpha} (fp32): PPL={ppl_taa:.4f}, delta={ppl_taa-ppl_base:+.4f} ({(ppl_taa-ppl_base)/ppl_base*100:+.2f}%)')
    else:
        loss_taa = F.cross_entropy(out_taa.logits[:, :-1, :].contiguous().view(-1, out_taa.logits.size(-1)), shift_labels.view(-1))
        ppl_taa = torch.exp(loss_taa).item()
        print(f'alpha={alpha}: PPL={ppl_taa:.4f}, delta={ppl_taa-ppl_base:+.4f} ({(ppl_taa-ppl_base)/ppl_base*100:+.2f}%)')
