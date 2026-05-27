import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '/root/autodl-tmp/Qwen2.5-7B-Instruct'
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Test SDPA with float attention_mask
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map='auto', trust_remote_code=True, attn_implementation='sdpa')
model.eval()

prompt = 'The history of artificial intelligence began in antiquity, with myths and stories of artificial beings.'
inputs = tok(prompt, return_tensors='pt').to(model.device)

# 1. Baseline - no TAA
with torch.no_grad():
    out_baseline = model(**inputs, use_cache=True)
shift_logits = out_baseline.logits[:, :-1, :].contiguous()
shift_labels = inputs.input_ids[:, 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
ppl_baseline = torch.exp(loss).item()
print(f'Baseline PPL: {ppl_baseline:.4f}, Loss: {loss.item():.4f}')
print(f'Has NaN: {torch.isnan(out_baseline.logits).any().item()}')

# 2. Test TAA via attention_mask modification
seq_len = inputs.input_ids.shape[1]
cost_vector = torch.zeros(seq_len, device=model.device)
cost_vector[:int(seq_len*0.7)] = 1.0

alpha = 0.1
mu = cost_vector.mean()
sigma = cost_vector.std()
bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)

# Get the model's internal causal mask first
with torch.no_grad():
    # Let the model create its mask, then we'll modify it
    # For SDPA, the model creates the mask internally via _update_causal_mask
    # We need to check if we can inject via a hook
    pass

# 3. Use forward hook on attention modules to inject bias
# Since the mask is created inside model.forward(), we need to intercept at the attention layer level
# The key is: does the self_attn module receive attention_mask as a kwarg?

# Let's check by inspecting the first layer's forward signature
import inspect
attn_module = model.model.layers[0].self_attn
sig = inspect.signature(attn_module.forward)
print(f'Attention forward params: {list(sig.parameters.keys())}')

# 4. Try register_forward_pre_hook approach with SDPA
hooks = []
modified_layers = 0
start_layer = len(model.model.layers) * 2 // 3

def make_hook(bias):
    def pre_hook(module, args, kwargs):
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            mask = kwargs['attention_mask']
            if mask.dim() == 4:
                bias_exp = bias.view(1, 1, 1, -1).to(mask.dtype)
                kwargs['attention_mask'] = mask + bias_exp
            elif mask.dim() == 2:
                bias_exp = bias.view(1, -1).to(mask.dtype)
                kwargs['attention_mask'] = mask + bias_exp
        return args, kwargs
    return pre_hook

for i in range(start_layer, len(model.model.layers)):
    h = model.model.layers[i].self_attn.register_forward_pre_hook(make_hook(bias_1d), with_kwargs=True)
    hooks.append(h)
    modified_layers += 1

print(f'Installed {modified_layers} TAA hooks on layers {start_layer}-{len(model.model.layers)-1}')

# 5. Run with TAA
with torch.no_grad():
    out_taa = model(**inputs, use_cache=True)

has_nan = torch.isnan(out_taa.logits).any().item()
print(f'TAA logits has NaN: {has_nan}')

if not has_nan:
    shift_logits_taa = out_taa.logits[:, :-1, :].contiguous()
    loss_taa = F.cross_entropy(shift_logits_taa.view(-1, shift_logits_taa.size(-1)), shift_labels.view(-1))
    ppl_taa = torch.exp(loss_taa).item()
    print(f'TAA PPL: {ppl_taa:.4f}, Loss: {loss_taa.item():.4f}')
    print(f'PPL delta: {ppl_taa - ppl_baseline:+.4f} ({(ppl_taa-ppl_baseline)/ppl_baseline*100:+.2f}%)')
else:
    print('TAA produces NaN - SDPA does not support float attention_mask modification')
    # Check what kind of mask reaches the attention layers
    # The issue might be that SDPA receives a 4D mask but adding float bias breaks it

# Clean up hooks
for h in hooks:
    h.remove()

# 6. Alternative: directly modify the attention_mask in the model call
# Get the processed mask from the model
print()
print('--- Testing direct attention_mask injection ---')

# Create a 4D causal mask manually
from transformers.masking_utils import create_causal_mask
causal_mask = create_causal_mask(
    inputs.input_ids.shape, 
    dtype=model.dtype, 
    device=model.device,
)
print(f'Causal mask shape: {causal_mask.shape}, dtype: {causal_mask.dtype}')

# Add TAA bias to the causal mask
if causal_mask.dim() == 4:
    taa_mask = causal_mask + bias_1d.view(1, 1, 1, -1).to(causal_mask.dtype)
else:
    taa_mask = causal_mask

# Pass modified mask directly
with torch.no_grad():
    out_direct = model(input_ids=inputs.input_ids, attention_mask=taa_mask, use_cache=True)

has_nan2 = torch.isnan(out_direct.logits).any().item()
print(f'Direct mask TAA logits has NaN: {has_nan2}')

if not has_nan2:
    shift_logits_d = out_direct.logits[:, :-1, :].contiguous()
    loss_d = F.cross_entropy(shift_logits_d.view(-1, shift_logits_d.size(-1)), shift_labels.view(-1))
    ppl_d = torch.exp(loss_d).item()
    print(f'Direct TAA PPL: {ppl_d:.4f}')
    print(f'PPL delta: {ppl_d - ppl_baseline:+.4f} ({(ppl_d-ppl_baseline)/ppl_baseline*100:+.2f}%)')
