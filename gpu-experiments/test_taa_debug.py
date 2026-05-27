import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '/root/autodl-tmp/Qwen2.5-7B-Instruct'
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map='auto', trust_remote_code=True, attn_implementation='sdpa')
model.eval()

prompt = 'The history of artificial intelligence began in antiquity, with myths and stories of artificial beings.'
inputs = tok(prompt, return_tensors='pt').to(model.device)
seq_len = inputs.input_ids.shape[1]

# Build TAA bias
cost_vector = torch.zeros(seq_len, device=model.device)
cost_vector[:int(seq_len*0.7)] = 1.0
alpha = 0.1
mu = cost_vector.mean()
sigma = cost_vector.std()
bias_1d = -alpha * torch.tanh((cost_vector - mu) / sigma)

# Debug: check what kwargs the attention module receives
debug_info = {}

def make_debug_hook(layer_idx):
    def pre_hook(module, args, kwargs):
        debug_info[layer_idx] = {
            'has_mask': 'attention_mask' in kwargs,
            'mask_type': type(kwargs.get('attention_mask')).__name__ if 'attention_mask' in kwargs else None,
            'mask_shape': kwargs['attention_mask'].shape if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None else None,
            'mask_dtype': kwargs['attention_mask'].dtype if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None else None,
            'mask_values_sample': kwargs['attention_mask'][0, 0, -1, :5].tolist() if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None and kwargs['attention_mask'].dim() == 4 else None,
            'kwargs_keys': list(kwargs.keys()),
        }
        return args, kwargs
    return pre_hook

# Install debug hooks on layer 18 (first TAA layer) and layer 27 (last)
hooks = []
hooks.append(model.model.layers[18].self_attn.register_forward_pre_hook(make_debug_hook(18), with_kwargs=True))
hooks.append(model.model.layers[27].self_attn.register_forward_pre_hook(make_debug_hook(27), with_kwargs=True))

with torch.no_grad():
    out = model(**inputs, use_cache=True)

for layer_idx, info in debug_info.items():
    print(f'Layer {layer_idx}: {info}')

# Clean up
for h in hooks:
    h.remove()

# Now test: if attention_mask IS being passed, add TAA bias via hook
# Use a VERY large alpha to make the effect obvious
alpha_large = 5.0  # Much larger than normal, just to see if PPL changes
bias_large = -alpha_large * torch.tanh((cost_vector - mu) / sigma)

taa_hooks = []
modified_layers = 0
start_layer = len(model.model.layers) * 2 // 3

def make_taa_hook(bias):
    def pre_hook(module, args, kwargs):
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            mask = kwargs['attention_mask']
            if mask.dim() == 4:
                bias_exp = bias.view(1, 1, 1, -1).to(mask.dtype)
                new_mask = mask + bias_exp
                kwargs['attention_mask'] = new_mask
        return args, kwargs
    return pre_hook

for i in range(start_layer, len(model.model.layers)):
    h = model.model.layers[i].self_attn.register_forward_pre_hook(make_taa_hook(bias_large), with_kwargs=True)
    taa_hooks.append(h)
    modified_layers += 1

print(f'Installed {modified_layers} TAA hooks with alpha={alpha_large}')

# Check debug again
debug_info2 = {}
hooks2 = []
def make_debug2(layer_idx):
    def pre_hook(module, args, kwargs):
        mask = kwargs.get('attention_mask')
        if mask is not None and mask.dim() == 4:
            debug_info2[layer_idx] = {
                'mask_last_row': mask[0, 0, -1, :5].tolist(),
                'mask_changed': True,
            }
        return args, kwargs
    return pre_hook

hooks2.append(model.model.layers[18].self_attn.register_forward_pre_hook(make_debug2(18), with_kwargs=True))

with torch.no_grad():
    out_taa = model(**inputs, use_cache=True)

print(f'TAA debug: {debug_info2}')

shift_labels = inputs.input_ids[:, 1:].contiguous()
loss_taa = F.cross_entropy(out_taa.logits[:, :-1, :].contiguous().view(-1, out_taa.logits.size(-1)), shift_labels.view(-1))
ppl_taa = torch.exp(loss_taa).item()

loss_base = F.cross_entropy(out.logits[:, :-1, :].contiguous().view(-1, out.logits.size(-1)), shift_labels.view(-1))
ppl_base = torch.exp(loss_base).item()

print(f'Baseline PPL: {ppl_base:.4f}')
print(f'TAA PPL (alpha={alpha_large}): {ppl_taa:.4f}')
print(f'Delta: {ppl_taa - ppl_base:+.4f}')

for h in taa_hooks + hooks2:
    h.remove()
