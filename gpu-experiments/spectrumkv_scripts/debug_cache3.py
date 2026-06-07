import torch, os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from spectrumkv_utils import load_model
model, tokenizer = load_model('qwen7b')
ids = tokenizer('hello', return_tensors='pt').input_ids.to(model.device)
out = model(ids, use_cache=True)
kv = out.past_key_values

print(f'TYPE: {type(kv).__name__}')
print(f'LEN: {len(kv)}')

# Check iteration and item structure
for i, item in enumerate(kv):
    print(f'Layer {i}: type={type(item).__name__} len={len(item)}')
    for j, sub in enumerate(item):
        if hasattr(sub, 'shape'):
            print(f'  [{j}] Tensor shape={sub.shape} dtype={sub.dtype}')
        elif isinstance(sub, (tuple, list)):
            print(f'  [{j}] {type(sub).__name__} len={len(sub)}')
        else:
            print(f'  [{j}] {type(sub).__name__} = {sub}')
    if i >= 1:  # Just first 2 layers
        print('...(remaining layers similar)')
        break

# Try update method signature
import inspect
print(f'\nUPDATE_SIG: {inspect.signature(kv.update)}')

# Try layers attribute
if hasattr(kv, 'layers'):
    print(f'LAYERS: type={type(kv.layers).__name__} len={len(kv.layers)}')

print('\n--- Testing model forward with past_key_values ---')
# Test: pass the KV back to the model
next_id = torch.tensor([[ids[0, -1].item()]], device=model.device)
try:
    out2 = model(next_id, past_key_values=kv, use_cache=True)
    print('FORWARD_WITH_KV: OK')
    kv2 = out2.past_key_values
    print(f'KV2_TYPE: {type(kv2).__name__} LEN: {len(kv2)}')
    # Check first layer again
    for i, item in enumerate(kv2):
        print(f'KV2 Layer {i}: type={type(item).__name__} len={len(item)}')
        for j, sub in enumerate(item):
            if hasattr(sub, 'shape'):
                print(f'  [{j}] Tensor shape={sub.shape}')
        break
except Exception as e:
    print(f'FORWARD_WITH_KV_ERROR: {e}')

del model, tokenizer
