import torch, os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from spectrumkv_utils import load_model
model, tokenizer = load_model('qwen7b')
ids = tokenizer('hello', return_tensors='pt').input_ids.to(model.device)
out = model(ids, use_cache=True)
kv = out.past_key_values

print(f'Cache type: {type(kv).__name__}')
print(f'Has .layers: {hasattr(kv, "layers")}')
if hasattr(kv, 'layers'):
    layer0 = kv.layers[0]
    print(f'Layer0 type: {type(layer0).__name__}')
    print(f'Layer0 dir: {[x for x in dir(layer0) if not x.startswith("_")]}')
    # Try common attribute names
    for attr in ['key', 'value', 'key_states', 'value_states', 'keys', 'values',
                 'self_attn_key', 'self_attn_value', 'k', 'v']:
        if hasattr(layer0, attr):
            val = getattr(layer0, attr)
            if hasattr(val, 'shape'):
                print(f'  layer0.{attr}: shape={val.shape}')
            else:
                print(f'  layer0.{attr}: type={type(val).__name__} val={val}')

# Try iterating over the layer object
if hasattr(kv, 'layers'):
    layer0 = kv.layers[0]
    try:
        for i, item in enumerate(layer0):
            if hasattr(item, 'shape'):
                print(f'  layer0[{i}]: shape={item.shape}')
            else:
                print(f'  layer0[{i}]: type={type(item).__name__}')
            if i >= 3:
                break
    except Exception as e:
        print(f'  iter error: {e}')

# Try the update method to understand cache structure
print(f'\nCache.get_seq_length(): {kv.get_seq_length()}')

# Try to access key/value directly through known patterns
if hasattr(kv, 'layers'):
    layer0 = kv.layers[0]
    # Check if it has __getattr__ for key/value
    try:
        k = layer0.key
        print(f'layer0.key shape: {k.shape}')
    except:
        pass
    try:
        v = layer0.value  
        print(f'layer0.value shape: {v.shape}')
    except:
        pass

del model, tokenizer
