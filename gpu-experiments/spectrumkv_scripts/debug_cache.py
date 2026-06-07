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
print(f'HAS_LEGACY: {hasattr(kv, "to_legacy_cache")}')
print(f'HAS_KEY_CACHE: {hasattr(kv, "key_cache")}')
item0 = kv[0]
print(f'ITEM0_TYPE: {type(item0).__name__}')
if hasattr(item0, '__len__'):
    print(f'ITEM0_LEN: {len(item0)}')
    if len(item0) >= 2:
        print(f'ITEM0_0_SHAPE: {item0[0].shape}')
        print(f'ITEM0_1_SHAPE: {item0[1].shape}')
print(f'DIR: {[x for x in dir(kv) if not x.startswith("_")]}')
# Try to_legacy_cache
if hasattr(kv, 'to_legacy_cache'):
    try:
        legacy = kv.to_legacy_cache()
        print(f'LEGACY_TYPE: {type(legacy).__name__}')
        print(f'LEGACY_LEN: {len(legacy)}')
        print(f'LEGACY_0_TYPE: {type(legacy[0]).__name__}')
    except Exception as e:
        print(f'LEGACY_ERROR: {e}')
