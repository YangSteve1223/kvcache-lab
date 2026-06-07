import torch, os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

from spectrumkv_utils import load_model
model, tokenizer = load_model('qwen7b')
ids = tokenizer('hello', return_tensors='pt').input_ids.to(model.device)
out = model(ids, use_cache=True)
kv = out.past_key_values

print(f'TYPE: {type(kv).__name__}')
print(f'MRO: {[c.__name__ for c in type(kv).__mro__]}')
print(f'LEN: {len(kv)}')

# List all public attributes/methods
attrs = [x for x in dir(kv) if not x.startswith('_')]
print(f'ATTRS: {attrs}')

# Check for specific attrs
for attr in ['to_legacy_cache', 'key_cache', 'value_cache', 'to_tuple', 
             'get_seq_length', 'get_max_length', 'update', 'get_usable_length']:
    print(f'HAS_{attr.upper()}: {hasattr(kv, attr)}')

# Try to get sequence length
if hasattr(kv, 'get_seq_length'):
    try:
        print(f'SEQ_LENGTH: {kv.get_seq_length()}')
    except Exception as e:
        print(f'SEQ_LENGTH_ERROR: {e}')

# Try iterating
try:
    for i, item in enumerate(kv):
        print(f'ITEM {i}: type={type(item).__name__}', end='')
        if hasattr(item, 'shape'):
            print(f' shape={item.shape}')
        elif hasattr(item, '__len__'):
            print(f' len={len(item)}')
            if len(item) > 0 and hasattr(item[0], 'shape'):
                print(f'  [0] shape={item[0].shape}')
        else:
            print()
        if i >= 2:
            print('...(truncated)')
            break
except Exception as e:
    print(f'ITER_ERROR: {e}')

# Try to_tuple
if hasattr(kv, 'to_tuple'):
    try:
        t = kv.to_tuple()
        print(f'TO_TUPLE_TYPE: {type(t).__name__}')
        print(f'TO_TUPLE_LEN: {len(t)}')
        print(f'TO_TUPLE_0_TYPE: {type(t[0]).__name__}')
        if hasattr(t[0], '__len__'):
            print(f'TO_TUPLE_0_LEN: {len(t[0])}')
            if len(t[0]) >= 2 and hasattr(t[0][0], 'shape'):
                print(f'TO_TUPLE_0_0_SHAPE: {t[0][0].shape}')
                print(f'TO_TUPLE_0_1_SHAPE: {t[0][1].shape}')
    except Exception as e:
        print(f'TO_TUPLE_ERROR: {e}')

del model, tokenizer
