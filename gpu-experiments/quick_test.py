import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '/root/autodl-tmp/Qwen2.5-7B-Instruct'
tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map='auto', trust_remote_code=True, attn_implementation='eager')
model.eval()

prompt = 'The history of artificial intelligence began in antiquity.'
inputs = tok(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    out = model(**inputs, use_cache=True, output_attentions=True)

print('Logits shape:', out.logits.shape)
print('Has NaN:', torch.isnan(out.logits).any().item())
if out.attentions:
    print('Attn shape:', out.attentions[0].shape)
    print('Attn sum:', out.attentions[-1][0, 0, -1, :].sum().item())

input_ids = inputs.input_ids
shift_logits = out.logits[:, :-1, :].contiguous()
shift_labels = input_ids[:, 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
ppl = torch.exp(loss).item()
print('PPL:', ppl)
print('Loss:', loss.item())
