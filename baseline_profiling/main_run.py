import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity
device = 'cuda'
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# warm-up
with torch.no_grad():
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=50)

# wait for all kernels in all streams on a CUDA device to complete
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50)

torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))