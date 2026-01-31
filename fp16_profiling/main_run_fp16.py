import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, record_function, ProfilerActivity
device = 'cuda'
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

model.half() # convert model to FP16

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# warm-up (gpu 준비)
# AMP autocast로 FP16 연산 사용 (mixed precision)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16): 
    for _ in range(3):
        _ = model.generate(**inputs, max_new_tokens=50)

# wait for all kernels in all streams on a CUDA device to complete
torch.cuda.synchronize()

start = time.time()
with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )
torch.cuda.synchronize()
end = time.time()
latency = end-start
print(f"FP16 Latency: {latency*1000:.2f} ms")

# profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        _ = model.generate(**inputs, max_new_tokens=50)

torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))