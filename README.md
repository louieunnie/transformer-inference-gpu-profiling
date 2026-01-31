# Optimization of Transformer Inference Latency on NVIDIA GPUs

## Introduction
This project investigates the inference latency of a Transformer-based language model on NVIDIA GPUs, 
with the goal of understanding and improving GPU efficiency in autoregressive inference workloads.


### Model Selection
I use [Huggingface GPT-2](https://huggingface.co/openai-community/gpt2) as the target model because its simple and well-documented Transformer architecture makes it suitable for analyzing inference-time GPU bottlenecks. 
In particular, GPT-2 represents a canonical autoregressive workload where attention and linear layers dominate execution, allowing clear attribution of performance costs.

## Environment
- GPU: NVIDIA GeForce RTX 3060 (12GB)
- Python: 3.10
- PyTorch: 2.10.0+cu128
- CUDA: 12.8

---

## Guideline (How I proceeded)
### Baseline Profiling
I first profiled the standard HuggingFace GPT-2 inference implementation without any optimization to understand how the workload executes on the GPU. (`baseling_profiling/main_run.py`)
Profiling was used to identify which CUDA kernels dominate execution time during autoregressive generation, rather than relying on intuition or source-level assumptions.

### Bottleneck Analysis
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm        17.85%     116.634ms        30.41%     198.705ms      82.794us      59.577ms        54.53%      59.577ms      24.824us          2400  
                                           aten::linear         0.03%     189.013us         0.60%       3.889ms      77.772us       0.000us         0.00%      22.173ms     443.469us            50  
                                           aten::matmul         0.08%     544.488us         0.50%       3.240ms      64.805us       0.000us         0.00%      22.173ms     443.469us            50  
                                               aten::mm         0.25%       1.655ms         0.36%       2.326ms      46.523us      22.173ms        20.29%      22.173ms     443.469us            50  
void gemv2T_kernel_val<int, int, float, float, float...         0.00%       0.000us         0.00%       0.000us       0.000us      22.173ms        20.29%      22.173ms     443.469us            50  
std::enable_if<!(false), void>::type internal::gemvx...         0.00%       0.000us         0.00%       0.000us       0.000us      18.589ms        17.01%      18.589ms      15.807us          1176  
void gemvNSP_kernel<float, float, float, float, 1, 8...         0.00%       0.000us         0.00%       0.000us       0.000us      18.548ms        16.98%      18.548ms      31.544us           588  
void gemvNSP_kernel<float, float, float, float, 1, 1...         0.00%       0.000us         0.00%       0.000us       0.000us      17.957ms        16.44%      17.957ms      30.540us           588  
                     aten::scaled_dot_product_attention         1.23%       8.032ms         8.29%      54.155ms      90.258us       0.000us         0.00%       7.811ms      13.019us           600  
          aten::_scaled_dot_product_efficient_attention         1.12%       7.339ms         7.06%      46.123ms      76.871us       0.000us         0.00%       7.811ms      13.019us           600  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 653.411ms
Self CUDA time total: 109.261ms
```
- The profiling results show that inference time is dominated by a collection of small GEMM and GEMV kernels, including `aten::addmm`, `aten::mm`, and multiple `gemv` variants. 
- Although each kernel invocation is relatively inexpensive, these operations are called repeatedly during token-by-token generation, making their cumulative cost the primary contributor to GPU execution time.
- This pattern indicates a memory-intensive inference workload with limited opportunity for large, highly optimized GEMM operations, which is characteristic of batch-1 autoregressive Transformer inference.