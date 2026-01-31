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
- Inference setting:
  - Batch size: 1
  - Autoregressive generation (model.generate)
  - Fixed prompt and token length for fair comparison
  - Warm-up runs were performed prior to profiling and timing to eliminate one-time initialization overhead.
  - torch.cuda.synchronize() was used to ensure accurate GPU timing due to asynchronous CUDA execution.

---

## Guideline (How I proceeded)
### Baseline Profiling
I first profiled the standard HuggingFace GPT-2 inference implementation without any optimization to understand how the workload executes on the GPU. (`baseling_profiling/main_run.py`)
Profiling was used to identify which CUDA kernels dominate execution time during autoregressive generation, rather than relying on intuition or source-level assumptions.
Before measuring latency and profiling, I performed several warm-up runs to eliminate one-time initialization overhead and capture steady-state inference performance.
#### Bottleneck Analysis
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     **Self CUDA**   **Self CUDA %**    CUDA total  **CUDA time avg**   ** # of Calls ** 
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm        17.85%     116.634ms        30.41%     198.705ms      82.794us     ** 59.577ms**        54.53%      59.577ms      24.824us          **2400**  
                                           aten::linear         0.03%     189.013us         0.60%       3.889ms      77.772us       0.000us         0.00%      22.173ms     443.469us            50  
                                           aten::matmul         0.08%     544.488us         0.50%       3.240ms      64.805us       0.000us         0.00%      22.173ms     443.469us            50  
                                               aten::mm         0.25%       1.655ms         0.36%       2.326ms      46.523us      22.173ms        20.29%      22.173ms     443.469us            50  
void **gemv2T_kernel_val**<int, int, float, float, float...         0.00%       0.000us         0.00%       0.000us       0.000us      22.173ms        20.29%      **22.173ms**     443.469us            **50**  
std::enable_if<!(false), void>::type **internal::gemvx**...         0.00%       0.000us         0.00%       0.000us       0.000us      18.589ms        17.01%      **18.589ms**      15.807us          **1176**  
void **gemvNSP_kernel**<float, float, float, float, 1, 8...         0.00%       0.000us         0.00%       0.000us       0.000us      18.548ms        16.98%      **18.548ms**      31.544us           **588**  
void **gemvNSP_kernel**<float, float, float, float, 1, 1...         0.00%       0.000us         0.00%       0.000us       0.000us      17.957ms        16.44%      **17.957ms**      30.540us           **588**  
                     aten::scaled_dot_product_attention         1.23%       8.032ms         8.29%      54.155ms      90.258us       0.000us         0.00%       7.811ms      13.019us           600  
          aten::_scaled_dot_product_efficient_attention         1.12%       7.339ms         7.06%      46.123ms      76.871us       0.000us         0.00%       7.811ms      13.019us           600  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 653.411ms
Self CUDA time total: 109.261ms
```
- The profiling results show that the inference workload is dominated by many small kernel invocations. 
- For example, `aten::addmm` is called **2,400 times**, while multiple GEMV-related kernels are each invoked hundreds of times.
- Although individual kernel executions are short (e.g., ~**25 µs** per `addmm` call), their cumulative CUDA time dominates execution due to **frequent memory access** during token-by-token generation (inference step).
- This behavior shows that batch-1 autoregressive inference mainly consists of many small operations (e.g., GEMV 😢) with frequent memory access, rather than large, highly efficient matrix computations (e.g., GEMM ☺️) that GPUs typically excel at.

### FP16 Mixed-Precision Inference
- Given that baseline profiling revealed a memory-intensive inference workload dominated by small GEMM/GEMV kernels, I evaluated FP16 mixed-precision inference as a practical optimization to reduce memory traffic. (`fp16_profiling/main_run_fp16.py`)
```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                            aten::addmm        13.20%     164.937ms        45.04%     562.969ms     134.040us      44.165ms        45.90%      86.319ms      **20.552us**          4200  
                                           aten::linear         0.07%     837.370us         1.12%      14.002ms     140.017us       0.000us         0.00%      22.417ms     224.168us           100  
                                       aten::layer_norm         2.52%      31.458ms        31.08%     388.488ms     155.395us       0.000us         0.00%      18.853ms       7.541us          2500  
                                            aten::copy_         6.55%      81.923ms        14.14%     176.677ms      27.056us      14.355ms        14.92%      14.357ms       2.199us          6530  
                                               aten::to         1.16%      14.492ms        25.60%     319.958ms      44.562us       0.000us         0.00%      14.130ms       1.968us          7180  
                                         aten::_to_copy         5.20%      64.962ms        24.44%     305.466ms      48.287us       0.000us         0.00%      14.130ms       2.234us          6326  
void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816...         0.00%       0.000us         0.00%       0.000us       0.000us      13.079ms        13.59%      13.079ms      22.243us           588  
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us      11.706ms        12.17%      11.706ms       2.660us          4400  
                                           aten::matmul         0.07%     896.726us         0.35%       4.361ms      87.226us       0.000us         0.00%      11.176ms     223.513us            50  
                                               aten::mm         0.17%       2.093ms         0.23%       2.838ms      56.768us      11.176ms        11.61%      11.176ms     223.513us            50  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.250s
Self CUDA time total: 96.224ms
```
- Total GPU execution time was reduced by approximately 12% (109ms -> 96ms)
- Per-kernel execution time for compute-intensive operations such as aten::addmm decreased (24.8 µs -> 20.6 µs)
- FP16 Tensor Core–backed kernels (e.g., CUTLASS FP16 kernels) appeared in the profiling results, indicating improved hardware utilization
- However, profiling also revealed increased **overhead from dtype conversion operations such as `aten::copy_` and `aten::to`**, reflecting the inherent **trade-offs of mixed-precision execution**.

---
## Discussion
- The results demonstrate that FP16 can provide a measurable latency improvement for batch-1 Transformer inference by reducing memory traffic and accelerating compute kernels.
- At the same time, the profiling results show **diminishing returns due to mixed-precision overheads**, particularly **frequent dtype conversions** required for numerically sensitive operations.
- These observations suggest that further optimization would likely require kernel fusion or architectural changes, rather than simple precision tuning alone.
---
## Possible Extensions
- Reducing dtype conversion overhead
- Kernel fusion for attention and linear layers
- Exploring optimized inference runtimes (e.g., TensorRT)