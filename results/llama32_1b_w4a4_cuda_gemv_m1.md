# Llama-3.2-1B direct-packed W4A4 CUDA M1 GEMV

## Result

The `packed_w4a4_cuda_gemv_m1` backend isolates the new M1 kernel: all prefill
shapes use the established tiled Triton W4A4 path, while only one-token decode
projections dispatch to the direct-packed CUDA GEMV. This makes the TPOT
comparison a one-factor experiment rather than mixing M16 prefill and M1
decode changes.

Three independent processes used the same model revision, environment, input
seed and token SHA-256. Each process measured seven steady prefill repetitions
and six timed decode transitions after separate warmup.

| metric | tiled Triton | CUDA M1 GEMV | speedup |
|---|---:|---:|---:|
| prefill median across trials | 189.20 ms | 187.18 ms | 1.022x |
| TTFT median across trials | 188.33 ms | 188.52 ms | 0.999x |
| TPOT median across trials | 170.01 ms | 94.53 ms | **1.803x** |

TPOT speedup range across the three independent trials: **1.790x–1.815x**.
The unchanged prefill/TTFT values are expected because the M1-only backend
deliberately falls back for M16.

Per trial runtime audit:

| path | calls |
|---|---:|
| custom CUDA M1 GEMV | 896 |
| custom CUDA M16 MMA | 0 |
| tiled prefill fallback | 1,232 |

Raw summary:
`results/raw/llama32_1b_w4a4_cuda_gemv_m1_only_trials_summary.json`.

## Kernel microbenchmark

Seven interleaved groups, each with 10 warmups and 100 measured iterations:

| projection shape `[M,K,N]` | CUDA M1 | tiled Triton | speedup |
|---|---:|---:|---:|
| `[1,2048,512]` K/V | 266.50 us | 327.70 us | 1.230x |
| `[1,2048,2048]` Q/O | 284.70 us | 575.54 us | 2.022x |
| `[1,2048,8192]` gate/up | 484.14 us | 1,905.66 us | 3.936x |
| `[1,8192,2048]` down | 1,130.65 us | 2,290.74 us | 2.026x |

Raw evidence: `results/raw/w4a4_cuda_gemv_m1_benchmark.json`.

The kernel assigns one four-lane subgroup to each output channel. Every lane
decodes two coordinates from one packed NQA4 block and the corresponding NQRT4
weight block, accumulates two products in FP32, and participates in a four-lane
reduction. This is a CUDA-core GEMV, not Tensor Core HMMA or INT4 GEMM.

## Correctness

- 112 / 112 real packed model projections executed;
- largest projection max absolute error versus tiled Triton: `0.0390625`;
- largest projection mean absolute error: `3.0517578125e-05`;
- unit M1 projection is checked against the tiled reference;
- unsupported M values and bias continue to fall back safely.

Raw projection audit:
`results/raw/w4a4_cuda_gemv_m1_all_projections.json`.

## Quality boundary

The M1-only backend uses tiled Triton for the prompt, so the measured prefill
logits are exactly equal to the baseline. Autoregressive tokens diverge after
the first custom decode transition. The projection audit shows no gross kernel
error; the divergence is consistent with accumulation-order differences
crossing later A4 quantization boundaries. Therefore the backend remains
explicit opt-in and this report claims decode latency improvement, not quality
equivalence.

## Claim boundary

Valid claim: on the recorded RTX 5070 single-request protocol, the direct
packed M1 GEMV reduced complete-model TPOT from 170.01 ms to 94.53 ms, a
three-trial median speedup of 1.803x.

Invalid extrapolations: production QPS, multi-request throughput, quality
preservation, INT4 Tensor Core execution, other GPUs, or unmeasured shapes.
