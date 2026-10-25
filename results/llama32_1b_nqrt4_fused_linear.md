# Llama-3.2-1B NQRT v4 packed-resident fused Linear

## Scope

This is the first kernel in the project that consumes packed NQRT v4 weights
and directly produces Linear outputs without materializing the complete BF16
weight matrix. The Triton program decodes each E8 block inside the K loop,
reuses that decoded block across a token tile, multiplies it by BF16 inputs,
and accumulates FP32 outputs.

The validated target is
`model.layers.0.self_attn.q_proj.weight` with shape 2048 x 2048 on an RTX 5070.
The input Hadamard transform is outside the benchmark. Packed streams and
compact metadata are prepared once and remain resident on the GPU.

## Storage and temporary memory

| Representation | Bytes | Relative to BF16 |
|---|---:|---:|
| resident NQRT streams + row norms + beta table | 2,236,432 | 26.66% |
| complete BF16 weight | 8,388,608 | 100% |

The fused call's measured peak increment is only its output: 4/32/64 KiB for
sequence lengths 1/8/16. Decode-then-GEMM increases allocated memory by about
10.13 MiB because it creates the full weight and decode intermediates.

## Latency and numerical agreement

Each path is warmed up and measured 30 times with CUDA synchronization.

| Sequence length | BF16 resident GEMM | Decode then GEMM | Packed fused | Fused vs decode+GEMM | Relative L2 error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0369 ms | 0.4667 ms | 0.5269 ms | 1.13x slower | 0.268% |
| 8 | 0.0321 ms | 0.4905 ms | 0.5454 ms | 1.11x slower | 0.247% |
| 16 | 0.0870 ms | 0.4994 ms | 0.5879 ms | 1.18x slower | 0.258% |

Maximum absolute error versus PyTorch BF16 GEMM is 0.03125-0.0625. The fused
kernel reconstructs the same packed weights, but its explicit FP32 reduction
order differs from cuBLAS BF16 GEMM, so bitwise equality is not expected.
Small synthetic CUDA tests require max error <= 0.0625 and relative L2 error
below 1%.

## Kernel design and current limitation

One program handles a tile of output rows and 1-16 tokens. Every E8 block is
decoded once per token tile rather than once per token. Automatic token-tile
selection makes token counts up to 16 the intended first-stage workload.
Longer sequences launch multiple token tiles and repeat weight decoding, so
this version is not a competitive prefill GEMM.

The kernel uses ordinary FP32 scalar/vector accumulation. It does not yet map
decoded tiles into Tensor Core matrix instructions, use shared-memory
double-buffering, or autotune row/token tile shapes. Consequently it remains
far slower than a resident BF16 cuBLAS GEMM and is 11-18% slower than the
existing decode-then-GEMM path. Its demonstrated benefit is eliminating full
BF16 weight materialization and reducing persistent plus temporary weight
memory, not latency.

## Claim boundary

The result supports the claim that the project has a real packed-weight fused
compute prototype: packed bytes directly produce Linear outputs. It does not
support claims of an optimized INT4 Tensor Core kernel, end-to-end model
speedup, production serving throughput, or exact cuBLAS numerical equivalence.
Machine-readable measurements are in
`results/raw/llama32_1b_nqrt4_qproj_fused_linear.json`.
