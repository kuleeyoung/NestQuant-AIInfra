# NQRT v4 stateless dither feasibility

Date: 2026-10-25  
GPU: NVIDIA GeForce RTX 5070  
Scope: dither generation only

## Correctness gate

- CPU and CUDA generation are bitwise equal for 4,096 E8 blocks.
- Full generation equals concatenated chunks with offsets 0, 11, and 30.
- Random access to blocks 11-29 equals the corresponding full-tensor slice.
- NQRT v4 reload is invariant across row chunk sizes 1, 2, 4, and 7.
- Triton-unpack hybrid decode agrees with the reference v4 chunked decoder.

## Shape-matched timing

The legacy side restores the generator state from the real NQRT v3 checkpoint.
The v4 side generates different, intentionally new values with the stateless
algorithm on GPU. Ten synchronized repetitions were used.

| Tensor shape | E8 blocks | v3 CPU replay mean | v4 GPU random access mean | Speedup |
|---:|---:|---:|---:|---:|
| 2048 x 2048 | 524,288 | 56.86 ms | 2.58 ms | 22.05x |
| 8192 x 2048 | 2,097,152 | 266.44 ms | 16.26 ms | 16.38x |
| 2048 x 8192 | 2,097,152 | 271.94 ms | 16.27 ms | 16.71x |

This establishes that the v3 sequential replay bottleneck can be replaced by
an addressable GPU operation. It does not establish v4 PPL, end-to-end tensor
decode speed, a fused low-bit GEMM, TTFT, or tokens/s. The next acceptance gate
is a fully re-quantized Llama-3.2-1B NQRT v4 checkpoint evaluated on the fixed
WikiText-2 protocol.
