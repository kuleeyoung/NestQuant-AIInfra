# Llama-3.2-1B NQRT Tensor Core block-dot prototype

## Scope

This prototype decodes packed NQRT v4 weights one K tile at a time inside a
Triton kernel, converts the reconstructed tile to BF16, and feeds it directly
to `tl.dot` with an FP32 accumulator. It never materializes the complete BF16
weight matrix.

The validated real weight is block-0 `q_proj`, shape 2048 x 2048, on an NVIDIA
GeForce RTX 5070. The search covers block N 16/32, block K 32/64/128, 4/8 warps,
and sequence lengths 1/8/16/128. All 48 combinations compile and run.

## E8 tile correctness

An initial implementation incorrectly passed a full K=32 vector to the E8
nearest-point helper, mixing four independent eight-dimensional parity/coset
decisions and producing a maximum output error of 44.25. The corrected kernel
represents decoded data as `[N, K/8, 8]`, performs every nearest-E8 operation
along the final eight-coordinate axis, then reshapes the reconstructed tile to
`[N, K]` for the block dot product.

The CUDA regression test requires maximum error <= 0.0625 and relative L2
error < 1% against a dequantized BF16 PyTorch Linear.

## Real-weight latency

Each configuration is warmed and measured 20 times with CUDA synchronization.

| Sequence length | Best dot config N/K/warps | Tensor Core dot | Scalar fused | Dot speedup | Resident BF16 cuBLAS | Relative L2 error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 16/32/8 | 0.2087 ms | 0.3747 ms | 1.795x | 0.0301 ms | 0.000000% |
| 8 | 16/64/4 | 0.2238 ms | 0.3440 ms | 1.537x | 0.0328 ms | 0.000446% |
| 16 | 16/64/8 | 0.2280 ms | 0.3828 ms | 1.679x | 0.0539 ms | 0.005358% |
| 128 | 16/32/4 | 1.1010 ms | 1.7809 ms | 1.617x | 0.0429 ms | 0.268700% |

Maximum absolute error for the four best configurations is 0/0.000977/0.015625/
0.0625. The dot path is substantially faster than the earlier explicit
elementwise multiply-and-reduce kernel, but still 4.2-25.7x slower than cuBLAS
with a complete BF16 weight already resident. Packed decode and E8
reconstruction remain inside every K tile and dominate more heavily as token
count grows.

## Tensor Core evidence

The generated Triton GPU IR contains an NVIDIA MMA layout and the generated PTX
contains instructions of the form:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

This is direct compiler-artifact evidence that the BF16 block dot maps to
Tensor Core matrix-multiply-accumulate instructions. It supports a Tensor Core
prototype claim; it does not imply an optimized INT4 Tensor Core kernel because
the packed NQRT values are reconstructed to BF16 before MMA.

## Projection-shape sweep and dispatch

The follow-up sweep covers all four distinct block-0 projection shapes. Every
one of the 192 shape/sequence/configuration combinations compiled and ran. The
best dot kernel is 1.537-2.315x faster than the scalar fused kernel at the
measured points.

| Projection class | Weight shape | Seq 1 | Seq 8 | Seq 16 | Seq 128 |
|---|---:|---:|---:|---:|---:|
| q/o | 2048 x 2048 | 1.795x | 1.537x | 1.679x | 1.617x |
| k/v | 512 x 2048 | 2.315x | 2.018x | 2.050x | 1.607x |
| gate/up | 8192 x 2048 | 1.600x | 1.641x | 1.650x | 1.711x |
| down | 2048 x 8192 | 2.003x | 2.049x | 2.063x | 1.691x |

`BLOCK_N=16` wins at every measured point. The checked-in dispatcher preserves
the measured K/warp choices for these shapes and uses K32/W4 as a conservative
fallback for unknown shapes or token counts above 16. It is an empirical RTX
5070 dispatch table, not a portable autotuner.

## Real decoder-block integration

`FusedNQRTLinear` now exposes `tensor_core_dot` and `scalar` compute backends,
with the Tensor Core path as the default. Replacing all seven projections in a
real Llama decoder block produces the following steady-state result (100 warmed,
synchronized repetitions per backend, run back-to-back):

| Sequence length | Resident BF16 block | Packed Tensor Core block | Packed/BF16 speed | Speedup over scalar packed block | Relative L2 error |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.1813 ms | 3.9220 ms | 0.811x | 1.726x | 0.139% |
| 8 | 3.1743 ms | 4.0625 ms | 0.781x | 1.626x | 0.930% |
| 16 | 3.2821 ms | 3.8935 ms | 0.843x | 1.571x | 1.620% |

The seven resident projection weights occupy 121,634,816 bytes as decoded BF16
and 32,403,568 bytes in packed form. The packed runtime therefore retains the
73.36% weight-memory reduction while reducing the scalar packed block's latency
by 36.3-42.1%. It remains 18.6-28.0% slower than its resident-BF16 block
baseline, so this is
not evidence of end-to-end model speedup.

## Decision and next step

The Tensor Core path replaces the scalar fused path for independent packed
Linears. Grouped QKV remains optional because its scalar grouped kernel does not
share this dot implementation. The next bottleneck is repeated E8 reconstruction
across M tiles, especially for longer prefill. A future kernel should either
reuse decoded weight tiles across more token rows or adopt a genuinely low-bit
MMA representation; the current kernel must still be described as packed NQRT
decode feeding BF16 Tensor Core MMA, not INT4 Tensor Core GEMM.

Machine-readable measurements are retained under `results/raw/` for q, k, gate,
and down projections plus `llama32_1b_nqrt4_tensor_core_block0.json`.

Reproduction entry points: `scripts/benchmark_triton_dot_linear.py` and
`scripts/benchmark_fused_nqrt_block.py`.
