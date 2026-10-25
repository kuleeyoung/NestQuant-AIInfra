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
the measured K/warp choices for these shapes. Unknown shapes use a conservative
fallback selected only by token count. It is an empirical RTX 5070 dispatch
table, not a portable autotuner.

### Long-prefill M-tile sweep

The first dispatch fixed `BLOCK_M=16`. That makes the same packed weight tile
be reconstructed by 2/4/8 independent programs at sequence lengths 32/64/128.
A follow-up sweep added M=32/64/128 across all four projection shapes. All 612
candidate executions compiled and ran without failure.

| Projection class | Seq 32: best M / speedup over M16 | Seq 64 | Seq 128 |
|---|---:|---:|---:|
| q/o | 128 / 1.799x | 64 / 3.109x | 128 / 4.307x |
| k/v | 64 / 1.146x | 64 / 1.415x | 128 / 2.254x |
| gate/up | 64 / 1.866x | 64 / 3.543x | 128 / 6.053x |
| down | 32 / 1.835x | 64 / 3.320x | 128 / 5.810x |

The result confirms repeated reconstruction across M tiles as a dominant
long-prefill cost. The runtime dispatcher now preserves the measured M/N/K and
warp choices through sequence length 128. Larger token counts reuse M128 as an
explicitly unverified fallback; another GPU requires retuning.

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

For long prefill, the large-M dispatch produces:

| Sequence length | Resident BF16 block | Packed Tensor Core block | Scalar packed block | Dot speedup over scalar | Dot overhead vs BF16 |
|---:|---:|---:|---:|---:|---:|
| 32 | 2.9408 ms | 3.5856 ms | 8.2970 ms | 2.314x | 21.9% |
| 64 | 3.0218 ms | 3.8294 ms | 14.5623 ms | 3.803x | 26.7% |
| 128 | 2.9787 ms | 4.3292 ms | 26.6830 ms | 6.164x | 45.3% |

The scalar-to-dot gain grows with token count and reaches an 83.8% latency
reduction at length 128. cuBLAS also becomes more efficient at these matrix
sizes, so eliminating repeated M-tile decode is necessary but not sufficient
to beat the resident-BF16 path.

## Kernel attribution

PyTorch Profiler was run for ten decoder-block calls after warm-up. At sequence
length 1, the 70 NQRT kernel events (seven projections times ten calls) account
for 85.4% of packed-path GPU self time. At length 128 they account for 79.3%.
No other individual event approaches this share, so Hadamard, attention, and
generic elementwise kernels are not the primary remaining bottleneck.

The isolated projection timings attribute about 46-48% of the seven fused
Linear time to gate/up, 25% to down, 17% to q/o, and 11% to k/v. The three MLP
projections therefore contribute about 71-73% and are the highest-value target.

A compile-time performance ablation preserves packed reads, coset construction,
beta and row scaling, and the same `tl.dot`, but removes stateless RNG and both
E8 nearest-point projections. Its output is intentionally invalid and is used
only for timing:

| Projection | Seq | Full kernel | No-RNG/E8 ablation | Time removed |
|---|---:|---:|---:|---:|
| q | 1 | 0.2095 ms | 0.0774 ms | 63.0% |
| gate | 1 | 0.6620 ms | 0.1868 ms | 71.8% |
| down | 1 | 0.6308 ms | 0.1502 ms | 76.2% |
| q | 128 | 0.2486 ms | 0.0960 ms | 61.4% |
| gate | 128 | 0.6962 ms | 0.1955 ms | 71.9% |
| down | 128 | 0.7036 ms | 0.2229 ms | 68.3% |

Deleting instructions can also change register pressure and occupancy, so the
percentages are an attribution experiment rather than an exact additive timing
model. They nevertheless identify stateless dither and E8 reconstruction—not
the BF16 Tensor Core dot—as the next optimization target.

## Decision and next step

The Tensor Core path replaces the scalar fused path for independent packed
Linears. Grouped QKV remains optional because its scalar grouped kernel does not
share this dot implementation. The large-M dispatch addresses repeated E8
reconstruction through length 128. The profile and compile-time ablation now
identify RNG/E8 reconstruction as the next target. The next experiment should
separately remove the first dither nearest-point projection and the second
modulo-q projection, then optimize the dominant one without changing the
bit-exact codec contract. The current kernel must still be
described as packed NQRT decode feeding BF16 Tensor Core MMA, not INT4 Tensor
Core GEMM.

Machine-readable measurements are retained under `results/raw/` for q, k, gate,
and down projections, the corresponding `block_m_sweep` runs, and short/long
decoder-block comparisons.

Reproduction entry points: `scripts/benchmark_triton_dot_linear.py` and
`scripts/benchmark_fused_nqrt_block.py`.
