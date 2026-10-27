# Llama-3.2-1B real-checkpoint J2 CUDA MMA integration

## Scope

The cooperative J2 CUDA Tensor Core kernel now consumes real packed records
from the 1,044,005,774-byte Llama-3.2-1B NQRT v4 checkpoint. A new
`cuda_mma_j2` backend is integrated into `FusedNQRTLinear`. It runs when the
flattened token count is exactly 16 and the BF16 q14 record satisfies the J2
layout constraints. Other token counts fall back to the existing Triton
Tensor-Core-oriented backend.

The checkpoint is
`results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc` and the
model revision is `4e20de362430cd3b72f300e6b0f18e50e7166e08`.

## Real-record E8 boundary correction

The first real-record run found exactly one discrepant output row in gate and
down projections. The four-lane E8 implementation used a butterfly sum for
the two coset errors, while the upstream PyTorch reference uses a particular
length-eight CUDA reduction order. Rare Voronoi-boundary blocks consequently
selected the other E8 coset.

J2 now gathers the eight squared residuals and evaluates
`((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7))` with explicit `add.rn.f32`
instructions, matching the PyTorch/Triton reference. Gate maximum error fell
from 0.131991 to 0.000645 and down maximum error from 0.259887 to 0.000519.

## Real packed Linear records

Inputs are deterministic synthetic BF16 activations with M=16. References use
the same real records decoded to BF16 followed by FP32 matrix multiplication.
Timing uses 25 warm-ups, 100 calls per sample, and seven samples.

| Projection | shape N x K | J2 BF16 | Triton packed BF16 | J2 speedup vs Triton | resident BF16 | J2 FP32 max error |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 2048 x 2048 | 105.507 us | 340.306 us | 3.23x | 15.163 us | 8.392e-04 |
| gate_proj | 8192 x 2048 | 369.574 us | 1055.84 us | 2.86x | 32.676 us | 6.451e-04 |
| down_proj | 2048 x 8192 | 373.389 us | 1353.42 us | 3.62x | 36.189 us | 5.193e-04 |

J2 remains slower than a fully decoded resident BF16 weight. The valid result
is a 2.86-3.62x improvement over the existing packed Triton path while retaining
the packed persistent representation, not a speedup over cuBLAS.

Machine-readable data:
`results/raw/llama32_1b_nqrt4_mma_j2_checkpoint.json`.

## Real decoder block, sequence length 16

All seven projections in decoder block 0 use real packed records through J2.
The block includes attention, RoPE, RMSNorm, MLP, residual paths, and the
checkpoint's online Hadamard transforms.

| Metric | decoded resident BF16 block | packed J2 block |
|---|---:|---:|
| median latency, 10 repeats | 3.198 ms | 3.360 ms |
| persistent projection weights | 121,634,816 B | 32,403,568 B |
| peak temporary increment | 1,179,648 B | 9,175,040 B |

The packed J2 block is 5.1% slower in the same run while reducing persistent
projection-weight bytes by 89,231,248 bytes, or 73.36%. Block output maximum
absolute error is 0.015625 and relative L2 error is 0.8896%.

The first call in this run took 34.87 seconds because it included JIT extension
compilation as well as seven record reads and GPU preparation. This is not a
clean checkpoint-loading measurement. A subsequent process with the compiled
extension cached measured 0.481 seconds for the corresponding preparation path.

Machine-readable data:
`results/raw/llama32_1b_nqrt4_mma_j2_block0_s16.json`.

## Fallback validation

Sequence lengths 1 and 8 do not satisfy the J2 M=16 specialization and
successfully use the existing Triton dot backend. Their maximum block errors
are 0.001953 and 0.023438. Only five timing repeats were used, so their noisy
latencies are fallback functional checks rather than performance claims.

Machine-readable data:
`results/raw/llama32_1b_nqrt4_mma_j2_block0_fallback_s1_s8.json`.

## Claim boundary and next work

This is a real packed decoder-block integration and a substantial packed-kernel
improvement. It is not full-model generation throughput and does not show a
speedup over resident BF16. The next step is to support M=1 for autoregressive
decode or general multi-tile M for prefill, then replace all model layers and
measure end-to-end tokens per second.
