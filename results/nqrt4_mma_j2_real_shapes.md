# NQRT4 J2 Tensor Core kernel on Llama-3.2-1B Linear shapes

## Scope

The J2 four-lane cooperative E8 decoder was generalized from the fixed
M=16, K=128, N=64 probe to runtime-variable K and N. The current kernel keeps
M=16 and requires K to be a multiple of 16 and N to be a multiple of 8. This
covers the principal Llama-3.2-1B projection shapes.

The benchmark uses deterministic synthetic packed q14 streams, not values read
from the real checkpoint. Therefore it establishes shape-level kernel behavior
and exact agreement with the Python reconstruction, but it is not yet a model
integration or end-to-end inference result.

## Correctness

The complete M=16, K=2048, N=2048 output was compared with chunked
Python/PyTorch reconstruction followed by FP32 matrix multiplication:

| Shape | max abs error | mean abs error | finite |
|---|---:|---:|:---:|
| 16x2048 @ 2048x2048 | 1.90735e-06 | 2.37555e-07 | yes |

The dynamic kernel also exactly matches the fixed J2 kernel on the K=128,
N=64 probe for all three tested seed/offset pairs. `ptxas` reports 40 registers
per thread, zero-byte stack frames, and zero spills.

## Stateless hash optimization

The original dither hash used 64-bit integer multiplication even though every
stage is reduced modulo 2^31. Replacing its internal arithmetic with unsigned
32-bit operations preserves the low 31 bits exactly. The independent probe
reported zero dither, reconstruction, and BF16 mismatches.

| Projection shape (N x K) | J2 before | J2 uint32 hash | latency reduction |
|---|---:|---:|---:|
| q_proj 2048 x 2048 | 361.646 us | 219.813 us | 39.2% |
| gate/up 8192 x 2048 | 593.923 us | 375.456 us | 36.8% |
| down 2048 x 8192 | 1035.87 us | 819.789 us | 20.9% |

## Current measurements

Timing uses 25 warm-ups, 200 measured calls, and the median of seven repeats.
The resident BF16 comparison has pre-decoded weights and performs no E8/dither
work, so it is a performance upper bound rather than an equal-work baseline.

| Projection | J2 median | resident BF16 median | J2 / BF16 | packed streams | BF16 weight |
|---|---:|---:|---:|---:|---:|
| q_proj 2048 x 2048 | 219.813 us | 17.9638 us | 12.236x | 2.125 MiB | 8 MiB |
| gate/up 8192 x 2048 | 375.456 us | 34.8944 us | 10.760x | 8.5 MiB | 32 MiB |
| down 2048 x 8192 | 819.789 us | 41.3832 us | 19.810x | 8.5 MiB | 32 MiB |

The packed footprint shown here includes code and beta-ID streams but excludes
the small row-norm and four-value beta metadata. It is 26.56% of the BF16
weight bytes for all three shapes.

## Interpretation and next decision

J2 removes redundant per-lane E8 decoding and the uint32 change removes
unnecessary 64-bit hash arithmetic. At real dimensions the remaining decode
work is still much more expensive than a pre-decoded cuBLAS GEMM. The especially
large down-projection cost shows that the serial K loop inside each output tile
is now a major bottleneck.

The next architectural choice is between:

1. preserving fully stateless reconstruction and parallelizing/splitting the K
   dimension, which requires partial-output reduction but keeps the smallest
   persistent representation; or
2. caching dither or partially decoded tiles, which spends additional persistent
   memory to remove repeated hash/E8 work.

No full-model speedup is claimed by this experiment.

## Reproduction

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 \
  ~/.venvs/nestquant/bin/python scripts/benchmark_nqrt4_mma_j2_real_shapes.py
```
