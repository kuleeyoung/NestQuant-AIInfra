# NQRT4 J2 deterministic two-pass Split-K

## Design

The stateless J2 kernel now has a two-pass Split-K path. Grid Y selects a
contiguous range of K tiles. Every first-pass block decodes only its assigned
packed E8 blocks and writes FP32 partial outputs in `[split, 16, N]` layout. A
second kernel sums splits in fixed ascending order. No atomics, cached dither,
or decoded persistent weights are used.

The first-pass kernel uses 48 registers per thread with zero stack and zero
spills. The reduction kernel uses 36 registers with zero stack and zero spills.

## Correctness and determinism

On the K=128, N=64 probe, split counts 1, 2, 4, and 8 all have maximum absolute
error at or below 9.53674e-07. Repeating each call produces bitwise-identical
outputs. Different split counts need not be bitwise identical to each other
because they use different FP32 addition groupings.

On the complete synthetic Llama q-projection shape, M=16, K=2048, N=2048:

| splits | max abs error | mean abs error |
|---:|---:|---:|
| unsplit | 1.90735e-06 | 2.37555e-07 |
| 2 | 1.43051e-06 | 1.58059e-07 |
| 4 | 1.90735e-06 | 1.60164e-07 |
| 8 | 1.43051e-06 | 1.21556e-07 |
| 16 | 9.53674e-07 | 1.01093e-07 |

## Shape tuning and automatic dispatch

The sweep covers split counts 1 through 64 for all shapes and extends through
128 and 256 for down projection. The measured choices are encoded in
`run_m16_auto`: split 16 for the 2048x2048 family, split 8 when N is at least
8192, and split 64 when K is at least 8192 and N is at most 2048.

| Projection (N x K) | unsplit J2 | auto Split-K | speedup | workspace | resident BF16 |
|---|---:|---:|---:|---:|---:|
| q_proj 2048 x 2048 | 219.945 us | 91.214 us | 2.411x | 2 MiB | 14.841 us |
| gate/up 8192 x 2048 | 381.216 us | 338.274 us | 1.127x | 4 MiB | 35.635 us |
| down 2048 x 8192 | 886.889 us | 298.190 us | 2.974x | 8 MiB | 36.850 us |

The temporary workspace is released with the returned operation state and does
not change the packed persistent weight representation. Packed code and beta-ID
streams remain 26.56% of BF16 weight bytes.

## Interpretation

Split-K strongly helps q and down because their 2048 output channels do not
provide enough independent output tiles to hide the long serial K loop. Gate/up
already has 8192 output channels and abundant output parallelism, so extra K
parallelism yields only a modest gain and eventually adds reduction overhead.

The kernel remains slower than a pre-decoded cuBLAS BF16 GEMM. This result proves
a substantial improvement to stateless packed execution at real layer shapes;
it does not prove end-to-end model acceleration. The next integration step is
to load real NQRT records, route M=16 projection calls through `run_m16_auto`,
and measure a complete decoder block before expanding M coverage.

## Reproduction

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 \
  ~/.venvs/nestquant/bin/python scripts/benchmark_nqrt4_mma_j2_real_shapes.py
```
