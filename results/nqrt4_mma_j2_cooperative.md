# NQRT4 J2 cooperative E8 decode

## Purpose

J1 was the correctness baseline: every CUDA lane independently decoded complete
eight-dimensional E8 blocks. J2 assigns one E8 block to each adjacent four-lane
group. Every lane owns two coordinates and the group uses warp shuffle operations
for parity, maximum-residual selection, and squared-error reduction.

## Environment

- GPU: NVIDIA GeForce RTX 5070, compute capability 12.0
- CUDA toolkit: 13.0.88
- MMA: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
- Workload: M=16, K=128, N=64; packed q14/NQRT4 weights
- Timing: 100 warm-up launches, 2000 measured launches, median of 9 repeats

## Correctness

Compared with the Python/PyTorch fake-quant reconstruction followed by FP32
matrix multiplication:

| seed | block offset | max abs error | mean abs error | finite |
|---:|---:|---:|---:|:---:|
| 58 | 37 | 9.53674e-07 | 5.82077e-09 | yes |
| 0 | 0 | 9.53674e-07 | 1.07102e-08 | yes |
| 1234567 | 4096 | 9.53674e-07 | 2.15368e-08 | yes |

The standalone cooperative E8 probe also matched the scalar CUDA reference on
4096 vectors / 32768 coordinates, including exact integer and half-integer
boundaries: zero mismatches and zero maximum absolute error.

## Kernel resources and latency

| implementation | registers/thread | stack/thread | spills | median latency |
|---|---:|---:|---:|---:|
| J1 redundant full-vector decode | 80 | 160 B | 0 | 21.624 us |
| J2 four-lane cooperative decode | 44 | 0 B | 0 | 15.1585 us |
| resident BF16 GEMM, J2 measurement run | - | - | - | 12.2259 us |

J2 is about 29.9% faster than J1 on this shape. Its fused-to-resident-BF16
latency ratio is 1.240x, down from J1's previously measured 1.674x. The result
isolates redundant decode work and per-thread local arrays as major J1 costs;
it does not yet establish a full-model speedup.

## Reproduction

```bash
CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 \
  ~/.venvs/nestquant/bin/python scripts/build_test_nqrt4_mma_j2.py
```
