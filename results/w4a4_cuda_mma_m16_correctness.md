# Direct-packed W4A4 CUDA MMA M16

## Result

The experimental CUDA backend now consumes packed NQA4 activations and packed
NQRT4 weights directly and executes Tensor Core
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` without materializing a
dense BF16 activation or weight matrix.

Validated projection:

- weight: `model.layers.0.self_attn.q_proj.weight`
- shape: `M=16, K=2048, N=2048`
- device: NVIDIA GeForce RTX 5070
- activation blocks compared: 4096
- CUDA versus upstream/Triton activation block mismatches: **0**
- CUDA MMA versus tiled Triton max absolute error: **0.0078125**
- CUDA MMA versus tiled Triton mean absolute error: **7.2133e-7**
- all outputs finite: yes

The backend is integrated as `packed_w4a4_cuda_mma`. Dispatch is deliberately
limited to the validated no-bias `M=16`, `K % 16 == 0`, `N % 8 == 0`, q14,
4-bit-code/2-bit-beta NQA4 and compatible NQRT4 v4 case. Unsupported shapes or
bias fall back to the tiled Triton backend.

## Compiled instruction audit

`scripts/verify_w4a4_mma_binary.py` loads the actual PyTorch CUDA extension,
hashes the shared object, runs CUDA 13.0 `cuobjdump --dump-sass`, and requires
the compiled binary to contain:

```text
HMMA.16816.F32.BF16 R12, R16, R30, R12
```

The current RTX 5070 artifact contains exactly one matching instruction and
has SHA-256
`e36008739a674b4d5acf2236e5c0b3fba737e8c8248e11f00729fef1a2fb4496`.
Machine-readable evidence is stored in
`results/raw/w4a4_cuda_mma_binary_audit.json`. This proves that the compiled
kernel contains BF16 Tensor Core HMMA SASS; correctness and speed remain
separate claims supported by the tests and benchmarks below.

## Stable microbenchmark

Seven interleaved groups, each with 10 warmups and 100 measured iterations:

| backend | median us | min us | max us |
|---|---:|---:|---:|
| direct-packed CUDA MMA | 411.41 | 408.53 | 413.90 |
| tiled Triton | 552.34 | 547.43 | 569.50 |

Median speedup: **1.343x** for this projection and shape.

This is a kernel microbenchmark, not an end-to-end model speedup claim.

Raw evidence:

- `results/raw/w4a4_cuda_mma_m16_cooperative_stable.json`
- `results/raw/w4a4_cuda_mma_m16_corrected.json`

## Root cause of the former boundary mismatch

Three of 4096 E8 activation blocks initially disagreed. The issue was not
packing, beta selection, MMA fragment layout, or general numerical instability.
It was FP32 non-associativity in the half-coset correction residual:

```text
upstream: (x - 0.5) - round(x - 0.5)
former CUDA: x - (round(x - 0.5) + 0.5)
```

The expressions are identical over real numbers but can differ by one FP32
bit. At exact Voronoi boundaries that bit changes `argmax`, so a different E8
coordinate is corrected. Preserving the upstream operation grouping reduced
activation mismatches from 3 to 0 and Linear max error from 1.546875 to
0.0078125.

## Performance optimization

The correctness-first decoder redundantly evaluated all eight E8 coordinates
in each of four lanes. It was replaced with a four-lane cooperative decoder:
each lane owns two coordinates, subgroup shuffles reduce parity, argmax, and
candidate error, and every lane produces one BF16 pair for MMA. This reduced
the measured CUDA latency from approximately 598 us to 411 us while preserving
zero activation mismatch.

## Rejected approaches

The following approaches were measured and reverted:

1. ordinary CUDA division instead of reciprocal multiplication: 116 differing
   blocks and 0.38467 Linear mean absolute error;
2. exact integer-domain correction-coordinate selection: 35 differing blocks
   and 0.21119 mean absolute error;
3. fixed `1e-7` tie tolerance: 30 differing blocks and 0.19769 mean absolute
   error;
4. compile-time versus runtime q changes: no correctness benefit after the
   residual grouping was fixed.

These results demonstrate why mathematical equivalence alone is insufficient
for a bit-sensitive low-precision codec implementation.

## Verification

- targeted CUDA/backend/dispatch tests: 4 passed;
- formal pytest suite: 305 passed plus 37 subtests in 16.19 seconds;
- boundary regression codes are permanently covered in
  `tests/unit/runtime/test_cuda_w4a4_mma.py`;
- model-level dispatch and unsupported-shape fallback are covered in
  `tests/unit/models/test_fused_linear.py`.

## Remaining scope

The backend currently accelerates only the validated M16 projection case.
Full-model dispatch counting, three-trial latency measurement, the 112
projection audit, and end-to-end W4A4KV4 integration are complete. Remaining
scope is the user-level route decision between a dedicated M1 decode/GEMV
kernel, wider prefill M tiles, or a dedicated long-text M16 quality protocol.
