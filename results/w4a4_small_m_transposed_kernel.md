# W4A4 M2/M4/M8 transposed Tensor Core kernel

## Implementation

The small-M CUDA path computes `W[N,K] x X^T[K,M]` with
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. This places up to eight
activation rows on the native eight-column side of the Tensor Core operation
and sixteen output channels on its sixteen-row side. One decoded weight
fragment is therefore reused across the active token columns.

The production dispatcher supports `M in {2,4,8}`. Unsupported shapes, biased
Linears, explicitly sensitive modules, and incompatible packed records fall
back to the tuned tiled Triton path. Callers can provide exact module names via
`w4a4_tiled_fallback_names`; the fallback is counted separately from custom
kernel dispatches.

## Correctness gates

All 112 Llama-3.2-1B projections were compared with tiled Triton for every
supported M:

| M | projections | max absolute error | max per-layer mean error | max relative L2 | quality gate |
|---:|---:|---:|---:|---:|---:|
| 2 | 112 | 0.095703125 | 1.5259e-5 | 0.001034 | 112/112 |
| 4 | 112 | 0.09765625 | 9.4045e-6 | 0.001078 | 112/112 |
| 8 | 112 | 0.11328125 | 1.0194e-5 | 0.001230 | 112/112 |

The gate is `torch.allclose(atol=0.125, rtol=0.01)`. The compiled kernel uses
56 registers per thread with zero stack, shared-memory, and local-memory bytes,
matching the existing M16 Tensor Core kernel's static resource usage.

## Tuned-runtime microbenchmarks

These comparisons use `select_w4a4_tiled_config`, the same Triton selector used
by the model runtime. Earlier fixed `16x16x32` Triton measurements are retained
only as diagnostics and are not used for the claim below.

| M | q `[2048,2048]` | k `[2048,512]` | gate `[2048,8192]` | down `[8192,2048]` |
|---:|---:|---:|---:|---:|
| 2 | 1.650x | 1.461x | 3.845x | 1.393x |
| 4 | 1.759x | 1.490x | 3.599x | 1.367x |
| 8 | 1.718x | 1.494x | 3.764x | 1.303x |

This establishes isolated kernel speedup for the tested RTX 5070 shapes. It is
not a claim that every mixed full-model policy has the same speedup.

## Short-context quality sensitivity

A deterministic WikiText-2 diagnostic used the first 128 non-overlapping
eight-token windows (896 predicted tokens). This is a sensitivity protocol,
not the formal full-corpus PPL result.

| policy | tiled fallback projections | PPL | relative to tiled |
|---|---:|---:|---:|
| all tiled | 112 | 387.092089 | reference |
| all small-M | 0 | 397.318971 | +2.642% |
| Attention tiled, MLP small-M | 64 | 385.892102 | -0.310% |
| MLP tiled, Attention small-M | 48 | 391.632729 | +1.173% |
| Q/K/V tiled | 48 | 389.154059 | +0.533% |
| O tiled | 16 | 390.060740 | +0.767% |

The conservative quality-safe policy is therefore to keep all Attention
projections on tiled Triton and use the custom small-M kernel for MLP
`gate_proj`, `up_proj`, and `down_proj`. The apparent -0.310% improvement is
treated as finite-sample noise, not a quality gain.

## Paired end-to-end M8 result

For Llama-3.2-1B, batch 1, sequence length 8, twenty steady-prefill repeats,
and the conservative Attention-fallback policy:

| path | steady prefill median | TTFT |
|---|---:|---:|
| all tuned tiled Triton | 193.794 ms | 183.486 ms |
| Attention tiled + MLP small-M | 136.413 ms | 140.507 ms |
| speedup | 1.4206x | 1.3059x |

The candidate recorded 1,152 small-M dispatches and 1,792 explicit tiled
fallbacks over the complete benchmark, so the result audits the intended mixed
path rather than silently running one backend.

## Active-subgroup decode optimization

The first M2/M4 family implementation kept all eight activation subgroups in
the E8 decoder to satisfy full-warp shuffle synchronization, then zeroed the
inactive Tensor Core columns. The decoder now accepts an explicit participation
mask: only the complete four-lane subgroups belonging to real token rows enter
the E8 projection, while all 32 lanes still converge before the MMA instruction.

Relative to the pre-mask custom kernel:

| M | q | k | gate | down |
|---:|---:|---:|---:|---:|
| 2 | 1.275x | 1.213x | 1.034x | 1.162x |
| 4 | 1.070x | 1.163x | 1.139x | 1.099x |

All comparisons retain the same Triton correctness bound. M8 is unchanged
because all eight activation subgroups are active already.

Under the conservative Attention-tiled/MLP-small-M policy, paired full-model
results with twenty steady-prefill repetitions are:

| sequence length | tiled prefill median | mixed prefill median | prefill speedup | tiled TTFT | mixed TTFT | TTFT speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 281.137 ms | 209.174 ms | 1.3440x | 287.833 ms | 213.497 ms | 1.3482x |
| 4 | 265.856 ms | 180.552 ms | 1.4725x | 263.831 ms | 201.803 ms | 1.3074x |
| 8 | 193.794 ms | 136.413 ms | 1.4206x | 183.486 ms | 140.507 ms | 1.3059x |

Absolute latency varies with GPU frequency and concurrent desktop load, so the
defensible result is the same-process paired speedup rather than comparisons
between unrelated runs.

## Evidence boundary and next work

- The full all-projection small-M path remains experimental because the short
  quality protocol detected a 2.642% PPL regression.
- The Attention-fallback policy passed the bounded diagnostic, but a formal
  full-corpus quality run has not been performed for eight-token windows and
  would not be comparable to the established context-2048 PPL protocol.
- The next kernel engineering decision is whether to test M1 in the transposed
  Tensor Core family or freeze M1 on the current packed-word GEMV and move to a
  Marlin/CuTe-style M32-M128 pipeline.

That decision was resolved in favor of the large-M route. The first multi-warp
structure results and their scaling limit are recorded in
`results/w4a4_large_m_multiwarp_probe.md`.
