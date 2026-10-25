# Llama-3.2-1B grouped packed gate/up kernel

## Scope

This experiment tests whether combining the two large MLP input projections
into one Triton launch improves the current packed fused runtime. Both real
block-0 weights have shape 8192 x 2048. The already shared Hadamard-transformed
input is excluded from both paths.

The grouped pair uses two projection planes of the same conditional packed
kernel used by grouped QKV. No third projection plane is launched or computed.
Gate and up retain independent payloads, beta tables, row norms, and dither
seeds.

## Correctness and latency

Every path is warmed and measured 50 times with CUDA synchronization.

| Sequence length | Two independent launches | One grouped launch | Grouped speed | Gate/up exact |
|---:|---:|---:|---:|---|
| 1 | 1.8892 ms | 1.8983 ms | 0.995x | true / true |
| 8 | 1.8774 ms | 1.9031 ms | 0.986x | true / true |
| 16 | 1.8796 ms | 1.9372 ms | 0.970x | true / true |

Both outputs are bitwise equal to the independent fused kernels, but grouping
is 0.5-3.0% slower. Unlike QKV, these are two equally large projections. Their
packed decode and scalar FP32 accumulation dominate one saved launch, while
conditional projection selection adds scheduling complexity.

## Decision

The grouped gate/up path will not be integrated into the model adapter. The
isolated benchmark already rejects it, so a block integration would add cache
lifecycle complexity without a supported performance benefit. Together with
the near-zero block-level effect of grouped QKV, this closes the launch-only
optimization branch for the current scalar kernel.

The next meaningful runtime direction is to redesign the decoded dot product
around `tl.dot`/Tensor Core-compatible tiles or another higher-throughput
matrix-multiply dataflow. Merely grouping more projections is not sufficient.

Machine-readable measurements:
`results/raw/llama32_1b_nqrt4_grouped_gate_up.json`.

Reproduction entry point: `scripts/benchmark_triton_grouped_gate_up.py`.
