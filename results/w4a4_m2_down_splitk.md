# M2 down-projection split-K promotion

## Outcome

The deterministic split-8 CUDA path is promoted only for the verified
`M=2, K=8192, N=2048` Llama-3.2-1B down-projection shape. All other small-M
shapes retain the original kernel.

`split-K` partitions the K reduction dimension across independent CUDA thread
blocks. Each block writes an FP32 partial result, followed by a fixed-order
reduction kernel. The fixed order keeps the result reproducible while exposing
enough parallel work for the otherwise serial long-K down projection.

## Isolated layer result

Protocol: RTX 5070, 10 warmups, 100 iterations per sample, 7 rotated-order
samples, packed NQRT4 weight and packed A4 activation.

| path | median latency | speedup vs unsplit |
|---|---:|---:|
| original small-M | 2094.68 us | 1.00x |
| split-2 | 1132.93 us | 1.85x |
| split-4 | 641.99 us | 3.26x |
| split-8 | 497.28 us | 4.21x |
| split-8 CUDA Graph replay | 484.11 us | 4.33x |

For split-8 versus the original path, maximum absolute error is `0.015625`,
mean absolute error is `1.77e-5`, cosine similarity rounds to `1.0`, and the
result passes `atol=0.05, rtol=0.01`.

## Full-model CUDA Graph result

The same PyTorch profiler protocol captures one normal and one page-boundary
decode Graph for the complete packed-W4A4 + NQKV Llama-3.2-1B runtime.

| metric across two Graphs | before | after | change |
|---|---:|---:|---:|
| Graphed GPU total | 150.385 ms | 117.654 ms | -21.77% |
| 32 down projections | 52.661 ms | 13.286 ms | 3.96x faster |
| all packed W4A4 kernels | 133.505 ms | 100.837 ms | -24.47% |

The 32 split-K reduction launches total only `0.0267 ms`. Both normal and
boundary Graphs replay successfully; boundary eager fallback remains zero.

In the four-request service smoke, three hot repetitions give a median 11.89
token/s and 181.67 ms TPOT P95 for NQKV4. A freshly rerun BF16-KV baseline with
the same W4A4 split-K path gives 14.17 token/s and 151.75 ms. Thus NQKV4 still
trails by 16.1% throughput and 19.7% TPOT on this bounded workload; the kernel
promotion narrows a data-plane bottleneck but does not erase the KV4 attention
cost.

## Verification and evidence

- CUDA tests: `16 passed`, including `18` parameterized subtests.
- Isolated raw result: `results/raw/w4a4_m2_down_splitk_probe.json`.
- Full-model capture metadata: `results/raw/nqkv_graph_splitk_probe.json`.
- Full-model event summary: `results/raw/nqkv_graph_splitk_torch_summary.json`.
- Projection breakdown: `results/raw/nqkv_graph_splitk_shape_breakdown.json`.
- NQKV4 service run: `results/raw/llama32_1b_w4a4kv4_service_splitk_smoke.json`.
- BF16-KV service run: `results/raw/llama32_1b_w4a4_bf16kv_service_splitk_smoke.json`.

## Claim boundary

This establishes a shape-specific kernel and full-model Graph improvement on
Llama-3.2-1B with capacity 2. It does not establish the same speedup for M4/M8,
other model dimensions, general serving workloads, or end-to-end network
latency. The current C++ wrapper allocates Graph-pool-backed partial storage;
an explicit reusable workspace remains a possible engineering cleanup, not a
correctness or Graph-replay blocker in this measured fixed-shape path.
