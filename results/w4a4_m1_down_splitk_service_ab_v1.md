# M1 down-projection split-K Graph and service A/B

## Integration

The experimental `enable_m1_down_splitk` flag is threaded through fused-linear
conversion, the W4A4KV4 server, and the fresh-process benchmark. Dispatch is
restricted to `M=1,K=8192,N=2048` and uses split-8. All other shapes retain the
existing packed-word kernel. The flag remains disabled by default.

## CUDA Graph gates

The split-K partial and fixed-order reduction kernels were captured by the
full-model Graph pool. A cross-page workload reported:

```text
graph_boundary_replays = 6
graph_boundary_fallbacks = 0
```

The PyTorch CUDA trace contained 128 M1 split-K partial launches and 128
reductions in the ordinary short-window run. Total self CUDA time fell from
864.37 to 780.57 ms (-9.7%). M1 W4A4 time fell from about 358.74 to 276.52 ms
(-22.9%); all reductions totaled only 0.105 ms.

## Four-request-per-second interleaved A/B

Five fresh-process pairs used 24 measured requests per process and the standard
three-output-token workload. All 240 requests completed without failure or
overload.

| metric, five-run mean | split-K off | split-K on | change |
|---|---:|---:|---:|
| request throughput | 3.898 req/s | 3.867 req/s | -0.8% |
| TTFT P95 | 490.33 ms | 428.58 ms | -12.6% |
| TPOT P95 | 67.21 ms | 61.11 ms | -9.1% |
| E2E P95 | 623.68 ms | 543.18 ms | -12.9% |

Pairwise TPOT deltas (`on - off`) were -18.42, +5.75, -2.34, -21.50, and
+6.01 ms. Three pairs improved and two regressed; the median pairwise
improvement was 2.34 ms. CUDA reserved memory remained in the same observed
allocation bucket for both policies.

## Two-request-per-second low-queue A/B

Three fresh-process pairs used 12 requests per process. All 72 requests
completed.

| metric, three-run mean | split-K off | split-K on | change |
|---|---:|---:|---:|
| request throughput | 2.073 req/s | 2.081 req/s | +0.4% |
| TTFT P95 | 179.28 ms | 181.16 ms | +1.0% |
| TPOT P95 | 58.12 ms | 48.16 ms | **-17.1%** |
| E2E P95 | 294.44 ms | 275.89 ms | **-6.3%** |

Every low-load TPOT pair improved by 8.50-10.71 ms and every E2E pair improved.
TTFT remained effectively unchanged, as expected because this kernel affects
decode rather than prompt prefill.

## Decision

The kernel passes isolated correctness, full Graph replay, boundary Graph, and
low-queue service gates. It also improves the five-run high-load mean, but its
high-load tail benefit is not pairwise consistent and measured throughput is
slightly lower. Keep the explicit opt-in flag rather than making it the default.

This is a useful negative-to-qualified production result: a 2.63x isolated
kernel speedup becomes a stable 17.1% low-load TPOT improvement, but only a
9.1% noisy high-load TPOT-mean improvement after scheduling and queue effects.

## Ten-pair promotion gate

A later same-protocol run expanded the 4 req/s experiment to ten interleaved
fresh-process pairs and alternated which policy ran first. All 480 requests
completed without failure or overload.

| metric, ten-run mean | split-K off | split-K on | change | improving pairs |
|---|---:|---:|---:|---:|
| request throughput | 3.833 req/s | 3.862 req/s | +0.77% | 8/10 |
| TTFT P95 | 496.60 ms | 434.63 ms | -12.48% | 9/10 |
| TPOT P95 | 69.69 ms | 65.86 ms | -5.50% | 7/10 |
| E2E P95 | 642.79 ms | 545.15 ms | -15.19% | 9/10 |

Candidate-second odd pairs improved TPOT P95 by 5.74 ms on average;
candidate-first even pairs improved it by 1.93 ms. The direction therefore
survived both execution orders. This closes the larger paired-run gate: the
exact `M=1,K=8192,N=2048` split-8 path passed this performance gate. A later
HTTP output-identity gate found autoregressive token divergence and therefore
revoked the default promotion. The path is now disabled by default and retained
only behind explicit `--enable-m1-down-splitk`; no other shape uses split-K.

## Mixed-length follow-up

Four fresh-process low-load pairs then replayed deterministic bimodal requests with
prompt lengths 4-32 and output lengths 2-8. Token throughput improved in 4/4 pairs
with a paired-median change of +0.62%; TPOT P95 improved in 3/4 pairs with a
paired-median change of -9.17%. TTFT improved in 2/4 pairs and E2E P95 improved in
2/4, with an E2E paired median of +1.90%, so the result does not support a stable
mixed-workload E2E latency claim. The hash-enabled fourth pair had identical workload
and generated-token SHA-256 off/on, and all runs reported zero NQKV boundary Graph
fallbacks. Decode-only mean step latency improved in 4/4 pairs with a paired-median
change of -15.90%; this explains why one request-level TPOT P95 can regress even though
the targeted compute phase improves, because scheduling changes which token intervals
land in mixed prefill/decode steps. This engine-level evidence remains useful for
performance attribution, but the stronger HTTP autoregressive correctness gate takes
precedence and keeps the candidate disabled by default.

Artifacts:

- `results/raw/http_m1_splitk_graph_smoke_v1/`
- `results/raw/http_m1_splitk_boundary_counter_v1_summary.json`
- `results/raw/http_m1_splitk_policy_ab_v1/`
- `results/raw/http_m1_splitk_lowload_ab_v1/`
- `results/raw/http_m1_splitk_policy_ab_v2/`
- `results/w4a4_m1_down_splitk_variance_analysis_v2.md`
- `results/w4a4_m1_down_splitk_mixed_ab_v1.md`
- `results/w4a4_m1_down_splitk_mixed_saturation_probe_v1.md`
- `results/w4a4_m1_down_splitk_http_correctness_v1.md`
