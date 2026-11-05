# Packed W4A4 dynamic-M chunked staged runtime

## Motivation and workload evidence

The fixed CUDA family covers M1/M2/M4/M8/M16/M32/M64/M128. A real
continuous-batching engine, however, owns eight physical slots and pads each
admitted prompt microbatch to its maximum prompt width. Its prefill Linear
shape is therefore `M = 8 * prompt_width`, while graph decode remains M8.

One repetition of all three official synthetic profiles and both admission
policies produced six completed runs, 105 prefill model forwards, and 260
decode forwards. The measured shape distribution shows:

| scope | forwards | exact fixed-kernel coverage | fallback forwards |
|---|---:|---:|---:|
| prefill | 105 | 8 (7.62%) | 97 (92.38%) |
| decode | 260 | 260 (100%) | 0 |
| combined | 365 | 268 (73.42%) | 97 (26.58%) |

Decode is already solved by M8. Prefill includes M40 through M384, with common
shapes such as M120, M192, M208, M216, M232, and M240. Adding a few isolated
fixed kernels would therefore leave most of the observed distribution
uncovered.

## Chunk-plus-padding design

The arbitrary-M wrapper divides packed activation rows into full M128 chunks
and pads only the final remainder to the smallest supported size. Packed A4
payloads are sliced on row-byte boundaries. Padding rows use zero row norms,
so their decoded activation is exactly zero without re-quantizing valid rows.
Outputs from padding rows are discarded and valid chunk outputs are
concatenated in original order.

Across the measured workload this policy covers every observed M with 11.19%
weighted extra rows. Prefill requires 179 CUDA chunk launches for 105 model
forwards. This is a structural simulation, not a latency claim; actual timing
below includes slicing, allocation, launch, trim, and concatenation overhead.

## Projection microbenchmarks

Same-process rotating-order measurements use layer-0 q-projection
`[N=2048,K=2048]`, ten warmups, one hundred iterations, and seven groups:

| M | chunked staged | tuned Triton | speedup |
|---:|---:|---:|---:|
| 40 | 580.94 us | 1085.58 us | 1.869x |
| 120 | 1178.16 us | 1737.72 us | 1.475x |
| 240 | 2343.48 us | 3238.56 us | 1.382x |
| 384 | 3495.93 us | 4311.69 us | 1.233x |

All four pass the benchmark numerical gate. M384 uses three chunks and remains
23.3% faster than tuned Triton despite Python-side composition overhead.

## Full-projection quality audit

The common M240 and maximum observed M384 were audited across all 112 packed
Llama-3.2-1B projections:

| M | strict-gate passes | maximum relative L2 | strict-gate failures |
|---:|---:|---:|---|
| 240 | 110/112 | 0.1223% | layer-5 MLP gate/up |
| 384 | 110/112 | 0.1167% | layer-5 MLP gate/up |

The failure set is unchanged from M32/M64/M128. Production dispatch therefore
retains exact-name tiled fallback for those two sensitive projections.

## Full-model validation

Llama-3.2-1B batch-one non-discrete prefill results use seven steady repeats,
the same packed checkpoint, and exact fallback for layer-5 gate/up:

| sequence | tiled prefill | chunked prefill | prefill speedup | TTFT speedup |
|---:|---:|---:|---:|---:|
| 40 | 265.18 ms | 182.48 ms | 1.453x | 1.578x |
| 120 | 424.48 ms | 304.60 ms | 1.394x | 1.481x |

Each CUDA run records 1,210 chunked dispatches and 28 tiled sensitive-layer
fallbacks. Exact large-M dispatch remains zero, proving these shapes use the
new arbitrary-M path. First-prefill logits are not bit-equivalent to tiled
Triton: relative L2 errors are 20.28% and 16.13% for seq40 and seq120. Greedy
tokens may diverge, so the path is not described as end-to-end quality lossless.

## Evidence and boundary

- `scripts/benchmark_service_workloads.py` records real engine M histograms.
- `scripts/summarize_service_m_distribution.py` computes coverage and padding.
- `results/raw/llama32_1b_service_m_distribution_{v1,summary_v1}.json` stores
  the six-run workload evidence.
- `results/raw/w4a4_m{40,120,240,384}_q_chunked_staged_v1.json` stores paired
  projection timings.
- `results/raw/llama32_1b_w4a4_m{240,384}_chunked_112_projections.json` stores
  quality audits.
- `results/raw/llama32_1b_w4a4_chunked_staged_e2e_seq40_120.json` stores the
  end-to-end comparison.

The service histogram benchmark used the established packed NQRT engine to
measure shapes; it did not benchmark service latency with the new W4A4 backend.
The end-to-end results establish batch-one prefill benefit, not production QPS,
multi-user SLA, multi-GPU scaling, or token-by-token equivalence.

## Continuous-batching service A/B

The service harness now accepts `packed_w4a4_tiled` and
`packed_w4a4_cuda_mma` backends plus exact sensitive-layer fallback names. A4
beta tables are registered as module buffers and moved with each converted
Linear, eliminating CPU-to-CUDA copies during CUDA Graph capture. This was
required for packed W4A4 graph decode; creating a beta tensor inside forward is
illegal during capture.

The selected `8/4` admission policy and steady 50 req/s synthetic profile were
run three times per backend with paired seeds and request/token counts:

| metric | paired median change | paired range |
|---|---:|---:|
| token throughput | **+51.96%** | +46.16% to +65.38% |
| TTFT P95 | **-37.00%** | -32.77% to -41.49% |
| TPOT P95 | **-31.76%** | -23.66% to -40.36% |
| E2E P95 | **-36.24%** | -33.40% to -41.92% |

All six backend runs completed all 48 requests with zero failures and no final
request-registry residue. All three paired CUDA runs improved token throughput.
The paired per-run throughput values were 20.42 to 31.03, 18.06 to 26.39, and
17.42 to 28.81 token/s.

The same three-pair protocol was then applied to the burst and bimodal
profiles:

| profile | token throughput | TTFT P95 | TPOT P95 | E2E P95 |
|---|---:|---:|---:|---:|
| steady 50 req/s | **+51.96%** | **-37.00%** | **-31.76%** | **-36.24%** |
| burst 16x3 | **+61.21%** | **-40.60%** | **-29.31%** | **-41.05%** |
| Poisson 20 req/s bimodal | **+49.84%** | **-37.55%** | **-40.25%** | **-37.61%** |

Across the additional twelve backend runs, every run again completed all 48
requests with zero failures. The table reports the median of three paired
relative changes per profile, never a ratio of unrelated aggregate medians.

Python dispatcher counters prove graph capture and prefill routing, but they do
not count each CUDA Graph decode replay because replay bypasses Python forward.
The service result is therefore supported by wall-clock request/token metrics,
while dispatch counters are used only as routing diagnostics.

Additional evidence:

- `results/raw/llama32_1b_service_w4a4_{tiled,cuda}_steady_selected3.json`
- `results/raw/llama32_1b_service_w4a4_cuda_vs_tiled_steady_selected3.json`
- `results/raw/llama32_1b_service_w4a4_{tiled,cuda}_burst_bimodal_selected3.json`
- `results/raw/llama32_1b_service_w4a4_cuda_vs_tiled_burst_bimodal_selected3.json`
- `scripts/compare_w4a4_service_backends.py` validates pairing and summarizes
  per-pair relative changes.

This remains a single-process synthetic token-ID engine benchmark without
tokenizer, network transport, random sampling, multi-GPU execution, or a
production SLA claim.
