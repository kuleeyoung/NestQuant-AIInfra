# Llama-3.2-1B packed W4A4KV4 continuous-batching smoke

## Scope

This is a bounded functional smoke run on one RTX 5070. The model uses all 112
packed W4A4 projections, the batched NQKV v2 paged KV4 cache, and the existing
host-driven continuous-batching scheduler. It does not include a tokenizer,
network server, random sampling, multi-GPU execution, or production SLA.

Configuration: capacity 2, maximum cache length 32, four requests, prompt
length 4--8, output length 2--3, one repetition. One repetition is not stable
performance evidence.

## Results

| Path | Decode execution | token/s | TTFT P95 | TPOT P95 | CUDA peak allocated |
|---|---|---:|---:|---:|---:|
| packed W4A4 + NQKV4, original | eager host-driven | 8.03 | 814.66 ms | 256.60 ms | 1034.93 MiB |
| packed W4A4 + NQKV4, current | normal + boundary CUDA Graph | 9.49 | 722.98 ms | 223.05 ms | 1099.92 MiB |
| packed W4A4 + NQKV4, M2 down split-8 | normal + boundary CUDA Graph | **11.89** | **591.14 ms** | **181.67 ms** | 1099.92 MiB |
| packed W4A4 + BF16 StaticCache | CUDA Graph | 10.45 | 646.80 ms | 203.23 ms | 1066.85 MiB |
| packed W4A4 + BF16 StaticCache, M2 down split-8 | CUDA Graph | **14.17** | **491.90 ms** | **151.75 ms** | 1066.85 MiB |

Both runs completed every request and ended with zero scheduler registrations.
The NQKV run also ended with zero backend cache registrations, proving terminal
page release and slot reuse do not require an extra engine step.

The first cold NQKV run reached only 2.79 token/s because an unseen M=14 shape
triggered JIT compilation inside the measured interval. The table uses the
second run after that kernel was cached. The NQKV dispatch audit recorded 896
small-M CUDA MMA calls, 224 chunked-staged CUDA calls, and zero tiled fallbacks.

The M2 down split-8 row reports the median of three hot repetitions under the
same smoke configuration. The earlier 9.49 token/s row was a single hot run,
so the apparent +25.3% throughput and -18.6% TPOT changes are a stronger
updated measurement, but not a paired alternating A/B confidence interval.

## Interpretation

The integrated service path now functionally realizes W4A4KV4 and captures two
fixed-capacity full-model graphs: a normal within-page decode Graph and a
boundary Graph that performs E8 multi-beta encoding, bit packing, fixed-pool
publication, page-table updates, attention and the remaining Llama forward.
The current NQKV path improves overall smoke throughput by about 18.1% over its
eager predecessor. Decode-only steps fell from approximately 126--127 ms to
80.8--83.8 ms. Mixed prefill/decode steps still dominate this tiny workload, so
total throughput remains about 9.2% below the BF16-KV Graph baseline.

After profiling identified the `M=2, K=8192, N=2048` down projection as the
largest Graph component, a deterministic split-8 kernel reduced that projection
by about 3.96x across the two profiled Graphs. The three-run service median is
11.89 token/s with 181.67 ms TPOT P95. A fresh three-run BF16-KV baseline using
the same split-K weight path reaches 14.17 token/s and 151.75 ms TPOT P95.
NQKV4 therefore remains 16.1% lower in token throughput and 19.7% higher in
TPOT on this small workload; the older 10.45 token/s value is retained only as
historical evidence.

The host still reserves a physical page ID before boundary replay and commits
the generation-checked handle afterward; it no longer executes the model or KV
encoding at the boundary. A real 1B prompt-15/output-3 boundary smoke completed
two requests, observed exactly one boundary Graph replay and zero eager
fallbacks, and returned both scheduler and backend registrations to zero. The
boundary step was 88.2 ms and the following normal Graph step was 86.2 ms,
removing the former 244.2 ms eager transition spike. The second Graph raises
peak allocated memory by roughly 32.5 MiB relative to the single-Graph version.

The 31.92 MiB peak-allocation difference is not a general KV compression ratio:
the context and capacity are tiny, while the BF16 baseline includes CUDA Graph
private allocations. Long-context paired measurements are required before a
memory-saving service claim.

## Evidence

- NQKV raw run: `raw/llama32_1b_w4a4kv4_service_smoke_hot.json`
- NQKV Graph run: `raw/llama32_1b_w4a4kv4_service_graph_smoke.json`
- NQKV Graph boundary run: `raw/llama32_1b_w4a4kv4_service_graph_boundary_smoke.json`
- NQKV dual-Graph run: `raw/llama32_1b_w4a4kv4_service_dual_graph_smoke.json`
- NQKV dual-Graph boundary run: `raw/llama32_1b_w4a4kv4_service_dual_graph_boundary_smoke.json`
- NQKV M2 split-K three-run smoke: `raw/llama32_1b_w4a4kv4_service_splitk_smoke.json`
- BF16-KV M2 split-K three-run smoke: `raw/llama32_1b_w4a4_bf16kv_service_splitk_smoke.json`
- M2 split-K kernel and Graph report: `w4a4_m2_down_splitk.md`
- BF16-KV raw run: `raw/llama32_1b_w4a4_bf16kv_service_smoke.json`
- Smoke config: `../configs/service_benchmark_nqkv_smoke.v1.json`
