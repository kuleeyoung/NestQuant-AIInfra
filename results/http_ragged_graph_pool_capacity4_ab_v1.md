# Capacity-4 ragged CUDA Graph pool A/B

## Change

The logical NQKV cache capacity remains four slots, but decode no longer has to
execute every request as an `M=4` model batch. The runner compacts active rows
and selects the smallest captured `M=1`, `M=2`, or `M=4` CUDA Graph. Each Graph
owns stable `slot_ids`, `active_mask`, input buffers, and a matching K1 attention
workspace. Its output is scattered back to the original logical slots, so the
continuous-batching scheduler contract does not change. Prefill is still the
fixed-capacity path in this revision.

## Correctness gates

- Compact execution rows can address non-contiguous logical KV slots.
- Active slot mappings must be unique; invalid aliases fail closed.
- M1/M2 active logits match the eager fixed-capacity oracle exactly.
- Inactive rows are not computed and are returned as zero logits.
- Ordinary and page-boundary Graphs pass the complete backend regression.
- Full project regression: `401 passed, 61 subtests passed` on 2026-11-03.

## Service protocol

Both sides use the same Llama-3.2-1B NQRT4 checkpoint, capacity four, packed
W4A4 compute, NQKV4 cache, 29-token cross-page prompt, three output tokens,
four offered requests/s, 12 client workers, two warmup requests, and
`min_prefill_batch=2,max_wait_steps=2`. The historical fixed-M4 side is one
fresh-process run. The ragged Graph-pool side is reported as the mean of three
fresh-process runs; therefore this is strong directional evidence, but not a
balanced multi-trial causal A/B or a production SLA.

## Result

| metric | fixed M4, 1 run | ragged M1/M2/M4, 3-run mean | change |
|---|---:|---:|---:|
| completed / failed / overloaded | 24 / 0 / 0 | 72 / 0 / 0 | all passed |
| request throughput | 3.076 req/s | 3.844 req/s | +25.0% |
| TTFT P95 | 1640.80 ms | 639.28 ms | -61.0% |
| TPOT P95 | 331.84 ms | 227.28 ms | -31.5% |
| E2E P95 | 2125.13 ms | 985.05 ms | -53.6% |

The three ragged-pool TPOT P95 values are 222.11, 232.68, and 227.04 ms
(CV 1.90%). TTFT P95 values are 634.40, 646.86, and 636.58 ms (CV 0.85%).

## Attribution and next gate

The compact decode architecture removes useless packed Linear and attention
work when fewer than four slots are active. The first ragged trial also reduced
mean command-queue time from 221.90 to 106.13 ms and mean prefill compute from
395.13 to 292.10 ms. Prefill code itself was not compacted, so the latter can be
an indirect scheduling/load effect and must not be described as a prefill-kernel
speedup.

Compact/ragged prefill with the same logical slot mapping was implemented in the
next revision; see `results/http_compact_prefill_capacity4_ab_v1.md`. A balanced
fixed-M4 versus ragged-pool A/B and Nsight attribution are still required before
changing the packed W4A4 kernels.

Raw artifacts:

- `results/raw/http_capacity4_4rps_probe_v1_summary.json`
- `results/raw/http_ragged_graph_pool_capacity4_4rps_repro_v1_summary.json`
- `results/raw/http_ragged_graph_pool_capacity4_4rps_repro_v1/`
