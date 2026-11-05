# Capacity-4 compact prefill follow-up

## Change

This follow-up keeps the ragged M1/M2/M4 decode Graph pool and also compacts
prefill rows before the model call. Compact rows carry explicit logical
`slot_ids`; NQKV writes use the general transactional writer, which reserves all
required pages before mutation and cancels reservations on failure. The
fixed-capacity GPU multi-page planner is not reused because it currently assumes
that execution row index equals logical slot index.

## Correctness

- Sparse-row, cross-page, multi-page, and rollback backend tests pass.
- Full regression: `401 passed, 61 subtests passed` on 2026-11-03.
- The external scheduler contract remains fixed-capacity; compact logits are
  scattered back to their logical rows and inactive rows are zero.

## Protocol

The workload is identical to the capacity-4 Graph-pool experiment: the same
Llama-3.2-1B NQRT4 checkpoint, packed W4A4, NQKV4, 29-token prompt, three output
tokens, four offered requests/s, 12 client workers, two warmups, 24 measured
requests, and `min_prefill_batch=2,max_wait_steps=2`. Both columns below use
three fresh service processes.

## Result

| metric | decode-only compact, mean | compact prefill + decode, mean | change |
|---|---:|---:|---:|
| successful requests | 72 / 72 | 72 / 72 | unchanged |
| request throughput | 3.844 req/s | 3.883 req/s | +1.0% |
| TTFT P95 | 639.28 ms | 521.80 ms | -18.4% |
| TPOT P95 | 227.28 ms | 107.91 ms | -52.5% |
| E2E P95 | 985.05 ms | 692.00 ms | -29.7% |

The compact-prefill TPOT P95 values were 187.79, 68.39, and 67.54 ms. Their
mean is 107.91 ms, median is 68.39 ms, and CV is 52.35%. TTFT P95 values were
591.88, 484.68, and 488.83 ms (mean 521.80 ms, CV 9.50%). The first process was
materially slower, so 68 ms must not be presented as stable production TPOT.

The corresponding prefill-compute means were 213.85, 179.17, and 178.15 ms.
This supports compact prefill as the likely source of the gain, while the TPOT
improvement is also affected by reduced GPU queue interference. A profiler run
or a balanced interleaved A/B is required for causal kernel attribution.

## Decision

Keep compact prefill: all correctness gates passed and all three runs improved
over the historical fixed-M4 service. Do not yet optimize its CPU writer loop.
First profile the slow-versus-fast process behavior and record M1/M2/M4 Graph
hits, because eliminating unexplained variance has higher evidentiary value than
optimizing the best run.

This variance gate was subsequently closed by a same-revision interleaved
`fixed` versus `full` policy A/B. All three `full` TPOT P95 values were between
68.50 and 74.82 ms and every pair favored `full`; see
`results/http_policy_interleaved_ab_v1.md`.

## Graph-width observability follow-up

A separate same-load run was used only to verify path selection, not added to
the three-run performance aggregate. It completed 24/24 requests and reported:

| runtime counter | value |
|---|---:|
| M1 Graph replays | 32 |
| M2 Graph replays | 10 |
| M4 Graph replays | 0 |
| compact-prefill calls | 21 |
| compact-prefill active rows | 26 |
| peak registered requests | 3 |

Thus all 42 measured decode replays used M1 or M2. This directly confirms that
the old fixed-M4 runner performed avoidable work for this workload; it does not
prove that M4 is unnecessary under a different arrival pattern.

Raw artifacts:

- `results/raw/http_ragged_graph_pool_capacity4_4rps_repro_v1_summary.json`
- `results/raw/http_compact_prefill_capacity4_4rps_repro_v1_summary.json`
- `results/raw/http_compact_prefill_capacity4_4rps_repro_v1/`
- `results/raw/http_graph_width_observability_v1_summary.json`
