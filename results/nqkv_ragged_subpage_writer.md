# GPU ragged sub-page NQKV writer

## Outcome

The NQKV prefill path now uses a Triton multi-token writer when each row has a
contiguous-prefix mask and every active slot remains strictly below the packed
page boundary. Mask validation and row counts happen once per model forward;
all Transformer layers reuse the result and launch fixed-address GPU writes.
Page-filling, cross-page and non-prefix inputs retain the transactional writer.

## Kernel contract

Input layout is `[batch, tokens, kv_heads, head_dim]` plus fixed `slot_ids` and
a Boolean `[batch, tokens]` write mask. The data kernel maps each active
`(batch row, token, KV head)` to `hot_length + token`; the metadata kernel sums
each row mask and updates hot/sequence lengths once. The low-level launch is
CUDA-Graph-safe, while the checked API rejects duplicate slots, non-prefix
masks, and writes that would fill or cross a page.

## One-layer probe

RTX 5070, shape `[2, 15, 8, 64]`, row counts `[15, 8]`, 10 warmups and 100
host-wall repetitions:

| path | median latency | speedup vs transactional |
|---|---:|---:|
| transactional writer | 315.34 us | 1.00x |
| checked GPU writer | 183.40 us | 1.72x |
| prevalidated GPU-only launch | 42.01 us | 7.51x |

The default backend uses the third structure after one forward-level
validation, while advancing each layer's host length mirror after launch.

## Service smoke

The existing four-request, capacity-2 token-ID smoke was repeated five times.
All requests completed with zero boundary fallback and zero retained scheduler
or backend-cache registrations.

| metric | previous three-run median | ragged-writer five-run median | change |
|---|---:|---:|---:|
| token throughput | 11.89 token/s | 12.41 token/s | +4.35% |
| TPOT P95 | 181.67 ms | 171.02 ms | -5.86% |
| TTFT P95 | 591.14 ms | 607.58 ms | +2.78% |

One retained run suffered an external scheduling stall: 3.98 token/s and
1.55-second TPOT P95, while its recorded GPU step classes remained near the
other runs and no fallback/leak occurred. Therefore medians are reported and
the outlier is not deleted. Compared with the earlier same-code BF16-KV median
of 14.17 token/s, the NQKV4 throughput deficit is now about 12.42%; these runs
were not alternated in one process and are not a confidence interval.

## Dispatch audit

One independent smoke recorded:

- 16 dense-prefill layer launches: one full 16-layer forward;
- 64 ragged-subpage layer launches: four partial-capacity 16-layer forwards;
- zero boundary fallback;
- zero terminal request/cache retention.

This proves the promoted path is exercised by the continuous-batching workload,
not merely present in source code.

## Same-process paired A/B

The writer was then evaluated in one loaded model and one CUDA context. Each
of 20 repetitions used identical requests/tokens for both arms; execution order
alternated by repetition. The transactional arm disabled only the ragged
sub-page dispatch, while dense prefill, W4A4, attention and Graph behavior were
unchanged. Ten thousand paired bootstrap resamples estimate the confidence
intervals.

| metric | transactional median | ragged median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 12.34 token/s | 13.61 token/s | +9.72% | [+7.84%, +11.49%] | 20/20 |
| TTFT P95 | 637.93 ms | 566.01 ms | -11.56% | [-12.56%, -8.91%] | 20/20 |
| TPOT P95 | 188.46 ms | 169.45 ms | -9.41% | [-11.13%, -7.89%] | 20/20 |
| E2E P95 | 744.17 ms | 672.89 ms | -9.28% | [-10.86%, -7.71%] | 20/20 |

All 20 ragged arms dispatched the new writer and all 20 transactional arms
reported zero ragged launches. Every confidence interval excludes zero, so the
path clears the predeclared 5% continuation threshold.

## Correctness and evidence

- Ragged `[2, 1]` rows match the transactional writer elementwise.
- Non-prefix masks and page-filling writes hard-fail before launch.
- NQKV regression: 67 tests plus 16 parameterized subtests passed.
- Full project: 352 tests plus 61 parameterized subtests passed.
- Probe: `results/raw/nqkv_ragged_writer_probe.json`.
- Five-run service: `results/raw/llama32_1b_w4a4kv4_service_ragged_writer_smoke.json`.
- Dispatch audit: `results/raw/llama32_1b_w4a4kv4_ragged_dispatch_audit.json`.
- Paired raw runs: `results/raw/nqkv_ragged_writer_paired_ab.json`.
- Paired summary: `results/raw/nqkv_ragged_writer_paired_ab_summary.json`.
- Paired report: `results/nqkv_ragged_writer_paired_ab.md`.

## Claim boundary

This path does not allocate or pack pages on GPU and intentionally refuses a
write that reaches the page boundary. It is a sub-page hot-tail optimization,
not a general multi-page GPU allocator, network-service SLA, or proof that all
arrival distributions gain the same throughput.
