# W4A4KV4 HTTP admission-policy A/B

Baseline is `min_prefill_batch=2, max_wait_steps=2`; candidate is immediate admission with `min_prefill_batch=1, max_wait_steps=0`.

| metric | baseline median | immediate median | relative change | immediate improved trials |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.88 | 3.76 | -3.1% | 0/3 |
| ttft_p95_ms | 439.55 | 488.53 | +11.1% | 0/3 |
| tpot_p95_ms | 66.74 | 148.96 | +123.2% | 0/3 |
| e2e_p95_ms | 572.57 | 764.19 | +33.5% | 0/3 |
| command_queue_mean_ms | 57.29 | 93.49 | +63.2% | 0/3 |
| waiting_queue_mean_ms | 83.28 | 36.54 | -56.1% | 3/3 |
| prefill_compute_mean_ms | 166.68 | 164.96 | -1.0% | 3/3 |

## Decision

Retain the baseline admission policy. Immediate admission reduces scheduler waiting but worsens TPOT, throughput and end-to-end latency.

Immediate admission moves time out of the scheduler waiting phase, but it creates smaller and more frequent prefill work. The worker remains busy longer, command ingestion is delayed, and decode batching efficiency deteriorates.

Claim boundary: Three sequential fresh-process trials per policy. Trial indices are compared for readability but were not interleaved paired experiments.

## Rejected intermediate probe

Keeping `min_prefill_batch=2` and changing only `max_wait_steps=2 -> 1` was also
tested once before expanding repetitions. It completed 24/24 requests, but achieved
only 3.44 request/s with TTFT/TPOT/E2E P95 of 1055.31/147.25/1306.25 ms. The direction
was decisively worse than the repeated baseline, so this probe was not promoted to a
repeated policy result. Its raw artifacts remain under
`results/raw/http_admission_wait1_4rps_probe_v1`.
