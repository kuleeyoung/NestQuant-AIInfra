# NQKV batched page encoding isolated paired A/B

Pairs: 20; paired bootstrap resamples: 10000.

| metric | per-slot pack median | batched pack median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 13.03 | 15.94 | 25.12% | [21.09%, 27.66%] | 20/20 |
| TTFT P95 | 648.16 | 505.44 | 22.14% | [19.16%, 24.19%] | 20/20 |
| TPOT P95 | 184.29 | 147.87 | 21.62% | [19.10%, 22.70%] | 20/20 |
| E2E P95 | 702.61 | 549.40 | 20.89% | [18.08%, 22.81%] | 20/20 |

## Protocol boundary

Both arms run in one process with one loaded model and one CUDA context. Each repetition shares identical requests/tokens; arm order alternates. The experiment excludes tokenizer/network I/O and is not a production SLA.

## Decision

The isolated batched page encoder passes the predeclared 5% promotion gate.
Relative to per-slot encoding inside the same segmented four-page path, median
token throughput improves by 25.12%, TTFT P95 falls by 22.14%, TPOT P95 falls by
21.62%, and E2E P95 falls by 20.89%. All 20 paired runs improve on all four
metrics and every paired bootstrap interval excludes zero.

The batched encoder is therefore the default encoder for general multi-page
prefill. The per-slot implementation remains selectable as a reference and
fallback. These measurements cover the 1B token-ID engine with capacity 2 and
prompt length 64; they are not network-service SLA or cross-model claims.
