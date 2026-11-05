# NQKV four-page batched-pack same-process paired A/B

Pairs: 20; paired bootstrap resamples: 10000.

| metric | transactional median | segmented batched-pack median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 12.08 | 14.58 | 24.24% | [18.86%, 29.62%] | 19/20 |
| TTFT P95 | 683.46 | 545.72 | 21.33% | [17.01%, 25.50%] | 19/20 |
| TPOT P95 | 198.95 | 153.53 | 19.69% | [14.56%, 24.43%] | 20/20 |
| E2E P95 | 734.56 | 597.02 | 20.28% | [16.52%, 23.85%] | 19/20 |

## Protocol boundary

Both arms run in one process with one loaded model and one CUDA context. Each repetition shares identical requests/tokens; arm order alternates. The experiment excludes tokenizer/network I/O and is not a production SLA.
