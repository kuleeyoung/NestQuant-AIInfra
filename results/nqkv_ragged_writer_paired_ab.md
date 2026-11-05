# NQKV ragged writer same-process paired A/B

Pairs: 20; paired bootstrap resamples: 10000.

| metric | transactional median | ragged median | paired median improvement | bootstrap 95% CI | improved pairs |
|---|---:|---:|---:|---:|---:|
| token throughput | 12.34 | 13.61 | 9.72% | [7.84%, 11.49%] | 20/20 |
| TTFT P95 | 637.93 | 566.01 | 11.56% | [8.91%, 12.56%] | 20/20 |
| TPOT P95 | 188.46 | 169.45 | 9.41% | [7.89%, 11.13%] | 20/20 |
| E2E P95 | 744.17 | 672.89 | 9.28% | [7.71%, 10.86%] | 20/20 |

## Protocol boundary

Both arms run in one process with one loaded model and one CUDA context. Each repetition shares identical requests/tokens; arm order alternates. The experiment excludes tokenizer/network I/O and is not a production SLA.
