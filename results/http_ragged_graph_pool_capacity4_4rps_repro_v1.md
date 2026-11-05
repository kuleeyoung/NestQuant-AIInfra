# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **3**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.849 | 3.822 | 3.861 | 0.42% |
| token_throughput_per_second | 11.547 | 11.467 | 11.584 | 0.42% |
| ttft_p50_ms | 512.698 | 487.000 | 525.681 | 3.16% |
| ttft_p95_ms | 636.583 | 634.400 | 646.856 | 0.85% |
| tpot_p50_ms | 67.767 | 67.536 | 82.822 | 9.84% |
| tpot_p95_ms | 227.038 | 222.109 | 232.680 | 1.90% |
| e2e_p50_ms | 729.434 | 713.265 | 736.430 | 1.34% |
| e2e_p95_ms | 983.483 | 969.278 | 1002.399 | 1.38% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
