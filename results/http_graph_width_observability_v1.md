# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **1**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.833 | 3.833 | 3.833 | 0.00% |
| token_throughput_per_second | 11.499 | 11.499 | 11.499 | 0.00% |
| ttft_p50_ms | 376.142 | 376.142 | 376.142 | 0.00% |
| ttft_p95_ms | 546.542 | 546.542 | 546.542 | 0.00% |
| tpot_p50_ms | 64.137 | 64.137 | 64.137 | 0.00% |
| tpot_p95_ms | 75.117 | 75.117 | 75.117 | 0.00% |
| e2e_p50_ms | 501.673 | 501.673 | 501.673 | 0.00% |
| e2e_p95_ms | 697.527 | 697.527 | 697.527 | 0.00% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
