# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **1**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 2.120 | 2.120 | 2.120 | 0.00% |
| token_throughput_per_second | 6.359 | 6.359 | 6.359 | 0.00% |
| ttft_p50_ms | 173.843 | 173.843 | 173.843 | 0.00% |
| ttft_p95_ms | 176.309 | 176.309 | 176.309 | 0.00% |
| tpot_p50_ms | 52.707 | 52.707 | 52.707 | 0.00% |
| tpot_p95_ms | 53.605 | 53.605 | 53.605 | 0.00% |
| e2e_p50_ms | 279.774 | 279.774 | 279.774 | 0.00% |
| e2e_p95_ms | 282.371 | 282.371 | 282.371 | 0.00% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
