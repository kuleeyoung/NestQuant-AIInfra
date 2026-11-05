# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **1**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.797 | 3.797 | 3.797 | 0.00% |
| token_throughput_per_second | 11.391 | 11.391 | 11.391 | 0.00% |
| ttft_p50_ms | 348.463 | 348.463 | 348.463 | 0.00% |
| ttft_p95_ms | 516.546 | 516.546 | 516.546 | 0.00% |
| tpot_p50_ms | 58.812 | 58.812 | 58.812 | 0.00% |
| tpot_p95_ms | 71.056 | 71.056 | 71.056 | 0.00% |
| e2e_p50_ms | 469.792 | 469.792 | 469.792 | 0.00% |
| e2e_p95_ms | 656.913 | 656.913 | 656.913 | 0.00% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
