# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **1**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.467 | 3.467 | 3.467 | 0.00% |
| token_throughput_per_second | 10.402 | 10.402 | 10.402 | 0.00% |
| ttft_p50_ms | 242.805 | 242.805 | 242.805 | 0.00% |
| ttft_p95_ms | 295.963 | 295.963 | 295.963 | 0.00% |
| tpot_p50_ms | 50.247 | 50.247 | 50.247 | 0.00% |
| tpot_p95_ms | 50.730 | 50.730 | 50.730 | 0.00% |
| e2e_p50_ms | 343.774 | 343.774 | 343.774 | 0.00% |
| e2e_p95_ms | 397.869 | 397.869 | 397.869 | 0.00% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
