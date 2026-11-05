# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **1**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.746 | 3.746 | 3.746 | 0.00% |
| token_throughput_per_second | 11.237 | 11.237 | 11.237 | 0.00% |
| ttft_p50_ms | 510.845 | 510.845 | 510.845 | 0.00% |
| ttft_p95_ms | 669.012 | 669.012 | 669.012 | 0.00% |
| tpot_p50_ms | 70.525 | 70.525 | 70.525 | 0.00% |
| tpot_p95_ms | 250.530 | 250.530 | 250.530 | 0.00% |
| e2e_p50_ms | 715.521 | 715.521 | 715.521 | 0.00% |
| e2e_p95_ms | 1026.560 | 1026.560 | 1026.560 | 0.00% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
