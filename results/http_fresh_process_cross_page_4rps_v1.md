# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **3**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.861 | 3.854 | 3.906 | 0.60% |
| token_throughput_per_second | 11.582 | 11.563 | 11.719 | 0.60% |
| ttft_p50_ms | 312.516 | 295.910 | 318.054 | 3.05% |
| ttft_p95_ms | 452.593 | 431.496 | 455.804 | 2.41% |
| tpot_p50_ms | 63.713 | 63.389 | 64.186 | 0.51% |
| tpot_p95_ms | 66.319 | 64.856 | 66.814 | 1.26% |
| e2e_p50_ms | 439.885 | 422.544 | 445.866 | 2.27% |
| e2e_p95_ms | 582.608 | 562.273 | 585.144 | 1.78% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
