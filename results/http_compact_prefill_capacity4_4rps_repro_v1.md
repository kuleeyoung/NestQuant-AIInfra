# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **3**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 3.898 | 3.831 | 3.919 | 0.97% |
| token_throughput_per_second | 11.694 | 11.492 | 11.757 | 0.97% |
| ttft_p50_ms | 321.275 | 315.294 | 356.747 | 5.53% |
| ttft_p95_ms | 488.833 | 484.677 | 591.881 | 9.50% |
| tpot_p50_ms | 58.092 | 57.854 | 67.519 | 7.36% |
| tpot_p95_ms | 68.390 | 67.537 | 187.789 | 52.35% |
| e2e_p50_ms | 438.265 | 431.540 | 495.021 | 6.26% |
| e2e_p95_ms | 621.983 | 617.947 | 836.069 | 14.72% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
