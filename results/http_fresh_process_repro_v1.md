# W4A4KV4 HTTP fresh-process reproducibility

- fresh service processes: **3**
- all trials passed: **True**

| metric | median | min | max | CV |
|---|---:|---:|---:|---:|
| request_throughput_per_second | 2.035 | 2.035 | 2.035 | 0.01% |
| token_throughput_per_second | 6.105 | 6.104 | 6.106 | 0.01% |
| ttft_p50_ms | 168.743 | 166.679 | 170.007 | 0.81% |
| ttft_p95_ms | 174.057 | 172.680 | 189.133 | 4.17% |
| tpot_p50_ms | 63.077 | 62.785 | 63.093 | 0.23% |
| tpot_p95_ms | 64.552 | 64.237 | 68.603 | 3.02% |
| e2e_p50_ms | 295.141 | 294.162 | 296.008 | 0.26% |
| e2e_p95_ms | 301.437 | 300.732 | 326.082 | 3.81% |

CV is the coefficient of variation: population standard deviation divided by the mean. Raw benchmark JSON, GPU telemetry and server logs are retained beside the summary.

## GPU state during the measured phase

| trial | samples | P-state | SM clock mean/range (MHz) | power mean (W) | utilization mean | max temperature |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 41 | P1 | 2888.0 / 2880-2895 | 88.7 | 49.2% | 45.3 C |
| 2 | 42 | P1 | 2886.6 / 2880-2895 | 88.6 | 41.7% | 46.4 C |
| 3 | 41 | P1 | 2885.7 / 2880-2895 | 87.9 | 44.1% | 47.1 C |

The three measured phases held the same P-state and nearly identical clocks and
power. With two explicit warmup requests per fresh process, TPOT P95 returned to
64.24-68.60 ms instead of the 174-203 ms observed in the earlier independent
capacity-sweep process. The earlier outlier is therefore not reproducible under
this controlled protocol. This evidence does not identify its historical root
cause; likely variables include missing explicit warmup or unrecorded competing
GPU load, so the older absolute latency remains excluded from the stable baseline.

Claim boundary: Fresh-process, single-host reproducibility experiment. It records GPU state and client-observed HTTP latency, but is not a production SLA.
