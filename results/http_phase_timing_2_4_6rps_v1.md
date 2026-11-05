# W4A4KV4 HTTP server-side TTFT phase decomposition

## Additive mean decomposition

| offered req/s | client TTFT mean | command queue | scheduler waiting | prefill compute | post-prefill | server TTFT mean | queue share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 188.19 | 0.35 | 0.01 | 184.66 | 0.06 | 185.08 | 0.2% |
| 4 | 303.65 | 58.89 | 76.89 | 165.51 | 0.05 | 301.34 | 45.1% |
| 6 | 298.46 | 53.41 | 75.93 | 166.64 | 0.05 | 296.03 | 43.7% |

## Tail decomposition

| offered req/s | client TTFT P95 | server TTFT P95 | command queue P95 | scheduler waiting P95 | prefill P95 | client-server P95 gap |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 194.60 | 190.90 | 0.42 | 0.01 | 189.79 | 3.70 |
| 4 | 439.20 | 436.88 | 146.69 | 126.97 | 170.50 | 2.32 |
| 6 | 427.44 | 424.59 | 131.36 | 130.35 | 170.66 | 2.85 |

Mean phases are additive for attribution. P95 columns describe each phase's own tail and are not added because their P95 values can come from different requests.

Claim boundary: Worker-owned monotonic timestamps for one finite single-host HTTP run per offered-load point; not a production SLA or a causal scheduler A/B.
