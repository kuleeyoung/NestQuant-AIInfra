# Llama-3.2-1B packed W4A4KV4 HTTP capacity sweep

| offered req/s | completed | HTTP 429 | overload rate | completed req/s | token/s | TTFT P50/P95 ms | TPOT P95 ms | E2E P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 24/24 | 0 | 0.0% | 2.03 | 6.10 | 166.34/174.02 | 63.99 | 299.73 |
| 4 | 24/24 | 0 | 0.0% | 3.90 | 11.70 | 287.72/434.33 | 64.87 | 562.59 |
| 6 | 24/24 | 0 | 0.0% | 5.66 | 16.99 | 389.60/552.27 | 83.26 | 708.34 |
| 8 | 23/24 | 1 | 4.2% | 5.96 | 17.87 | 795.89/980.17 | 84.79 | 1109.14 |
| 10 | 21/24 | 3 | 12.5% | 6.36 | 19.09 | 761.11/1070.04 | 67.38 | 1201.93 |
| 12 | 19/24 | 5 | 20.8% | 6.29 | 18.88 | 845.10/1056.39 | 65.03 | 1185.93 |

## Decision

The highest lossless offered rate in this finite sweep is 6 request/s. The first point with explicit HTTP 429 backpressure is 8 request/s. Peak completed throughput is 6.36 request/s or 19.09 token/s.

## Boundary

Finite single-host HTTP/1.1 NDJSON offered-load sweep; not a steady-state production SLA.
Each point contains only 24 scheduled requests; the sweep locates a local knee but does not establish a long-duration SLA.
