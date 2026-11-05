# Llama-3.2-1B packed W4A4KV4 HTTP capacity sweep

| offered req/s | completed | HTTP 429 | overload rate | completed req/s | token/s | TTFT P50/P95 ms | TPOT P95 ms | E2E P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 24/24 | 0 | 0.0% | 2.05 | 6.14 | 110.50/114.93 | 56.45 | 226.39 |
| 4 | 24/24 | 0 | 0.0% | 4.02 | 12.05 | 108.92/110.46 | 56.81 | 223.86 |
| 8 | 24/24 | 0 | 0.0% | 7.71 | 23.14 | 205.22/301.53 | 59.33 | 415.82 |
| 16 | 19/24 | 5 | 20.8% | 8.46 | 25.39 | 606.49/771.16 | 60.14 | 886.96 |
| 32 | 13/24 | 11 | 45.8% | 8.24 | 24.72 | 623.84/778.19 | 58.92 | 894.50 |

## Decision

The highest lossless offered rate in this finite sweep is 8 request/s. The first point with explicit HTTP 429 backpressure is 16 request/s. Peak completed throughput is 8.46 request/s or 25.39 token/s.

## Boundary

Finite five-point single-host HTTP/1.1 NDJSON capacity sweep; not a steady-state production SLA.
Each point contains only 24 scheduled requests; the sweep locates a local knee but does not establish a long-duration SLA.
