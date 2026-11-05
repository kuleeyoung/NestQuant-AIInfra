# Llama-3.2-1B packed W4A4KV4 HTTP capacity sweep

| offered req/s | completed | HTTP 429 | overload rate | completed req/s | token/s | TTFT P50/P95 ms | TPOT P95 ms | E2E P95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24/24 | 0 | 0.0% | 1.02 | 3.05 | 431.47/561.86 | 179.69 | 914.28 |
| 2 | 24/24 | 0 | 0.0% | 1.93 | 5.78 | 690.98/1120.84 | 174.29 | 1452.89 |
| 4 | 21/24 | 3 | 12.5% | 2.34 | 7.01 | 1882.03/3140.63 | 198.21 | 3473.44 |
| 8 | 15/24 | 9 | 37.5% | 2.63 | 7.89 | 2173.43/2591.67 | 203.17 | 2961.16 |

## Decision

The highest lossless offered rate in this finite sweep is 2 request/s. The first point with explicit HTTP 429 backpressure is 4 request/s. Peak completed throughput is 2.63 request/s or 7.89 token/s.

## Boundary

Finite single-host HTTP/1.1 NDJSON offered-load sweep; not a steady-state production SLA.
Each point contains only 24 scheduled requests; the sweep locates a local knee but does not establish a long-duration SLA.

## Reproduction note

This service process measured TPOT P95 around 174-203 ms, whereas the preceding
independent 120-second cross-page soak process measured 60.90 ms with the same
prompt and backend configuration. The within-process offered-load trend remains
valid because all four points share one loaded model and CUDA context, but the
absolute latency is not promoted as a stable cross-process baseline. A future
formal latency claim should record GPU clocks/power/competing load and use paired
or repeated fresh-process trials.
