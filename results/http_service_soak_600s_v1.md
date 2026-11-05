# Llama-3.2-1B packed W4A4KV4 HTTP 600-second soak

## Protocol

- Single local RTX 5070 service process.
- Packed W4A4 CUDA MMA with NQKV4, capacity 2 and max-inflight 8.
- HTTP/1.1 NDJSON clients at 8 offered request/s for 600 seconds.
- Fixed prompt `The capital of France is`; three output tokens per request.
- Runtime resources sampled every five seconds; stability ranges exclude the
  first 60 seconds.

## Result

| metric | result |
|---|---:|
| wall time | 600.17 s |
| scheduled / completed | 4,800 / 4,792 |
| HTTP 429 | 8 (0.167%) |
| failed / timed out | 0 / 0 |
| completion rate | 99.833% |
| completed request throughput | 7.984 request/s |
| completed token throughput | 23.953 token/s |
| TTFT P50 / P95 / P99 | 289.73 / 765.23 / 862.87 ms |
| TPOT P50 / P95 / P99 | 60.26 / 65.98 / 69.27 ms |
| E2E P50 / P95 / P99 | 409.09 / 892.65 / 992.16 ms |

## Resource stability after 60 seconds

| resource | minimum | maximum | range | final |
|---|---:|---:|---:|---:|
| CPU RSS | 2,621,083,648 B | 2,621,407,232 B | **323,584 B** | 2,621,276,160 B |
| CUDA allocated | 1,151,823,360 B | 1,151,823,360 B | **0 B** | 1,151,823,360 B |
| CUDA reserved | 1,182,793,728 B | 1,182,793,728 B | **0 B** | 1,182,793,728 B |
| registered requests | 0 | 7 | 7 | **0** |
| NQKV allocated cold pages | 0 | 0 | 0 | **0** |
| NQKV free cold pages | 192 | 192 | 0 | **192** |

Final service state was inflight=0, registered=0, waiting=0, running=0 and
allocated cold pages=0. The process exited cleanly after the report was saved.

## Claim boundary

This run establishes a ten-minute single-host HTTP lifecycle and hot-tail
resource-stability result. The short prompt plus three generated tokens did not
fill a 16-token NQKV page, so the run did not repeatedly allocate cold packed
pages. It must not be cited as a cross-page NQKV page-pool soak or a production
SLA. The machine-readable samples are in
`results/raw/http_service_soak_600s_v1.json`.
