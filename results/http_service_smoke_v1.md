# Llama-3.2-1B packed W4A4KV4 HTTP open-loop smoke

## Result

On 2026-11-02, the local RTX 5070 service ran with packed W4A4 CUDA MMA,
NQKV4, capacity 2, page size 16 and max cache length 96. Twelve clients arrived
at fixed 100 ms intervals and requested three output tokens through the NDJSON
streaming endpoint.

| metric | result |
|---|---:|
| completed / scheduled | 12 / 12 |
| failed / overloaded | 0 / 0 |
| request throughput | 6.75 request/s |
| token throughput | 20.24 token/s |
| TTFT P50 / P95 / P99 | 429.58 / 550.31 / 561.72 ms |
| TPOT P50 / P95 / P99 | 58.34 / 66.01 / 73.08 ms |
| E2E P50 / P95 / P99 | 546.86 / 665.79 / 675.73 ms |

## Boundary

Timings are observed by the local HTTP client and include tokenizer, socket,
queueing, packed model execution and NDJSON delivery. This is a twelve-request
functional/performance smoke on one machine, not a steady-state capacity claim
or production SLA. The raw per-request timings are stored in
`results/raw/http_service_smoke_v1.json`.
