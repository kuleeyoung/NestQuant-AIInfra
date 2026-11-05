# Llama-3.2-1B packed W4A4KV4 cross-page HTTP soak

## Protocol

- 120-second single-host HTTP/1.1 NDJSON run at 2 offered request/s.
- 240 scheduled requests, three output tokens per request.
- The fixed prompt is 29 tokenizer tokens, so every request crosses the
  16-token NQKV page boundary and owns at least one packed cold page per layer.
- `--require-nqkv-pages` makes the run fail unless the worker-observed cold-page
  allocation high-water mark is greater than zero.

## Result

| metric | result |
|---|---:|
| scheduled / completed | **240 / 240** |
| failed / timed out / HTTP 429 | **0 / 0 / 0** |
| completed request throughput | 2.004 request/s |
| completed token throughput | 6.011 token/s |
| TTFT P50 / P95 / P99 / max | 174.01 / 195.32 / 1,526.29 / 2,280.21 ms |
| TPOT P50 / P95 / P99 / max | 56.28 / 60.90 / 66.23 / 92.83 ms |
| E2E P50 / P95 / P99 / max | 287.79 / 319.10 / 1,643.69 / 2,429.73 ms |

The high TTFT/E2E P99 records a small number of startup or transient tail
events and is intentionally retained rather than removed from the report.

## Page lifecycle evidence

| counter | value |
|---|---:|
| registered request high-water mark | 5 |
| NQKV allocated-page high-water mark | **32** |
| final NQKV allocated pages | **0** |
| final NQKV free pages | **192** |
| final inflight / registered / waiting / running | **0 / 0 / 0 / 0** |

There are 16 transformer layers and service capacity is two, so an observed
32-page peak is exactly consistent with two concurrent requests each owning one
cold page in every layer. All pages returned to the pool after drain.

After 60 seconds, CPU RSS varied by 69,632 bytes; CUDA allocated and reserved
memory ranges were both zero. The final process state was clean, and the server
exited with code zero.

## Claim boundary

This is a 120-second, 240-request cross-page lifecycle soak at one low offered
load. It proves repeated packed-page allocation, use and release without leaked
page ownership in this protocol. It is not a saturated cross-page capacity
sweep, a ten-minute cross-page soak, or a production SLA. Raw samples are in
`results/raw/http_cross_page_soak_120s_v1.json`.
