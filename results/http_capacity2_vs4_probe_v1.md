# W4A4KV4 fixed-capacity 2 versus 4 probe

## Protocol

Both runs use the same Llama-3.2-1B NQRT4 checkpoint, packed W4A4 compute,
NQKV4 cache, 29-token cross-page prompt, three output tokens, 4 offered
request/s, 12 client workers, two explicit warmup requests and the default
`min_prefill_batch=2,max_wait_steps=2` admission policy. The only intended
service configuration change is fixed execution capacity 2 versus 4.

## Result

| metric | capacity 2 | capacity 4 | capacity 4 / capacity 2 |
|---|---:|---:|---:|
| completed / failed / overloaded | 24 / 0 / 0 | 24 / 0 / 0 | - |
| request throughput | 3.90 req/s | 3.08 req/s | 0.79x |
| TTFT P95 | 439.20 ms | 1640.80 ms | 3.74x |
| TPOT P95 | 65.53 ms | 331.84 ms | 5.06x |
| E2E P95 | 568.97 ms | 2125.13 ms | 3.74x |
| command-queue mean | 58.89 ms | 221.90 ms | 3.77x |
| scheduler-waiting mean | 76.89 ms | 332.68 ms | 4.33x |
| prefill-compute mean | 165.51 ms | 395.13 ms | 2.39x |
| benchmark GPU utilization mean | 83.6% | 87.9% | 1.05x |
| benchmark GPU power mean | 116.4 W | 104.8 W | 0.90x |
| observed VRAM maximum | 2552 MiB | 2751 MiB | +199 MiB |

## Decision

Do not promote fixed capacity 4. The current runner requires every model input
batch to equal cache capacity. `active_mask` preserves KV semantics for inactive
slots, but the packed Linear stack still executes the fixed M4 shape. Higher GPU
utilization therefore represents more fixed-width work rather than more useful
completed requests.

The result motivates either a ragged multi-capacity Graph Pool that selects a
compact M1/M2/M4 execution shape while preserving stable slot/page ownership, or
substantial M4 kernel optimization. It does not show that continuous batching in
general is harmful, and one run per capacity is insufficient for a production
capacity claim. The regression is large enough to reject expanding this probe
without first changing the execution architecture.

Raw artifacts are retained under `results/raw/http_phase_4rps_v1` and
`results/raw/http_capacity4_4rps_probe_v1`.
