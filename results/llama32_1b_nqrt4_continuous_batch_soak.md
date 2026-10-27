# Llama-3.2-1B NQRT4 continuous-batching 10-minute soak

## Outcome

The packed Llama-3.2-1B continuous-batching engine passed a 600-second real-GPU
soak on one NVIDIA GeForce RTX 5070. The run kept a target backlog of 16 requests
over 8 physical KV-cache slots, with prompt lengths sampled from 4 to 32 tokens
and output lengths sampled from 2 to 8 tokens.

| Metric | Result |
|---|---:|
| requested duration | 600 s |
| elapsed including final drain | 600.511 s |
| submitted / finished requests | 17,264 / 17,264 |
| generated tokens | 86,348 |
| engine steps | 15,834 |
| mean token throughput | 143.79 token/s |
| periodic finite-logit checks | 368 / 368 passed |
| terminal requests retained after drain | 0 |
| peak CUDA allocated memory | 1,203,838,464 bytes |
| sampled CUDA allocated range | 512 bytes |
| sampled CUDA reserved range | 195,035,136 bytes |

The reserved-memory range was caused by allocator growth during the first minute.
After that growth, reserved memory stayed at 1,514,143,744 bytes. CPU RSS grew by
153,145,344 bytes from the first, pre-steady-state sample to the final sample; after
the 60-second sample its total range was 11,505,664 bytes and appeared as bounded
steps followed by plateaus, not per-request linear growth.

All 17,264 terminal requests were removed from the scheduler registry. Three
completed requests were replayed with independent full-sequence greedy forwards.
Their token sequences matched exactly in the formal run, and every selected token
had zero gap from the maximum logit. The latter is the primary check because BF16
can round two candidate logits to an exact tie, for which different argmax execution
paths may legally select different tied indices.

## Reproduction

```bash
python scripts/soak_continuous_batch_engine.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --duration-seconds 600 \
  --sample-interval-seconds 30 \
  --offline \
  --output results/raw/continuous_batch_soak_10min.json
```

Machine-readable samples and correctness details are in
`results/raw/continuous_batch_soak_10min.json`.

## What changed to make long-lived execution safe

`ContinuousBatchScheduler.forget_terminal()` removes finished or cancelled requests
from the long-lived request registry while refusing to remove waiting or running
requests. The soak driver calls it after consuming each completion. This closes the
CPU-state retention path that short functional tests would not expose.

## Evidence boundary

This validates a single-process, host-driven greedy prototype using packed NQRT4
weights, BF16 accumulation, CUDA Graph decode, a fixed capacity of 8, and synthetic
token-ID requests. It does not validate tokenizer or HTTP handling, random sampling,
cancellation under load, multi-GPU execution, production SLA/QPS, or 3B/8B models.
The 10-minute trace found no unbounded request-state or CUDA-memory growth; it is not
a proof that no leak can exist over arbitrarily long production runs.
