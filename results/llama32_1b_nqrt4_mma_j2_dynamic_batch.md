# Llama-3.2-1B packed J2 dynamic-batch feasibility

## Runtime behavior

The `cuda_mma_j2` backend now accepts flattened token counts from 1 through 16.
M=1 uses the dedicated one-row kernel. M=16 uses the full Tensor Core tile.
Counts 2-15 are zero-padded to 16, processed by one M16 packed decode, and
cropped back to the valid rows. Thus a partially filled scheduling window does
not repeat packed-weight reconstruction for every request.

This is the model-execution substrate for dynamic batching, not yet a networked
request scheduler. The benchmark uses synchronized fixed batches whose members
advance one decode step together.

## Long decode confirmation at batch 16

With 64 generated tokens, TPOT averages 62 measured cache-enabled transitions.

| Metric | original BF16 | packed J2 | J2 / BF16 |
|---|---:|---:|---:|
| TPOT | 13.063 ms | 44.674 ms | 3.42x slower |
| per-request step rate | 76.55/s | 22.38/s | 0.292x |
| aggregate throughput | 1224.87 token/s | 358.15 token/s | 0.292x |
| warm CUDA allocation | 2.334 GiB | 1.005 GiB | 43.0% |

Raw data: `results/raw/llama32_1b_nqrt4_mma_j2_full_b16_confirm64.json`.

## Partial-batch sweep

The same 112 packed projections are used at every batch size. Each case uses a
one-token prompt and 32 generated tokens; TPOT averages 30 transitions.

| Batch | J2 TPOT | J2 per-request rate | J2 aggregate throughput | BF16 aggregate throughput | J2 warm allocation |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.674 ms | 21.89/s | 21.89 token/s | 92.76 token/s | 1.005 GiB |
| 2 | 45.966 ms | 21.76/s | 43.51 token/s | 171.76 token/s | 1.005 GiB |
| 4 | 45.207 ms | 22.12/s | 88.48 token/s | 396.54 token/s | 1.005 GiB |
| 8 | 47.191 ms | 21.19/s | 169.52 token/s | 643.07 token/s | 1.005 GiB |
| 16 | 45.484 ms | 21.99/s | 351.77 token/s | 1544.90 token/s | 1.005 GiB |

Packed aggregate throughput scales 16.07x from batch 1 to batch 16 while TPOT
stays in a narrow 45.2-47.2 ms range. This confirms that one decoded weight tile
serves all active rows in a batch and that partial batching does not duplicate
decode work. It does not close the per-step gap to resident BF16: the packed
path reaches 22.8% of BF16 aggregate throughput in this sweep.

Raw data: `results/raw/llama32_1b_nqrt4_mma_j2_dynamic_batch_sweep.json`.

## Metric correction

The full-model report now records both `decode_tokens_per_second_per_request`
and `aggregate_decode_tokens_per_second`. The legacy
`decode_tokens_per_second` field remains the per-request step rate for backward
compatibility. Aggregate throughput is `batch_size / TPOT`.

## Current bottleneck and claim boundary

Batching amortizes each packed reconstruction across up to 16 requests and
achieves near-linear scaling, but every model step still executes 112 packed
projections, stateless hashes, two E8 decisions per block, Split-K partial
kernels, and reductions. This is fixed-batch execution evidence on one GPU,
not yet an arrival-aware continuous-batching service. It supports a throughput
scaling claim, not a claim that packed J2 outperforms original BF16 inference.
