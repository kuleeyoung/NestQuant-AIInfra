# Llama-3.2-1B grouped packed QKV kernel

## Scope

This experiment compares three independent packed fused Linear launches with
one grouped Triton launch for the real block-0 Q/K/V records. The already
shared Hadamard-transformed input is identical for both paths, so this isolates
kernel organization rather than input-transform work.

Real weight shapes are:

- Q: 2048 x 2048;
- K: 512 x 2048;
- V: 512 x 2048.

Q, K, and V retain separate code streams, beta streams, row norms, beta tables,
and stateless-dither seeds. In particular, their real seeds are different, so
the grouped kernel selects the seed by projection while keeping counters local
to each record. Treating concatenated QKV rows as one dither coordinate space
would reconstruct different weights and is intentionally rejected by design.

## Correctness and latency

Every path is warmed and measured 50 times with CUDA synchronization.

| Sequence length | Three independent launches | One grouped launch | Median speedup | Q/K/V exact |
|---:|---:|---:|---:|---|
| 1 | 0.8315 ms | 0.7417 ms | 1.121x | true / true / true |
| 8 | 0.8272 ms | 0.7584 ms | 1.091x | true / true / true |
| 16 | 0.8095 ms | 0.7599 ms | 1.065x | true / true / true |

All three grouped outputs are bitwise equal to their independent fused-kernel
counterparts. The 6.5-12.1% latency reduction confirms that consolidating the
launch grid is useful, especially at token length 1 where fixed launch overhead
is a larger fraction of work.

## Claim boundary

This result proves that three real packed NQRT projections can share one Triton
launch without changing their per-record quantization semantics. It does not
yet include rotary embeddings, attention, output projection, MLP, or a complete
decoder block. It also remains a scalar/FP32-accumulation kernel rather than a
Tensor Core implementation.

Machine-readable measurements:
`results/raw/llama32_1b_nqrt4_grouped_qkv.json`.

Reproduction entry point: `scripts/benchmark_triton_grouped_qkv.py`.
