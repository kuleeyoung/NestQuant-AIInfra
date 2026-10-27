# Llama-3.2-1B J2 M=1 autoregressive decode

## M=1 kernel

The J2 CUDA backend now supports one-token inputs. The m16n8k16 MMA tile places
the real activation in row zero and supplies zero fragments for rows 1-15;
only output row zero is stored. Packed E8 weights are still decoded once, so
the unused MMA rows do not multiply decode work by sixteen. Split-K partials
use `[split, N]` FP32 workspace rather than `[split, 16, N]`.

## Real packed projection records

| Projection | shape N x K | J2 M1 | Triton packed M1 | speedup | resident BF16 | max FP32 error |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 2048 x 2048 | 87.046 us | 330.201 us | 3.79x | 14.086 us | 3.004e-04 |
| gate_proj | 8192 x 2048 | 394.676 us | 1176.42 us | 2.98x | 40.880 us | 4.675e-04 |
| down_proj | 2048 x 8192 | 315.317 us | 1299.99 us | 4.12x | 34.521 us | 2.877e-04 |

Raw data: `results/raw/llama32_1b_nqrt4_mma_j2_checkpoint_m1.json`.

## Decoder block confirmation

All seven packed projections in block 0 use the M1 kernel. In the 50-repeat
confirmation, the packed block median/mean is 2.846/3.239 ms versus
3.677/3.973 ms for the decoded rotated-weight BF16 reference. Median and mean
speedups are 1.292x and 1.227x. Maximum absolute error is 0.001953 and relative
L2 error is 0.1124%. Packed persistent projection bytes remain 73.36% below
the seven BF16 matrices.

This reference includes the same online Hadamard coordinate transformation as
the packed checkpoint and therefore isolates packed compute from materialized
quantized-weight execution. It is not the original unrotated BF16 model path.

Raw data: `results/raw/llama32_1b_nqrt4_mma_j2_block0_m1_confirm50.json`.

## Full-model autoregressive result

All 112 projections across 16 decoder layers were replaced. Batch size and
prompt length are one; TPOT averages 14 measured decode transitions after
separate cache-enabled warm-up calls.

| Metric | original BF16 model | packed J2 M1 |
|---|---:|---:|
| steady one-token prefill median | 9.610 ms | 39.064 ms |
| TTFT | 8.941 ms | 45.631 ms |
| TPOT | 10.783 ms | 39.197 ms |
| decode throughput | 92.74 token/s | 25.51 token/s |
| CUDA allocated after warm-up | 2,506,233,344 B | 1,078,588,928 B |
| packed projection payloads | 0 | 518,457,088 B |

The packed path reduces warm CUDA allocation by 1,427,644,416 bytes, or 56.96%,
but is 3.64x slower in TPOT. Generated token sequences diverge after the second
token, as expected when comparing the quantized checkpoint with the original
BF16 model. First-forward logit relative L2 difference is 14.83%; existing
full WikiText-2 evaluation remains the appropriate quality evidence rather
than this one-token random prompt.

Raw data: `results/raw/llama32_1b_nqrt4_mma_j2_full_m1.json`.

## Interpretation

M1 establishes a functioning packed end-to-end generation path and materially
outperforms the previous packed Triton kernels in isolation. It does not beat
the original resident BF16 model. Full-model execution performs 112 separate
packed projection calls per token, repeatedly regenerates stateless dither and
E8 weights, allocates Split-K workspaces, and launches partial/reduction kernels.
The next optimization must reduce cross-projection or cross-token repeated work
and launch count; further isolated nearest-E8 micro-optimizations are unlikely
to close a 3.64x TPOT gap by themselves.
