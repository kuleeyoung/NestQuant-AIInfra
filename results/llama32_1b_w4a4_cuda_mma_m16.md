# Llama-3.2-1B direct-packed W4A4 CUDA MMA M16

## Scope

This result evaluates the new `packed_w4a4_cuda_mma` backend on the complete
16-layer Llama-3.2-1B model at batch 1 and sequence length 16. All 112
projections are replaced with packed NQRT modules. M16 projections use the
custom CUDA Tensor Core kernel; subsequent M1 decode steps safely fall back to
the tiled Triton W4A4 kernel.

Raw evidence is stored in the three `..._trial1.json` through
`..._trial3.json` files and the machine-validated aggregate
`results/raw/llama32_1b_w4a4_cuda_mma_m16_trials_summary.json`.

The summary records Python 3.12.3, PyTorch 2.14.1+cu130, CUDA runtime 13.0,
Triton 3.8.0, Transformers 5.20.2, RTX 5070, input seed `20261025`, and the
shared input-token SHA-256. The summarizer rejects protocol or token drift.

## End-to-end latency

Three independent processes, each with seven repeated steady prefill measurements.
The table reports the median of trial medians and the range across trials:

| backend | median prefill | trial range | TTFT median | TPOT median |
|---|---:|---:|---:|---:|
| tiled Triton | 178.93 ms | 178.46–180.33 ms | 180.12 ms | 163.27 ms |
| CUDA MMA M16 | 117.17 ms | 115.59–118.81 ms | 115.77 ms | 164.62 ms |

- steady M16 prefill speedup median: **1.527x** (range 1.518x–1.544x);
- median TTFT reduction: **35.7%**;
- TPOT is unchanged within noise because M1 decode falls back to Triton;
- model/runtime allocated CUDA memory is identical: 1,078,588,928 bytes;
- packed weight bytes are identical: 518,457,088 bytes.

Runtime dispatch counters provide a direct execution audit rather than merely
inferring the selected path from shapes:

| path | calls |
|---|---:|
| custom CUDA MMA M16 | 1,232 per trial |
| tiled Triton fallback (M1 decode) | 448 per trial |

The counts cover one cold prefill, the benchmark warmup/repeats, and the
cache-enabled prefill/decode measurements across all 112 projections.

Cold prefill is not used for the speed claim because it includes backend JIT
compilation and lazy packed-weight preparation.

## Correctness and quality boundary

- The custom activation decoder matches the upstream/Python and Triton packed
  activation exactly for all 4096 blocks in the real q-projection probe.
- All 112 real model projections were independently compared with tiled
  Triton. The largest projection max absolute error was `0.1597900390625`; the
  largest projection mean absolute error was `1.08617e-5`.
- Full-model logits are not numerically equivalent after 16 layers:
  max absolute logit difference `4.96875`, relative L2 difference `0.20639`.
  The generated sequence first diverged on the third reported token.

The last point is expected to be investigated as a quality question rather
than hidden. BF16 Tensor Core accumulation order differs between the custom
K16 MMA loop and Triton's tiled dot. Small per-layer differences can cross a
later A4 quantization boundary and produce different low-bit codes. Therefore:

- the **1.527x median M16 prefill latency result is valid for the measured execution**;
- no end-to-end quality-equivalence claim is made yet;
- the backend remains explicit opt-in and is not the default W4A4 backend;
- the established full-corpus W4A4 PPL claim remains the tiled result
  `10.683118`.

## Projection-wide audit

`scripts/validate_w4a4_mma_projections.py` checks every packed projection with
an independent deterministic M16 activation. Raw output is stored in
`results/raw/w4a4_cuda_mma_all_projections.json`.

| check | result |
|---|---:|
| projections executed | 112 / 112 |
| non-finite failures | 0 |
| largest max absolute error | 0.1597900390625 |
| largest mean absolute error | 1.08617e-5 |

## Verification

- targeted CUDA/backend/dispatch tests: 4 passed;
- formal pytest suite: 305 passed plus 37 subtests in 16.19 seconds;
- unsupported shapes and bias automatically fall back to tiled Triton;
- no model checkpoint or formal tiled-quality artifact was modified.

## Deferred route decision

The next kernel investment should be selected deliberately:

1. build M1 first to accelerate autoregressive decode;
2. build M32/M64/M128 to cover longer prefill tiles;
3. establish a dedicated M16 long-text quality protocol before expanding
   kernel shapes.

This choice affects project direction and is intentionally deferred for the
user rather than selected as an implementation detail.
