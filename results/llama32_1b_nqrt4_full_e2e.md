# Llama-3.2-1B NQRT full-model end-to-end benchmark

## Scope and protocol

This report extends the real block-0 Tensor Core packed prototype to all 16
decoder layers of the fixed `meta-llama/Llama-3.2-1B` revision
`4e20de362430cd3b72f300e6b0f18e50e7166e08`. All 112 attention/MLP projection
weights are replaced by `FusedNQRTLinear`; embedding, RMSNorm, rotary
embedding, attention, KV cache, and `lm_head` remain the base BF16 model.

The benchmark uses batch size 1, the same randomly generated token IDs for all
three paths at each sequence length, inference mode, CUDA synchronization around
each timed region, 20 steady-state repetitions, and eight generated tokens.
`TTFT` (Time To First Token) is measured as one warmed prefill forward with KV
cache enabled. `TPOT` (Time Per Output Token) is the mean of the following
single-token decode forwards using the returned KV cache after one untimed
decode warmup. The first packed
forward also records cold preparation, including reading 112 NQMC records and
copying packed streams to CUDA.

The reproducible entry point is
`scripts/benchmark_full_nqrt_model.py`; raw outputs are retained at
`results/raw/llama32_1b_full_e2e_seq1_128_warmdecode.json` and
`results/raw/llama32_1b_full_e2e_seq128_512_warmdecode.json`.

## Results

| Seq | Path | Steady prefill | TTFT | TPOT | Decode tok/s | Model/runtime allocated | Packed projection streams | Logit relative L2 error |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | BF16 resident | 10.380 ms | 9.671 ms | 12.258 ms | 81.58 | 2,390.1 MB | — | 0 |
| 1 | Scalar packed | 116.472 ms | 111.880 ms | 110.330 ms | 9.06 | 1,028.6 MB | 494.4 MB | 15.51% |
| 1 | Tensor Core packed | 65.946 ms | 66.553 ms | 65.310 ms | 15.31 | 1,028.6 MB | 494.4 MB | 14.47% |
| 128 | BF16 resident | 12.849 ms | 10.746 ms | 12.309 ms | 81.24 | 2,390.1 MB | — | 0 |
| 128 | Scalar packed | 509.092 ms | 497.826 ms | 96.220 ms | 10.39 | 1,028.6 MB | 494.4 MB | 12.17% |
| 128 | Tensor Core packed | 64.465 ms | 65.263 ms | 59.002 ms | 16.95 | 1,028.6 MB | 494.4 MB | 11.85% |
| 512 | BF16 resident | 26.206 ms | 26.593 ms | 12.840 ms | 77.88 | 2,390.1 MB | — | 0 |
| 512 | Scalar packed | 1,849.901 ms | 1,852.016 ms | 95.044 ms | 10.52 | 1,028.6 MB | 494.4 MB | 11.96% |
| 512 | Tensor Core packed | 172.981 ms | 172.354 ms | 56.749 ms | 17.62 | 1,028.6 MB | 494.4 MB | 11.87% |

The Tensor Core packed path accelerates the scalar packed path by 1.77x, 7.90x,
and 10.69x at sequence lengths 1, 128, and 512 for steady prefill. It also
reduces packed TPOT from 110.330 to 65.310 ms at length 1, from 96.220 to
59.002 ms at length 128, and from 95.044 to 56.749 ms at length 512. The large
long-context prefill gain comes from the measured M-tile reuse and Tensor Core
dot path.

The packed path does not yet beat the resident BF16 path: Tensor Core packed
prefill is 6.35x/5.02x/6.60x slower at lengths 1/128/512, and TPOT is
5.33x/4.79x/4.42x slower. This gap is expected from per-tile NQRT unpacking, dither replay, two
E8 nearest-point projections, and kernel launches; the BF16 path starts with
all weights already decoded and uses the highly optimized dense-model path.

The full model still gains a substantial memory reduction. Model/runtime CUDA
allocation after warmup falls from 2,390.1 MB to 1,028.6 MB, a 57.0% reduction
for the complete model process under this protocol. The packed projection
streams themselves occupy 494.4 MB; the remainder is unquantized model state,
CUDA runtime state, activations, and cache metadata. This is lower than the
73.36% projection-weight-only reduction because embeddings, normalization
parameters, `lm_head`, and other dense state remain BF16.

The packed logits have relative L2 error of 11.85%-15.51% on the tested random
inputs, and generated token sequences can diverge after an initially matching
prefix. This is quality/accuracy evidence for the selected NQRT checkpoint,
not evidence of exact output equivalence or acceptable application quality.
Perplexity and downstream task results remain the authoritative quality
measurements.

## Cold preparation and evidence boundary

At sequence length 1, cold packed prefill takes several seconds because the
first call reads and prepares all 112 packed records and compiles/warms Triton
variants. This cost is separated from steady-state latency. It is not included
in the BF16-vs-packed steady-state comparison.

This is a batch-size-one, single-GPU end-to-end inference result for one fixed
Llama revision. It is not a multi-user serving benchmark, multi-GPU result,
production INT4 GEMM result, or claim that packed inference is faster than
BF16. The demonstrated engineering result is: a complete 1B model can execute
with all quantized projection streams resident in packed form, Tensor Core
packed compute substantially improves over scalar packed compute, and the
memory/latency trade-off is now measured at model level rather than only at one
decoder block.
