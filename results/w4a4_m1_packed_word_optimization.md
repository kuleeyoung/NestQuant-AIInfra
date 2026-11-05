# W4A4 M1 packed-word decode optimization

## Outcome

The production `M=1` packed W4A4 CUDA path now loads each packed E8 codeword
once and distributes it to the four decoder lanes with warp shuffles. The
pre-optimization kernel remains callable as a legacy regression baseline.

This is a QuIP-inspired packed-bit decode optimization, not a claim that the
QuIP# kernel was copied or reproduced. NestQuant's E8 codebook, beta selection,
dither, and exact reconstruction semantics remain unchanged.

## Correctness and binary audit

- 112/112 Llama-3.2-1B projection layers are BF16 bit-exact against the legacy
  production M1 kernel (`production_max_abs_error = 0`).
- Against the independent tiled Triton implementation, the worst projection
  has maximum absolute error `0.0390625` and maximum per-layer mean absolute
  error `3.0517578125e-05`, matching the established accumulation-order bound.
- Both compiled kernels use 40 registers per thread, zero stack bytes, zero
  shared-memory bytes, and zero local-memory bytes.
- Five targeted CUDA unit tests pass after rebuilding the extension.

Raw evidence:

- `raw/llama32_1b_w4a4_m1_packed_word_112_projections.json`
- `raw/w4a4_m1_binary_resources.json`

## Kernel-only performance

Three independent processes used 10 warmups, 100 timed launches per group, and
seven interleaved groups per process. Every measured shape improved.

| Real projection shape `[M,K,N]` | Minimum speedup | Median speedup | Maximum speedup |
|---|---:|---:|---:|
| `[1,2048,2048]` q_proj | 1.0931x | 1.0932x | 1.0985x |
| `[1,2048,512]` k_proj | 1.0891x | 1.0896x | 1.0917x |
| `[1,2048,8192]` gate_proj | 1.0998x | 1.1000x | 1.1014x |
| `[1,8192,2048]` down_proj | 1.0953x | 1.0985x | 1.1016x |

Raw evidence: `raw/w4a4_m1_packed_word_probe_trial1.json` through `trial3.json`.

## End-to-end result

The controlled comparison uses the same Llama-3.2-1B checkpoint, batch 1,
prompt length 1, 12 generated tokens, 20 prefill repetitions, and all 112
projection layers dispatched to the selected M1 kernel.

| Kernel | TTFT | TPOT | Decode throughput |
|---|---:|---:|---:|
| Legacy M1 | 91.029 ms | 89.877 ms | 11.126 token/s |
| Production packed-word M1 | 84.973 ms | 85.944 ms | 11.635 token/s |
| Improvement | 1.0713x | 1.0458x | +4.57% |

Raw evidence:

- `raw/llama32_1b_m1_production_full_e2e_trial2.json`
- `raw/llama32_1b_m1_legacy_full_e2e_trial2.json`

An earlier short run reported a much larger TPOT difference, but it was not
stable after swapping process order and increasing the sample count, so it is
retained only as raw diagnostic data and is not used as a project claim.

## Profiler boundary and next decision

Nsight Systems produced a report file in this WSL environment but captured no
CUDA kernel or GPU-operation records. Nsight Compute is not installed. The
current conclusion therefore comes from controlled CUDA-event A/B timing,
bit-exact validation, and static resource inspection—not from hardware-counter
attribution of memory, integer-pipeline, or arithmetic stalls.

The packed-word change removes a measurable part of M1 decode overhead, but a
roughly 9-10% isolated improvement shows that E8 byte extraction is not the only
bottleneck. The next planned experiment remains a fixed
`M=8, N=2048, K=2048` tinygemm-style Tensor Core probe to test cross-row weight
reuse before implementing an `M=2..8` kernel family.
