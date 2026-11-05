# Direct-packed W4A4 CUDA MMA reproduction

## Requirements

- NVIDIA GPU with BF16 Tensor Core support;
- CUDA toolkit and a working PyTorch CUDA extension toolchain;
- Triton, used as the correctness and performance baseline;
- the local NQRT4 Llama-3.2-1B checkpoint for real-model commands.

The extension is compiled lazily on the first call. Compilation time must not
be included in steady kernel timing.

## Boundary regression and dispatch tests

```bash
python -m pytest -q \
  tests/unit/runtime/test_cuda_w4a4_mma.py \
  tests/unit/models/test_fused_linear.py
```

The activation regression includes the three E8 code blocks that exposed the
FP32 grouping mismatch. The model test covers M16 dispatch and unsupported
shape fallback.

## Verify compiled Tensor Core instruction

```bash
python scripts/verify_w4a4_mma_binary.py \
  --output results/raw/w4a4_cuda_mma_binary_audit.json
```

This hashes the loaded extension and uses `cuobjdump --dump-sass` to require
`HMMA.16816.F32.BF16` in the actual GPU binary. Source-level inline PTX alone
is not accepted as compiled-instruction evidence.

## Real q-projection correctness and stable microbenchmark

```bash
python scripts/build_test_w4a4_mma.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --warmup 10 \
  --iterations 100 \
  --groups 7 \
  --output results/raw/w4a4_cuda_mma_m16_cooperative_stable.json
```

Acceptance gates:

- `cuda_triton_differing_block_count == 0`;
- `activation_decode_max_abs_error == 0`;
- all outputs finite;
- CUDA/Triton output error remains within the checked BF16 tolerance;
- report median and the complete timing sample arrays, not the best run.

## Audit every model projection

```bash
python scripts/validate_w4a4_mma_projections.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --output results/raw/w4a4_cuda_mma_all_projections.json
```

This executes all 112 packed projection records with deterministic M16
activations and compares the custom CUDA output against tiled Triton.

For the M1 GEMV audit, add `--tokens 1` and write to
`results/raw/w4a4_cuda_gemv_m1_all_projections.json`.

## M1 GEMV microbenchmark

```bash
python scripts/benchmark_w4a4_gemv_m1.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --warmup 10 --iterations 100 --groups 7 \
  --output results/raw/w4a4_cuda_gemv_m1_benchmark.json
```

## Isolated full-model M1 decode benchmark

```bash
python scripts/benchmark_full_nqrt_model.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --sequence-length 16 --batch-size 1 --decode-tokens 8 --repeats 7 \
  --modes packed_w4a4_tiled packed_w4a4_cuda_gemv_m1 \
  --offline \
  --output results/raw/llama32_1b_w4a4_cuda_gemv_m1_only_trial1.json
```

Run three independent trials sequentially and aggregate them with
`scripts/summarize_w4a4_mma_trials.py`. Do not run trials concurrently on one
GPU: resource contention invalidates the comparison.

## Complete-model M16 benchmark

```bash
python scripts/benchmark_full_nqrt_model.py \
  results/raw/llama32_1b_paper_hadamard_w4_shared128_ldlq_nqrt4.nqmc \
  --sequence-length 16 \
  --batch-size 1 \
  --decode-tokens 4 \
  --repeats 7 \
  --modes packed_w4a4_tiled packed_w4a4_cuda_mma \
  --offline \
  --output results/raw/llama32_1b_w4a4_cuda_mma_m16_trial1.json
```

Repeat the complete process twice more as `trial2.json` and `trial3.json`, then
validate that the protocol, environment, seed, input-token hash, and modes are
identical and aggregate the independent-process medians:

```bash
python scripts/summarize_w4a4_mma_trials.py \
  results/raw/llama32_1b_w4a4_cuda_mma_m16_trial1.json \
  results/raw/llama32_1b_w4a4_cuda_mma_m16_trial2.json \
  results/raw/llama32_1b_w4a4_cuda_mma_m16_trial3.json \
  --output results/raw/llama32_1b_w4a4_cuda_mma_m16_trials_summary.json
```

The new backend is explicit opt-in. M16 prefill uses CUDA MMA; M1 decode falls
back to tiled Triton. Do not interpret generated-token equality as a kernel
correctness test: repeated A4 requantization can amplify legal BF16 accumulation
order differences. Kernel correctness, end-to-end latency, and model quality
must be reported as separate evidence.

## Audited local result

- q-projection CUDA median: 411.41 us;
- q-projection tiled Triton median: 552.34 us;
- kernel speedup: 1.343x;
- three-trial median M16 prefill: 117.17 ms versus 178.93 ms;
- three-trial median prefill speedup: 1.527x (range 1.518x to 1.544x);
- M16 execution audit: 1,232 custom dispatches per trial;
- M1-only TPOT: 94.53 ms versus 170.01 ms, median speedup 1.803x;
- M1 execution audit: 896 custom dispatches and zero decode fallbacks per trial;
- formal test suite: 305 passed plus 37 subtests.

See `results/w4a4_cuda_mma_m16_correctness.md` and
`results/llama32_1b_w4a4_cuda_mma_m16.md` for evidence boundaries.
