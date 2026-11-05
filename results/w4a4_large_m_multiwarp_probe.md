# W4A4 M32/M64/M128 multi-warp structure probe

## Goal and mapping

This experiment tests the first large-M step after the production M1-M16
family. One CTA owns sixteen output channels. Each warp computes eight token
rows with `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, so M32,
M64, and M128 launch four, eight, and sixteen warps per CTA respectively.

The probe consumes packed NQRT v4 weights and packed NQA4 activations directly.
It does not materialize BF16 matrices. This first structure probe deliberately
keeps weight decode warp-local: it measures whether the multi-warp mapping is
useful before adding shared-memory staging and pipeline complexity.

## Correctness and binary gates

- CUDA unit suite: `10 passed, 4 subtests passed`.
- M32, M64, and M128 are BF16 bit-exact with concatenated production M8
  results for the same rows.
- The compiled large-M kernel uses 56 registers per thread and zero stack,
  shared-memory, and local-memory bytes.
- All four representative real shapes are finite and pass the benchmark's
  `atol=0.04, rtol=0.01` Triton comparison.

Across all 112 Llama-3.2-1B projections at M32, the maximum relative L2 error
against tiled Triton is 0.001303 and the maximum mean absolute error is
9.44e-6. Two layer-5 MLP projections fail the stricter per-element
`atol=0.125, rtol=0.01` gate: their maximum absolute errors are 0.1875 and
0.15979. The multi-warp output is bit-exact with four M8 calls, so this is an
accumulation-order boundary versus Triton's K=32 tiling rather than an address
mapping error. The probe is therefore not production-dispatched yet.

## Tuned Triton comparison on RTX 5070

Each entry is `Triton time / custom-kernel time`; values above one favor the
custom kernel. Measurements use alternating order, seven groups, ten warmups,
and one hundred iterations per group. Triton configuration comes from the
runtime's `select_w4a4_tiled_config`.

| M | q `[K=2048,N=2048]` | k `[2048,512]` | gate `[2048,8192]` | down `[8192,2048]` |
|---:|---:|---:|---:|---:|
| 32 | 1.464x | 1.393x | 1.505x | 1.434x |
| 64 | 1.263x | 1.493x | 1.063x | 1.279x |
| 128 | 1.048x | 1.207x | 0.986x | 1.094x |

Raw evidence is stored in
`results/raw/w4a4_m{32,64,128}_{q,k,gate,down}_multiwarp_v1.json`; the M32
112-projection audit is
`results/raw/llama32_1b_w4a4_m32_multiwarp_112_projections.json`.

## Interpretation and next gate

The mapping is worthwhile at M32 and still shape-dependent at M64. It reaches
its scaling limit by M128: sixteen warps repeat the same weight decode, and the
gate projection is already slower than tuned Triton. Merely increasing CTA
size is therefore rejected as the production M128 design.

The next experiment must reuse decoded weight fragments across consumer warps.
A production candidate needs staged shared memory, overlap between decode/load
and MMA, and a tile scheduler; CUTLASS/CuTe is preferred for that pipeline.
The local environment currently has CUDA 13.0 and PyTorch CUDA extension
support but no installed CUTLASS/CuTe headers, so dependency integration is a
separate, explicit engineering step.

No end-to-end latency or quality claim is made from this isolated probe.

## M128 decoded-weight shared-memory staging

The follow-up M128 kernel decodes each 16x16 BF16 weight fragment once in warp
0, stores it in shared memory, and lets all sixteen token warps consume the
same fragment. This reduces weight recovery from sixteen copies per K tile to
one. The correctness-first implementation uses two full-CTA barriers per
K=16 tile.

| M128 shape | unstaged vs Triton | staged vs Triton | staged improvement over unstaged |
|---|---:|---:|---:|
| q `[2048,2048]` | 1.048x | 1.510x | 1.461x |
| k `[2048,512]` | 1.207x | 1.370x | 1.174x |
| gate `[2048,8192]` | 0.986x | 1.593x | 1.626x |
| down `[8192,2048]` | 1.094x | 1.582x | 1.455x |

The compiled staged kernel uses 64 registers per thread, 1536 bytes of static
shared memory, and zero stack/local bytes. It is BF16 bit-exact with the
unstaged kernel. The gate result directly establishes repeated E8 weight
recovery as the previous M128 bottleneck.

## Rejected double-buffer pipeline

A second probe used one producer warp, sixteen consumer warps, and two
alternating decoded-weight buffers. It overlaps production of tile `t+1` with
consumer work on tile `t`, reducing the steady-state protocol to one CTA
barrier per tile. It remains bit-exact, but gate latency regressed from
3701.0 us to 4274.1 us. The pipeline uses 72 registers, 544 threads, and 2048
shared bytes, so its scheduling/resource cost exceeds the removed barrier cost
on the tested RTX 5070. It is retained as a negative control, not promoted.

## Production dispatch and model-level gates

The staged kernel is production-dispatched only for exact M=128, compatible
NQRT v4/NQA4 records, no bias, and no explicit tiled fallback. It has a
dedicated dispatch counter. Existing M1-M16 behavior is unchanged, and the
combined CUDA plus model-dispatch suite reports `21 passed, 4 subtests passed`.

Across 112 real projections, two layer-5 MLP projections exceed the strict
per-element comparison while the maximum relative L2 remains 0.001299 and the
maximum mean absolute error remains 9.06e-6. Exact-name fallback is supported
for sensitivity policies.

For Llama-3.2-1B, batch 1, sequence length 128, ten steady-prefill repeats,
and exact fallback of layer-5 gate/up:

| path | steady prefill median | TTFT | speedup |
|---|---:|---:|---:|
| all tiled Triton | 449.37 ms | 440.78 ms | reference |
| staged M128 candidate | 311.38 ms | 309.20 ms | 1.443x prefill / 1.426x TTFT |

The generated first token matched, but the single-prompt logit relative L2
was 0.2088. This is recorded as a sensitivity warning rather than treated as
a task-quality metric.

A deterministic 128-window WikiText-2 diagnostic at context length 128
evaluated 16,256 predicted tokens:

| path | PPL | elapsed | relative PPL |
|---|---:|---:|---:|
| tiled Triton | 26.43752 | 64.48 s | reference |
| staged M128 | 26.44796 | 45.50 s | +0.0395% |

The bounded task metric is effectively stable and ran 1.417x faster. This is
not the formal 2258-window full-corpus result, and no quality improvement is
claimed.

Raw additions:

- `results/raw/w4a4_m128_{q,k,gate,down}_staged_v1.json`
- `results/raw/w4a4_m128_gate_pipeline_v1.json`
- `results/raw/llama32_1b_w4a4_m128_staged_112_projections.json`
- `results/raw/llama32_1b_w4a4_m128_staged_e2e_seq128.json`
- `results/raw/llama32_1b_w4a4_{tiled,m128_staged}_wikitext_ctx128_128.json`

CUTLASS/CuTe 4.8.0 was selected for the future layout/scheduler refactor, but
the pinned source download was blocked by the current GitHub connection. The
custom shared-memory experiment therefore proceeded without adding an
unversioned or partial third-party dependency.

## CUTLASS CuTe M128 layout probe

CUTLASS v4.8.0 was subsequently downloaded from the fixed GitHub codeload tag,
verified with SHA-256
`b7bb2ba0473a5a5d699c62e5feb42caa11cb109ec58ba82ca458af9cd139f39b`,
and compiled with CUDA 13.0 for `sm_120`. A standalone smoke kernel confirmed
`CUTLASS 4.8.0` and a 16x16 CuTe layout with 256 elements on the RTX 5070.

The isolated M128 CuTe extension keeps the established E8 decoders and
`m16n8k16` MMA, while expressing the shared weight fragment, warp-to-token
mapping, and output tensor through CuTe layouts. It is BF16 bit-exact with the
production staged kernel.

Same-process rotating-order results:

| M128 shape | CuTe vs Triton | CuTe / staged speed |
|---|---:|---:|
| q `[2048,2048]` | 1.490x | 0.993x |
| k `[2048,512]` | 1.365x | 1.002x |
| gate `[2048,8192]` | 1.609x | 1.000x |
| down `[8192,2048]` | 1.573x | 1.001x |

The CuTe kernel and hand-written staged kernel both compile to 64 registers,
1536 shared bytes, and zero stack/local bytes. The abstraction therefore adds
no measurable resource or performance cost, but it is not promoted yet: the
hand-written production kernel remains dependency-free, while CuTe stays an
optional independently built candidate.

Reproduction assets:

- `scripts/bootstrap_cutlass.py` pins URL, version, and SHA-256.
- `scripts/probe_cutlass_cute.cu` verifies host headers plus an SM120 kernel.
- `nestquant/runtime/nqrt4_w4a4_cute_probe.cu` contains the M128 layout probe.
- `results/raw/w4a4_m128_{q,k,gate,down}_cute_paired_v1.json` contains raw A/B.

## Production staged family: M32/M64/M128

The CuTe family experiment showed that the same decoded-weight shared-memory
layout works across M32, M64, and M128. The dependency-free CUDA kernel was
therefore generalized to those three exact row counts and promoted into the
`packed_w4a4_cuda_mma` dispatcher. Each CTA still owns sixteen output channels;
M32, M64, and M128 use four, eight, and sixteen warps respectively. Sensitive
layers retain the existing tiled fallback.

The production implementation does not require CUTLASS at runtime. CUTLASS/CuTe
remains an optional layout-validation path, while the shipped kernel uses the
same shared-memory staging directly in CUDA. All three templates compile to
64 registers per thread, 1536 bytes of static shared memory, and zero
stack/local bytes.

For layer-0 q-projection `[N=2048,K=2048]`, same-process alternating-order
measurements on RTX 5070 are:

| rows | production staged | tuned Triton | speedup vs Triton |
|---:|---:|---:|---:|
| M32 | 493.08 us | 809.63 us | 1.642x |
| M64 | 572.22 us | 1092.10 us | 1.909x |
| M128 | 1200.50 us | 1803.86 us | 1.503x |

The CUDA runtime/model regression suite passes `23 tests` plus `11 subtests`.
The family test verifies M32/M64/M128 are BF16 bit-exact with the prior
unstaged CUDA implementation. Model-level tests additionally verify that all
three row counts increment the large-M staged dispatch counter and remain
within the established tiled-Triton numerical gate.

Raw measurements:

- `results/raw/w4a4_m32_q_staged_family_v1.json`
- `results/raw/w4a4_m64_q_staged_family_v1.json`
- `results/raw/w4a4_m128_q_staged_family_v1.json`

## Full-model M32/M64 production validation

The production dispatcher was evaluated on the complete Llama-3.2-1B model
with all 112 projections converted to packed W4A4. The two projections that
miss the strict per-element audit gate, layer-5 MLP gate/up, were forced to the
tiled Triton fallback by exact module name. All other projections used the
custom CUDA family whenever their row count matched a supported M.

| prefill shape | tiled prefill | staged prefill | prefill speedup | tiled TTFT | staged TTFT | TTFT speedup |
|---|---:|---:|---:|---:|---:|---:|
| batch 1, seq 32 | 220.28 ms | 150.42 ms | 1.464x | 217.48 ms | 153.73 ms | 1.415x |
| batch 1, seq 64 | 263.56 ms | 164.46 ms | 1.603x | 263.68 ms | 166.08 ms | 1.588x |

Each CUDA run recorded 1,210 large-M staged dispatches and 28 tiled fallback
dispatches across warmup, seven steady-prefill repetitions, TTFT, and decode.
The decode portion also exercises the production M1 kernel, so its TPOT
speedup is not attributed to large-M alone.

The first-prefill logits are not bit-equivalent to tiled Triton. Relative L2
logit error is 20.08% at seq32 and 16.51% at seq64, and later greedy tokens can
diverge. Small per-projection accumulation-order differences therefore compose
across 16 layers; the kernel must not be described as end-to-end quality
equivalent on the basis of its layer-local error gate.

A fixed 128-window WikiText-2 check provides a statistical quality boundary:

| context | tiled PPL | staged PPL | relative PPL change | tiled time | staged time | evaluation speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 80.2768 | 79.3699 | -1.130% | 31.44 s | 21.98 s | 1.430x |
| 64 | 44.6012 | 44.7422 | +0.316% | 38.80 s | 25.65 s | 1.513x |
| 64, 512 windows | 44.5997 | 44.6756 | +0.170% | 143.37 s | 92.10 s | 1.557x |

These short-context, 128-window checks are not full-corpus PPL results. The
negative M32 delta is not evidence that a different accumulation order improves
the model; finite-window perturbations may move PPL in either direction. The
supported claim is that this limited protocol found no catastrophic quality
regression while confirming the end-to-end performance benefit.

The M64 protocol was expanded from 128 to 512 identical-prefix windows
(32,256 predicted tokens). The relative PPL delta contracted from +0.316% to
+0.170%, while evaluation speedup remained 1.557x. Both runs record the same
token SHA-256 and the same 1076.28 MiB peak CUDA allocation. This satisfies the
stage acceptance rule: at least 512 paired windows, absolute relative PPL
delta below 0.5%, no growth versus the 128-window checkpoint, and speedup above
one. A 4,516-window full-context-64 sweep remains optional stronger evidence,
not a blocker for promoting this kernel family.

Raw evidence:

- `results/raw/llama32_1b_w4a4_large_m_staged_e2e_seq32_64.json`
- `results/raw/llama32_1b_w4a4_m{32,64}_staged_112_projections.json`
- `results/raw/llama32_1b_w4a4_{tiled,m32_staged}_wikitext_ctx32_128.json`
- `results/raw/llama32_1b_w4a4_{tiled,m64_staged}_wikitext_ctx64_128.json`
- `results/raw/llama32_1b_w4a4_{tiled,m64_staged}_wikitext_ctx64_512.json`
- `scripts/verify_w4a4_quality_convergence.py` enforces paired-input,
  quality-delta, speedup, and convergence gates.
- `results/raw/llama32_1b_w4a4_m64_staged_quality_gate.json` records the
  machine-checked accepted decision.
