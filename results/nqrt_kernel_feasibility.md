# NQRT compiled decoder feasibility gate

Date: 2026-10-25  
Hardware: NVIDIA GeForce RTX 5070, WSL2, PyTorch 2.14.1+cu130, Triton 3.8.0

## Outcome

The prototype proves that NQPK fixed-width streams can be decoded correctly by
a compiled Triton kernel, but it also falsifies the hypothesis that unpacking is
the dominant blocker for the current NQRT v3 runtime.

- Triton unpack is bitwise equal to the reference unpacker.
- The isolated q_proj unpack is about 21x faster with compact int16 output.
- End-to-end tensor decode improves by only 1.09x to 1.14x on three real Llama
  tensors.
- Hybrid decode increases peak temporary CUDA allocation by about 10 MB to
  48 MB because the complete expanded code streams are materialized.
- CPU dither replay consumes 75% to 89% of measured hybrid tensor-decode time.

This is a compiled primitive and feasibility result, not a fused low-bit GEMM
or a production inference speedup claim.

## Measurements

All cases use real records from the W4, shared-128, LDLQ, NQRT v3
Llama-3.2-1B checkpoint. End-to-end output is BF16 and bitwise identical to the
reference path.

| Tensor | Shape | Reference | Hybrid | Speedup | Reference peak | Hybrid peak | CPU dither share |
|---|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 2048 x 2048 | 95.96 ms | 84.31 ms | 1.138x | 21.78 MiB | 31.53 MiB | 75.0% |
| up_proj | 8192 x 2048 | 407.74 ms | 371.18 ms | 1.098x | 45.78 MiB | 91.53 MiB | 77.5% |
| down_proj | 2048 x 8192 | 334.71 ms | 306.02 ms | 1.094x | 87.13 MiB | 126.13 MiB | 89.1% |

The phase percentages come from one synchronized profiled call per tensor.
They are diagnostic shares, not statistically averaged benchmark values.

## Why NQRT v3 reaches a structural limit

NQRT v3 stores the state of a portable PyTorch CPU random generator rather
than independently addressable dither values. Exact decode restores that state
and reproduces the original `torch.rand` stream before E8 reconstruction.
The existing implementation deliberately generates the entire tensor stream:
splitting one large random call into smaller calls is not guaranteed to preserve
the same sequence.

Consequently, an exact v3 decoder cannot map arbitrary weight tiles directly to
independent GPU program instances without either:

1. reproducing PyTorch's CPU generator and floating-point E8 dither construction
   inside the GPU kernel, including version-sensitive semantics; or
2. generating the full dither tensor on CPU and transferring it, which keeps the
   measured bottleneck and prevents a clean packed-to-GEMM fused path.

The E8 reconstruction itself is regular eight-dimensional arithmetic and is a
reasonable Triton target. It is not the main measured cost while v3 dither
replay remains on the critical path.

## Environment boundary

- Triton works on the installed CUDA stack and RTX 5070.
- `g++` and Ninja are installed.
- `nvcc` is absent, so a conventional PyTorch C++/CUDA extension is not
  currently buildable without changing the environment.
- Triton is therefore the verified compiled-kernel route on this machine.

## Morning decision

### Route A - NQRT v4 kernel route (recommended)

Replace generator-state dither with a stateless, counter-addressable definition
keyed by tensor seed and global E8 block index. Re-quantize a checkpoint, rerun
the fixed WikiText-2 quality protocol, then implement tiled Triton E8 decode and
finally evaluate a fused decode/GEMM design.

Benefits: removes the measured structural blocker, enables independent GPU
tiles, and creates the strongest AI-infrastructure story. Costs: v4 artifacts
will not be bitwise compatible with v3, quality must be revalidated, and a true
fused GEMM remains substantial engineering work.

Acceptance gate: v4 dither is chunk-invariant; CPU and Triton decoders agree;
W4 perplexity remains within a predeclared tolerance of v3; full-model lazy
forward shows lower decode time without higher transient memory.

### Route B - NQRT v3 compatibility route

Keep the current file format and implement Triton E8 reconstruction while
retaining CPU dither replay. Optimize allocations and caching, but do not claim
a fused low-bit runtime.

Benefits: existing checkpoints remain bitwise compatible and implementation risk
is lower. Costs: the 75% to 89% CPU bottleneck remains, so expected end-to-end
headroom is limited and the architecture cannot cleanly become tile-fused.

Acceptance gate: retain exact v3 output and demonstrate a measurable full-model
forward improvement at bounded cache capacity and no peak-memory regression.

### Route C - Freeze runtime work and package the research prototype

Stop kernel development. Consolidate the reproducible quantization, packed file
format, lazy runtime, profiling, tests, and evidence boundaries into interview
documentation and a public-facing experiment report.

Benefits: fastest path to a polished, defensible portfolio. Costs: the project
must be described as a packed/lazy runtime prototype, not genuine low-bit
inference acceleration.

## Recommendation

Choose Route A if the project is intended to be the primary AI-infrastructure
portfolio project and several more weeks are available. Choose Route C if the
near-term priority is applications and interview preparation. Route B is useful
only when preserving existing v3 artifacts is itself a requirement; the current
measurements make it the weakest default investment.

## Reproduction

```bash
python scripts/audit_kernel_environment.py --output results/raw/kernel_environment.json
python scripts/benchmark_triton_unpack.py CHECKPOINT WEIGHT --repeats 20 --device cuda
python scripts/benchmark_triton_hybrid_decode.py CHECKPOINT WEIGHT... --repeats 5 --rows-per-chunk 128 --device cuda
python -m unittest tests.unit.runtime.test_triton_unpack -v
```

Raw benchmark JSON is intentionally kept under `results/raw/` and is not used
as a standalone résumé claim.
