# Llama-3.2-1B block-0 packed fused NQRT runtime

## Scope

This experiment replaces all seven Linear projections in decoder block 0 with
the packed NQRT v4 fused Triton module:

- attention: q, k, v, and output projections;
- MLP: gate, up, and down projections.

The script first performs a normal model call and captures the exact decoder
block inputs, attention mask, position IDs, and rotary position embeddings
constructed by Transformers 5.20.2. It then benchmarks the same block call with
two implementations of the same quantized weights:

1. all seven NQRT weights decoded to resident BF16 matrices, followed by the
   standard PyTorch/cuBLAS Linear path;
2. all seven packed streams resident on the GPU and consumed directly by the
   fused Triton kernels without full BF16 weight materialization.

Both paths apply the checkpoint's online input Hadamard transforms. This is a
real Transformer-block integration result, but it is not a full-model
generation or throughput benchmark.

## Resident weight memory

| Representation for seven projections | Bytes | Relative to BF16 |
|---|---:|---:|
| decoded BF16 matrices | 121,634,816 | 100% |
| packed code/beta streams and metadata | 32,403,568 | 26.64% |

The packed form removes 89,231,248 bytes, or 73.36%, from the seven resident
projection weights. The first fused block call, including seven ext4 NQMC
record reads, parsing, and GPU preparation, takes 1.532 seconds. That cold-start
cost is excluded from the steady-state timings below.

## Steady-state block latency and numerical agreement

Every warmed path is measured 30 times with CUDA synchronization.

| Sequence length | Resident BF16 block | Packed fused block | Fused / BF16 speed | Max absolute error | Relative L2 error |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.9380 ms | 5.9617 ms | 0.661x | 0.007813 | 0.876% |
| 8 | 3.6101 ms | 5.9670 ms | 0.605x | 0.023438 | 0.936% |
| 16 | 4.0053 ms | 6.0214 ms | 0.665x | 0.062500 | 1.135% |

The packed block is 1.50-1.65x slower than a block whose decoded BF16 weights
are already resident. The isolated q-projection kernel previously beat a path
that decoded a full matrix on every call; that comparison does not imply it can
beat cuBLAS after the decode cost has been amortized. Integrating seven fused
modules makes this distinction explicit.

The measured temporary allocation increment is the same for both paths at
72/576/1152 KiB for sequence lengths 1/8/16. This measurement excludes the
already-resident weight state; the material difference is the 121.6 MB versus
32.4 MB persistent projection representation above.

## Engineering conclusion and next bottleneck

The model adapter works: real attention and MLP code call packed weights
directly, all seven modules preserve shapes and residual semantics, and the
block output remains close to the decoded-weight reference. The result also
shows that avoiding materialization alone is insufficient for a competitive
steady-state runtime. The next kernel step should combine projections that
share the same input (Q/K/V and gate/up) and move the dot products toward
Tensor Core operations; merely replacing more PyTorch Linears with the current
scalar-accumulation kernel will increase end-to-end latency.

Machine-readable measurements are in
`results/raw/llama32_1b_nqrt4_fused_block0.json`. Reproduction entry point:
`scripts/benchmark_fused_nqrt_block.py`.
