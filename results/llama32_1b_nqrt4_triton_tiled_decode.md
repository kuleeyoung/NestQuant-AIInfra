# Llama-3.2-1B NQRT v4 tiled Triton decode

## Scope

This experiment validates a compiled packed-weight decode path on an RTX 5070.
One Triton kernel reads NQRT v4 fixed-width code and beta streams, regenerates
the stateless dither, performs nested E8 reconstruction, applies beta and row
scaling, and writes the final BF16 weight. It does not fuse the following GEMM.

Environment: WSL2, PyTorch 2.14.1+cu130, Triton 3.8.0, NVIDIA GeForce RTX 5070.
The checkpoint is the 1,044,005,774-byte Llama-3.2-1B paper-Hadamard,
shared128-LDLQ W4 NQMC whose full WikiText-2 PPL is 10.070482.

## Correctness

Every one of the 112 packed model weights was decoded through both the
chunked PyTorch reference and the tiled Triton path as BF16. All 112 tensors
were elementwise and bitwise equal; the mismatch count and maximum absolute
error were both zero.

The last correctness issue was an E8 Voronoi-boundary decision. PyTorch's
length-eight CUDA reduction pairs lanes `(0,4)`, `(2,6)`, `(1,5)`, and `(3,7)`
before combining the even and odd partial sums. Explicit `add.rn.f32` PTX
operations reproduce that order and prevent compiler reassociation. A blanket
epsilon comparison was rejected because it corrected four boundary blocks but
changed additional valid decisions.

## Standalone tensor benchmark

Each operation was warmed up and then measured 20 times with CUDA
synchronization. Times cover BF16 weight materialization only; GEMM is excluded.

| Weight | Shape | Reference mean | Tiled mean | Speedup | Reference peak | Tiled peak |
|---|---:|---:|---:|---:|---:|---:|
| q_proj | 2048 x 2048 | 45.731 ms | 0.489 ms | 93.48x | 23.66 MiB | 10.13 MiB |
| up_proj | 8192 x 2048 | 200.264 ms | 1.511 ms | 132.54x | 47.66 MiB | 40.53 MiB |
| down_proj | 2048 x 8192 | 68.391 ms | 1.500 ms | 45.59x | 94.63 MiB | 40.51 MiB |

All three benchmark outputs were bitwise equal to the reference decoder.
Machine-readable measurements are in
`results/raw/llama32_1b_nqrt4_triton_tiled_decode.json`.

## Full lazy-model diagnostic

With a 256 MiB decoded-weight LRU and sequence length 8:

| Metric | Result |
|---|---:|
| compiled warm-up, excluded | 4.089 s |
| profiled forward | 2.290 s |
| record read | 1.914 s (83.56%) |
| unpack/dequant | 0.166 s (7.25%) |
| input transform | 0.084 s (3.67%) |
| record parse | 0.076 s (3.33%) |
| GEMM output cast | 0.015 s (0.68%) |
| CUDA peak allocated | 846,217,728 bytes |
| output checksum | 3,037,640.5 |

The checksum matches the reference lazy forward. The profile deliberately
synchronizes every GPU phase, so it is diagnostic rather than a serving
throughput measurement. The dominant cost is now record I/O from the
Windows-mounted checkpoint, not E8 decode. Raw data are in
`results/raw/llama32_1b_nqrt4_tiled_lazy_model_phases.json`.

A separate non-profiling forward compared a fully dequantized baseline with
the tiled lazy model. Its logits were bitwise equal with zero maximum and mean
absolute error. Baseline versus lazy CUDA peak allocation was 2,508,363,264
versus 846,217,728 bytes (66.26% lower), while forward time was 48.98 ms versus
2.759 s. The lazy path is slower end to end because it synchronously reads 112
records from the Windows-mounted file and has no decode/GEMM overlap. This is
an honest memory/latency trade-off, not an end-to-end speedup result.

## Claim boundary

The result supports a resume claim about implementing and validating a fused
packed-stream decoder, including bit-exact boundary handling and measured
standalone decode acceleration. It does not support claims of INT4 Tensor Core
GEMM, no-BF16-materialization inference, production serving throughput, or
end-to-end token-generation speedup. The next runtime stage is to overlap
record reads with compute or fuse decode directly into matrix multiplication.
