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

## Storage-path ablation

The same checkpoint was copied from the Windows `/mnt/d` mount to WSL's native
ext4 filesystem. Both files have SHA-256
`dcecdf797ca08ad435612c2bef4376a098ba1a6107e9a390dc8e9e9e0cbb9aa4`.
A dedicated benchmark alternated the two paths for seven repetitions and read
the same 112 quantized record blobs (518,506,818 bytes per repetition), while
excluding parse and decode:

| Path | Mean | Median | Observed throughput |
|---|---:|---:|---:|
| Windows `/mnt/d` | 1.485 s | 1.523 s | 333 MiB/s |
| WSL ext4 | 0.121 s | 0.100 s | 4,100 MiB/s |

Operating-system page caches were not dropped, so these are observed warm
runtime-path measurements rather than raw-device cold-read bandwidth. Under
the same synchronized model profile, ext4 reduced record-read time from
1.914 s to 0.0618 s and total forward time from 2.290 s to 0.367 s.

In the non-profiling benchmark on ext4, the first lazy forward took 0.692 s;
the next two took 0.298 s and 0.314 s. Logits remained bitwise equal and CUDA
peak allocation remained 846,217,728 bytes. The fully materialized baseline
forward in that run took 78.18 ms, so the low-memory lazy path is still about
four times slower after warm-up. Storage placement removes most of the prior
I/O penalty but does not eliminate synchronous read/decode overhead.

## One-record-ahead CPU prefetch

An experimental `--record-prefetch` mode uses one background thread to read
the next Linear's serialized record while the current record is parsed,
decoded, and consumed. Model registration order defines the prefetch sequence;
checkpoint manifest order is not assumed to be execution order.

On ext4, synchronized profiling reduced visible record wait from 61.8 ms to
1.67 ms, but parse and decode time increased enough that total forward changed
only from 366.5 ms to 350.1 ms. In the non-profiling runs, no-prefetch warm
times were 298/314 ms and prefetch warm times were 353/325 ms, so this is not a
demonstrated ext4 latency improvement and remains disabled by default.

On `/mnt/d`, prefetch reduced the first measured lazy forward from 2.759 s in
the earlier no-prefetch run to 1.975 s; subsequent prefetch runs were
1.500/1.415 s. This mitigates a slow storage path but remains far slower than
putting the checkpoint on ext4. The engineering decision is therefore to keep
CPU prefetch as an opt-in diagnostic feature, recommend ext4 placement first,
and avoid claiming a general prefetch speedup from these measurements.

## Claim boundary

The result supports a resume claim about implementing and validating a fused
packed-stream decoder, including bit-exact boundary handling and measured
standalone decode acceleration. It does not support claims of INT4 Tensor Core
GEMM, no-BF16-materialization inference, production serving throughput, or
end-to-end token-generation speedup. The next runtime stage is to overlap
record reads with compute or fuse decode directly into matrix multiplication.
