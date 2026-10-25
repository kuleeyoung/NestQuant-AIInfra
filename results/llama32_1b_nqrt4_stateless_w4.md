# Llama-3.2-1B NQRT v4 stateless-dither W4

Date: 2026-10-25

## Result

The paper-style local Hadamard + shared128 block-LDLQ W4 checkpoint was fully
re-quantized with NQRT v4 random-access dither and evaluated after independent
NQMC reload.

| Metric | NQRT v3 | NQRT v4 |
|---|---:|---:|
| WikiText-2 PPL | 10.076886 | **10.070482** |
| Predicted tokens | 288,627 | 288,627 |
| Quantized NQRT records | 112 | 112 |
| Dither representation | CPU generator state | stateless seed |
| Checkpoint bytes | 1,044,566,690 | 1,044,005,774 |
| Quantized-record effective rate | 4.267428 bit/parameter | 4.262816 bit/parameter |

The PPL delta is -0.006404 for this fixed seed. It is a measured outcome, not a
claim that stateless dither is generally more accurate.

The v4 checkpoint SHA-256 is:

```text
dcecdf797ca08ad435612c2bef4376a098ba1a6107e9a390dc8e9e9e0cbb9aa4
```

Inspection verified payload checksum, full-file SHA-256, 112/112 NQRT v4
records, 112/112 `stateless_seed` records, q=14, four beta choices, and the
upstream E8 tie semantics.

## Runtime evidence

### Representative tensor decode

Triton fixed-width unpack plus PyTorch v4 dither/E8 reconstruction is bitwise
equal to the reference v4 chunked decoder.

| Tensor | Reference | Hybrid | Speedup | Hybrid peak allocation |
|---|---:|---:|---:|---:|
| q_proj 2048 x 2048 | 45.20 ms | 31.49 ms | 1.435x | 33.41 MiB |
| up_proj 8192 x 2048 | 179.03 ms | 126.42 ms | 1.416x | 93.41 MiB |
| down_proj 2048 x 8192 | 66.73 ms | 33.35 ms | 2.001x | 133.63 MiB |

Only unpack is compiled. Dither generation, E8 reconstruction, scaling, BF16
materialization, and GEMM are still PyTorch operations.

### Full lazy-model diagnostic forward

The same synchronized one-forward protocol with sequence length 8, row chunks
of 128, and a 256 MiB bounded cache produced:

| Metric | NQRT v3 | NQRT v4 | Change |
|---|---:|---:|---:|
| Profiled forward wall time | 22.071 s | 10.741 s | 2.055x faster |
| Unpack/dequant total | 20.224 s | 8.938 s | 2.263x faster |
| Record read | 1.554 s | 1.640 s | approximately unchanged |
| Decode share of forward | 91.63% | 83.21% | still dominant |

This comparison demonstrates removal of the v3 CPU replay bottleneck in the
reference lazy runtime. It is not a serving throughput benchmark: cache misses
still materialize BF16 weights, GEMMs remain BF16, and synchronization is added
for phase accounting.

## Next kernel gate

The next implementation target is one tiled Triton operation that consumes
packed code/beta bytes and performs stateless dither plus E8 reconstruction
without materializing complete expanded integer streams. The current hybrid
path increases transient CUDA allocation, so memory reduction is an explicit
acceptance criterion alongside latency and bitwise agreement.
