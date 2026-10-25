# Llama-3.2-1B NQRT Tensor Core block-dot prototype

## Scope

This prototype decodes packed NQRT v4 weights one K tile at a time inside a
Triton kernel, converts the reconstructed tile to BF16, and feeds it directly
to `tl.dot` with an FP32 accumulator. It never materializes the complete BF16
weight matrix.

The validated real weight is block-0 `q_proj`, shape 2048 x 2048, on an NVIDIA
GeForce RTX 5070. The search covers block N 16/32, block K 32/64/128, 4/8 warps,
and sequence lengths 1/8/16/128. All 48 combinations compile and run.

## E8 tile correctness

An initial implementation incorrectly passed a full K=32 vector to the E8
nearest-point helper, mixing four independent eight-dimensional parity/coset
decisions and producing a maximum output error of 44.25. The corrected kernel
represents decoded data as `[N, K/8, 8]`, performs every nearest-E8 operation
along the final eight-coordinate axis, then reshapes the reconstructed tile to
`[N, K]` for the block dot product.

The CUDA regression test requires maximum error <= 0.0625 and relative L2
error < 1% against a dequantized BF16 PyTorch Linear.

## Real-weight latency

Each configuration is warmed and measured 20 times with CUDA synchronization.

| Sequence length | Best dot config N/K/warps | Tensor Core dot | Scalar fused | Dot speedup | Resident BF16 cuBLAS | Relative L2 error |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 16/32/8 | 0.2087 ms | 0.3747 ms | 1.795x | 0.0301 ms | 0.000000% |
| 8 | 16/64/4 | 0.2238 ms | 0.3440 ms | 1.537x | 0.0328 ms | 0.000446% |
| 16 | 16/64/8 | 0.2280 ms | 0.3828 ms | 1.679x | 0.0539 ms | 0.005358% |
| 128 | 16/32/4 | 1.1010 ms | 1.7809 ms | 1.617x | 0.0429 ms | 0.268700% |

Maximum absolute error for the four best configurations is 0/0.000977/0.015625/
0.0625. The dot path is substantially faster than the earlier explicit
elementwise multiply-and-reduce kernel, but still 4.2-25.7x slower than cuBLAS
with a complete BF16 weight already resident. Packed decode and E8
reconstruction remain inside every K tile and dominate more heavily as token
count grows.

## Tensor Core evidence

The generated Triton GPU IR contains an NVIDIA MMA layout and the generated PTX
contains instructions of the form:

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
```

This is direct compiler-artifact evidence that the BF16 block dot maps to
Tensor Core matrix-multiply-accumulate instructions. It supports a Tensor Core
prototype claim; it does not imply an optimized INT4 Tensor Core kernel because
the packed NQRT values are reconstructed to BF16 before MMA.

## Decision and next step

This positive result reopens model integration with a stronger compute path.
The best configuration varies by sequence length, so the prototype should not
replace the existing default until a small token-count dispatch policy is
confirmed on q/k/v/o and gate/up/down shapes. The next experiment should test
the dot kernel on all distinct Llama projection shapes and then rerun block 0.

Machine-readable measurements:
`results/raw/llama32_1b_nqrt4_dot_qproj.json`.

Reproduction entry point: `scripts/benchmark_triton_dot_linear.py`.
