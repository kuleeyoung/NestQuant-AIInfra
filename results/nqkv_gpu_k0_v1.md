# NQKV GPU page pipeline and K0 attention

## Result

NQKV v1 now extends beyond a CPU container: packed streams remain in a
fixed-address CUDA pool, the GPU writer produces canonical bytes without a CPU
round trip, and K0 attention consumes one cold page at a time with FP32 online
softmax.

```bash
python scripts/benchmark_nqkv_gpu.py --iterations 200 \
  --output results/nqkv_gpu_k0_v1.json
```

| Metric | RTX 5070 result |
|---|---:|
| Page shape per K or V | `[16, 8, 64]` |
| Context | 8 pages / 128 tokens |
| Query heads | 32 (GQA 4:1) |
| BF16 K+V bytes per page | 32,768 B |
| Packed NQKV bytes per page | 9,728 B |
| Resident page reduction | **70.3125%** |
| Fused GPU quantize + pack | **792.76 us/page** |
| Triton page decode | **52.58 us/page** |
| K0 page-streamed attention | **3,864.00 us/query** |

The subsequent K1 path fuses packed E8 decode, page-local QK/softmax/PV and a
second-stage online-softmax reduction. With a fixed-address workspace and GPU-
dynamic page/hot lengths, the 128-token measurement was 90.12 us eager and
67.54 us under CUDA Graph versus 4,093.50 us for K0 in the same run. K1 was
45.42x faster than K0. A captured Graph remained correct after the live cache
crossed from two hot tokens to one packed cold page plus a new hot tail.

The initial correctness writer took about 21.05 ms/page. Removing repeated
host synchronization reduced it to 11.11 ms/page; fusing four-beta E8
selection in Triton reduced it to 0.793 ms/page, about 26.6x faster than the
first version. Canonical code/beta bytes still match the CPU format exactly.

Implemented layers: generation-checked physical page pool, logical page table,
BF16 hot tail and packed cold pages, GPU-only writer, Triton decoder,
page-streamed GQA oracle, two-stage fused K1 Attention, reusable workspace and
dynamic CUDA Graph.

## Evidence boundary

This proves real packed KV residency, a GPU hot-to-cold lifecycle and a fused
Attention microbenchmark. It is not yet an end-to-end Llama Attention
replacement. NQKV v1 also shares one beta table between K and V, whereas the
reviewed 1B quality protocol uses distinct global Key and Value tables. That
format mismatch must be resolved before claiming equality with the completed
W4KV4/W4A4KV4 results. Non-KV activations remain BF16-resident, so this is not
yet a complete W4A4KV4 low-bit execution engine.

## NQKV v2 resolution

NQKV v2 now stores independent Key and Value beta tables and keeps v1 readable.
Using the reviewed Llama-3.2-1B tables, the physical payload remains 9,728 B
and the serialized container grows by only 18 B. On RTX 5070 at 128 tokens,
the v2 writer/page decoder/K0/K1/Graph measurements were 966.21/62.89/
4,172.32/67.87/65.70 us respectively; K1 was 61.48x faster than K0. CPU and
CUDA floating reductions may make different quantization decisions on exact
Voronoi boundaries, so each writer is checked against its same-device public
oracle while the byte layout remains common.
