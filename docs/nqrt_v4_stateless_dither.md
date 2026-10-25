# NQRT v4 random-access dither

NQRT v4 replaces the default serialized PyTorch CPU generator state with a
versioned stateless dither descriptor:

```text
dither_mode = 3
dither_algorithm = "lcg31_mix_v1"
dither_seed = integer in [0, 2^31)
```

Old NQRT v1-v3 records remain readable. Generator-state dither also remains an
explicit compatibility option, but new model quantization exports default to
the v4 stateless mode.

## Addressing rule

Every scalar uniform sample is identified by its global E8 block and coordinate:

```text
counter = global_block_index * 8 + coordinate_index
```

`lcg31_mix_v1(seed, counter)` is a specified integer hash implemented only with
signed int64 arithmetic whose intermediate values remain below `2^63`. The
31-bit output is converted to an FP32 sample in `[0, 2)`, then mapped into the
E8 Voronoi region with the recorded codec tie-break semantics. Casting to the
recorded tensor dtype happens after this FP32 construction.

The algorithm ID is part of the file semantics. Changing constants, float
construction, E8 tie behavior, or counter layout requires a new algorithm ID.

## Required invariants

- one full call equals any concatenation of adjacent chunk calls;
- a block can be generated without visiting earlier blocks;
- execution order does not affect output;
- CPU and CUDA outputs are bitwise equal in the supported runtime;
- `rows_per_chunk` does not change reconstructed weights;
- v4 artifacts do not silently claim bitwise equivalence with v3 artifacts.

The tests cover CPU/CUDA equality, chunk/order invariance, NQRT round-trip,
multiple decode chunk sizes, model-level LDLQ export, and the Triton-unpack
hybrid path.

## Why v4 is necessary

NQRT v3 restores a PyTorch CPU generator state and replays one full random
stream. A GPU program decoding an arbitrary tile cannot directly seek to its
dither values. Real Llama tensor profiling showed that this CPU replay consumed
75%-89% of hybrid decode time after compiled unpack was introduced.

In an isolated shape-matched benchmark, v4 GPU random-access generation was
16.4x-22.1x faster than v3 CPU replay. This result removes the serialization
dependency, but does not by itself establish checkpoint quality, full decode
speed, fused GEMM speed, or model throughput. Those claims require v4
re-quantization and the fixed evaluation protocol.
