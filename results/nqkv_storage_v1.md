# NQKV v1 packed-page storage smoke

## Result

The first fixed-capacity NQKV page implementation matches the established
KV4 floating-reconstruction oracle exactly after BF16 output casting. This is
a synthetic storage and codec smoke, not an attention or end-to-end model
benchmark.

Command, executed in the pinned WSL project environment:

```bash
python scripts/benchmark_nqkv_storage.py \
  --output results/raw/nqkv_storage_v1.json
```

Configuration:

| Field | Value |
|---|---:|
| Shape | `[16, 8, 64]` per K or V |
| Page size | 16 tokens |
| Quantization | `q=14`, four betas |
| Codec | `e8_upstream_half_coset_tie_v1` |
| Layout | NHD |
| Norm | FP32, one per token/head row |
| Seed | 20261027 |

Measured fixed-page accounting for K and V together:

| Region | Bytes |
|---|---:|
| Residue codes | 8,192 |
| Beta IDs | 512 |
| Row norms | 1,024 |
| Physical page payload | **9,728** |
| Serialized self-checking container | **9,914** |
| Physical entries | 16,384 |
| Payload rate | **4.75 bits/entry** |

Quality smoke:

| Check | Result |
|---|---:|
| Packed reconstruction equals existing oracle | **true** |
| Key relative squared error vs original BF16 | 0.0177655 |
| Value relative squared error vs original BF16 | 0.0177304 |
| Key maximum absolute error | 0.427734 |
| Value maximum absolute error | 0.421875 |

## Evidence boundary

This result proves that KV4 has a deterministic, versioned low-bit page
representation whose real payload size agrees with the previous 4.75-bit
estimate. The later [GPU pipeline milestone](nqkv_gpu_k0_v1.md) now adds
fixed-address device residency, a fused writer, compiled decoder and K0
page-streamed attention while retaining this canonical byte ordering. Neither
result yet proves end-to-end Llama integration or a fused attention speedup.
