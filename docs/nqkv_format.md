# NQKV packed KV page contract

## Scope

NQKV is the fixed-capacity canonical format shared by the CPU reference and GPU packed-KV
runtime stage. It stores post-RoPE keys and values in NHD order
`[token, kv_head, head_dim]` and uses the same upstream E8 codec, row-wise L2
normalization, multi-beta selection, and float32 arithmetic as the established
KV4 fake-quant oracle.

Version 1 stores one beta table shared by Key and Value. Version 2 adds an
independent Value beta table while retaining the v1 header/parser. The code,
beta-ID and norm page payload is unchanged; with four Key and four Value betas,
v2 adds only 18 serialized bytes (one uint16 count and four FP32 values).

This milestone proves a real low-bit storage representation. It does **not**
yet prove GPU-resident page allocation, a packed attention kernel, lower peak
model VRAM, or faster decoding.

## Frozen v1 semantics

- codec: `e8_upstream_half_coset_tie_v1`;
- logical layout: NHD;
- Key position: post-RoPE;
- E8 block width: 8 consecutive head-dimension entries;
- quantization rows: one `[head_dim]` row per token and KV head;
- input arithmetic: BF16/FP16 inputs are promoted to float32 before quantization;
- code width: `ceil(log2(q))`, four bits for `q=14`;
- beta ID width: `ceil(log2(k))`, two bits for four betas;
- norm dtype: float32;
- physical capacity: `page_size` tokens, with zero-padded unused rows;
- integrity: SHA-256 over the header and all payload fields.

The beta table is embedded as float32 values. Future calibration work may
change which beta values are supplied, but a serialized page is self-contained
and cannot silently inherit a different beta table.

## Physical fields

The serialized container is:

```text
fixed header
beta table
packed K indices (NQPK)
K float32 row norms
packed V indices (NQPK)
V float32 row norms
SHA-256 checksum
```

The logical page payload used for bit accounting excludes the nested container
headers, beta dictionary, and checksum. It includes all fixed-capacity K/V code
bytes, beta-ID bytes, and row-norm bytes. For `q=14`, four betas, and
`head_dim=64`, this is:

```text
4.00 bits/entry residue code
0.25 bits/entry beta ID
0.50 bits/entry float32 row norm
--------------------------------
4.75 bits/entry physical page payload
```

Container overhead is still reported separately by `storage_report()` and must
not be hidden when measuring very small test pages.

## Required compatibility checks

All CUDA/Triton writers and readers added later must match
`PackedNQKVPage.from_tensors()` and `PackedNQKVPage.dequantize()` on:

- ordinary random rows;
- zero rows and unused page capacity;
- token counts at `1`, `page_size-1`, and `page_size`;
- all beta IDs;
- E8 Voronoi tie boundaries;
- overload cases;
- non-contiguous physical page indices.

The Python container is the correctness oracle. A compiled implementation may
use a header-free GPU page pool, but its code, beta, and norm byte regions must
have the same logical ordering and measured payload size.
