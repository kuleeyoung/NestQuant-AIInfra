# NestQuant paper-to-runtime semantics freeze

## Purpose

This document freezes the algorithm boundary used by the packed W4/KV4/A4
runtime work. A compiled implementation is correct only when it matches these
semantics and the corresponding Python oracle; a faster but different
quantizer is a separate ablation, not an optimization of the same path.

## Paper-to-code map

| Paper/runtime concept | Canonical local implementation | Current status |
|---|---|---|
| E8 nearest lattice point | `nestquant/core/lattice.py` | frozen and boundary-tested |
| nested E8 residue codec | `nestquant/core/codec.py` | `e8_upstream_half_coset_tie_v1` |
| row L2 normalization | `nestquant/core/quantizer.py::quantize_rows` | frozen oracle |
| 8D block quantization | `nestquant/core/quantizer.py::quantize_e8_blocks` | frozen oracle |
| fixed-width residue/beta packing | `nestquant/core/packing.py` | NQPK v1 |
| local Hadamard coordinates | `nestquant/core/rotation.py` | validated W4/A4 path |
| beta candidate selection | `nestquant/core/beta_search.py` | experiment input, recorded per run |
| block LDLQ / QA-LDLQ | `nestquant/core/ldlq.py` | implemented; QA result remains an ablation |
| activation quality oracle | `nestquant/core/runtime_quantization.py` | oracle plus NQA4 CUDA packing |
| post-RoPE KV quality oracle | `nestquant/evaluation/kv_cache.py` | oracle plus packed NQKV runtime |
| packed W4 tensor/checkpoint | `tensor_checkpoint.py`, NQMC/NQRT | implemented |
| packed KV page | `nestquant/core/kv_format.py` | backward-compatible NQKV v1/v2 CPU reference |
| packed KV GPU page pool | `nestquant/runtime/nqkv_pool.py` | NQKV v2 fixed-address pool implemented |
| E8 paged attention | `nqkv_attention.py`, `nqkv_attention_k1.py` | K0 oracle and K1 decode path implemented |
| packed activation compute | `triton_w4a4_linear.py`, `cuda_w4a4_mma.py` | tiled full-model path and opt-in CUDA M16 path implemented |

## Frozen KV4 main path

- model: Llama-3.2-1B only;
- Key position: post-RoPE;
- Query/Key rotation: the established matched per-head Hadamard path;
- Key and Value row: one KV head at one token, length `head_dim`;
- E8 grouping: consecutive groups of eight along `head_dim`;
- arithmetic: promote BF16/FP16 rows to float32 for quantization;
- normalization: row L2 norm and `sqrt(head_dim)` scaling;
- beta selection: minimum block reconstruction squared error from the supplied
  beta dictionary;
- codec: `e8_upstream_half_coset_tie_v1`;
- dither: disabled for runtime activation/KV Algorithm-3 semantics;
- packed v1 norm: float32;
- logical cache layout: NHD `[token, kv_head, head_dim]`.

Changing pre/post-RoPE placement, the normalization row, beta source, norm
precision, dither, codec tie-break, or layout creates a new experiment/config
version and requires a quality regression. It must not silently reuse an NQKV
v1 result label.

## Evidence boundary

NQKV v2 now has a GPU page pool, K0/K1 readers, a Hugging Face cache adapter,
and a complete 1B generation smoke test. This proves that the measured runtime
consumes packed KV pages; it does not prove a TPOT speedup, production serving,
or support for every batching, padding, beam-reorder, and prefill shape.

The CUDA W4A4 path decodes packed 4-bit E8 residues into BF16 fragments and
uses BF16 Tensor Core MMA. It is therefore a true packed-storage/packed-input
runtime, but it must not be described as an integer INT4 Tensor Core GEMM.
