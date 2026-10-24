# NestQuant tensor serialization formats

## NQPK v1 packed index streams

This format is a working prototype for the two integer streams produced by the E8 quantizer. It is not a complete model checkpoint format and is not connected to a low-bit inference kernel.

## Contents

The container stores one quantized tensor's residue codes and beta IDs. For each value it uses the smallest supported fixed width that covers the declared cardinality:

- residue width: `ceil(log2(q))` bits per residue (`q=14` uses 4 bits);
- beta-index width: `ceil(log2(beta_count))` bits per E8 block (`beta_count=4` uses 2 bits);
- packing order: values are packed least-significant bits first within each byte;
- supported widths: 1, 2, 4, and 8 bits. These widths divide a byte, so this version has no cross-byte spill case;
- final partial byte: unused high bits must be zero.

The little-endian header uses `struct` format `<4sBBHHQQ`:

| Field | Type | Meaning |
|---|---|---|
| magic | 4 bytes | `NQPK` |
| version | uint8 | `1` |
| ndim | uint8 | Number of residue tensor dimensions |
| q | uint16 | Residue cardinality |
| beta_count | uint16 | Number of selected beta values |
| code_length | uint64 | Residue payload length in bytes |
| beta_length | uint64 | Beta-index payload length in bytes |

It is followed by `ndim` little-endian uint64 dimensions, then the code payload and beta-index payload. The residue tensor must end in dimension 8; the beta-index shape is `codes.shape[:-1]`.

## NQRT v3 single-tensor artifact

`PackedNestQuantTensor` wraps the NQPK stream with the data needed to reconstruct one quantized matrix tensor: UTF-8 tensor name, q, beta cardinality and float beta values, row/feature dimensions, output dtype, per-row norms, dither reproduction data, and codec semantics. NQRT v3 has a fixed little-endian length header followed by compact JSON metadata, the NQPK container, row-norm bytes, and an optional dither payload. Float tensors support float16, bfloat16, float32, and float64, preserving their recorded dtype exactly.

NQRT v3 writes `codec_semantics=e8_upstream_half_coset_tie_v1`, matching the fixed author implementation when the integer and half-integer E8 cosets are equidistant. Old NQRT v1/v2 records lack this field and are read as `e8_legacy_integer_coset_tie_v1`; this prevents historical residue codes from being silently decoded to a different boundary representative. See [codec semantics](codec_semantics.md).

NQRT v2/v3 supports three dither modes: absent, exact tensor values, or a CPU `torch.Generator` state captured immediately before generation. State mode stores the compact RNG state and PyTorch version rather than one FP32 value per weight; the random stream is drawn on CPU, while the E8 nearest-point operation runs on the requested model device. The exact realization is reproduced on CPU during reconstruction. CPU and CUDA dither values matched bit-for-bit in a 4096-block check on the current runtime. Matching the recorded PyTorch version is required for a reproducibility claim because RNG behavior is runtime-version dependent. The parser also retains backward reading support for NQRT v1 tensor-dither files.

This is a self-contained **single-tensor serialization**, not a full model checkpoint. It has no model-wide tensor index, tied-weight handling, architecture/config file, tokenizer, optimizer state, or inference kernel. Its size must not be presented as a model-level compression ratio.

## NQMC v1 model container

`PackedNestQuantModel` stores one model state dict as a manifest and payload. Each unique tensor is either a dense record (raw bytes plus shape/dtype) or an NQRT record; a JSON manifest stores names, offsets, lengths, model metadata, alias groups, and a SHA-256 payload checksum. `from_state_dict()` detects exact tied tensors using storage pointer, offset, shape, stride, and dtype. If any alias is quantized, one NQRT record represents the group; `state_dict()` restores all names pointing at the same tensor object. Dense entries support bool, integer types, float16, bfloat16, float32, and float64.

```python
checkpoint = PackedNestQuantModel.from_state_dict(
    model.state_dict(),
    {"model.layers.0.self_attn.q_proj.weight": packed_q_proj},
    model_metadata={"model_id": "example/base-model"},
)
blob = checkpoint.to_bytes()
loaded = PackedNestQuantModel.from_bytes(blob)
restored_state = loaded.state_dict()
model.load_state_dict(restored_state)
```

NQMC v1 currently requires a little-endian host and restores quantized tensors by dequantizing them to their recorded floating dtype. The WikiText-2 evaluator can export NQMC and evaluate PPL after reloading it, or evaluate a previously exported file with `--load-checkpoint`. `model_metadata.encoder` distinguishes independent E8-block encoding from block-LDLQ encoding; LDLQ also records the calibration protocol and Hessian artifact hashes. Hessian is an encoder-only input and is not needed to decode the stored final codes. This is a serialization container, not an integer matmul format or speedup implementation. Real Llama-3.2-1B model-level results are recorded for [ordinary W4](../results/llama32_1b_nqmc_w4_full.md) and [train64 block-LDLQ W4](../results/llama32_1b_w4_ldlq_nqmc.md); low-bit inference performance remains unmeasured.

模型导出使用 `PackedNestQuantModel.write_to_file()` 逐记录写入临时 payload、增量计算 SHA-256，再以 8 MiB 分块组成临时目标文件并原子替换正式路径，因此不需要额外拼接一个完整 NQMC `bytes` 副本。模型回载使用 `NQMCFileReader` 分块校验 payload SHA-256，并按记录读取、验证和反量化。写入模型前先检查完整记录集合、state-dict 名称、shape、dtype 与 alias，因此格式错误不会留下半加载模型。该路径降低的是导出/加载期间的主机瞬时副本：最终模型仍以浮点权重常驻并执行。`scripts/benchmark_nqmc_stream_load.py` 只测检查点回载耗时和峰值 RSS，不运行 PPL，也不能作为推理加速证据。

`scripts/inspect_nqmc.py` 在同一个 seek-based reader 上提供只读审计。默认验证 manifest、entry bounds 和 payload checksum；`--records` 逐记录聚合 NQRT version、codec semantics、q、beta count 与 dither 模式，`--file-sha256` 额外计算完整文件哈希。该路径不加载模型或执行反量化，详见 [NQMC inspection](nqmc_inspection.md)。

## API and checks

```python
packed = pack_nestquant_indices(
    quantized.codes,
    quantized.beta_indices,
    q=quantized.q,
    beta_count=len(quantized.betas),
)
blob = packed.to_bytes()
restored = PackedNestQuantIndices.from_bytes(blob)
codes, beta_indices = restored.unpack()

tensor_file = PackedNestQuantTensor.from_quantized_rows("layer.weight", quantized)
blob = tensor_file.to_bytes()
restored_tensor = PackedNestQuantTensor.from_bytes(blob)
weight_hat = restored_tensor.dequantize()
```

Packing rejects out-of-range residues/indices and shape mismatches. Parsing checks the magic, version, shape, declared payload sizes, and truncation/trailing bytes; `unpack()` checks residue/index ranges and nonzero padding bits. Unit tests cover byte-level packing at all supported widths, tensor reconstruction with and without dither, dense model tensors, tied aliases, checksums, and malformed input. Model-level file size and PPL are measured separately; these tests do not establish low-bit inference performance.
