"""Portable, versioned serialization of one fake-quantized NestQuant tensor.

This stores enough data to reconstruct a floating-point tensor. It is not a
low-bit inference format: loading dequantizes codes into the recorded dtype.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import struct

import torch

from .codec import (
    CURRENT_CODEC_SEMANTICS,
    LEGACY_CODEC_SEMANTICS,
    UPSTREAM_CODEC_SEMANTICS,
    generate_e8_dither,
    generate_e8_dither_stateless,
    STATELESS_DITHER_ALGORITHM,
)
from .packing import PackedNestQuantIndices, pack_nestquant_indices
from .quantizer import QuantizedRows


_HEADER_V1 = struct.Struct("<4sBIQQQ")
_HEADER_V2 = struct.Struct("<4sBBIQQQ")
_MAGIC = b"NQRT"
_VERSION = 4
_DTYPE_TO_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


@dataclass(frozen=True)
class PackedNestQuantTensor:
    """Self-contained serialized data for one quantized matrix tensor."""

    tensor_name: str
    indices: PackedNestQuantIndices
    row_norms: torch.Tensor
    betas: tuple[float, ...]
    feature_count: int
    output_dtype: torch.dtype
    dither: torch.Tensor | None = None
    dither_generator_state: bytes | None = None
    dither_generator_device: str | None = None
    dither_torch_version: str | None = None
    dither_seed: int | None = None
    dither_algorithm: str | None = None
    codec_semantics: str = CURRENT_CODEC_SEMANTICS

    @classmethod
    def from_quantized_rows(
        cls,
        tensor_name: str,
        quantized: QuantizedRows,
        *,
        dither_generator_state: bytes | None = None,
        dither_generator_device: str = "cpu",
        dither_torch_version: str | None = None,
        dither_seed: int | None = None,
        dither_algorithm: str = STATELESS_DITHER_ALGORITHM,
        verify_generator_state: bool = True,
    ) -> "PackedNestQuantTensor":
        if not tensor_name or not isinstance(tensor_name, str):
            raise ValueError("tensor_name must be a nonempty string")
        if dither_generator_state is not None and dither_seed is not None:
            raise ValueError("choose generator-state or stateless dither, not both")
        stored_dither = None if dither_generator_state is not None or dither_seed is not None else (
            None if quantized.dither is None else quantized.dither.detach().cpu().contiguous()
        )
        if dither_generator_state is not None:
            if quantized.dither is not None and verify_generator_state:
                regenerated = _generate_dither_from_state(
                    dither_generator_state, quantized.codes.shape,
                    device=dither_generator_device, dtype=quantized.dither.dtype,
                    output_device=quantized.dither.device,
                    codec_semantics=quantized.codec_semantics,
                )
                if not torch.equal(regenerated.cpu(), quantized.dither.detach().cpu()):
                    raise ValueError("generator state does not reproduce the quantizer dither tensor")
        if dither_seed is not None:
            _validate_stateless_dither(dither_seed, dither_algorithm)
            if quantized.dither is not None and verify_generator_state:
                regenerated = _generate_dither_from_seed(
                    dither_seed, quantized.codes.shape, dtype=quantized.dither.dtype,
                    output_device=quantized.dither.device,
                    codec_semantics=quantized.codec_semantics,
                    algorithm=dither_algorithm,
                )
                if not torch.equal(regenerated, quantized.dither):
                    raise ValueError("stateless seed does not reproduce the quantizer dither tensor")
        return cls(
            tensor_name=tensor_name,
            indices=pack_nestquant_indices(
                quantized.codes, quantized.beta_indices,
                q=quantized.q, beta_count=len(quantized.betas),
            ),
            row_norms=quantized.row_norms.detach().cpu().contiguous(),
            betas=tuple(float(value) for value in quantized.betas),
            feature_count=int(quantized.feature_count),
            output_dtype=quantized.output_dtype,
            dither=stored_dither,
            dither_generator_state=dither_generator_state,
            dither_generator_device=dither_generator_device if dither_generator_state is not None else None,
            dither_torch_version=(dither_torch_version or str(torch.__version__))
            if dither_generator_state is not None else None,
            dither_seed=dither_seed,
            dither_algorithm=dither_algorithm if dither_seed is not None else None,
            codec_semantics=quantized.codec_semantics,
        )

    @property
    def serialized_bytes(self) -> int:
        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        _validate_metadata(self.tensor_name, self.indices, self.row_norms,
                           self.betas, self.feature_count, self.output_dtype, self.dither)
        dither_sources = sum(value is not None for value in (
            self.dither, self.dither_generator_state, self.dither_seed,
        ))
        if dither_sources > 1:
            raise ValueError("store exactly one dither representation")
        if self.codec_semantics not in {UPSTREAM_CODEC_SEMANTICS, LEGACY_CODEC_SEMANTICS}:
            raise ValueError("unsupported tensor checkpoint codec semantics")
        if self.dither_generator_state is not None:
            _validate_generator_state(self.dither_generator_state, self.dither_generator_device)
        if self.dither_seed is not None:
            _validate_stateless_dither(self.dither_seed, self.dither_algorithm)
        if self.dither is not None:
            dither_mode = 1
        elif self.dither_generator_state is not None:
            dither_mode = 2
        elif self.dither_seed is not None:
            dither_mode = 3
        else:
            dither_mode = 0
        metadata = json.dumps({
            "tensor_name": self.tensor_name,
            "q": self.indices.q,
            "beta_count": self.indices.beta_count,
            "betas": list(self.betas),
            "rows": self.indices.shape[0],
            "feature_count": self.feature_count,
            "output_dtype": _DTYPE_TO_NAME[self.output_dtype],
            "dither_generator_device": self.dither_generator_device,
            "dither_torch_version": self.dither_torch_version,
            "dither_seed": self.dither_seed,
            "dither_algorithm": self.dither_algorithm,
            "codec_semantics": self.codec_semantics,
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        indices_blob = self.indices.to_bytes()
        norms_blob = _tensor_to_bytes(self.row_norms)
        dither_blob = (self.dither_generator_state if dither_mode == 2 else
                       b"" if self.dither is None else _tensor_to_bytes(self.dither))
        header = _HEADER_V2.pack(_MAGIC, _VERSION, dither_mode, len(metadata),
                                 len(indices_blob), len(norms_blob), len(dither_blob))
        return header + metadata + indices_blob + norms_blob + dither_blob

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedNestQuantTensor":
        view = memoryview(data)
        if len(view) < 5:
            raise ValueError("input is shorter than the tensor checkpoint header")
        magic = bytes(view[:4])
        version = int(view[4])
        if magic != _MAGIC:
            raise ValueError("invalid tensor checkpoint magic")
        if version == 1:
            if len(view) < _HEADER_V1.size:
                raise ValueError("input is shorter than the v1 tensor checkpoint header")
            _, _, metadata_len, indices_len, norms_len, dither_len = _HEADER_V1.unpack_from(view)
            dither_mode = 1 if dither_len else 0
            offset = _HEADER_V1.size
        elif version in (2, 3, 4):
            if len(view) < _HEADER_V2.size:
                raise ValueError("input is shorter than the v2 tensor checkpoint header")
            _, _, dither_mode, metadata_len, indices_len, norms_len, dither_len = _HEADER_V2.unpack_from(view)
            if dither_mode not in (0, 1, 2, 3):
                raise ValueError("invalid dither storage mode")
            offset = _HEADER_V2.size
        else:
            raise ValueError(f"unsupported tensor checkpoint version: {version}")
        lengths = (metadata_len, indices_len, norms_len, dither_len)
        if offset + sum(lengths) != len(view):
            raise ValueError("tensor checkpoint payload lengths do not match file size")
        try:
            metadata = json.loads(bytes(view[offset:offset + metadata_len]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid tensor checkpoint metadata") from error
        offset += metadata_len
        indices = PackedNestQuantIndices.from_bytes(view[offset:offset + indices_len])
        offset += indices_len
        norms_blob = bytes(view[offset:offset + norms_len])
        offset += norms_len
        dither_blob = bytes(view[offset:offset + dither_len])

        try:
            dtype = _NAME_TO_DTYPE[metadata["output_dtype"]]
            rows = int(metadata["rows"])
            features = int(metadata["feature_count"])
            name = metadata["tensor_name"]
            betas = tuple(float(value) for value in metadata["betas"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("missing or invalid tensor checkpoint metadata fields") from error
        if not isinstance(name, str):
            raise ValueError("invalid tensor name")
        if version == 1 and metadata.get("has_dither") != bool(dither_mode):
            raise ValueError("v1 dither flag disagrees with payload")
        if metadata.get("q") != indices.q or metadata.get("beta_count") != indices.beta_count:
            raise ValueError("metadata disagrees with packed index parameters")
        if indices.shape != (rows, features // 8, 8) or features <= 0 or features % 8:
            raise ValueError("metadata feature shape disagrees with packed indices")
        if len(norms_blob) != rows * torch.empty((), dtype=dtype).element_size():
            raise ValueError("row norm payload size disagrees with metadata")
        if dither_mode == 1:
            expected_dither = math.prod(indices.shape) * torch.empty((), dtype=dtype).element_size()
            if len(dither_blob) != expected_dither:
                raise ValueError("dither payload size disagrees with metadata")
        elif dither_mode in (0, 3) and dither_blob:
            raise ValueError("dither payload exists for a payload-free mode")
        norms = _tensor_from_bytes(norms_blob, (rows,), dtype)
        dither = _tensor_from_bytes(dither_blob, indices.shape, dtype) if dither_mode == 1 else None
        generator_state = dither_blob if dither_mode == 2 else None
        generator_device = metadata.get("dither_generator_device") if generator_state is not None else None
        torch_version = metadata.get("dither_torch_version") if generator_state is not None else None
        dither_seed = metadata.get("dither_seed") if dither_mode == 3 else None
        dither_algorithm = metadata.get("dither_algorithm") if dither_mode == 3 else None
        codec_semantics = (
            metadata.get("codec_semantics") if version >= 3 else LEGACY_CODEC_SEMANTICS
        )
        if codec_semantics not in {UPSTREAM_CODEC_SEMANTICS, LEGACY_CODEC_SEMANTICS}:
            raise ValueError("unsupported tensor checkpoint codec semantics")
        if generator_state is not None:
            _validate_generator_state(generator_state, generator_device)
        if dither_mode == 3:
            if version < 4:
                raise ValueError("stateless dither requires NQRT v4")
            _validate_stateless_dither(dither_seed, dither_algorithm)
        _validate_metadata(name, indices, norms, betas, features, dtype, dither)
        return cls(
            tensor_name=name, indices=indices, row_norms=norms, betas=betas,
            feature_count=features, output_dtype=dtype, dither=dither,
            dither_generator_state=generator_state,
            dither_generator_device=generator_device,
            dither_torch_version=torch_version, dither_seed=dither_seed,
            dither_algorithm=dither_algorithm, codec_semantics=codec_semantics,
        )

    def to_quantized_rows(self, *, device: torch.device | str = "cpu") -> QuantizedRows:
        codes, beta_indices = self.indices.unpack(device=device)
        dither = None if self.dither is None else self.dither.to(device)
        if self.dither_generator_state is not None:
            dither = _generate_dither_from_state(
                self.dither_generator_state, self.indices.shape,
                device=self.dither_generator_device, dtype=self.output_dtype,
                output_device="cpu", codec_semantics=self.codec_semantics,
            ).to(device)
        elif self.dither_seed is not None:
            dither = _generate_dither_from_seed(
                self.dither_seed, self.indices.shape, dtype=self.output_dtype,
                output_device=device, codec_semantics=self.codec_semantics,
                algorithm=self.dither_algorithm,
            )
        return QuantizedRows(
            codes=codes,
            beta_indices=beta_indices,
            dither=dither,
            row_norms=self.row_norms.to(device),
            q=self.indices.q,
            betas=self.betas,
            feature_count=self.feature_count,
            output_dtype=self.output_dtype,
            codec_semantics=self.codec_semantics,
        )

    def dequantize(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return self.to_quantized_rows(device=device).dequantize()

    def dequantize_chunked(
        self,
        *,
        device: torch.device | str = "cpu",
        output_dtype: torch.dtype | None = None,
        rows_per_chunk: int = 128,
    ) -> torch.Tensor:
        """Decode row chunks and immediately store them in the target dtype."""
        if isinstance(rows_per_chunk, bool) or not isinstance(rows_per_chunk, int) or rows_per_chunk < 1:
            raise ValueError("rows_per_chunk must be a positive integer")
        target_dtype = self.output_dtype if output_dtype is None else output_dtype
        if target_dtype not in _DTYPE_TO_NAME:
            raise ValueError("output_dtype must be float16, bfloat16, float32, or float64")
        target_device = torch.device(device)
        rows = self.indices.shape[0]
        output = torch.empty(
            (rows, self.feature_count), device=target_device, dtype=target_dtype
        )
        generated_dither = None
        if self.dither_generator_state is not None:
            # A single draw with the original full shape is required for exact
            # portable replay. CPU RNG output is not guaranteed to be invariant
            # when one large torch.rand call is split into smaller calls.
            generated_dither = _generate_dither_from_state(
                self.dither_generator_state, self.indices.shape,
                device=self.dither_generator_device, dtype=self.output_dtype,
                output_device="cpu", codec_semantics=self.codec_semantics,
            )
        for start in range(0, rows, rows_per_chunk):
            end = min(start + rows_per_chunk, rows)
            codes, beta_indices = self.indices.unpack_rows(
                start, end, device=target_device
            )
            dither = None
            if self.dither is not None:
                dither = self.dither[start:end].to(target_device)
            elif generated_dither is not None:
                dither = generated_dither[start:end].to(target_device)
            elif self.dither_seed is not None:
                blocks_per_row = self.indices.shape[1]
                dither = generate_e8_dither_stateless(
                    (end - start) * blocks_per_row,
                    self.dither_seed,
                    block_offset=start * blocks_per_row,
                    device=target_device,
                    dtype=self.output_dtype,
                    codec_semantics=self.codec_semantics,
                    algorithm=self.dither_algorithm,
                ).reshape(end - start, blocks_per_row, 8)
            chunk = QuantizedRows(
                codes=codes,
                beta_indices=beta_indices,
                dither=dither,
                row_norms=self.row_norms[start:end].to(target_device),
                q=self.indices.q,
                betas=self.betas,
                feature_count=self.feature_count,
                output_dtype=self.output_dtype,
                codec_semantics=self.codec_semantics,
            ).dequantize()
            output[start:end].copy_(chunk.to(target_dtype))
        return output


def _validate_metadata(tensor_name, indices, row_norms, betas, feature_count, output_dtype, dither):
    if not isinstance(tensor_name, str) or not tensor_name:
        raise ValueError("tensor_name must be a nonempty string")
    if output_dtype not in _DTYPE_TO_NAME:
        raise ValueError("output_dtype must be float16, bfloat16, float32, or float64")
    if feature_count <= 0 or feature_count % 8 or indices.shape != (
            indices.shape[0], feature_count // 8, 8):
        raise ValueError("feature_count and packed code shape are inconsistent")
    if row_norms.shape != (indices.shape[0],) or row_norms.dtype != output_dtype:
        raise ValueError("row_norms must have shape [rows] and match output_dtype")
    if not torch.isfinite(row_norms).all().item() or (row_norms < 0).any().item():
        raise ValueError("row_norms must be finite and nonnegative")
    if len(betas) != indices.beta_count or any(not math.isfinite(value) or value <= 0 for value in betas):
        raise ValueError("beta table must contain beta_count positive finite values")
    if dither is not None:
        if dither.shape != indices.shape or dither.dtype != output_dtype:
            raise ValueError("dither must match code shape and output_dtype")
        if not torch.isfinite(dither).all().item():
            raise ValueError("dither must contain only finite values")


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    byte_view = tensor.detach().cpu().contiguous().view(torch.uint8)
    return byte_view.numpy().tobytes(order="C")


def _tensor_from_bytes(blob: bytes, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    byte_tensor = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
    return byte_tensor.view(dtype).reshape(shape)


def _validate_generator_state(state: bytes, device: str | None) -> None:
    if not isinstance(state, bytes) or not state:
        raise ValueError("dither generator state must be nonempty bytes")
    if device != "cpu":
        raise ValueError("NQRT v2/v3 supports portable CPU dither generator states only")
    generator = torch.Generator(device="cpu")
    try:
        generator.set_state(torch.frombuffer(bytearray(state), dtype=torch.uint8).clone())
    except RuntimeError as error:
        raise ValueError("invalid CPU dither generator state") from error


def _validate_stateless_dither(seed: int, algorithm: str | None) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("stateless dither seed must be an integer in [0, 2**31)")
    if algorithm != STATELESS_DITHER_ALGORITHM:
        raise ValueError("unsupported stateless dither algorithm")


def _generate_dither_from_state(
    state: bytes, shape: tuple[int, ...], *, device: str | None,
    dtype: torch.dtype, output_device: torch.device | str = "cpu",
    codec_semantics: str = CURRENT_CODEC_SEMANTICS,
) -> torch.Tensor:
    _validate_generator_state(state, device)
    generator = torch.Generator(device="cpu")
    generator.set_state(torch.frombuffer(bytearray(state), dtype=torch.uint8).clone())
    count = math.prod(shape[:-1])
    return generate_e8_dither(
        count, generator, device=output_device, dtype=dtype,
        codec_semantics=codec_semantics,
    ).reshape(shape)


def _generate_dither_from_seed(
    seed: int, shape: tuple[int, ...], *, dtype: torch.dtype,
    output_device: torch.device | str,
    codec_semantics: str = CURRENT_CODEC_SEMANTICS,
    algorithm: str | None = STATELESS_DITHER_ALGORITHM,
) -> torch.Tensor:
    _validate_stateless_dither(seed, algorithm)
    return generate_e8_dither_stateless(
        math.prod(shape[:-1]), seed, device=output_device, dtype=dtype,
        codec_semantics=codec_semantics, algorithm=algorithm,
    ).reshape(shape)
