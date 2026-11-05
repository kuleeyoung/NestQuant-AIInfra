"""Versioned packed NestQuant KV-cache pages.

NQKV v1 is the CPU reference contract for future CUDA/Triton page writers and
attention readers.  A page has fixed physical capacity and NHD logical layout
(``[token, kv_head, head_dim]``).  Keys and values use the same paper-style
row normalization and E8 codec as :func:`quantize_rows`, but their residue
codes and beta IDs are stored in fixed-width byte streams rather than floating
reconstruction tensors.

Version 1 deliberately keeps row norms in float32.  This matches the existing
KV fake-quant oracle before introducing a separate FP16-norm quality ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Sequence

import torch

from .codec import CURRENT_CODEC_SEMANTICS, UPSTREAM_CODEC_SEMANTICS
from .packing import PackedNestQuantIndices, pack_nestquant_indices
from .quantizer import QuantizedRows, quantize_rows


_MAGIC = b"NQKV"
_VERSION_V1 = 1
_VERSION_V2 = 2
_LAYOUT_NHD = 1
_NORM_FP32 = 1
_CODEC_UPSTREAM = 1
_CHECKSUM_BYTES = 32
_HEADER_V1 = struct.Struct("<4sBBBBHHHHHHHIIII")
_HEADER_V2 = struct.Struct("<4sBBBBHHHHHHHHIIII")

_DTYPE_TO_ID = {
    torch.bfloat16: 1,
    torch.float16: 2,
    torch.float32: 3,
    torch.float64: 4,
}
_ID_TO_DTYPE = {value: key for key, value in _DTYPE_TO_ID.items()}


@dataclass(frozen=True)
class PackedNQKVPage:
    """One fixed-capacity packed Key/Value page in NHD layout.

    ``key_indices`` and ``value_indices`` always cover the full physical page,
    including zero-padded rows after ``valid_tokens``.  This makes the payload
    size independent of the current fill level and suitable for a page pool.
    """

    page_size: int
    valid_tokens: int
    num_kv_heads: int
    head_dim: int
    q: int
    betas: tuple[float, ...]
    output_dtype: torch.dtype
    key_indices: PackedNestQuantIndices
    key_norms: torch.Tensor
    value_indices: PackedNestQuantIndices
    value_norms: torch.Tensor
    codec_semantics: str = CURRENT_CODEC_SEMANTICS
    value_betas: tuple[float, ...] | None = None

    @classmethod
    def from_tensors(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_size: int,
        q: int,
        betas: Sequence[float] | None = None,
        key_betas: Sequence[float] | None = None,
        value_betas: Sequence[float] | None = None,
    ) -> "PackedNQKVPage":
        """Quantize an NHD Key/Value prefix into a fixed-capacity page."""
        _validate_input_tensors(key, value, page_size)
        if betas is not None:
            if key_betas is not None or value_betas is not None:
                raise ValueError("pass legacy betas or key_betas/value_betas, not both")
            key_beta_values = _validate_betas(betas)
            value_beta_values = key_beta_values
            serialized_value_betas = None
        else:
            if key_betas is None or value_betas is None:
                raise ValueError("v2 requires both key_betas and value_betas")
            key_beta_values = _validate_betas(key_betas)
            value_beta_values = _validate_betas(value_betas)
            serialized_value_betas = value_beta_values
        valid_tokens, num_heads, head_dim = key.shape

        # The established KV oracle promotes BF16/FP16 rows to float32 before
        # quantization.  Keep that rule here so packed-vs-fake comparisons do
        # not accidentally measure an arithmetic-dtype change.
        key_rows = torch.zeros(
            (page_size * num_heads, head_dim), device=key.device, dtype=torch.float32
        )
        value_rows = torch.zeros_like(key_rows)
        row_count = valid_tokens * num_heads
        key_rows[:row_count].copy_(key.reshape(row_count, head_dim).float())
        value_rows[:row_count].copy_(value.reshape(row_count, head_dim).float())

        quantized_key = quantize_rows(key_rows, q=q, betas=key_beta_values)
        quantized_value = quantize_rows(value_rows, q=q, betas=value_beta_values)
        return cls(
            page_size=page_size,
            valid_tokens=valid_tokens,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q=q,
            betas=key_beta_values,
            output_dtype=key.dtype,
            key_indices=pack_nestquant_indices(
                quantized_key.codes, quantized_key.beta_indices,
                q=q, beta_count=len(key_beta_values),
            ),
            key_norms=quantized_key.row_norms.detach().cpu().to(torch.float32).contiguous(),
            value_indices=pack_nestquant_indices(
                quantized_value.codes, quantized_value.beta_indices,
                q=q, beta_count=len(value_beta_values),
            ),
            value_norms=quantized_value.row_norms.detach().cpu().to(torch.float32).contiguous(),
            value_betas=serialized_value_betas,
        )

    @property
    def format_version(self) -> int:
        return _VERSION_V2 if self.value_betas is not None else _VERSION_V1

    @property
    def resolved_value_betas(self) -> tuple[float, ...]:
        return self.betas if self.value_betas is None else self.value_betas

    @property
    def physical_rows(self) -> int:
        return self.page_size * self.num_kv_heads

    @property
    def payload_bytes(self) -> int:
        """Bytes resident in the logical page payload, excluding container metadata."""
        return (
            len(self.key_indices.code_payload) + len(self.key_indices.beta_payload)
            + self.key_norms.numel() * self.key_norms.element_size()
            + len(self.value_indices.code_payload) + len(self.value_indices.beta_payload)
            + self.value_norms.numel() * self.value_norms.element_size()
        )

    @property
    def serialized_bytes(self) -> int:
        return len(self.to_bytes())

    def storage_report(self) -> dict[str, int | float | str]:
        """Return measured payload accounting for this physical page."""
        entries = 2 * self.page_size * self.num_kv_heads * self.head_dim
        code_bytes = len(self.key_indices.code_payload) + len(self.value_indices.code_payload)
        beta_bytes = len(self.key_indices.beta_payload) + len(self.value_indices.beta_payload)
        norm_bytes = (
            self.key_norms.numel() * self.key_norms.element_size()
            + self.value_norms.numel() * self.value_norms.element_size()
        )
        return {
            "format": f"NQKV-v{self.format_version}",
            "layout": "NHD",
            "page_size": self.page_size,
            "valid_tokens": self.valid_tokens,
            "physical_entries": entries,
            "code_bytes": code_bytes,
            "beta_bytes": beta_bytes,
            "norm_bytes": norm_bytes,
            "payload_bytes": self.payload_bytes,
            "container_bytes": self.serialized_bytes,
            "payload_bits_per_entry": self.payload_bytes * 8 / entries,
        }

    def dequantize(
        self, *, device: torch.device | str = "cpu",
        output_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the valid NHD Key/Value prefix from packed streams."""
        self._validate()
        target = torch.device(device)
        dtype = self.output_dtype if output_dtype is None else output_dtype
        if dtype not in _DTYPE_TO_ID:
            raise TypeError("output_dtype must be bfloat16, float16, float32, or float64")
        key = self._dequantize_one(
            self.key_indices, self.key_norms, self.betas, target, dtype
        )
        value = self._dequantize_one(
            self.value_indices, self.value_norms, self.resolved_value_betas, target, dtype
        )
        shape = (self.page_size, self.num_kv_heads, self.head_dim)
        return key.reshape(shape)[:self.valid_tokens], value.reshape(shape)[:self.valid_tokens]

    def _dequantize_one(
        self,
        indices: PackedNestQuantIndices,
        norms: torch.Tensor,
        betas: tuple[float, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        codes, beta_indices = indices.unpack(device=device)
        quantized = QuantizedRows(
            codes=codes,
            beta_indices=beta_indices,
            dither=None,
            row_norms=norms.to(device=device),
            q=self.q,
            betas=betas,
            feature_count=self.head_dim,
            output_dtype=torch.float32,
            codec_semantics=self.codec_semantics,
        )
        return quantized.dequantize().to(dtype=dtype)

    def to_bytes(self) -> bytes:
        """Serialize the page with a SHA-256 checksum over header and payload."""
        self._validate()
        key_beta_blob = struct.pack("<" + "f" * len(self.betas), *self.betas)
        value_beta_values = self.resolved_value_betas
        value_beta_blob = (
            b"" if self.format_version == _VERSION_V1 else
            struct.pack("<" + "f" * len(value_beta_values), *value_beta_values)
        )
        key_indices_blob = self.key_indices.to_bytes()
        value_indices_blob = self.value_indices.to_bytes()
        key_norms_blob = _float32_tensor_bytes(self.key_norms)
        value_norms_blob = _float32_tensor_bytes(self.value_norms)
        common = (
            _MAGIC, self.format_version, _LAYOUT_NHD, _NORM_FP32,
            _DTYPE_TO_ID[self.output_dtype], _CODEC_UPSTREAM,
            self.page_size, self.valid_tokens, self.num_kv_heads, self.head_dim,
            self.q, len(self.betas),
        )
        lengths = (
            len(key_indices_blob), len(key_norms_blob),
            len(value_indices_blob), len(value_norms_blob),
        )
        header = (
            _HEADER_V1.pack(*common, *lengths)
            if self.format_version == _VERSION_V1
            else _HEADER_V2.pack(*common, len(value_beta_values), *lengths)
        )
        body = (
            header + key_beta_blob + value_beta_blob + key_indices_blob
            + key_norms_blob + value_indices_blob + value_norms_blob
        )
        return body + hashlib.sha256(body).digest()

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedNQKVPage":
        view = memoryview(data)
        if len(view) < _HEADER_V1.size + _CHECKSUM_BYTES:
            raise ValueError("input is shorter than the NQKV header and checksum")
        magic, version = struct.unpack_from("<4sB", view)
        if magic != _MAGIC:
            raise ValueError("invalid NQKV magic")
        if version == _VERSION_V1:
            header = _HEADER_V1
            fields = header.unpack_from(view)
            (magic, version, layout, norm_dtype, output_dtype_id, codec_id,
             page_size, valid_tokens, num_heads, head_dim, q, key_beta_count,
             key_indices_len, key_norms_len, value_indices_len, value_norms_len) = fields
            value_beta_count = key_beta_count
        elif version == _VERSION_V2:
            header = _HEADER_V2
            fields = header.unpack_from(view)
            (magic, version, layout, norm_dtype, output_dtype_id, codec_id,
             page_size, valid_tokens, num_heads, head_dim, q, key_beta_count,
             value_beta_count, key_indices_len, key_norms_len,
             value_indices_len, value_norms_len) = fields
        else:
            raise ValueError(f"unsupported NQKV version: {version}")
        if layout != _LAYOUT_NHD:
            raise ValueError("unsupported NQKV layout")
        if norm_dtype != _NORM_FP32:
            raise ValueError("unsupported NQKV norm dtype")
        if codec_id != _CODEC_UPSTREAM:
            raise ValueError("unsupported NQKV codec semantics")
        if output_dtype_id not in _ID_TO_DTYPE:
            raise ValueError("unsupported NQKV output dtype")
        key_beta_len = key_beta_count * 4
        value_beta_len = 0 if version == _VERSION_V1 else value_beta_count * 4
        expected = (
            header.size + key_beta_len + value_beta_len + key_indices_len + key_norms_len
            + value_indices_len + value_norms_len + _CHECKSUM_BYTES
        )
        if len(view) != expected:
            raise ValueError("NQKV payload lengths do not match the container size")
        body = bytes(view[:-_CHECKSUM_BYTES])
        if hashlib.sha256(body).digest() != bytes(view[-_CHECKSUM_BYTES:]):
            raise ValueError("NQKV checksum mismatch")

        offset = header.size
        key_betas = struct.unpack_from("<" + "f" * key_beta_count, view, offset)
        offset += key_beta_len
        parsed_value_betas = None
        if version == _VERSION_V2:
            parsed_value_betas = struct.unpack_from(
                "<" + "f" * value_beta_count, view, offset
            )
            offset += value_beta_len
        key_indices = PackedNestQuantIndices.from_bytes(view[offset:offset + key_indices_len])
        offset += key_indices_len
        key_norms = _float32_tensor_from_bytes(bytes(view[offset:offset + key_norms_len]))
        offset += key_norms_len
        value_indices = PackedNestQuantIndices.from_bytes(view[offset:offset + value_indices_len])
        offset += value_indices_len
        value_norms = _float32_tensor_from_bytes(bytes(view[offset:offset + value_norms_len]))
        page = cls(
            page_size=page_size,
            valid_tokens=valid_tokens,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            q=q,
            betas=tuple(float(beta) for beta in key_betas),
            output_dtype=_ID_TO_DTYPE[output_dtype_id],
            key_indices=key_indices,
            key_norms=key_norms,
            value_indices=value_indices,
            value_norms=value_norms,
            codec_semantics=UPSTREAM_CODEC_SEMANTICS,
            value_betas=(
                None if parsed_value_betas is None
                else tuple(float(beta) for beta in parsed_value_betas)
            ),
        )
        page._validate()
        return page

    def _validate(self) -> None:
        if self.codec_semantics != UPSTREAM_CODEC_SEMANTICS:
            raise ValueError("NQKV requires upstream E8 codec semantics")
        if self.output_dtype not in _DTYPE_TO_ID:
            raise TypeError("unsupported NQKV output dtype")
        if not 1 <= self.valid_tokens <= self.page_size <= 65535:
            raise ValueError("valid_tokens must be in [1, page_size]")
        if not 1 <= self.num_kv_heads <= 65535:
            raise ValueError("num_kv_heads must be in [1, 65535]")
        if not 1 <= self.head_dim <= 65535 or self.head_dim % 8:
            raise ValueError("head_dim must be positive and divisible by 8")
        key_beta_values = _validate_betas(self.betas)
        value_beta_values = _validate_betas(self.resolved_value_betas)
        if len(key_beta_values) > 65535 or len(value_beta_values) > 65535:
            raise ValueError("NQKV supports at most 65535 beta values per table")
        expected_shape = (self.physical_rows, self.head_dim // 8, 8)
        expected_norms = (self.physical_rows,)
        for name, indices, norms, beta_values in (
            ("key", self.key_indices, self.key_norms, key_beta_values),
            ("value", self.value_indices, self.value_norms, value_beta_values),
        ):
            if indices.shape != expected_shape:
                raise ValueError(f"{name} packed shape disagrees with page metadata")
            if indices.q != self.q or indices.beta_count != len(beta_values):
                raise ValueError(f"{name} packing parameters disagree with page metadata")
            if norms.shape != expected_norms or norms.dtype != torch.float32 or norms.device.type != "cpu":
                raise ValueError(f"{name} norms must be a contiguous CPU float32 vector")
            if not norms.is_contiguous() or not torch.isfinite(norms).all().item():
                raise ValueError(f"{name} norms must be contiguous and finite")


def _validate_input_tensors(key: torch.Tensor, value: torch.Tensor, page_size: int) -> None:
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError("key and value must be torch tensors")
    if not key.is_floating_point() or not value.is_floating_point():
        raise TypeError("key and value must be floating-point tensors")
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("key and value must share NHD shape [tokens, kv_heads, head_dim]")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError("key and value must share device and dtype")
    if key.dtype not in _DTYPE_TO_ID:
        raise TypeError("key/value dtype must be bfloat16, float16, float32, or float64")
    if not 1 <= key.shape[0] <= page_size:
        raise ValueError("token count must be in [1, page_size]")
    if page_size > 65535:
        raise ValueError("page_size must be at most 65535")
    if key.shape[1] < 1 or key.shape[1] > 65535:
        raise ValueError("kv head count must be in [1, 65535]")
    if key.shape[2] < 1 or key.shape[2] > 65535 or key.shape[2] % 8:
        raise ValueError("head_dim must be positive and divisible by 8")
    if not torch.isfinite(key).all().item() or not torch.isfinite(value).all().item():
        raise ValueError("key and value must contain only finite values")


def _validate_betas(betas: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(float(beta) for beta in betas)
    except (TypeError, ValueError) as error:
        raise TypeError("betas must be a nonempty sequence of numbers") from error
    if not values or any(not math.isfinite(beta) or beta <= 0 for beta in values):
        raise ValueError("betas must contain positive finite values")
    return values


def _float32_tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _float32_tensor_from_bytes(data: bytes) -> torch.Tensor:
    # bytearray gives torch a writable owner and avoids non-writable-buffer warnings.
    return torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
