"""Versioned fixed-width packing for NestQuant residue and beta-index tensors.

This container stores integer code streams only. Row norms, selected beta
values, model tensor names, and dither generation metadata are external.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Sequence

import numpy as np
import torch


_HEADER = struct.Struct("<4sBBHHQQ")
_MAGIC = b"NQPK"
_VERSION = 1


@dataclass(frozen=True)
class PackedNestQuantIndices:
    """Packed residue and beta-index streams for one quantized tensor."""

    shape: tuple[int, ...]
    q: int
    beta_count: int
    code_payload: bytes
    beta_payload: bytes

    @property
    def code_bit_width(self) -> int:
        return _bits_for_cardinality(self.q)

    @property
    def beta_index_bit_width(self) -> int:
        return _bits_for_cardinality(self.beta_count)

    @property
    def header_bytes(self) -> int:
        return _HEADER.size + 8 * len(self.shape)

    @property
    def serialized_bytes(self) -> int:
        return self.header_bytes + len(self.code_payload) + len(self.beta_payload)

    def to_bytes(self) -> bytes:
        if len(self.shape) < 2 or len(self.shape) > 255 or any(dim < 1 for dim in self.shape) or self.shape[-1] != 8:
            raise ValueError("shape must contain 2 to 255 positive dimensions and end in 8")
        _validate_header_cardinalities(self.q, self.beta_count)
        expected_codes = _ceil_bytes(math.prod(self.shape), self.code_bit_width)
        expected_beta = _ceil_bytes(math.prod(self.shape[:-1]), self.beta_index_bit_width)
        if len(self.code_payload) != expected_codes or len(self.beta_payload) != expected_beta:
            raise ValueError("payload lengths do not match tensor shape and bit widths")
        header = _HEADER.pack(_MAGIC, _VERSION, len(self.shape), self.q,
                              self.beta_count, len(self.code_payload), len(self.beta_payload))
        dims = struct.pack("<" + "Q" * len(self.shape), *self.shape)
        return header + dims + self.code_payload + self.beta_payload

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedNestQuantIndices":
        view = memoryview(data)
        if len(view) < _HEADER.size:
            raise ValueError("input is shorter than the NestQuant packing header")
        magic, version, ndim, q, beta_count, code_len, beta_len = _HEADER.unpack_from(view)
        if magic != _MAGIC:
            raise ValueError("invalid NestQuant packing magic")
        if version != _VERSION:
            raise ValueError(f"unsupported NestQuant packing version: {version}")
        if ndim < 2:
            raise ValueError("packed code shape must include rows and 8D block dimension")
        _validate_header_cardinalities(q, beta_count)
        shape_end = _HEADER.size + 8 * ndim
        if len(view) < shape_end:
            raise ValueError("truncated tensor shape in packing header")
        shape = struct.unpack_from("<" + "Q" * ndim, view, _HEADER.size)
        if any(dim < 1 for dim in shape) or shape[-1] != 8:
            raise ValueError("shape must be positive and end with E8 dimension 8")
        code_width = _bits_for_cardinality(q)
        beta_width = _bits_for_cardinality(beta_count)
        if code_len != _ceil_bytes(math.prod(shape), code_width):
            raise ValueError("residue payload length disagrees with header shape")
        if beta_len != _ceil_bytes(math.prod(shape[:-1]), beta_width):
            raise ValueError("beta-index payload length disagrees with header shape")
        expected_size = shape_end + code_len + beta_len
        if len(view) != expected_size:
            raise ValueError("packed container has trailing or truncated payload bytes")
        code_start = shape_end
        beta_start = code_start + code_len
        return cls(tuple(shape), q, beta_count,
                   bytes(view[code_start:beta_start]), bytes(view[beta_start:]))

    def unpack(self, *, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        code_count = math.prod(self.shape)
        beta_shape = self.shape[:-1]
        beta_count = math.prod(beta_shape)
        codes = _unpack_unsigned(self.code_payload, code_count, self.code_bit_width)
        beta_indices = _unpack_unsigned(self.beta_payload, beta_count, self.beta_index_bit_width)
        if codes.numel() and int(codes.max()) >= self.q:
            raise ValueError("decoded residue is outside [0, q)")
        if beta_indices.numel() and int(beta_indices.max()) >= self.beta_count:
            raise ValueError("decoded beta index is outside [0, beta_count)")
        return (codes.reshape(self.shape).to(device),
                beta_indices.reshape(beta_shape).to(device))

    def unpack_rows(
        self, start: int, end: int, *, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpack a contiguous range along the leading row dimension."""
        rows = self.shape[0]
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise TypeError("row bounds must be integers")
        if start < 0 or end < start or end > rows:
            raise ValueError("row range is outside packed tensor bounds")
        code_values_per_row = math.prod(self.shape[1:])
        beta_values_per_row = math.prod(self.shape[1:-1])
        codes = _unpack_unsigned_range(
            self.code_payload, start * code_values_per_row, end * code_values_per_row,
            math.prod(self.shape), self.code_bit_width,
        )
        beta_indices = _unpack_unsigned_range(
            self.beta_payload, start * beta_values_per_row, end * beta_values_per_row,
            math.prod(self.shape[:-1]), self.beta_index_bit_width,
        )
        selected_rows = end - start
        code_shape = (selected_rows, *self.shape[1:])
        beta_shape = (selected_rows, *self.shape[1:-1])
        if codes.numel() and int(codes.max()) >= self.q:
            raise ValueError("decoded residue is outside [0, q)")
        if beta_indices.numel() and int(beta_indices.max()) >= self.beta_count:
            raise ValueError("decoded beta index is outside [0, beta_count)")
        return codes.reshape(code_shape).to(device), beta_indices.reshape(beta_shape).to(device)


def pack_nestquant_indices(
    codes: torch.Tensor,
    beta_indices: torch.Tensor,
    *,
    q: int,
    beta_count: int,
) -> PackedNestQuantIndices:
    """Pack E8 residue codes and selected-beta IDs into byte-aligned streams.

    Each code uses ``ceil(log2(q))`` bits and each beta ID uses
    ``ceil(log2(beta_count))`` bits. Supported widths are 1, 2, 4, or 8,
    which divide a byte and allow vectorized packing without bit spill logic.
    Bits are ordered least-significant first within each byte.
    """
    if not isinstance(codes, torch.Tensor) or codes.dtype not in (torch.int8, torch.uint8, torch.int16,
                                                                    torch.int32, torch.int64):
        raise TypeError("codes must be an integer torch.Tensor")
    if not isinstance(beta_indices, torch.Tensor) or beta_indices.dtype not in (
            torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise TypeError("beta_indices must be an integer torch.Tensor")
    if codes.ndim < 2 or codes.shape[-1] != 8:
        raise ValueError("codes must end in E8 dimension 8")
    if tuple(beta_indices.shape) != tuple(codes.shape[:-1]):
        raise ValueError("beta_indices shape must equal codes.shape[:-1]")
    if q < 2 or q > 256:
        raise ValueError("q must be in [2, 256]")
    if beta_count < 1 or beta_count > 256:
        raise ValueError("beta_count must be in [1, 256]")
    code_width = _bits_for_cardinality(q)
    beta_width = _bits_for_cardinality(beta_count)
    if code_width not in (1, 2, 4, 8) or beta_width not in (1, 2, 4, 8):
        raise ValueError("this format version supports only bit widths 1, 2, 4, or 8")
    code_payload = _pack_unsigned(codes, q, code_width)
    beta_payload = _pack_unsigned(beta_indices, beta_count, beta_width)
    return PackedNestQuantIndices(tuple(codes.shape), q, beta_count, code_payload, beta_payload)


def _bits_for_cardinality(cardinality: int) -> int:
    if cardinality < 1:
        raise ValueError("cardinality must be positive")
    return max(1, math.ceil(math.log2(cardinality)))


def _validate_header_cardinalities(q: int, beta_count: int) -> None:
    if q < 2 or q > 256 or beta_count < 1 or beta_count > 256:
        raise ValueError("q must be in [2, 256] and beta_count in [1, 256]")
    if _bits_for_cardinality(q) not in (1, 2, 4, 8) or _bits_for_cardinality(beta_count) not in (1, 2, 4, 8):
        raise ValueError("packing format supports only 1, 2, 4, or 8 bits per value")


def _ceil_bytes(value_count: int, bit_width: int) -> int:
    return (value_count * bit_width + 7) // 8


def _pack_unsigned(values: torch.Tensor, cardinality: int, width: int) -> bytes:
    source = values.detach()
    if source.numel() and (int(source.min().item()) < 0 or int(source.max().item()) >= cardinality):
        raise ValueError(f"all values must be in [0, {cardinality})")
    # Codes and beta IDs are at most 8 bits. Convert on the source device
    # before transfer so GPU serialization does not copy an int64 per value.
    flat = source.to(dtype=torch.uint8).to(device="cpu").contiguous().view(-1).numpy()
    per_byte = 8 // width
    if flat.size == 0:
        return b""
    # NumPy's uint8 strided operations avoid a very costly CPU int64 tensor
    # expansion on large matrices while preserving the NQPK byte order.
    padded = np.zeros(((flat.size + per_byte - 1) // per_byte) * per_byte, dtype=np.uint8)
    padded[:flat.size] = flat
    packed = np.zeros(padded.size // per_byte, dtype=np.uint8)
    for offset in range(per_byte):
        packed |= padded[offset::per_byte] << (offset * width)
    return packed.tobytes()


def _unpack_unsigned(payload: bytes, value_count: int, width: int) -> torch.Tensor:
    expected = _ceil_bytes(value_count, width)
    if len(payload) != expected:
        raise ValueError("payload size does not match expected packed value count")
    if value_count == 0:
        return torch.empty((0,), dtype=torch.int64)
    packed = np.frombuffer(payload, dtype=np.uint8)
    per_byte = 8 // width
    remainder = value_count % per_byte
    if remainder and int(packed[-1]) >> (remainder * width):
        raise ValueError("nonzero padding bits in packed payload")
    values = np.empty((packed.size, per_byte), dtype=np.uint8)
    for offset in range(per_byte):
        values[:, offset] = (packed >> (offset * width)) & ((1 << width) - 1)
    return torch.from_numpy(values.reshape(-1)[:value_count].copy()).to(torch.int64)


def _unpack_unsigned_range(
    payload: bytes, start: int, end: int, total_count: int, width: int
) -> torch.Tensor:
    """Unpack only ``[start, end)`` from a fixed-width byte stream."""
    if start < 0 or end < start or end > total_count:
        raise ValueError("packed value range is out of bounds")
    if start == end:
        return torch.empty((0,), dtype=torch.int64)
    per_byte = 8 // width
    first_byte = start // per_byte
    last_byte = (end + per_byte - 1) // per_byte
    packed = np.frombuffer(payload[first_byte:last_byte], dtype=np.uint8)
    values = np.empty((packed.size, per_byte), dtype=np.uint8)
    mask = (1 << width) - 1
    for offset in range(per_byte):
        values[:, offset] = (packed >> (offset * width)) & mask
    local_start = start - first_byte * per_byte
    count = end - start
    return torch.from_numpy(
        values.reshape(-1)[local_start:local_start + count].copy()
    ).to(torch.int64)
