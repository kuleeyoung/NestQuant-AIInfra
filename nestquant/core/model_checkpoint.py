"""Model-level container for dense tensors and NestQuant single-tensor files.

NQMC stores complete state-dict coverage but does not implement low-bit model
execution. Quantized entries are dequantized to their recorded dtype on load.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
import sys
from typing import Mapping

import torch

from .tensor_checkpoint import PackedNestQuantTensor, _tensor_from_bytes, _tensor_to_bytes


_HEADER = struct.Struct("<4sBQQ")
_MAGIC = b"NQMC"
_VERSION = 1
_DTYPE_TO_NAME = {
    torch.bool: "bool", torch.uint8: "uint8", torch.int8: "int8",
    torch.int16: "int16", torch.int32: "int32", torch.int64: "int64",
    torch.float16: "float16", torch.bfloat16: "bfloat16",
    torch.float32: "float32", torch.float64: "float64",
}
_NAME_TO_DTYPE = {name: dtype for dtype, name in _DTYPE_TO_NAME.items()}


@dataclass(frozen=True)
class PackedNestQuantModel:
    """A model state dict serialized as dense and quantized tensor records."""

    dense_tensors: Mapping[str, torch.Tensor]
    quantized_tensors: Mapping[str, PackedNestQuantTensor]
    aliases: Mapping[str, str]
    model_metadata: Mapping[str, object]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        quantized_tensors: Mapping[str, PackedNestQuantTensor],
        *,
        model_metadata: Mapping[str, object] | None = None,
    ) -> "PackedNestQuantModel":
        """Build a complete model container, deduplicating exact tied tensors.

        ``quantized_tensors`` maps state-dict names to single-tensor NestQuant
        records. Exact aliases (same storage, offset, shape, stride, and dtype)
        are grouped; if any member is quantized, that one record represents
        the entire alias group.
        """
        if sys.byteorder != "little":
            raise RuntimeError("NQMC v1 currently requires a little-endian host")
        if not state_dict:
            raise ValueError("state_dict must not be empty")
        if set(quantized_tensors) - set(state_dict):
            raise ValueError("quantized tensor names must exist in state_dict")
        if any(not isinstance(name, str) or not name for name in state_dict):
            raise ValueError("state_dict keys must be nonempty strings")
        if any(not isinstance(tensor, torch.Tensor) for tensor in state_dict.values()):
            raise TypeError("all state_dict values must be torch.Tensor instances")

        groups: dict[tuple, list[str]] = {}
        for name, tensor in state_dict.items():
            if tensor.layout != torch.strided:
                raise ValueError(f"unsupported non-strided tensor: {name}")
            pointer_key = tensor.data_ptr() if tensor.numel() else id(tensor)
            key = (str(tensor.device), pointer_key, tensor.storage_offset(),
                   tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
            groups.setdefault(key, []).append(name)

        dense: dict[str, torch.Tensor] = {}
        quantized: dict[str, PackedNestQuantTensor] = {}
        aliases: dict[str, str] = {}
        for members in groups.values():
            quantized_members = [name for name in members if name in quantized_tensors]
            if len(quantized_members) > 1:
                raise ValueError(f"only one quantized record may represent tied weights: {members}")
            canonical = quantized_members[0] if quantized_members else members[0]
            for name in members:
                if name != canonical:
                    aliases[name] = canonical
            if quantized_members:
                record = quantized_tensors[canonical]
                if record.tensor_name != canonical:
                    from dataclasses import replace
                    record = replace(record, tensor_name=canonical)
                if (record.indices.shape[0], record.feature_count) != tuple(state_dict[canonical].shape):
                    raise ValueError(f"quantized tensor shape mismatch: {canonical}")
                quantized[canonical] = record
            else:
                dense[canonical] = state_dict[canonical].detach().cpu().contiguous().clone()
        return cls(dense, quantized, aliases, dict(model_metadata or {}))

    @property
    def tensor_count(self) -> int:
        return len(self.dense_tensors) + len(self.quantized_tensors) + len(self.aliases)

    def to_bytes(self) -> bytes:
        entries = []
        payload_parts = []
        payload_offset = 0
        all_names = set(self.dense_tensors) | set(self.quantized_tensors)
        if all_names & set(self.aliases):
            raise ValueError("alias names cannot also be stored tensor names")
        for name in sorted(self.dense_tensors):
            tensor = self.dense_tensors[name]
            if tensor.layout != torch.strided or tensor.dtype not in _DTYPE_TO_NAME:
                raise ValueError(f"unsupported dense tensor layout or dtype: {name}")
            blob = _tensor_to_bytes(tensor)
            expected = tensor.numel() * tensor.element_size()
            if len(blob) != expected:
                raise ValueError(f"dense tensor payload has invalid byte length: {name}")
            entries.append({"name": name, "kind": "dense", "offset": payload_offset,
                            "length": len(blob), "shape": list(tensor.shape),
                            "dtype": _DTYPE_TO_NAME[tensor.dtype]})
            payload_parts.append(blob)
            payload_offset += len(blob)
        for name in sorted(self.quantized_tensors):
            record = self.quantized_tensors[name]
            if record.tensor_name != name:
                raise ValueError(f"quantized record name mismatch: {name}")
            blob = record.to_bytes()
            entries.append({"name": name, "kind": "nestquant", "offset": payload_offset,
                            "length": len(blob), "shape": [record.indices.shape[0], record.feature_count],
                            "dtype": _DTYPE_TO_NAME[record.output_dtype]})
            payload_parts.append(blob)
            payload_offset += len(blob)

        payload = b"".join(payload_parts)
        manifest = json.dumps({
            "format": "NQMC", "version": _VERSION,
            "entries": entries, "aliases": dict(self.aliases),
            "model_metadata": dict(self.model_metadata),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return _HEADER.pack(_MAGIC, _VERSION, len(manifest), len(payload)) + manifest + payload

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "PackedNestQuantModel":
        view = memoryview(data)
        if len(view) < _HEADER.size:
            raise ValueError("input is shorter than the model checkpoint header")
        magic, version, manifest_len, payload_len = _HEADER.unpack_from(view)
        if magic != _MAGIC:
            raise ValueError("invalid model checkpoint magic")
        if version != _VERSION:
            raise ValueError(f"unsupported model checkpoint version: {version}")
        payload_start = _HEADER.size + manifest_len
        if payload_start + payload_len != len(view):
            raise ValueError("model checkpoint lengths do not match file size")
        try:
            manifest = json.loads(bytes(view[_HEADER.size:payload_start]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid model checkpoint manifest") from error
        if not isinstance(manifest, dict) or manifest.get("format") != "NQMC" or manifest.get("version") != _VERSION:
            raise ValueError("unsupported or malformed model checkpoint manifest")
        payload = view[payload_start:]
        if hashlib.sha256(payload).hexdigest() != manifest.get("payload_sha256"):
            raise ValueError("model checkpoint payload checksum mismatch")
        dense: dict[str, torch.Tensor] = {}
        quantized: dict[str, PackedNestQuantTensor] = {}
        seen_names = set()
        entries = manifest.get("entries")
        if not isinstance(entries, list):
            raise ValueError("model checkpoint entries must be a list")
        for entry in entries:
            try:
                name, kind = entry["name"], entry["kind"]
                offset, length = int(entry["offset"]), int(entry["length"])
                shape = tuple(int(dim) for dim in entry["shape"])
                dtype = _NAME_TO_DTYPE[entry["dtype"]]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid model checkpoint entry") from error
            if not isinstance(name, str) or not name or name in seen_names:
                raise ValueError("tensor names must be unique nonempty strings")
            if offset < 0 or length < 0 or offset + length > len(payload) or any(dim < 0 for dim in shape):
                raise ValueError("model checkpoint entry is out of payload bounds")
            blob = bytes(payload[offset:offset + length])
            if kind == "dense":
                expected_length = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                if len(blob) != expected_length:
                    raise ValueError("dense tensor payload size disagrees with manifest")
                dense[name] = _tensor_from_bytes(blob, shape, dtype)
            elif kind == "nestquant":
                record = PackedNestQuantTensor.from_bytes(blob)
                if (record.tensor_name != name
                        or (record.indices.shape[0], record.feature_count) != shape
                        or record.output_dtype != dtype):
                    raise ValueError("quantized record disagrees with model manifest")
                quantized[name] = record
            else:
                raise ValueError(f"unsupported tensor record kind: {kind}")
            seen_names.add(name)

        aliases = manifest.get("aliases")
        metadata = manifest.get("model_metadata")
        if not isinstance(aliases, dict) or not isinstance(metadata, dict):
            raise ValueError("aliases and model_metadata must be objects")
        stored_names = set(dense) | set(quantized)
        for alias, canonical in aliases.items():
            if not isinstance(alias, str) or not isinstance(canonical, str) or alias in stored_names:
                raise ValueError("invalid alias entry")
            if canonical not in stored_names:
                raise ValueError(f"alias target does not exist: {canonical}")
        return cls(dense, quantized, aliases, metadata)

    def state_dict(self, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Restore all names, sharing tensor objects for declared aliases."""
        result = {name: tensor.to(device) for name, tensor in self.dense_tensors.items()}
        result.update({name: record.dequantize(device=device)
                       for name, record in self.quantized_tensors.items()})
        for alias, canonical in self.aliases.items():
            if canonical not in result:
                raise ValueError(f"alias target does not exist: {canonical}")
            result[alias] = result[canonical]
        if len(result) != self.tensor_count:
            raise ValueError("model tensor names are duplicated or missing")
        return result
