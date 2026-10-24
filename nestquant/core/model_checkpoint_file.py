"""Bounded-memory file access for NQMC model checkpoints.

The in-memory ``PackedNestQuantModel`` API remains useful for small artifacts.
This reader validates a file first, then exposes one stored tensor record at a
time so a model checkpoint does not need to be duplicated in host memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterator

import torch

from .model_checkpoint import _HEADER, _MAGIC, _NAME_TO_DTYPE, _VERSION
from .tensor_checkpoint import PackedNestQuantTensor, _tensor_from_bytes


@dataclass(frozen=True)
class NQMCEntry:
    name: str
    kind: str
    offset: int
    length: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class NQMCFileReader:
    """Validated, seek-based reader for an NQMC v1 file."""

    def __init__(self, path: str | Path, *, verify_checksum: bool = True,
                 chunk_bytes: int = 8 * 1024 * 1024):
        self.path = Path(path)
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        self.file_bytes = self.path.stat().st_size
        with self.path.open("rb") as stream:
            header = stream.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError("input is shorter than the model checkpoint header")
            magic, version, manifest_len, payload_len = _HEADER.unpack(header)
            if magic != _MAGIC:
                raise ValueError("invalid model checkpoint magic")
            if version != _VERSION:
                raise ValueError(f"unsupported model checkpoint version: {version}")
            self.payload_start = _HEADER.size + manifest_len
            self.payload_bytes = payload_len
            if self.payload_start + payload_len != self.file_bytes:
                raise ValueError("model checkpoint lengths do not match file size")
            manifest_blob = stream.read(manifest_len)
            if len(manifest_blob) != manifest_len:
                raise ValueError("truncated model checkpoint manifest")
        try:
            manifest = json.loads(manifest_blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid model checkpoint manifest") from error
        self.entries, self.aliases, self.model_metadata, expected_sha = self._validate_manifest(
            manifest
        )
        self._entries_by_name = {entry.name: entry for entry in self.entries}
        if verify_checksum:
            digest = hashlib.sha256()
            with self.path.open("rb") as stream:
                stream.seek(self.payload_start)
                remaining = self.payload_bytes
                while remaining:
                    block = stream.read(min(chunk_bytes, remaining))
                    if not block:
                        raise ValueError("truncated model checkpoint payload")
                    digest.update(block)
                    remaining -= len(block)
            if digest.hexdigest() != expected_sha:
                raise ValueError("model checkpoint payload checksum mismatch")

    def _validate_manifest(self, manifest):
        if (not isinstance(manifest, dict) or manifest.get("format") != "NQMC"
                or manifest.get("version") != _VERSION):
            raise ValueError("unsupported or malformed model checkpoint manifest")
        raw_entries = manifest.get("entries")
        aliases = manifest.get("aliases")
        metadata = manifest.get("model_metadata")
        checksum = manifest.get("payload_sha256")
        if not isinstance(raw_entries, list):
            raise ValueError("model checkpoint entries must be a list")
        if not isinstance(aliases, dict) or not isinstance(metadata, dict):
            raise ValueError("aliases and model_metadata must be objects")
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(char not in "0123456789abcdef" for char in checksum)):
            raise ValueError("invalid model checkpoint payload checksum")

        entries = []
        seen = set()
        spans = []
        for raw in raw_entries:
            try:
                name, kind = raw["name"], raw["kind"]
                offset, length = int(raw["offset"]), int(raw["length"])
                shape = tuple(int(dim) for dim in raw["shape"])
                dtype = _NAME_TO_DTYPE[raw["dtype"]]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid model checkpoint entry") from error
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("tensor names must be unique nonempty strings")
            if kind not in {"dense", "nestquant"}:
                raise ValueError(f"unsupported tensor record kind: {kind}")
            if (offset < 0 or length < 0 or offset + length > self.payload_bytes
                    or any(dim < 0 for dim in shape)):
                raise ValueError("model checkpoint entry is out of payload bounds")
            if kind == "dense":
                expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                if length != expected:
                    raise ValueError("dense tensor payload size disagrees with manifest")
            entries.append(NQMCEntry(name, kind, offset, length, shape, dtype))
            spans.append((offset, offset + length))
            seen.add(name)
        if any(left_end > right_start for (_, left_end), (right_start, _) in
               zip(sorted(spans), sorted(spans)[1:])):
            raise ValueError("model checkpoint entries overlap")
        for alias, canonical in aliases.items():
            if (not isinstance(alias, str) or not alias or not isinstance(canonical, str)
                    or alias in seen or alias in aliases and alias == canonical):
                raise ValueError("invalid alias entry")
            if canonical not in seen:
                raise ValueError(f"alias target does not exist: {canonical}")
        return tuple(entries), dict(aliases), dict(metadata), checksum

    @property
    def tensor_count(self) -> int:
        return len(self.entries) + len(self.aliases)

    def canonical_name(self, name: str) -> str:
        """Resolve a state-dict name to the unique stored record name."""
        if name in self._entries_by_name:
            return name
        try:
            return self.aliases[name]
        except KeyError as error:
            raise KeyError(f"tensor is not present in NQMC: {name}") from error

    def get_entry(self, name: str) -> NQMCEntry:
        """Return manifest metadata for a tensor or tied alias in O(1)."""
        return self._entries_by_name[self.canonical_name(name)]

    def read_named_record(
        self, name: str
    ) -> tuple[NQMCEntry, torch.Tensor | PackedNestQuantTensor]:
        """Read one stored record by tensor name without decoding quantized values."""
        entry = self.get_entry(name)
        return entry, self.read_entry(entry)

    def decode_tensor(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        output_dtype: torch.dtype | None = None,
        rows_per_chunk: int | None = None,
    ) -> torch.Tensor:
        """Read and decode exactly one named tensor, resolving tied aliases."""
        _entry, record = self.read_named_record(name)
        if isinstance(record, PackedNestQuantTensor):
            if rows_per_chunk is not None:
                return record.dequantize_chunked(
                    device=device, output_dtype=output_dtype,
                    rows_per_chunk=rows_per_chunk,
                )
            value = record.dequantize(device=device)
        else:
            value = record.to(device)
        return value if output_dtype is None else value.to(output_dtype)

    def read_entry(self, entry: NQMCEntry) -> torch.Tensor | PackedNestQuantTensor:
        """Read and parse one manifest entry."""
        return self.parse_entry_blob(entry, self.read_entry_blob(entry))

    def read_entry_blob(self, entry: NQMCEntry) -> bytes:
        """Read the exact serialized bytes for one validated manifest entry."""
        with self.path.open("rb") as stream:
            stream.seek(self.payload_start + entry.offset)
            blob = stream.read(entry.length)
        if len(blob) != entry.length:
            raise ValueError(f"truncated tensor record: {entry.name}")
        return blob

    def parse_entry_blob(
        self, entry: NQMCEntry, blob: bytes
    ) -> torch.Tensor | PackedNestQuantTensor:
        """Parse and cross-check bytes previously read for ``entry``."""
        if not isinstance(blob, bytes):
            raise TypeError("entry blob must be bytes")
        if len(blob) != entry.length:
            raise ValueError(f"tensor record length mismatch: {entry.name}")
        if entry.kind == "dense":
            return _tensor_from_bytes(blob, entry.shape, entry.dtype)
        record = PackedNestQuantTensor.from_bytes(blob)
        if (record.tensor_name != entry.name
                or (record.indices.shape[0], record.feature_count) != entry.shape
                or record.output_dtype != entry.dtype):
            raise ValueError("quantized record disagrees with model manifest")
        return record

    def iter_records(self) -> Iterator[tuple[NQMCEntry, torch.Tensor | PackedNestQuantTensor]]:
        for entry in self.entries:
            yield entry, self.read_entry(entry)

    def load_into_model(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device | str,
        record_validator: Callable[[NQMCEntry, torch.Tensor | PackedNestQuantTensor], None]
        | None = None,
    ) -> None:
        """Copy records into an existing model while retaining one record at a time."""
        targets = model.state_dict()
        expected_names = {entry.name for entry in self.entries} | set(self.aliases)
        if set(targets) != expected_names:
            missing = sorted(set(targets) - expected_names)
            unexpected = sorted(expected_names - set(targets))
            raise ValueError(f"state-dict mismatch; missing={missing}, unexpected={unexpected}")

        # Validate every embedded record and target before the first model mutation.
        for entry, record in self.iter_records():
            target = targets[entry.name]
            dtype_compatible = (
                target.dtype == entry.dtype
                or (torch.empty((), dtype=target.dtype).is_floating_point()
                    and torch.empty((), dtype=entry.dtype).is_floating_point())
            )
            if tuple(target.shape) != entry.shape or not dtype_compatible:
                raise ValueError(f"model tensor metadata mismatch: {entry.name}")
            if entry.kind == "nestquant":
                assert isinstance(record, PackedNestQuantTensor)
            if record_validator is not None:
                record_validator(entry, record)
        for alias, canonical in self.aliases.items():
            if tuple(targets[alias].shape) != tuple(targets[canonical].shape):
                raise ValueError(f"alias tensor shape mismatch: {alias}")
            if targets[alias].dtype != targets[canonical].dtype:
                raise ValueError(f"alias tensor dtype mismatch: {alias}")

        with torch.no_grad():
            for entry, record in self.iter_records():
                value = (record.dequantize(device=device)
                         if isinstance(record, PackedNestQuantTensor) else record.to(device))
                targets[entry.name].copy_(value.to(targets[entry.name].device))
                del value, record
            for alias, canonical in self.aliases.items():
                if targets[alias].data_ptr() != targets[canonical].data_ptr():
                    targets[alias].copy_(targets[canonical])

