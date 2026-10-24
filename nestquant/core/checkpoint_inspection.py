"""Read-only accounting and semantic inspection for NQMC checkpoints."""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
from typing import Any

from .model_checkpoint_file import NQMCFileReader
from .tensor_checkpoint import PackedNestQuantTensor


def inspect_nqmc(
    path: str | Path,
    *,
    verify_checksum: bool = True,
    inspect_records: bool = False,
    compute_file_sha256: bool = False,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Return auditable NQMC metadata without dequantizing model weights.

    Manifest-only accounting is bounded-memory and does not parse tensor
    records. ``inspect_records`` reads one record at a time to validate NQRT
    metadata and aggregate codec semantics, q, beta counts, and dither modes.
    """
    checkpoint = Path(path)
    reader = NQMCFileReader(
        checkpoint, verify_checksum=verify_checksum, chunk_bytes=chunk_bytes
    )
    dense = [entry for entry in reader.entries if entry.kind == "dense"]
    quantized = [entry for entry in reader.entries if entry.kind == "nestquant"]
    dense_bytes = sum(entry.length for entry in dense)
    quantized_bytes = sum(entry.length for entry in quantized)
    dense_values = sum(math.prod(entry.shape) for entry in dense)
    quantized_values = sum(math.prod(entry.shape) for entry in quantized)
    bf16_reference_bytes = dense_bytes + 2 * quantized_values

    report: dict[str, Any] = {
        "format": "NQMC",
        "version": 1,
        "checkpoint": str(checkpoint.resolve()),
        "file_bytes": reader.file_bytes,
        "payload_bytes": reader.payload_bytes,
        "payload_checksum_verified": verify_checksum,
        "tensor_count_including_aliases": reader.tensor_count,
        "unique_tensor_record_count": len(reader.entries),
        "quantized_tensor_count": len(quantized),
        "quantized_parameter_count": quantized_values,
        "quantized_record_bytes": quantized_bytes,
        "effective_bits_per_quantized_parameter_including_record_metadata": (
            8 * quantized_bytes / quantized_values if quantized_values else None
        ),
        "dense_tensor_count": len(dense),
        "dense_unique_value_count": dense_values,
        "dense_payload_bytes": dense_bytes,
        "tied_alias_count": len(reader.aliases),
        "aliases": dict(reader.aliases),
        "container_header_manifest_bytes": (
            reader.file_bytes - dense_bytes - quantized_bytes
        ),
        "raw_bf16_equivalent_payload_bytes": bf16_reference_bytes,
        "file_to_raw_bf16_payload_ratio": (
            reader.file_bytes / bf16_reference_bytes if bf16_reference_bytes else None
        ),
        "model_metadata": dict(reader.model_metadata),
        "record_inspection": None,
        "limitations": [
            "The BF16 denominator is computed from unique state values, not a measured BF16 checkpoint file.",
            "NQMC stores packed weights but the current runtime dequantizes them for floating-point Linear execution.",
            "Storage accounting and format inspection do not establish low-bit inference speed or device-memory residency.",
        ],
    }

    if compute_file_sha256:
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(chunk_bytes), b""):
                digest.update(block)
        report["file_sha256"] = digest.hexdigest()

    if inspect_records:
        semantics: Counter[str] = Counter()
        q_values: Counter[int] = Counter()
        beta_counts: Counter[int] = Counter()
        dither_storage: Counter[str] = Counter()
        versions: Counter[int] = Counter()
        for entry, record in reader.iter_records():
            if entry.kind != "nestquant":
                continue
            if not isinstance(record, PackedNestQuantTensor):
                raise TypeError(f"expected packed NestQuant tensor: {entry.name}")
            semantics[record.codec_semantics] += 1
            q_values[record.indices.q] += 1
            beta_counts[record.indices.beta_count] += 1
            dither_storage[_dither_mode(record)] += 1
            versions[_read_nqrt_version(reader, entry)] += 1
        report["record_inspection"] = {
            "quantized_records_parsed": sum(semantics.values()),
            "codec_semantics_counts": _string_key_counts(semantics),
            "nqrt_version_counts": _string_key_counts(versions),
            "q_counts": _string_key_counts(q_values),
            "beta_count_counts": _string_key_counts(beta_counts),
            "dither_storage_counts": _string_key_counts(dither_storage),
        }
    return report


def _dither_mode(record: PackedNestQuantTensor) -> str:
    if record.dither is not None:
        return "tensor"
    if record.dither_generator_state is not None:
        return "generator_state"
    return "none"


def _read_nqrt_version(reader: NQMCFileReader, entry) -> int:
    with reader.path.open("rb") as stream:
        stream.seek(reader.payload_start + entry.offset + 4)
        raw = stream.read(1)
    if len(raw) != 1:
        raise ValueError(f"truncated tensor checkpoint version: {entry.name}")
    return raw[0]


def _string_key_counts(values: Counter) -> dict[str, int]:
    return {str(key): values[key] for key in sorted(values, key=str)}
