"""Account for actual NQMC bytes without dequantizing model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


HEADER = struct.Struct("<4sBQQ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blob = args.checkpoint.read_bytes()
    if len(blob) < HEADER.size:
        raise ValueError("truncated NQMC header")
    magic, version, manifest_length, payload_length = HEADER.unpack_from(blob)
    if magic != b"NQMC" or version != 1:
        raise ValueError("expected NQMC v1")
    if HEADER.size + manifest_length + payload_length != len(blob):
        raise ValueError("NQMC length mismatch")
    manifest = json.loads(blob[HEADER.size:HEADER.size + manifest_length])
    payload = memoryview(blob)[HEADER.size + manifest_length:]
    if hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]:
        raise ValueError("NQMC payload checksum mismatch")
    dense = [entry for entry in manifest["entries"] if entry["kind"] == "dense"]
    quantized = [entry for entry in manifest["entries"] if entry["kind"] == "nestquant"]
    if len(dense) + len(quantized) != len(manifest["entries"]):
        raise ValueError("unknown NQMC entry kind")
    dense_bytes = sum(entry["length"] for entry in dense)
    quantized_bytes = sum(entry["length"] for entry in quantized)
    quantized_values = sum(math.prod(entry["shape"]) for entry in quantized)
    dense_values = sum(math.prod(entry["shape"]) for entry in dense)
    # This is the raw storage of the same unique state values if the quantized
    # matrices were BF16. It is not a measured BF16 checkpoint file size.
    reference_payload_bytes = dense_bytes + 2 * quantized_values
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "file_sha256": hashlib.sha256(blob).hexdigest(),
        "file_bytes": len(blob),
        "quantized_tensor_count": len(quantized),
        "quantized_parameter_count": quantized_values,
        "quantized_record_bytes": quantized_bytes,
        "effective_bits_per_quantized_parameter_including_record_metadata": 8 * quantized_bytes / quantized_values,
        "dense_tensor_count": len(dense),
        "dense_unique_value_count": dense_values,
        "dense_payload_bytes": dense_bytes,
        "tied_alias_count": len(manifest["aliases"]),
        "container_header_manifest_bytes": len(blob) - dense_bytes - quantized_bytes,
        "raw_bf16_equivalent_payload_bytes": reference_payload_bytes,
        "file_to_raw_bf16_payload_ratio": len(blob) / reference_payload_bytes,
        "limitations": [
            "The BF16 denominator is computed from unique state values, not a measured BF16 checkpoint file.",
            "NQMC stores packed weights but dequantizes them for BF16 inference; no low-bit kernel or speedup is claimed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
