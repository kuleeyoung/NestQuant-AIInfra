"""CLI for bounded-memory NQMC checkpoint inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..core.checkpoint_inspection import inspect_nqmc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an NQMC checkpoint without loading a model or dequantizing weights."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument(
        "--records", action="store_true",
        help="parse every NQRT record and aggregate codec/version/dither metadata",
    )
    parser.add_argument(
        "--file-sha256", action="store_true",
        help="compute the SHA-256 of the entire NQMC file (adds one full read pass)",
    )
    parser.add_argument(
        "--skip-payload-checksum", action="store_true",
        help="skip NQMC payload checksum verification for faster trusted-file inspection",
    )
    parser.add_argument("--chunk-mib", type=int, default=8)
    args = parser.parse_args()
    if args.chunk_mib < 1:
        parser.error("--chunk-mib must be positive")
    report = inspect_nqmc(
        args.checkpoint,
        verify_checksum=not args.skip_payload_checksum,
        inspect_records=args.records,
        compute_file_sha256=args.file_sha256,
        chunk_bytes=args.chunk_mib * 1024 * 1024,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
