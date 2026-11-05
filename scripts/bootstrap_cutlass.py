"""Download and verify the pinned CUTLASS headers used by CuTe probes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request


VERSION = "4.8.0"
URL = f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v{VERSION}"
SHA256 = "b7bb2ba0473a5a5d699c62e5feb42caa11cb109ec58ba82ca458af9cd139f39b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_root(root: Path) -> bool:
    version_header = root / "include/cutlass/version.h"
    cute_header = root / "include/cute/tensor.hpp"
    sm120_examples = root / "examples/79_blackwell_geforce_gemm"
    if not version_header.is_file() or not cute_header.is_file():
        return False
    if not sm120_examples.is_dir():
        return False
    text = version_header.read_text(encoding="utf-8")
    return all(
        definition in text
        for definition in (
            "#define CUTLASS_MAJOR 4",
            "#define CUTLASS_MINOR 8",
            "#define CUTLASS_PATCH 0",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "tmp/vendor",
    )
    args = parser.parse_args()
    vendor = args.vendor_dir.resolve()
    archive = vendor / f"cutlass-v{VERSION}.tar.gz"
    root = vendor / f"cutlass-{VERSION}"
    if _valid_root(root):
        print(f"CUTLASS {VERSION} already verified at {root}")
        return

    vendor.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or _sha256(archive) != SHA256:
        partial = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading {URL} -> {partial}", flush=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = _sha256(partial)
        if actual != SHA256:
            raise RuntimeError(
                f"CUTLASS archive SHA-256 mismatch: expected {SHA256}, got {actual}"
            )
        partial.replace(archive)

    actual = _sha256(archive)
    if actual != SHA256:
        raise RuntimeError(
            f"CUTLASS archive SHA-256 mismatch: expected {SHA256}, got {actual}"
        )
    print(f"Extracting verified archive {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(vendor, filter="data")
    if not _valid_root(root):
        raise RuntimeError("extracted CUTLASS tree failed version/header validation")
    print(f"CUTLASS {VERSION} verified at {root}")


if __name__ == "__main__":
    main()
