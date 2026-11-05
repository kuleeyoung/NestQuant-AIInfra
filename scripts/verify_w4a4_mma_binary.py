"""Verify that the compiled W4A4 CUDA extension contains BF16 HMMA SASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.runtime.cuda_w4a4_mma import _load_extension


HMMA_PATTERN = re.compile(r"\bHMMA\.16816\.F32\.BF16\b")


def find_hmma_lines(sass: str) -> list[str]:
    return [line.strip() for line in sass.splitlines() if HMMA_PATTERN.search(line)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    extension_path = Path(_load_extension().__file__).resolve()
    cuda_home = getattr(torch.utils.cpp_extension, "CUDA_HOME", None)
    default_tool = None if cuda_home is None else Path(cuda_home) / "bin/cuobjdump"
    tool = args.cuobjdump or default_tool
    if tool is None or not tool.is_file():
        parser.error("cuobjdump was not found; pass --cuobjdump explicitly")

    version = subprocess.run(
        [str(tool), "--version"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()
    sass = subprocess.run(
        [str(tool), "--dump-sass", str(extension_path)], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    matches = find_hmma_lines(sass)
    if not matches:
        raise RuntimeError("compiled extension does not contain HMMA.16816.F32.BF16")

    report = {
        "status": "passed",
        "extension_path": str(extension_path),
        "extension_sha256": _sha256(extension_path),
        "device_name": torch.cuda.get_device_name(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuobjdump": str(tool.resolve()),
        "cuobjdump_version": version,
        "required_instruction": HMMA_PATTERN.pattern,
        "match_count": len(matches),
        "matched_sass": matches,
        "claim_boundary": (
            "This proves that the compiled extension contains BF16 Tensor Core "
            "HMMA SASS. It does not by itself prove speedup or model quality."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
