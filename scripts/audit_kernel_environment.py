"""Report local prerequisites for Triton and PyTorch CUDA extension work."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess

import torch


def _version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    text = (result.stdout or result.stderr).strip()
    return text or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    triton_spec = importlib.util.find_spec("triton")
    triton_version = None
    if triton_spec is not None:
        import triton
        triton_version = triton.__version__
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None
        ),
        "triton_installed": triton_spec is not None,
        "triton_version": triton_version,
        "nvcc": _version(["nvcc", "--version"]),
        "cxx": _version(["g++", "--version"]),
        "ninja": _version(["ninja", "--version"]),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
