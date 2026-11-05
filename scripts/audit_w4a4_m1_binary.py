"""Audit compiled resource usage for production and legacy W4A4 M1 kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

import torch

from nestquant.runtime.cuda_w4a4_mma import _load_extension


KERNELS = {
    "legacy": "nqrt4_w4a4_gemv_m1_kernel",
    "production_packed_word": "nqrt4_w4a4_gemv_m1_packed_word_probe_kernel",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuobjdump", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    extension = Path(_load_extension().__file__).resolve()
    cuda_home = getattr(torch.utils.cpp_extension, "CUDA_HOME", None)
    tool = args.cuobjdump or (
        None if cuda_home is None else Path(cuda_home) / "bin/cuobjdump"
    )
    if tool is None or not tool.is_file():
        parser.error("cuobjdump was not found; pass --cuobjdump explicitly")
    raw = subprocess.run(
        [str(tool), "--dump-resource-usage", str(extension)], check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    rows = {}
    for label, kernel in KERNELS.items():
        match = re.search(
            rf"Function [^\n]*{re.escape(kernel)}[^\n]*:\s*\n"
            rf"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
            raw,
        )
        if match is None:
            raise RuntimeError(f"resource record not found for {kernel}")
        rows[label] = dict(zip(
            ("registers", "stack_bytes", "shared_bytes", "local_bytes"),
            map(int, match.groups()),
        ))
    report = {
        "extension": str(extension),
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cuobjdump": str(tool.resolve()),
        "kernels": rows,
        "resource_usage_equal": rows["legacy"] == rows["production_packed_word"],
        "claim_boundary": (
            "Static binary resource usage does not identify runtime bottlenecks "
            "or replace hardware-counter profiling."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
