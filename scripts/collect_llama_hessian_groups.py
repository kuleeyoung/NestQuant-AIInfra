"""Collect all Llama-3.2-1B Hessians in memory-safe layer groups."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = PROJECT_ROOT / "scripts" / "collect_llama_hessian.py"


def layer_modules(start: int, stop: int) -> list[str]:
    modules: list[str] = []
    for layer in range(start, stop):
        prefix = f"model.layers.{layer}"
        modules.extend(
            [
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.down_proj",
            ]
        )
    return modules


def run_collection(
    modules: list[str], output: Path, chunks: int, offline: bool,
    calibration_manifest: Path | None,
) -> None:
    command = [
        sys.executable,
        str(COLLECTOR),
        "--modules",
        *modules,
        "--chunks",
        str(chunks),
        "--output",
        str(output),
    ]
    if offline:
        command.append("--offline")
    if calibration_manifest is not None:
        command.extend(["--calibration-manifest", str(calibration_manifest)])
    print(f"Collecting {len(modules)} unique inputs -> {output}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--layers-per-group", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw/calibration"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--calibration-manifest", type=Path,
                        help="shared versioned calibration-window manifest")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace complete tensor/manifest pairs"
    )
    args = parser.parse_args()
    if args.chunks < 1 or args.layers < 1 or args.layers_per_group < 1:
        raise ValueError("chunks, layers, and layers-per-group must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[list[str], Path]] = []
    for start in range(0, args.layers, args.layers_per_group):
        stop = min(start + args.layers_per_group, args.layers)
        output = output_dir / f"hessian_layers{start:02d}_{stop - 1:02d}_train{args.chunks}.pt"
        jobs.append((layer_modules(start, stop), output))
    jobs.append((["lm_head"], output_dir / f"hessian_lm_head_train{args.chunks}.pt"))

    for modules, output in jobs:
        manifest = output.with_suffix(output.suffix + ".json")
        if output.exists() or manifest.exists():
            if output.exists() and manifest.exists() and not args.overwrite:
                print(f"Skipping complete pair: {output}", flush=True)
                continue
            if not args.overwrite:
                raise FileExistsError(
                    f"incomplete artifact pair for {output}; inspect it or pass --overwrite"
                )
            output.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
        run_collection(
            modules, output, args.chunks, args.offline, args.calibration_manifest
        )


if __name__ == "__main__":
    main()
