"""Compare legacy CPU replay with NQRT v4 random-access GPU dither."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.codec import generate_e8_dither_stateless
from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import (
    PackedNestQuantTensor,
    _generate_dither_from_state,
)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _measure(operation, repeats: int, device: torch.device | None = None):
    operation()
    if device is not None:
        torch.cuda.synchronize(device)
    values = []
    for _ in range(repeats):
        if device is not None:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = operation()
        if device is not None:
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - started)
        del result
    return _stats(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("weight_name", nargs="+")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    device = torch.device(args.device)
    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    tensors = []
    for name in args.weight_name:
        entry, record = reader.read_named_record(name)
        if not isinstance(record, PackedNestQuantTensor):
            raise ValueError(f"not a packed record: {name}")
        if record.dither_generator_state is None:
            raise ValueError(f"legacy comparison requires generator-state dither: {name}")
        block_count = record.indices.shape[0] * record.indices.shape[1]
        legacy = _measure(
            lambda: _generate_dither_from_state(
                record.dither_generator_state,
                record.indices.shape,
                device=record.dither_generator_device,
                dtype=record.output_dtype,
                output_device="cpu",
                codec_semantics=record.codec_semantics,
            ),
            args.repeats,
        )
        stateless = _measure(
            lambda: generate_e8_dither_stateless(
                block_count, args.seed, device=device,
                dtype=record.output_dtype,
                codec_semantics=record.codec_semantics,
            ),
            args.repeats,
            device,
        )
        tensors.append({
            "weight_name": name,
            "shape": list(entry.shape),
            "e8_block_count": block_count,
            "legacy_cpu_generator_replay_seconds": legacy,
            "v4_gpu_random_access_seconds": stateless,
            "mean_speedup": legacy["mean"] / stateless["mean"],
        })
    report = {
        "scope": "dither generation only; v4 values intentionally differ from v3",
        "checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "repeats": args.repeats,
        "device_name": torch.cuda.get_device_name(device),
        "tensors": tensors,
        "claim_boundary": (
            "This isolates addressable dither generation. It does not establish "
            "v4 checkpoint quality, full tensor decode speed, or GEMM speed."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
