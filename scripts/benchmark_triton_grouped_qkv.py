"""Compare one grouped QKV launch with three independent packed launches."""

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

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.tensor_checkpoint import PackedNestQuantTensor
from nestquant.runtime.triton_fused_linear import (
    linear_fused_triton,
    prepare_fused_linear,
)
from nestquant.runtime.triton_grouped_linear import linear_fused_qkv_triton


def _measure(operation, repeats: int) -> dict[str, float]:
    operation()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = operation()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
        del outputs
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.layer_index < 0 or args.repeats < 1:
        parser.error("layer-index must be nonnegative and repeats must be positive")
    if any(length < 1 for length in args.sequence_length):
        parser.error("sequence lengths must be positive")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    names = tuple(
        f"model.layers.{args.layer_index}.self_attn.{kind}_proj.weight"
        for kind in ("q", "k", "v")
    )
    prepared = []
    for name in names:
        _, record = reader.read_named_record(name)
        if not isinstance(record, PackedNestQuantTensor):
            raise TypeError(f"expected packed projection: {name}")
        prepared.append(prepare_fused_linear(record))
    prepared_tuple = tuple(prepared)
    feature_count = prepared[0].record.feature_count
    generator = torch.Generator(device="cuda").manual_seed(20261025)
    results = []
    for sequence_length in args.sequence_length:
        # This represents the already-shared XH input. Both paths receive the
        # same contiguous BF16 tensor, so the benchmark isolates launch layout.
        inputs = torch.randn(
            (1, sequence_length, feature_count),
            device="cuda", dtype=torch.bfloat16, generator=generator,
        )
        independent = lambda: tuple(
            linear_fused_triton(inputs, item) for item in prepared_tuple
        )
        grouped = lambda: linear_fused_qkv_triton(inputs, prepared_tuple)
        expected = independent()
        actual = grouped()
        torch.cuda.synchronize()
        exact = [
            bool(torch.equal(left, right))
            for left, right in zip(actual, expected, strict=True)
        ]
        independent_stats = _measure(independent, args.repeats)
        grouped_stats = _measure(grouped, args.repeats)
        results.append({
            "sequence_length": sequence_length,
            "three_independent_launches_seconds": independent_stats,
            "one_grouped_launch_seconds": grouped_stats,
            "median_speedup": independent_stats["median"] / grouped_stats["median"],
            "outputs_bitwise_equal": exact,
        })

    report = {
        "scope": "real Llama Q/K/V packed fused kernels; shared input transform excluded",
        "checkpoint": str(args.checkpoint.resolve()),
        "weight_names": list(names),
        "weight_shapes": [
            [item.record.indices.shape[0], item.record.feature_count]
            for item in prepared_tuple
        ],
        "device_name": torch.cuda.get_device_name(),
        "repeats": args.repeats,
        "results": results,
        "claim_boundary": (
            "This isolates QKV kernel launch organization. It is not a full "
            "attention, decoder-block, or end-to-end model benchmark."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
