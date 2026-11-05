"""Summarize service Linear-M coverage and a chunk-plus-padding policy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


SUPPORTED_M = (1, 2, 4, 8, 16, 32, 64, 128)


def _merge(runs: list[dict], field: str) -> Counter[int]:
    merged: Counter[int] = Counter()
    for run in runs:
        merged.update({int(key): int(value) for key, value in run[field].items()})
    return merged


def _padded_rows(m_size: int) -> tuple[int, int]:
    full_chunks, remainder = divmod(m_size, 128)
    if remainder == 0:
        return m_size, full_chunks
    padded_tail = next(size for size in SUPPORTED_M if size >= remainder)
    return full_chunks * 128 + padded_tail, full_chunks + 1


def _summary(histogram: Counter[int]) -> dict:
    forwards = sum(histogram.values())
    exact_forwards = sum(
        count for m_size, count in histogram.items() if m_size in SUPPORTED_M
    )
    useful_rows = sum(m_size * count for m_size, count in histogram.items())
    padded_rows = 0
    launches = 0
    decisions = {}
    for m_size, count in sorted(histogram.items()):
        physical_rows, per_forward_launch_count = _padded_rows(m_size)
        padded_rows += physical_rows * count
        launches += per_forward_launch_count * count
        decisions[str(m_size)] = {
            "forward_count": count,
            "exact_supported": m_size in SUPPORTED_M,
            "physical_rows_after_chunk_padding": physical_rows,
            "kernel_launches_per_forward": per_forward_launch_count,
            "padding_overhead_fraction": physical_rows / m_size - 1.0,
        }
    return {
        "forward_count": forwards,
        "exact_supported_forward_count": exact_forwards,
        "fallback_forward_count": forwards - exact_forwards,
        "exact_supported_forward_fraction": (
            exact_forwards / forwards if forwards else 1.0
        ),
        "useful_rows": useful_rows,
        "physical_rows_after_chunk_padding": padded_rows,
        "weighted_padding_overhead_fraction": (
            padded_rows / useful_rows - 1.0 if useful_rows else 0.0
        ),
        "chunk_padding_kernel_launch_count": launches,
        "histogram": {str(key): value for key, value in sorted(histogram.items())},
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    runs = benchmark.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("benchmark must contain nonempty runs")

    prefill = _merge(runs, "prefill_m_forward_histogram")
    decode = _merge(runs, "decode_m_forward_histogram")
    combined = prefill + decode
    payload = {
        "schema_version": 1,
        "source": str(args.benchmark),
        "run_count": len(runs),
        "supported_m": list(SUPPORTED_M),
        "prefill": _summary(prefill),
        "decode": _summary(decode),
        "combined": _summary(combined),
        "claim_boundary": (
            "Counts model forwards, not individual Linear launches or GPU time. "
            "Chunk-plus-padding overhead is structural simulation only; it is not "
            "a measured latency claim."
        ),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
