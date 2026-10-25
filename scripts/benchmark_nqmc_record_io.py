"""Compare sequential NQMC quantized-record reads across storage paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.checkpoint) < 2:
        parser.error("provide at least two checkpoint paths")
    if args.repeats < 1:
        parser.error("repeats must be positive")

    readers = [NQMCFileReader(path, verify_checksum=False) for path in args.checkpoint]
    entries = [tuple(e for e in reader.entries if e.kind == "nestquant") for reader in readers]
    signatures = [tuple((e.name, e.length) for e in group) for group in entries]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("checkpoints do not contain the same quantized records")
    total_bytes = sum(entry.length for entry in entries[0])
    timings: list[list[float]] = [[] for _ in readers]

    # Alternate path order by repetition to reduce systematic first/last bias.
    for repetition in range(args.repeats):
        order = range(len(readers)) if repetition % 2 == 0 else reversed(range(len(readers)))
        for index in order:
            started = time.perf_counter()
            consumed = 0
            for entry in entries[index]:
                blob = readers[index].read_entry_blob(entry)
                consumed += len(blob)
            elapsed = time.perf_counter() - started
            if consumed != total_bytes:
                raise RuntimeError("record byte count changed during benchmark")
            timings[index].append(elapsed)

    results = []
    for path, values in zip(args.checkpoint, timings):
        stats = _summary(values)
        results.append({
            "path": str(path.resolve()),
            "seconds": stats,
            "throughput_mib_per_second_from_mean": (
                total_bytes / (1024 * 1024) / stats["mean"]
            ),
            "per_repeat_seconds": values,
        })
    report = {
        "scope": "sequential reads of all NQMC quantized record blobs; parse/decode excluded",
        "record_count": len(entries[0]),
        "bytes_per_repeat": total_bytes,
        "repeats": args.repeats,
        "alternating_path_order": True,
        "results": results,
        "methodology_limit": (
            "Operating-system page caches are not dropped; this measures the observed "
            "warm/runtime path rather than raw-device cold-read bandwidth."
        ),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
