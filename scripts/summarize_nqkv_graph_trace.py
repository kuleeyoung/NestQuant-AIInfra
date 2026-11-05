"""Summarize packed W4A4 kernels from a PyTorch Chrome trace."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


BASE_KERNEL = "nqrt4_w4a4_mma_small_m_kernel"
SPLIT_KERNEL = "nqrt4_w4a4_mma_small_m_splitk_partial_kernel"
REDUCE_KERNEL = "nqrt4_w4a4_splitk_reduce_kernel"
KERNELS = (BASE_KERNEL, SPLIT_KERNEL, REDUCE_KERNEL)
PROJECTION_ORDER = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
EXPECTED_GRIDS = (128, 32, 32, 128, 512, 512, 128)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    grouped: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    by_graph: dict[int, list[tuple[float, str, int, float]]] = defaultdict(list)
    all_kernel_durations: list[float] = []
    for event in payload.get("traceEvents", []):
        name = event.get("name", "")
        kind = next((kernel for kernel in KERNELS if kernel in name), None)
        if event.get("cat") != "kernel" or kind is None:
            continue
        metadata = event.get("args", {})
        grid = metadata.get("grid")
        if not isinstance(grid, list) or not grid:
            raise ValueError("W4A4 kernel event is missing grid metadata")
        graph_id = int(metadata.get("graph id", -1))
        grid_x = int(grid[0])
        duration = float(event["dur"])
        grouped[(graph_id, kind, grid_x)].append(duration)
        by_graph[graph_id].append((float(event["ts"]), kind, grid_x, duration))
        all_kernel_durations.append(duration)
    if not all_kernel_durations:
        raise RuntimeError("trace contains no packed W4A4 kernel events")

    groups = []
    for (graph_id, kind, grid_x), values in sorted(grouped.items()):
        groups.append({
            "graph_id": graph_id,
            "kernel": kind,
            "grid_x": grid_x,
            "inferred_output_width": (
                None if kind == REDUCE_KERNEL else grid_x * 16
            ),
            "count": len(values),
            "total_us": sum(values),
            "mean_us": statistics.fmean(values),
            "median_us": statistics.median(values),
            "min_us": min(values),
            "max_us": max(values),
        })
    projection_groups = []
    for graph_id, events in sorted(by_graph.items()):
        ordered = sorted(events)
        projection_values: dict[str, list[float]] = defaultdict(list)
        index = 0
        while index < len(ordered):
            for projection, expected_grid in zip(PROJECTION_ORDER, EXPECTED_GRIDS):
                if index >= len(ordered):
                    raise RuntimeError(f"graph {graph_id} ended mid-layer")
                _, kind, grid_x, duration = ordered[index]
                if grid_x != expected_grid:
                    raise RuntimeError(
                        f"graph {graph_id} projection {projection} mismatch at "
                        f"{index}: expected grid {expected_grid}, got {grid_x}"
                    )
                if projection != "down_proj":
                    if kind != BASE_KERNEL:
                        raise RuntimeError(
                            f"graph {graph_id} expected base kernel for {projection}"
                        )
                    projection_values[projection].append(duration)
                    index += 1
                    continue
                if kind == BASE_KERNEL:
                    projection_values[projection].append(duration)
                    index += 1
                    continue
                if kind != SPLIT_KERNEL or index + 1 >= len(ordered):
                    raise RuntimeError(
                        f"graph {graph_id} has malformed split-K down projection"
                    )
                _, reduce_kind, _, reduce_duration = ordered[index + 1]
                if reduce_kind != REDUCE_KERNEL:
                    raise RuntimeError(
                        f"graph {graph_id} split-K partial is not followed by reduction"
                    )
                projection_values[projection].append(duration + reduce_duration)
                index += 2
        for projection in PROJECTION_ORDER:
            values = projection_values[projection]
            projection_groups.append({
                "graph_id": graph_id,
                "projection": projection,
                "count": len(values),
                "total_us": sum(values),
                "mean_us": statistics.fmean(values),
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
            })

    result = {
        "status": "passed",
        "kernels": list(KERNELS),
        "total_count": len(all_kernel_durations),
        "total_us": sum(all_kernel_durations),
        "groups": groups,
        "projection_groups": projection_groups,
        "shape_mapping": {
            "grid_x_32": "N=512: k_proj/v_proj",
            "grid_x_128": "N=2048: q_proj/o_proj/down_proj",
            "grid_x_512": "N=8192: gate_proj/up_proj",
        },
        "claim_boundary": (
            "Output width is inferred from the current kernel's 16-column "
            "grid. Projection families are assigned only after validating "
            "the repeated Q/K/V/O/gate/up/down grid sequence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
