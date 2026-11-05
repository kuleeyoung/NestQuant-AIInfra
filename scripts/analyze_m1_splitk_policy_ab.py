from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = sum((x - x_mean) ** 2 for x in xs)
    y_norm = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_norm * y_norm)
    return numerator / denominator if denominator else None


def _gpu_summary(path: Path) -> dict[str, float | int | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = [sample for sample in payload["samples"] if sample["phase"] == "benchmark"]
    fields = {
        "gpu_utilization_percent": "utilization_percent",
        "power_w": "power_w",
        "sm_clock_mhz": "sm_clock_mhz",
        "temperature_c": "temperature_c",
        "memory_used_mib": "memory_used_mib",
    }
    result: dict[str, float | int | None] = {"sample_count": len(samples)}
    for output_name, input_name in fields.items():
        values = [float(sample[input_name]) for sample in samples if sample.get(input_name) is not None]
        result[f"{output_name}_mean"] = _mean(values) if values else None
        result[f"{output_name}_min"] = min(values) if values else None
        result[f"{output_name}_max"] = max(values) if values else None
    return result


def _load_run(root: Path, mode: str, pair: int) -> dict[str, Any]:
    summary_path = root / f"{mode}_{pair}_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    trial = payload["trials"][0]
    benchmark = trial["benchmark"]
    runtime = trial["runtime_metrics"]["runtime"]
    graph_m1 = int(runtime.get("graph_m1_replays", 0))
    graph_m2 = int(runtime.get("graph_m2_replays", 0))
    graph_m4 = int(runtime.get("graph_m4_replays", 0))
    graph_total = graph_m1 + graph_m2 + graph_m4
    return {
        "mode": mode,
        "pair": pair,
        "request_throughput_per_second": float(benchmark["request_throughput_per_second"]),
        "ttft_p95_ms": float(benchmark["ttft_ms"]["p95"]),
        "tpot_p95_ms": float(benchmark["tpot_ms"]["p95"]),
        "e2e_p95_ms": float(benchmark["e2e_ms"]["p95"]),
        "scheduled_lag_mean_ms": float(benchmark["scheduled_lag_ms"]["mean"]),
        "command_queue_mean_ms": float(benchmark["server_timing_ms"]["command_queue"]["mean"]),
        "waiting_queue_mean_ms": float(benchmark["server_timing_ms"]["waiting_queue"]["mean"]),
        "prefill_compute_mean_ms": float(benchmark["server_timing_ms"]["prefill_compute"]["mean"]),
        "graph_m1_replays": graph_m1,
        "graph_m2_replays": graph_m2,
        "graph_m4_replays": graph_m4,
        "graph_m1_share": graph_m1 / graph_total if graph_total else None,
        "compact_prefill_calls": int(runtime.get("compact_prefill_calls", 0)),
        "compact_prefill_rows": int(runtime.get("compact_prefill_rows", 0)),
        "workload_identity_sha256": benchmark.get("workload_identity_sha256"),
        "output_identity_sha256": benchmark.get("output_identity_sha256"),
        "gpu": _gpu_summary(root / f"{mode}_{pair}" / "trial_01.gpu.json"),
    }


def _delta(candidate: float, baseline: float) -> float:
    return candidate - baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired M1 split-K service A/B telemetry.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/raw/http_m1_splitk_policy_ab_v1"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/raw/http_m1_splitk_policy_ab_v1_analysis.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("results/w4a4_m1_down_splitk_variance_analysis_v1.md"),
    )
    args = parser.parse_args()

    pair_ids = sorted(
        int(path.stem.removeprefix("off_").removesuffix("_summary"))
        for path in args.input_dir.glob("off_*_summary.json")
    )
    pairs: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        off = _load_run(args.input_dir, "off", pair_id)
        on = _load_run(args.input_dir, "on", pair_id)
        for hash_name in ("workload_identity_sha256", "output_identity_sha256"):
            if off[hash_name] is not None or on[hash_name] is not None:
                if off[hash_name] != on[hash_name]:
                    raise ValueError(f"pair {pair_id} {hash_name} mismatch")
        delta = {
            "request_throughput_per_second": _delta(
                on["request_throughput_per_second"], off["request_throughput_per_second"]
            ),
            "ttft_p95_ms": _delta(on["ttft_p95_ms"], off["ttft_p95_ms"]),
            "tpot_p95_ms": _delta(on["tpot_p95_ms"], off["tpot_p95_ms"]),
            "e2e_p95_ms": _delta(on["e2e_p95_ms"], off["e2e_p95_ms"]),
            "scheduled_lag_mean_ms": _delta(
                on["scheduled_lag_mean_ms"], off["scheduled_lag_mean_ms"]
            ),
            "command_queue_mean_ms": _delta(
                on["command_queue_mean_ms"], off["command_queue_mean_ms"]
            ),
            "waiting_queue_mean_ms": _delta(
                on["waiting_queue_mean_ms"], off["waiting_queue_mean_ms"]
            ),
            "prefill_compute_mean_ms": _delta(
                on["prefill_compute_mean_ms"], off["prefill_compute_mean_ms"]
            ),
            "graph_m1_replays": on["graph_m1_replays"] - off["graph_m1_replays"],
            "graph_m2_replays": on["graph_m2_replays"] - off["graph_m2_replays"],
            "graph_m1_share": _delta(on["graph_m1_share"], off["graph_m1_share"]),
        }
        for key in ("gpu_utilization_percent_mean", "power_w_mean", "sm_clock_mhz_mean", "temperature_c_mean"):
            delta[key] = _delta(float(on["gpu"][key]), float(off["gpu"][key]))
        pairs.append({"pair": pair_id, "off": off, "on": on, "delta_on_minus_off": delta})

    correlation_inputs = [
        "graph_m1_replays",
        "graph_m1_share",
        "scheduled_lag_mean_ms",
        "command_queue_mean_ms",
        "waiting_queue_mean_ms",
        "prefill_compute_mean_ms",
        "gpu_utilization_percent_mean",
        "power_w_mean",
        "sm_clock_mhz_mean",
        "temperature_c_mean",
    ]
    tpot_deltas = [pair["delta_on_minus_off"]["tpot_p95_ms"] for pair in pairs]
    correlations = {
        key: _pearson(
            [float(pair["delta_on_minus_off"][key]) for pair in pairs], tpot_deltas
        )
        for key in correlation_inputs
    }
    metric_names = (
        "request_throughput_per_second",
        "ttft_p95_ms",
        "tpot_p95_ms",
        "e2e_p95_ms",
    )
    aggregate = {}
    for key in metric_names:
        off_values = [float(pair["off"][key]) for pair in pairs]
        on_values = [float(pair["on"][key]) for pair in pairs]
        deltas = [float(pair["delta_on_minus_off"][key]) for pair in pairs]
        off_mean = _mean(off_values)
        on_mean = _mean(on_values)
        aggregate[key] = {
            "off_mean": off_mean,
            "on_mean": on_mean,
            "mean_delta": _mean(deltas),
            "median_delta": statistics.median(deltas),
            "relative_mean_change_percent": (on_mean / off_mean - 1.0) * 100.0,
            "improving_pair_count": sum(
                delta > 0 if key == "request_throughput_per_second" else delta < 0
                for delta in deltas
            ),
        }
    order_groups = {}
    for candidate_first, label in ((False, "candidate_second_odd_pairs"), (True, "candidate_first_even_pairs")):
        selected = [pair for pair in pairs if (pair["pair"] % 2 == 0) == candidate_first]
        order_groups[label] = {
            "pair_count": len(selected),
            "tpot_p95_mean_delta_ms": _mean(
                [float(pair["delta_on_minus_off"]["tpot_p95_ms"]) for pair in selected]
            ),
            "ttft_p95_mean_delta_ms": _mean(
                [float(pair["delta_on_minus_off"]["ttft_p95_ms"]) for pair in selected]
            ),
        }
    output = {
        "schema_version": 1,
        "input_dir": str(args.input_dir),
        "pair_count": len(pairs),
        "pairs": pairs,
        "aggregate": aggregate,
        "execution_order_groups": order_groups,
        "pearson_correlation_with_tpot_p95_delta": correlations,
        "claim_boundary": (
            "Descriptive diagnosis over five paired fresh-process runs. Correlations are hypothesis "
            "signals only; n=5 is too small for causal or statistical claims."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# M1 down-projection split-K variance analysis ({len(pairs)} pairs)",
        "",
        f"This report diagnoses {len(pairs)} 4 req/s paired runs. A positive delta means the split-K",
        "candidate is larger than baseline; for latency, positive is a regression.",
        "",
        "## Aggregate result",
        "",
        "| Metric | split-K off mean | split-K on mean | mean change | paired median delta | improving pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "request_throughput_per_second": ("request throughput (req/s)", ""),
        "ttft_p95_ms": ("TTFT P95 (ms)", " ms"),
        "tpot_p95_ms": ("TPOT P95 (ms)", " ms"),
        "e2e_p95_ms": ("E2E P95 (ms)", " ms"),
    }
    for key in metric_names:
        data = aggregate[key]
        label, suffix = labels[key]
        lines.append(
            f"| {label} | {data['off_mean']:.3f} | {data['on_mean']:.3f} | "
            f"{data['relative_mean_change_percent']:+.2f}% | "
            f"{data['median_delta']:+.3f}{suffix} | {data['improving_pair_count']}/{len(pairs)} |"
        )
    lines.extend(
        [
            "",
            "Candidate-second odd pairs: TPOT mean delta "
            f"{order_groups['candidate_second_odd_pairs']['tpot_p95_mean_delta_ms']:+.3f} ms. "
            "Candidate-first even pairs: TPOT mean delta "
            f"{order_groups['candidate_first_even_pairs']['tpot_p95_mean_delta_ms']:+.3f} ms.",
            "",
            "## Pair-aligned evidence",
        "",
        "| Pair | TPOT P95 delta ms | TTFT P95 delta ms | M1 replay delta | M1 share delta | "
        "command queue delta ms | prefill delta ms | GPU util delta pp | power delta W | SM clock delta MHz |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        d = pair["delta_on_minus_off"]
        lines.append(
            f"| {pair['pair']} | {d['tpot_p95_ms']:+.3f} | {d['ttft_p95_ms']:+.3f} | "
            f"{d['graph_m1_replays']:+d} | {d['graph_m1_share']:+.4f} | "
            f"{d['command_queue_mean_ms']:+.3f} | {d['prefill_compute_mean_ms']:+.3f} | "
            f"{d['gpu_utilization_percent_mean']:+.2f} | {d['power_w_mean']:+.2f} | "
            f"{d['sm_clock_mhz_mean']:+.1f} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive correlations with TPOT P95 delta",
            "",
            f"Pearson r ranges from -1 to +1. Because there are only {len(pairs)} pairs, these values are",
            "used to rank follow-up hypotheses, not to prove causality.",
            "",
            "| Candidate signal delta | Pearson r |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(
        correlations.items(), key=lambda item: abs(item[1]) if item[1] is not None else -1, reverse=True
    ):
        lines.append(f"| `{key}` | {value:+.3f} |" if value is not None else f"| `{key}` | n/a |")
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            f"This is a descriptive diagnosis over {len(pairs)} fresh-process pairs. The sample is too small",
            "for a statistical or causal claim. Any production-default change still requires a larger",
            "paired run or a deterministic replay that controls the scheduling trace.",
            "",
            f"Machine-readable evidence: `{args.output_json.as_posix()}`",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(args.output_json), "report": str(args.output_report)}, indent=2))


if __name__ == "__main__":
    main()
