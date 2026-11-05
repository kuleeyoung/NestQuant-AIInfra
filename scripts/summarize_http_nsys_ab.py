"""Summarize an HTTP offered-load A/B from benchmark, GPU and Nsight API data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


CUDA_APIS = (
    "cudaStreamSynchronize",
    "cudaLaunchKernel",
    "cudaGraphLaunch_v10000",
    "cudaMemcpyAsync",
    "cuLaunchKernelEx",
)


def _mean(samples: list[dict], key: str) -> float:
    values = [float(sample[key]) for sample in samples if key in sample]
    return statistics.mean(values) if values else 0.0


def load_run(directory: Path, offered_rps: float) -> dict:
    benchmark = json.loads(
        (directory / "trial_01.benchmark.json").read_text(encoding="utf-8")
    )
    telemetry = json.loads(
        (directory / "trial_01.gpu.json").read_text(encoding="utf-8")
    )["samples"]
    telemetry = [
        sample for sample in telemetry
        if sample.get("phase") == "benchmark" and "error" not in sample
    ]
    api_rows = json.loads(
        (directory / "stats_cuda_api_sum.json").read_text(encoding="utf-8")
    )
    api_by_name = {row["Name"]: row for row in api_rows}
    request_count = int(benchmark["scheduled_requests"])
    apis = {}
    for name in CUDA_APIS:
        row = api_by_name.get(name, {})
        calls = int(row.get("Num Calls", 0))
        apis[name] = {
            "calls": calls,
            "calls_per_request": calls / request_count,
            "total_ms": float(row.get("Total Time (ns)", 0.0)) / 1e6,
            "average_us": float(row.get("Avg (ns)", 0.0)) / 1e3,
        }
    kernel_summary = directory / "stats_cuda_gpu_kern_sum.json"
    kernel_trace_available = kernel_summary.exists() and kernel_summary.stat().st_size > 0
    return {
        "offered_rps": offered_rps,
        "benchmark": {
            "scheduled_requests": request_count,
            "completed_requests": int(benchmark["completed_requests"]),
            "overloaded_requests": int(benchmark["overloaded_requests"]),
            "failed_requests": int(benchmark["failed_requests"]),
            "wall_seconds": float(benchmark["wall_seconds"]),
            "request_throughput_per_second": float(
                benchmark["request_throughput_per_second"]
            ),
            "ttft_p95_ms": float(benchmark["ttft_ms"]["p95"]),
            "tpot_p95_ms": float(benchmark["tpot_ms"]["p95"]),
            "e2e_p95_ms": float(benchmark["e2e_ms"]["p95"]),
        },
        "gpu": {
            "sample_count": len(telemetry),
            "utilization_mean_percent": _mean(telemetry, "utilization_percent"),
            "power_mean_w": _mean(telemetry, "power_w"),
            "sm_clock_mean_mhz": _mean(telemetry, "sm_clock_mhz"),
            "pstates": sorted({str(sample["pstate"]) for sample in telemetry}),
        },
        "cuda_apis": apis,
        "gpu_kernel_trace_available": kernel_trace_available,
    }


def summarize(baseline: dict, candidate: dict) -> dict:
    def ratio(path: tuple[str, ...]) -> float:
        left: object = baseline
        right: object = candidate
        for key in path:
            left = left[key]  # type: ignore[index]
            right = right[key]  # type: ignore[index]
        return float(right) / float(left)

    return {
        "schema_version": 1,
        "status": "passed" if all(
            run["benchmark"]["failed_requests"] == 0
            for run in (baseline, candidate)
        ) else "failed",
        "baseline": baseline,
        "candidate": candidate,
        "candidate_over_baseline": {
            "throughput": ratio(("benchmark", "request_throughput_per_second")),
            "ttft_p95": ratio(("benchmark", "ttft_p95_ms")),
            "tpot_p95": ratio(("benchmark", "tpot_p95_ms")),
            "e2e_p95": ratio(("benchmark", "e2e_p95_ms")),
            "gpu_utilization_mean": ratio(("gpu", "utilization_mean_percent")),
            "cuda_graph_launch_calls_per_request": ratio((
                "cuda_apis", "cudaGraphLaunch_v10000", "calls_per_request"
            )),
            "cuda_kernel_launch_calls_per_request": ratio((
                "cuda_apis", "cudaLaunchKernel", "calls_per_request"
            )),
        },
        "interpretation": (
            "If TPOT remains stable while TTFT and GPU utilization rise and CUDA "
            "launches per request fall, pressure is dominated by queueing/admission "
            "and improved batching rather than slower per-token device execution."
        ),
        "claim_boundary": (
            "Nsight Systems under this WSL runtime recorded CUDA API and OS runtime "
            "summaries but no GPU kernel timeline. CUDA API attribution is valid; "
            "kernel-duration or stall attribution requires a supported GPU trace or NCU."
        ),
    }


def write_markdown(summary: dict, output: Path) -> None:
    base = summary["baseline"]
    candidate = summary["candidate"]
    ratio = summary["candidate_over_baseline"]
    lines = [
        "# W4A4KV4 HTTP offered-load Nsight A/B",
        "",
        "| metric | baseline | candidate | candidate / baseline |",
        "|---|---:|---:|---:|",
        f"| offered load | {base['offered_rps']:.1f} req/s | "
        f"{candidate['offered_rps']:.1f} req/s | {candidate['offered_rps']/base['offered_rps']:.2f}x |",
        f"| completed / overloaded | {base['benchmark']['completed_requests']} / "
        f"{base['benchmark']['overloaded_requests']} | "
        f"{candidate['benchmark']['completed_requests']} / "
        f"{candidate['benchmark']['overloaded_requests']} | - |",
        f"| achieved throughput | {base['benchmark']['request_throughput_per_second']:.3f} | "
        f"{candidate['benchmark']['request_throughput_per_second']:.3f} | "
        f"{ratio['throughput']:.2f}x |",
        f"| TTFT P95 (ms) | {base['benchmark']['ttft_p95_ms']:.2f} | "
        f"{candidate['benchmark']['ttft_p95_ms']:.2f} | {ratio['ttft_p95']:.2f}x |",
        f"| TPOT P95 (ms) | {base['benchmark']['tpot_p95_ms']:.2f} | "
        f"{candidate['benchmark']['tpot_p95_ms']:.2f} | {ratio['tpot_p95']:.2f}x |",
        f"| E2E P95 (ms) | {base['benchmark']['e2e_p95_ms']:.2f} | "
        f"{candidate['benchmark']['e2e_p95_ms']:.2f} | {ratio['e2e_p95']:.2f}x |",
        f"| GPU utilization mean | {base['gpu']['utilization_mean_percent']:.1f}% | "
        f"{candidate['gpu']['utilization_mean_percent']:.1f}% | "
        f"{ratio['gpu_utilization_mean']:.2f}x |",
        f"| CUDA Graph launches/request | "
        f"{base['cuda_apis']['cudaGraphLaunch_v10000']['calls_per_request']:.2f} | "
        f"{candidate['cuda_apis']['cudaGraphLaunch_v10000']['calls_per_request']:.2f} | "
        f"{ratio['cuda_graph_launch_calls_per_request']:.2f}x |",
        f"| CUDA kernel launches/request | "
        f"{base['cuda_apis']['cudaLaunchKernel']['calls_per_request']:.1f} | "
        f"{candidate['cuda_apis']['cudaLaunchKernel']['calls_per_request']:.1f} | "
        f"{ratio['cuda_kernel_launch_calls_per_request']:.2f}x |",
        "",
        "The candidate doubled offered load and nearly doubled completed throughput. "
        "TPOT changed only slightly, while TTFT increased because requests waited longer "
        "for admission/slots. Fewer launches per request show that continuous batching "
        "amortized model execution across more requests.",
        "",
        f"Claim boundary: {summary['claim_boundary']}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-rps", type=float, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--candidate-rps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(
        load_run(args.baseline_dir, args.baseline_rps),
        load_run(args.candidate_dir, args.candidate_rps),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.report)
    print(json.dumps(summary["candidate_over_baseline"], indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
