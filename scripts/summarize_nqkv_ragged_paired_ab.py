"""Summarize same-process paired transactional versus ragged NQKV writer A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.runtime.service_benchmark import percentile


METRICS = {
    "token_throughput_per_second": "higher",
    "ttft_ms_p95": "lower",
    "tpot_ms_p95": "lower",
    "e2e_ms_p95": "lower",
}


def _bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(draw))
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--baseline-policy", default="transactional_2_0")
    parser.add_argument("--candidate-policy", default="ragged_gpu_2_0")
    parser.add_argument("--baseline-label", default="transactional")
    parser.add_argument("--candidate-label", default="ragged")
    parser.add_argument("--title", default="NQKV ragged writer same-process paired A/B")
    parser.add_argument(
        "--dispatch-field",
        default="nqkv_ragged_subpage_layer_launches",
        help="Per-run counter proving that only the candidate used the target path.",
    )
    parser.add_argument(
        "--skip-full-page-check",
        action="store_true",
        help="Do not additionally require the legacy full-page pack counter.",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    by_key = {
        (run["policy"], int(run["repetition"])): run
        for run in payload["runs"]
    }
    repetitions = sorted({int(run["repetition"]) for run in payload["runs"]})
    pairs = []
    for repetition in repetitions:
        transactional = by_key[(args.baseline_policy, repetition)]
        ragged = by_key[(args.candidate_policy, repetition)]
        if int(transactional[args.dispatch_field]) != 0:
            raise RuntimeError("baseline unexpectedly used target dispatch")
        if int(ragged[args.dispatch_field]) <= 0:
            raise RuntimeError("candidate did not use target dispatch")
        if (
            not args.skip_full_page_check
            and "nqkv_ragged_full_page_layer_packs" in ragged
        ):
            if int(transactional["nqkv_ragged_full_page_layer_packs"]) != 0:
                raise RuntimeError("baseline unexpectedly used full-page fusion")
            if int(ragged["nqkv_ragged_full_page_layer_packs"]) <= 0:
                raise RuntimeError("candidate did not dispatch full-page fusion")
        pairs.append((transactional, ragged))

    metrics = {}
    for metric_index, (metric, direction) in enumerate(METRICS.items()):
        baseline = [float(left[metric]) for left, _ in pairs]
        candidate = [float(right[metric]) for _, right in pairs]
        if direction == "higher":
            relative = [right / left - 1.0 for left, right in zip(baseline, candidate)]
            wins = sum(right > left for left, right in zip(baseline, candidate))
        else:
            relative = [1.0 - right / left for left, right in zip(baseline, candidate)]
            wins = sum(right < left for left, right in zip(baseline, candidate))
        relative_percent = [value * 100.0 for value in relative]
        metrics[metric] = {
            "direction": direction,
            "baseline_median": statistics.median(baseline),
            "candidate_median": statistics.median(candidate),
            "paired_improvement_percent_median": statistics.median(relative_percent),
            "paired_improvement_percent_bootstrap_95_ci": _bootstrap_ci(
                relative_percent,
                samples=args.bootstrap_samples,
                seed=20261102 + metric_index,
            ),
            "improved_pairs": wins,
            "total_pairs": len(pairs),
            "paired_improvement_percent_values": relative_percent,
        }

    result = {
        "status": "passed",
        "source": str(args.input),
        "pairs": len(pairs),
        "bootstrap_samples": args.bootstrap_samples,
        "metrics": metrics,
        "dispatch": {
            "field": args.dispatch_field,
            "baseline_launches": [
                int(left[args.dispatch_field])
                for left, _ in pairs
            ],
            "candidate_launches": [
                int(right[args.dispatch_field])
                for _, right in pairs
            ],
        },
        "claim_boundary": (
            "Same process/model/GPU and paired requests with alternating arm order. "
            "Bootstrap resamples paired relative improvements; token-ID engine only."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    labels = {
        "token_throughput_per_second": "token throughput",
        "ttft_ms_p95": "TTFT P95",
        "tpot_ms_p95": "TPOT P95",
        "e2e_ms_p95": "E2E P95",
    }
    lines = [
        f"# {args.title}", "",
        f"Pairs: {len(pairs)}; paired bootstrap resamples: {args.bootstrap_samples}.", "",
        f"| metric | {args.baseline_label} median | {args.candidate_label} median | paired median improvement | bootstrap 95% CI | improved pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, data in metrics.items():
        ci = data["paired_improvement_percent_bootstrap_95_ci"]
        lines.append(
            f"| {labels[metric]} | {data['baseline_median']:.2f} | "
            f"{data['candidate_median']:.2f} | "
            f"{data['paired_improvement_percent_median']:.2f}% | "
            f"[{ci[0]:.2f}%, {ci[1]:.2f}%] | "
            f"{data['improved_pairs']}/{data['total_pairs']} |"
        )
    lines.extend([
        "", "## Protocol boundary", "",
        "Both arms run in one process with one loaded model and one CUDA context. "
        "Each repetition shares identical requests/tokens; arm order alternates. "
        "The experiment excludes tokenizer/network I/O and is not a production SLA.", "",
    ])
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "pairs": result["pairs"], "metrics": metrics
    }, indent=2))


if __name__ == "__main__":
    main()
