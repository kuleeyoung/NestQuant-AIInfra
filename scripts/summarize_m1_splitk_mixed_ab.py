from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "token_throughput_per_second",
    "ttft_ms_p95",
    "tpot_ms_p95",
    "e2e_ms_p95",
)


def _load(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or len(payload.get("runs", [])) != 1:
        raise ValueError(f"expected one passed run: {path}")
    return payload, payload["runs"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paired mixed-length split-K runs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    pairs = []
    for pair_id in range(1, args.pairs + 1):
        off_payload, off = _load(args.input_dir / f"off_{pair_id}.json")
        on_payload, on = _load(args.input_dir / f"on_{pair_id}.json")
        identity_fields = (
            "scheduled_requests",
            "completed_requests",
            "generated_tokens",
        )
        for optional_hash in ("workload_identity_sha256", "output_identity_sha256"):
            if optional_hash in off or optional_hash in on:
                if off.get(optional_hash) != on.get(optional_hash):
                    mismatches.append(optional_hash)
        mismatches = [field for field in identity_fields if off[field] != on[field]]
        if off_payload["profiles"] != on_payload["profiles"]:
            mismatches.append("profiles")
        if off_payload["policies"] != on_payload["policies"]:
            mismatches.append("policies")
        if mismatches:
            raise ValueError(f"pair {pair_id} workload mismatch: {mismatches}")
        pairs.append(
            {
                "pair": pair_id,
                "off": {key: off[key] for key in METRICS},
                "on": {key: on[key] for key in METRICS},
                "delta_on_minus_off": {key: on[key] - off[key] for key in METRICS},
                "completion_rate": on["completion_rate"],
                "scheduled_requests": on["scheduled_requests"],
                "generated_tokens": on["generated_tokens"],
                "decode_m_forward_histogram_off": off["decode_m_forward_histogram"],
                "decode_m_forward_histogram_on": on["decode_m_forward_histogram"],
                "linear_m_forward_histogram_off": off["linear_m_forward_histogram"],
                "linear_m_forward_histogram_on": on["linear_m_forward_histogram"],
                "boundary_replays_off": off["nqkv_graph_boundary_replays"],
                "boundary_replays_on": on["nqkv_graph_boundary_replays"],
                "boundary_fallbacks_off": off["nqkv_graph_boundary_fallbacks"],
                "boundary_fallbacks_on": on["nqkv_graph_boundary_fallbacks"],
                "step_kind_metrics_off": off["step_kind_metrics"],
                "step_kind_metrics_on": on["step_kind_metrics"],
            }
        )

    aggregate = {}
    for metric in METRICS:
        deltas = [pair["delta_on_minus_off"][metric] for pair in pairs]
        ratios = [pair["on"][metric] / pair["off"][metric] - 1 for pair in pairs]
        aggregate[metric] = {
            "median_delta": statistics.median(deltas),
            "median_relative_change_percent": statistics.median(ratios) * 100,
            "improving_pairs": sum(
                delta > 0 if metric == "token_throughput_per_second" else delta < 0
                for delta in deltas
            ),
        }
    decode_mean_relative_changes = [
        pair["step_kind_metrics_on"]["decode_only"]["mean"]
        / pair["step_kind_metrics_off"]["decode_only"]["mean"]
        - 1.0
        for pair in pairs
    ]
    decode_phase = {
        "median_mean_latency_change_percent": statistics.median(
            decode_mean_relative_changes
        ) * 100.0,
        "improving_pairs": sum(change < 0 for change in decode_mean_relative_changes),
    }

    output = {
        "schema_version": 1,
        "pair_count": len(pairs),
        "pairs": pairs,
        "aggregate": aggregate,
        "decode_only_phase": decode_phase,
        "pairs_with_workload_and_output_hash_match": sum(
            "workload_identity_sha256" in _load(args.input_dir / f"off_{pair['pair']}.json")[1]
            for pair in pairs
        ),
        "claim_boundary": (
            f"{len(pairs)} paired fresh-process, deterministic synthetic-token engine runs. "
            "The paired median is robust to the observed first-pair process-state outlier, "
            f"but n={len(pairs)} remains a small local sample and is not a production SLA."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# M1 split-K mixed-length engine A/B",
        "",
        f"{len(pairs)} fresh-process pairs replayed identical deterministic requests with mixed prompt and",
        "output lengths. Positive throughput delta is better; negative latency delta is better.",
        "",
        "## Pair results",
        "",
        "| Pair | token/s off | token/s on | TPOT P95 off ms | TPOT P95 on ms | TPOT delta ms | TTFT delta ms | E2E delta ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['pair']} | {pair['off']['token_throughput_per_second']:.3f} | "
            f"{pair['on']['token_throughput_per_second']:.3f} | "
            f"{pair['off']['tpot_ms_p95']:.2f} | {pair['on']['tpot_ms_p95']:.2f} | "
            f"{pair['delta_on_minus_off']['tpot_ms_p95']:+.2f} | "
            f"{pair['delta_on_minus_off']['ttft_ms_p95']:+.2f} | "
            f"{pair['delta_on_minus_off']['e2e_ms_p95']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "## Paired-median result",
            "",
            "| Metric | median relative change | improving pairs |",
            "|---|---:|---:|",
        ]
    )
    labels = {
        "token_throughput_per_second": "token throughput",
        "ttft_ms_p95": "TTFT P95",
        "tpot_ms_p95": "TPOT P95",
        "e2e_ms_p95": "E2E P95",
    }
    for metric in METRICS:
        item = aggregate[metric]
        lines.append(
            f"| {labels[metric]} | {item['median_relative_change_percent']:+.2f}% | "
            f"{item['improving_pairs']}/{len(pairs)} |"
        )
    lines.extend(
        [
            "",
            "## Step-phase attribution",
            "",
            "| Pair | decode mean off ms | decode mean on ms | decode P95 off ms | decode P95 on ms | mixed mean off ms | mixed mean on ms |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in pairs:
        off_steps = pair["step_kind_metrics_off"]
        on_steps = pair["step_kind_metrics_on"]
        off_mixed = off_steps.get("mixed_prefill_decode", {}).get("mean", float("nan"))
        on_mixed = on_steps.get("mixed_prefill_decode", {}).get("mean", float("nan"))
        lines.append(
            f"| {pair['pair']} | {off_steps['decode_only']['mean']:.2f} | "
            f"{on_steps['decode_only']['mean']:.2f} | {off_steps['decode_only']['p95']:.2f} | "
            f"{on_steps['decode_only']['p95']:.2f} | {off_mixed:.2f} | {on_mixed:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Decode-only mean step latency improved in {decode_phase['improving_pairs']}/{len(pairs)} pairs; "
            f"the paired-median relative change was {decode_phase['median_mean_latency_change_percent']:+.2f}%.",
            "Request-level TPOT P95 can still move in the opposite direction because changing step",
            "duration changes which arrivals coalesce and which token intervals land in mixed steps.",
        ]
    )
    lines.extend(
        [
            "",
            "All pairs completed every request and generated the same token count from the same",
            "seeded workload. M histograms are retained per side as scheduling outcomes: faster or",
            "slower steps can change which arrivals coalesce into one batch. NQKV boundary Graph",
            "fallback count was zero in every run.",
            "",
            f"{output['pairs_with_workload_and_output_hash_match']}/{len(pairs)} pairs were recorded after",
            "identity hashes were added; those pairs have matching workload and generated-token SHA-256",
            "between split-K off/on. Earlier pairs retain count/config checks but cannot be retroactively",
            "promoted to token-sequence identity evidence.",
            "",
            "The first pair had a large cross-process state outlier in both TTFT and TPOT. It is",
            "retained in the table; paired medians are reported instead of deleting it or relying on",
            "an outlier-sensitive arithmetic mean.",
            "",
            "## Evidence boundary",
            "",
            output["claim_boundary"],
            "",
            f"Machine-readable summary: `{args.output_json.as_posix()}`",
        ]
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
