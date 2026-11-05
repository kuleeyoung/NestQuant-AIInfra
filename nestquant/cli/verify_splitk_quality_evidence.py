"""Verify a promoted M1 split-K quality gate against its bootstrap analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def verify_splitk_quality_evidence(
    gate: dict[str, Any],
    analysis: dict[str, Any],
    *,
    maximum_margin: float = 0.01,
    minimum_contexts: int = 2048,
) -> dict[str, Any]:
    """Validate the release contract and cross-check gate/analysis metrics."""
    if maximum_margin <= 0 or minimum_contexts < 1:
        raise ValueError("verification thresholds must be positive")
    if gate.get("schema_version") != 1 or analysis.get("schema_version") != 1:
        raise ValueError("unsupported evidence schema")
    if gate.get("status") != "passed" or gate.get("accepted") is not True:
        raise ValueError("split-K quality gate is not accepted")

    thresholds = gate.get("thresholds")
    checks = gate.get("checks")
    evidence = gate.get("evidence")
    if not all(isinstance(value, dict) for value in (thresholds, checks, evidence)):
        raise ValueError("gate thresholds, checks and evidence must be objects")

    margin = _finite_number(
        thresholds.get("relative_ppl_non_inferiority_margin"), "gate margin"
    )
    confidence = _finite_number(thresholds.get("confidence_level"), "confidence level")
    declared_contexts = thresholds.get("minimum_contexts")
    if margin > maximum_margin:
        raise ValueError("gate margin exceeds the release maximum")
    if confidence != 0.95:
        raise ValueError("quality gate must use a 95% confidence interval")
    if not isinstance(declared_contexts, int) or declared_contexts < minimum_contexts:
        raise ValueError("gate declares too few contexts")
    if checks.get("minimum_contexts") is not True:
        raise ValueError("minimum-context check did not pass")
    if checks.get("ci_upper_within_margin") is not True:
        raise ValueError("CI upper-bound check did not pass")
    if checks.get("deterministic_replay") is not True:
        raise ValueError("deterministic replay check did not pass")

    contexts = evidence.get("sample_count")
    predicted_tokens = evidence.get("predicted_token_count")
    if not isinstance(contexts, int) or contexts < minimum_contexts:
        raise ValueError("quality evidence has too few contexts")
    if not isinstance(predicted_tokens, int) or predicted_tokens < contexts:
        raise ValueError("quality evidence has an invalid token count")
    interval = evidence.get("relative_ppl_delta_bootstrap_95ci")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("quality evidence must contain a two-sided PPL interval")
    lower = _finite_number(interval[0], "PPL CI lower bound")
    upper = _finite_number(interval[1], "PPL CI upper bound")
    point = _finite_number(evidence.get("relative_ppl_delta"), "PPL delta")
    if lower > point or point > upper:
        raise ValueError("PPL point estimate lies outside its confidence interval")
    if upper > margin or upper > maximum_margin:
        raise ValueError("PPL confidence interval exceeds the release margin")

    comparisons = {
        "sample_count": analysis.get("sample_count"),
        "predicted_token_count": analysis.get("predicted_token_count"),
        "relative_ppl_delta": analysis.get("relative_ppl_delta"),
        "relative_ppl_delta_bootstrap_95ci": analysis.get(
            "relative_ppl_delta_bootstrap_95ci"
        ),
        "argmax_agreement_rate": analysis.get("argmax_agreement_rate"),
        "mean_kl": analysis.get("mean_kl"),
    }
    for name, value in comparisons.items():
        if evidence.get(name) != value:
            raise ValueError(f"gate {name} does not match bootstrap analysis")
    if analysis.get("reproducibility_max_abs_logit_error") != 0.0:
        raise ValueError("bootstrap analysis did not reproduce deterministically")

    return {
        "status": "passed",
        "accepted": True,
        "sample_count": contexts,
        "predicted_token_count": predicted_tokens,
        "relative_ppl_delta": point,
        "relative_ppl_delta_bootstrap_95ci": [lower, upper],
        "maximum_release_margin": maximum_margin,
        "deterministic_replay": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--maximum-margin", type=float, default=0.01)
    parser.add_argument("--minimum-contexts", type=int, default=2048)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    result = verify_splitk_quality_evidence(
        gate,
        analysis,
        maximum_margin=args.maximum_margin,
        minimum_contexts=args.minimum_contexts,
    )
    result["gate_sha256"] = sha256_file(args.gate)
    result["analysis_sha256"] = sha256_file(args.analysis)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
