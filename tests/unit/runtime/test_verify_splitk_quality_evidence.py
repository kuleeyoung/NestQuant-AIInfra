from __future__ import annotations

import copy

import pytest

from nestquant.cli.verify_splitk_quality_evidence import verify_splitk_quality_evidence


def valid_analysis() -> dict:
    return {
        "schema_version": 1,
        "sample_count": 2048,
        "predicted_token_count": 65536,
        "relative_ppl_delta": -0.0009,
        "relative_ppl_delta_bootstrap_95ci": [-0.0056, 0.0037],
        "argmax_agreement_rate": 0.777,
        "mean_kl": 0.138,
        "reproducibility_max_abs_logit_error": 0.0,
    }


def valid_gate() -> dict:
    analysis = valid_analysis()
    return {
        "schema_version": 1,
        "status": "passed",
        "accepted": True,
        "thresholds": {
            "relative_ppl_non_inferiority_margin": 0.01,
            "minimum_contexts": 2048,
            "confidence_level": 0.95,
        },
        "checks": {
            "minimum_contexts": True,
            "ci_upper_within_margin": True,
            "ci_lower_above_margin": False,
            "deterministic_replay": True,
        },
        "evidence": {
            name: analysis[name]
            for name in (
                "sample_count",
                "predicted_token_count",
                "relative_ppl_delta",
                "relative_ppl_delta_bootstrap_95ci",
                "argmax_agreement_rate",
                "mean_kl",
            )
        },
    }


def test_verify_accepts_release_quality_contract() -> None:
    result = verify_splitk_quality_evidence(valid_gate(), valid_analysis())

    assert result["status"] == "passed"
    assert result["sample_count"] == 2048
    assert result["deterministic_replay"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("status", "not accepted"),
        ("margin", "exceeds"),
        ("contexts", "too few"),
        ("upper", "exceeds"),
        ("replay", "did not pass"),
        ("mismatch", "does not match"),
    ],
)
def test_verify_rejects_broken_quality_evidence(mutation: str, message: str) -> None:
    gate = copy.deepcopy(valid_gate())
    analysis = copy.deepcopy(valid_analysis())
    if mutation == "status":
        gate["status"] = "inconclusive"
        gate["accepted"] = False
    elif mutation == "margin":
        gate["thresholds"]["relative_ppl_non_inferiority_margin"] = 0.02
    elif mutation == "contexts":
        gate["thresholds"]["minimum_contexts"] = 1024
    elif mutation == "upper":
        gate["evidence"]["relative_ppl_delta_bootstrap_95ci"] = [-0.0056, 0.012]
        analysis["relative_ppl_delta_bootstrap_95ci"] = [-0.0056, 0.012]
    elif mutation == "replay":
        gate["checks"]["deterministic_replay"] = False
    elif mutation == "mismatch":
        analysis["mean_kl"] = 0.2
    with pytest.raises(ValueError, match=message):
        verify_splitk_quality_evidence(gate, analysis)
