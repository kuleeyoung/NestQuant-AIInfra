from __future__ import annotations

import pytest

from scripts.analyze_m1_splitk_teacher_forced import analyze, percentile


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 3.0], 0.5) == 2.0


def test_analysis_resamples_context_deltas() -> None:
    payload = {
        "protocol": "source",
        "sample_plan_sha256": "plan",
        "token_ids_sha256": "tokens",
        "summary": {
            "predicted_token_count": 4,
            "argmax_agreement_rate": 0.75,
            "mean_kl": 0.01,
            "reproducibility_max_abs_logit_error": 0.0,
        },
        "samples": [
            {"steps": [
                {"baseline_nll": 1.0, "candidate_nll": 1.1},
                {"baseline_nll": 2.0, "candidate_nll": 2.1},
            ]},
            {"steps": [
                {"baseline_nll": 3.0, "candidate_nll": 2.8},
                {"baseline_nll": 4.0, "candidate_nll": 3.8},
            ]},
        ],
    }
    result = analyze(payload, seed=1, bootstrap_repetitions=1000)

    assert result["sample_count"] == 2
    assert result["candidate_worse_context_count"] == 1
    assert result["candidate_better_context_count"] == 1
    assert result["paired_context_mean_nll_delta"] == pytest.approx(-0.05)
