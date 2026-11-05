from __future__ import annotations

from scripts.verify_m1_splitk_quality_gate import verify


def _analysis(lower: float, upper: float, samples: int = 2048) -> dict:
    return {
        "relative_ppl_delta_bootstrap_95ci": [lower, upper],
        "sample_count": samples,
        "predicted_token_count": samples * 32,
        "relative_ppl_delta": (lower + upper) / 2,
        "argmax_agreement_rate": 0.8,
        "mean_kl": 0.1,
        "reproducibility_max_abs_logit_error": 0.0,
    }


def test_gate_passes_when_upper_bound_is_within_margin() -> None:
    result = verify(_analysis(-0.01, 0.009), margin=0.01, min_contexts=2048)
    assert result["status"] == "passed"
    assert result["accepted"]


def test_gate_fails_when_lower_bound_exceeds_margin() -> None:
    result = verify(_analysis(0.011, 0.02), margin=0.01, min_contexts=2048)
    assert result["status"] == "failed"
    assert not result["accepted"]


def test_gate_is_inconclusive_for_crossing_or_small_sample() -> None:
    crossing = verify(_analysis(-0.01, 0.02), margin=0.01, min_contexts=2048)
    small = verify(
        _analysis(-0.01, 0.009, samples=1024), margin=0.01,
        min_contexts=2048,
    )
    assert crossing["status"] == "inconclusive"
    assert small["status"] == "inconclusive"
