from __future__ import annotations

import math

import torch

from scripts.evaluate_m1_splitk_teacher_forced import (
    compare_forced_logits,
    max_trace_difference,
    summarize,
    write_json_atomic,
)


def test_compare_forced_logits_uses_identical_targets() -> None:
    baseline = [torch.tensor([[1.0, 2.0, 0.0]])]
    candidate = [torch.tensor([[1.0, 1.5, 2.5]])]
    result = compare_forced_logits(baseline, candidate, (1,))

    assert result["steps"][0]["target_token_id"] == 1
    assert not result["steps"][0]["argmax_equal"]
    assert result["steps"][0]["candidate_nll"] > result["steps"][0]["baseline_nll"]


def test_summarize_reports_paired_nll_and_agreement() -> None:
    samples = [{"steps": [
        {
            "baseline_nll": 1.0,
            "candidate_nll": 1.1,
            "argmax_equal": True,
            "baseline_to_candidate_kl": 0.01,
            "relative_l2_logit_error": 0.02,
            "max_abs_logit_error": 0.5,
        },
        {
            "baseline_nll": 2.0,
            "candidate_nll": 1.9,
            "argmax_equal": False,
            "baseline_to_candidate_kl": 0.03,
            "relative_l2_logit_error": 0.04,
            "max_abs_logit_error": 0.7,
        },
    ]}]
    result = summarize(samples)

    assert result["predicted_token_count"] == 2
    assert result["mean_nll_delta"] == 0.0
    assert result["argmax_agreement_rate"] == 0.5
    assert result["max_abs_logit_error"] == 0.7
    assert result["relative_perplexity_delta"] == math.exp(0.0) - 1.0


def test_max_trace_difference_detects_replay_drift() -> None:
    left = [torch.tensor([[1.0, 2.0]])]
    right = [torch.tensor([[1.0, 2.25]])]
    assert max_trace_difference(left, right) == 0.25


def test_write_json_atomic_replaces_destination(tmp_path) -> None:
    destination = tmp_path / "progress.json"
    write_json_atomic(destination, {"completed": 1})
    write_json_atomic(destination, {"completed": 2})
    assert destination.read_text(encoding="utf-8") == '{\n  "completed": 2\n}\n'
    assert not destination.with_suffix(".json.tmp").exists()
