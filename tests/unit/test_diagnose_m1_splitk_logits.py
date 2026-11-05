from __future__ import annotations

import torch

from scripts.diagnose_m1_splitk_logits import compare_traces


def _step(step: int, token: int, logits: list[float], inputs=None) -> dict:
    tensor = torch.tensor([logits])
    values, indices = torch.topk(tensor, k=2, dim=-1)
    return {
        "step": step,
        "input_token_ids": inputs if inputs is not None else [[step]],
        "selected_token_id": token,
        "top1_top2_margin": float((values[0, 0] - values[0, 1]).item()),
        "logits": tensor,
    }


def test_compare_traces_reports_first_divergence_and_same_input() -> None:
    baseline = [
        _step(0, 0, [2.0, 1.0], [[4, 5]]),
        _step(1, 0, [1.01, 1.0], [[0]]),
    ]
    candidate = [
        _step(0, 0, [2.0, 1.0], [[4, 5]]),
        _step(1, 1, [1.0, 1.01], [[0]]),
    ]

    result = compare_traces(baseline, candidate)

    assert result["first_token_divergence_step"] == 1
    assert not result["all_tokens_equal"]
    assert result["steps"][1]["same_input"]
    assert result["steps"][1]["max_abs_logit_error"] > 0.0


def test_compare_traces_marks_later_inputs_as_different() -> None:
    baseline = [_step(0, 0, [2.0, 1.0]), _step(1, 0, [2.0, 1.0], [[7]])]
    candidate = [_step(0, 0, [2.0, 1.0]), _step(1, 0, [2.0, 1.0], [[8]])]

    result = compare_traces(baseline, candidate)

    assert result["all_tokens_equal"]
    assert not result["steps"][1]["same_input"]
