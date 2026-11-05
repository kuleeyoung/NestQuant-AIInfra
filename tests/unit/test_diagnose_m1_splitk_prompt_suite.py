from __future__ import annotations

import torch

from scripts.diagnose_m1_splitk_prompt_suite import policy_summary


def _step(step: int, token: int, logits: list[float], inputs: list[list[int]]) -> dict:
    tensor = torch.tensor([logits])
    values, _ = torch.topk(tensor, k=2, dim=-1)
    return {
        "step": step,
        "input_token_ids": inputs,
        "selected_token_id": token,
        "top1_top2_margin": float(values[0, 0] - values[0, 1]),
        "logits": tensor,
    }


def test_policy_summary_limits_error_to_same_input_prefix() -> None:
    baseline = [
        _step(0, 0, [2.0, 1.0], [[4]]),
        _step(1, 0, [2.0, 1.0], [[0]]),
        _step(2, 0, [2.0, 1.0], [[0]]),
    ]
    candidate = [
        _step(0, 0, [2.0, 1.0], [[4]]),
        _step(1, 1, [1.0, 2.0], [[0]]),
        _step(2, 1, [100.0, 0.0], [[1]]),
    ]

    result = policy_summary(baseline, candidate)

    assert result["first_divergence_step"] == 1
    assert result["max_abs_logit_error_on_same_input"] == 1.0
    assert not result["tokens_equal"]
