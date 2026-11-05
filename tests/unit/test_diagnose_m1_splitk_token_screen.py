from __future__ import annotations

import pytest

from scripts.diagnose_m1_splitk_token_screen import generate_token_prompts


def test_generate_token_prompts_is_deterministic_and_length_stratified() -> None:
    first = generate_token_prompts(
        vocab_size=4096, lengths=(2, 5), samples_per_length=2, seed=17
    )
    second = generate_token_prompts(
        vocab_size=4096, lengths=(2, 5), samples_per_length=2, seed=17
    )

    assert first == second
    assert [len(tokens) for _, tokens in first] == [2, 2, 5, 5]
    assert all(1000 <= token < 4095 for _, tokens in first for token in tokens)


def test_generate_token_prompts_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_token_prompts(
            vocab_size=4096, lengths=(0,), samples_per_length=1, seed=17
        )
