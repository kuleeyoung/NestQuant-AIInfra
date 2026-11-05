from __future__ import annotations

import pytest

import torch

from scripts.diagnose_m1_splitk_numeric_stability import (
    distribution,
    percentile,
    reduce_partials,
)


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 9.0], 0.95) == pytest.approx(8.3)


def test_distribution_reports_tail() -> None:
    result = distribution([1.0, 2.0, 9.0])
    assert result["count"] == 3
    assert result["p50"] == 2.0
    assert result["max"] == 9.0


def test_reduce_partials_exposes_all_orders() -> None:
    partials = torch.tensor([[1.0, 2.0], [3.0, -1.0], [4.0, 5.0], [-2.0, 1.0]])
    result = reduce_partials(partials)

    assert set(result) == {"fp32_sequential", "fp32_pairwise", "fp32_kahan", "fp64_sum"}
    for value in result.values():
        assert torch.equal(value, torch.tensor([6.0, 7.0]))
