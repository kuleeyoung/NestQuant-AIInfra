import pytest

from nestquant.runtime.hybrid_residency import (
    ResidencyGroup,
    atomic_projection_groups,
    nondominated_indices,
    select_residency_plan,
)


def llama_layer_names(layer: int = 0) -> tuple[str, ...]:
    root = f"model.layers.{layer}"
    return tuple(f"{root}.{suffix}" for suffix in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ))


def test_atomic_groups_preserve_shared_transform_families() -> None:
    groups = atomic_projection_groups(llama_layer_names())
    assert [len(group) for group in groups] == [3, 1, 2, 1]
    assert groups[0][0].endswith("q_proj")
    assert groups[2][0].endswith("gate_proj")


def test_plan_uses_latency_saved_per_incremental_mib() -> None:
    names = ("a", "b", "c")
    mib = 1024 * 1024
    groups = (
        ResidencyGroup("a", ("a",), 3 * mib, mib, 4.0),
        ResidencyGroup("b", ("b",), 2 * mib, mib, 3.0),
        ResidencyGroup("c", ("c",), 2 * mib, mib, -1.0),
    )
    plan = select_residency_plan(
        groups, budget_bytes=2 * mib, all_projection_names=names
    )
    assert plan.selected_group_names == ("b",)
    assert plan.resident_projection_names == ("b",)
    assert plan.packed_projection_names == ("a", "c")
    assert plan.predicted_extra_device_bytes == mib
    assert plan.predicted_saved_ms == 3.0


def test_plan_rejects_overlap_and_invalid_budget() -> None:
    groups = (
        ResidencyGroup("ab", ("a", "b"), 4, 2, 1.0),
        ResidencyGroup("b", ("b",), 2, 1, 1.0),
    )
    with pytest.raises(ValueError, match="partition"):
        select_residency_plan(groups, budget_bytes=2, all_projection_names=("a", "b"))
    with pytest.raises(ValueError, match="budget_bytes"):
        select_residency_plan((), budget_bytes=True, all_projection_names=())


def test_nondominated_indices_handles_tradeoffs_and_dominance() -> None:
    points = (
        {"memory": 1.0, "latency": 4.0},
        {"memory": 2.0, "latency": 3.0},
        {"memory": 3.0, "latency": 5.0},
    )
    assert nondominated_indices(
        points, metric_names=("memory", "latency")
    ) == (0, 1)
