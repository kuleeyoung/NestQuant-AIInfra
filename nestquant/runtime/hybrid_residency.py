"""Deterministic packed/BF16 residency planning for Llama projections."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class ResidencyGroup:
    """One atomic set of projections that switch residency together."""

    name: str
    projection_names: tuple[str, ...]
    dense_bytes: int
    packed_bytes: int
    predicted_saved_ms: float

    def __post_init__(self) -> None:
        if not self.name or not self.projection_names:
            raise ValueError("residency groups require names and projections")
        if len(set(self.projection_names)) != len(self.projection_names):
            raise ValueError("projection names must be unique within a group")
        if min(self.dense_bytes, self.packed_bytes) < 0:
            raise ValueError("residency byte counts must be nonnegative")
        if not math.isfinite(self.predicted_saved_ms):
            raise ValueError("predicted_saved_ms must be finite")

    @property
    def extra_device_bytes(self) -> int:
        """Incremental bytes when packed streams are replaced by BF16 weights."""
        return max(0, self.dense_bytes - self.packed_bytes)

    @property
    def saved_ms_per_mib(self) -> float:
        extra_mib = self.extra_device_bytes / (1024 * 1024)
        if extra_mib == 0:
            return math.inf if self.predicted_saved_ms > 0 else 0.0
        return self.predicted_saved_ms / extra_mib


@dataclass(frozen=True)
class ResidencyPlan:
    budget_bytes: int
    selected_group_names: tuple[str, ...]
    resident_projection_names: tuple[str, ...]
    packed_projection_names: tuple[str, ...]
    predicted_extra_device_bytes: int
    predicted_saved_ms: float


def atomic_projection_groups(names: Iterable[str]) -> tuple[tuple[str, ...], ...]:
    """Group projections that share an input transform into atomic choices."""
    ordered = tuple(names)
    if not ordered or any(not isinstance(name, str) or not name for name in ordered):
        raise ValueError("projection names must be nonempty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("projection names must be unique")

    remaining = set(ordered)
    result: list[tuple[str, ...]] = []
    suffix_groups = (
        ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        ("self_attn.o_proj",),
        ("mlp.gate_proj", "mlp.up_proj"),
        ("mlp.down_proj",),
    )
    prefixes = sorted({name.rsplit(".", 2)[0] for name in ordered})
    for prefix in prefixes:
        for suffixes in suffix_groups:
            group = tuple(f"{prefix}.{suffix}" for suffix in suffixes)
            present = tuple(name for name in group if name in remaining)
            if present:
                result.append(present)
                remaining.difference_update(present)
    result.extend((name,) for name in sorted(remaining))
    return tuple(result)


def make_residency_groups(
    projection_names: Iterable[str],
    *,
    dense_bytes: Mapping[str, int],
    packed_bytes: Mapping[str, int],
    predicted_saved_ms: Mapping[str, float],
) -> tuple[ResidencyGroup, ...]:
    groups = []
    for members in atomic_projection_groups(projection_names):
        missing = [
            name for name in members
            if name not in dense_bytes or name not in packed_bytes
            or name not in predicted_saved_ms
        ]
        if missing:
            raise ValueError(f"projection metrics are incomplete: {missing}")
        groups.append(ResidencyGroup(
            name="+".join(members),
            projection_names=members,
            dense_bytes=sum(int(dense_bytes[name]) for name in members),
            packed_bytes=sum(int(packed_bytes[name]) for name in members),
            predicted_saved_ms=sum(float(predicted_saved_ms[name]) for name in members),
        ))
    return tuple(groups)


def residency_groups_from_payload(
    payload: Iterable[Mapping[str, object]],
) -> tuple[ResidencyGroup, ...]:
    groups = []
    for item in payload:
        try:
            groups.append(ResidencyGroup(
                name=str(item["name"]),
                projection_names=tuple(item["projection_names"]),
                dense_bytes=int(item["dense_bytes"]),
                packed_bytes=int(item["packed_device_bytes"]),
                predicted_saved_ms=float(item["weighted_predicted_saved_ms"]),
            ))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid residency group payload") from error
    if not groups:
        raise ValueError("residency group payload must not be empty")
    return tuple(groups)


def select_residency_plan(
    groups: Iterable[ResidencyGroup],
    *,
    budget_bytes: int,
    all_projection_names: Iterable[str],
) -> ResidencyPlan:
    """Greedily select positive-benefit groups by saved milliseconds per MiB."""
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 0:
        raise ValueError("budget_bytes must be a nonnegative integer")
    all_names = tuple(all_projection_names)
    if len(set(all_names)) != len(all_names):
        raise ValueError("all_projection_names must be unique")
    known = set(all_names)
    candidates = tuple(groups)
    grouped = [name for group in candidates for name in group.projection_names]
    if len(set(grouped)) != len(grouped) or set(grouped) != known:
        raise ValueError("residency groups must partition all projections")

    ranked = sorted(
        (group for group in candidates if group.predicted_saved_ms > 0),
        key=lambda group: (
            -group.saved_ms_per_mib,
            -group.predicted_saved_ms,
            group.extra_device_bytes,
            group.name,
        ),
    )
    selected: list[ResidencyGroup] = []
    used = 0
    for group in ranked:
        if used + group.extra_device_bytes <= budget_bytes:
            selected.append(group)
            used += group.extra_device_bytes
    resident = {name for group in selected for name in group.projection_names}
    return ResidencyPlan(
        budget_bytes=budget_bytes,
        selected_group_names=tuple(group.name for group in selected),
        resident_projection_names=tuple(name for name in all_names if name in resident),
        packed_projection_names=tuple(name for name in all_names if name not in resident),
        predicted_extra_device_bytes=used,
        predicted_saved_ms=sum(group.predicted_saved_ms for group in selected),
    )


def nondominated_indices(
    points: Sequence[Mapping[str, float]], *, metric_names: Sequence[str]
) -> tuple[int, ...]:
    """Return indices not dominated when every named metric is minimized."""
    if not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError("metric_names must be nonempty and unique")
    values = []
    for point in points:
        row = tuple(float(point[name]) for name in metric_names)
        if not all(math.isfinite(value) for value in row):
            raise ValueError("Pareto metrics must be finite")
        values.append(row)
    keep = []
    for index, current in enumerate(values):
        if not any(
            other_index != index
            and all(left <= right for left, right in zip(other, current))
            and any(left < right for left, right in zip(other, current))
            for other_index, other in enumerate(values)
        ):
            keep.append(index)
    return tuple(keep)
