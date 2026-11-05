"""Pure planning for prefix-contiguous ragged NQKV multi-page appends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class NQKVWriteSegment:
    """One contiguous source range written into one hot-page destination."""

    source_start: int
    token_count: int
    destination_start: int
    flush_after: bool
    page_ordinal: int


@dataclass(frozen=True)
class NQKVRowAppendPlan:
    """All write segments and resulting metadata for one fixed batch row."""

    row: int
    initial_hot_length: int
    token_count: int
    initial_page_count: int
    required_pages: int
    final_hot_length: int
    segments: tuple[NQKVWriteSegment, ...]


@dataclass(frozen=True)
class NQKVRaggedAppendPlan:
    """Immutable multi-row plan consumed round-by-round by the GPU backend."""

    page_size: int
    rows: tuple[NQKVRowAppendPlan, ...]
    max_rounds: int
    total_required_pages: int

    @property
    def required_pages_per_row(self) -> tuple[int, ...]:
        return tuple(row.required_pages for row in self.rows)

    @property
    def final_hot_lengths(self) -> tuple[int, ...]:
        return tuple(row.final_hot_length for row in self.rows)

    def round(self, index: int) -> tuple[NQKVWriteSegment | None, ...]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("round index must be an integer")
        if not 0 <= index < self.max_rounds:
            raise IndexError("round index is outside the append plan")
        return tuple(
            row.segments[index] if index < len(row.segments) else None
            for row in self.rows
        )


def plan_ragged_multi_page_append(
    hot_lengths: Sequence[int],
    row_counts: Sequence[int],
    page_counts: Sequence[int],
    *,
    page_size: int,
    max_pages_per_sequence: int,
) -> NQKVRaggedAppendPlan:
    """Split every row into page-local segments without mutating cache state."""
    if isinstance(page_size, bool) or not isinstance(page_size, int):
        raise TypeError("page_size must be an integer")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if (
        isinstance(max_pages_per_sequence, bool)
        or not isinstance(max_pages_per_sequence, int)
    ):
        raise TypeError("max_pages_per_sequence must be an integer")
    if max_pages_per_sequence < 1:
        raise ValueError("max_pages_per_sequence must be positive")
    hot = _integer_tuple("hot_lengths", hot_lengths)
    counts = _integer_tuple("row_counts", row_counts)
    pages = _integer_tuple("page_counts", page_counts)
    if not (len(hot) == len(counts) == len(pages)):
        raise ValueError("hot_lengths, row_counts and page_counts must align")
    if not hot:
        raise ValueError("append plan requires at least one row")

    rows = []
    for row, (initial_hot, token_count, initial_pages) in enumerate(
        zip(hot, counts, pages, strict=True)
    ):
        if not 0 <= initial_hot < page_size:
            raise ValueError("hot length must be inside one partial page")
        if token_count < 0:
            raise ValueError("row token count cannot be negative")
        if not 0 <= initial_pages <= max_pages_per_sequence:
            raise ValueError("page count is outside sequence capacity")

        source_start = 0
        current_hot = initial_hot
        required_pages = 0
        segments = []
        while source_start < token_count:
            available = page_size - current_hot
            take = min(available, token_count - source_start)
            flush_after = current_hot + take == page_size
            segments.append(
                NQKVWriteSegment(
                    source_start=source_start,
                    token_count=take,
                    destination_start=current_hot,
                    flush_after=flush_after,
                    page_ordinal=required_pages,
                )
            )
            source_start += take
            if flush_after:
                required_pages += 1
                current_hot = 0
            else:
                current_hot += take

        if initial_pages + required_pages > max_pages_per_sequence:
            raise RuntimeError("append exceeds max_pages_per_sequence")
        rows.append(
            NQKVRowAppendPlan(
                row=row,
                initial_hot_length=initial_hot,
                token_count=token_count,
                initial_page_count=initial_pages,
                required_pages=required_pages,
                final_hot_length=current_hot,
                segments=tuple(segments),
            )
        )

    resolved = tuple(rows)
    return NQKVRaggedAppendPlan(
        page_size=page_size,
        rows=resolved,
        max_rounds=max(len(row.segments) for row in resolved),
        total_required_pages=sum(row.required_pages for row in resolved),
    )


def _integer_tuple(name: str, values: Sequence[int]) -> tuple[int, ...]:
    resolved = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in resolved):
        raise TypeError(f"{name} must contain integers")
    return resolved
