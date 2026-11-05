"""Transactional multi-layer lifecycle for batched paged NQKV caches."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import cast

import torch

from .nqkv_attention_k1 import (
    BatchedNQKVAttentionWorkspace,
    _launch_nqkv_decode_attention_batched_k1,
    nqkv_decode_attention_batched_k1,
)
from .nqkv_batched_writer import BatchedNQKVWriteResult, BatchedNQKVWriter
from .nqkv_hot_cache import NQKVHotColdCache
from .nqkv_segment_plan import (
    NQKVRaggedAppendPlan,
    plan_ragged_multi_page_append,
)
from .nqkv_graph_writer import (
    _launch_append_hot_ragged_segment_graph_safe,
    _launch_append_hot_ragged_subpage_graph_safe,
)


_EMPTY_SLOT = object()


class BatchedNQKVCacheManager:
    """Keep request/slot ownership identical across every Transformer layer."""

    def __init__(
        self,
        layer_caches: Sequence[NQKVHotColdCache],
        *,
        query_heads: int,
    ) -> None:
        caches = tuple(layer_caches)
        if not caches or any(not isinstance(cache, NQKVHotColdCache) for cache in caches):
            raise ValueError("layer_caches must contain at least one NQKVHotColdCache")
        reference = self._configuration(caches[0])
        if any(self._configuration(cache) != reference for cache in caches[1:]):
            raise ValueError("all NQKV layer caches must share one configuration")
        capacity, _, _, num_kv_heads, _, _, _, _, _ = reference
        if query_heads < 1 or query_heads % num_kv_heads:
            raise ValueError("query_heads must be a positive multiple of num_kv_heads")
        self.layer_caches = caches
        self.capacity = capacity
        self.num_layers = len(caches)
        self.query_heads = query_heads
        self.device = caches[0].cold_cache.pool.device
        self.writers = tuple(BatchedNQKVWriter(cache) for cache in caches)
        self.workspaces = tuple(
            BatchedNQKVAttentionWorkspace.create(
                cache,
                batch_size=capacity,
                query_heads=query_heads,
                output_dtype=cache.cold_cache.pool.output_dtype,
            )
            for cache in caches
        )
        self._attention_workspaces = {
            (layer_idx, capacity, workspace.output.dtype): workspace
            for layer_idx, workspace in enumerate(self.workspaces)
        }
        self._requests: dict[Hashable, int] = {}
        self._slots: list[object] = [_EMPTY_SLOT] * capacity
        self._pending_graph_pages: dict[
            tuple[int, int], tuple[Hashable, object, int]
        ] = {}
        self._pending_multi_pages: dict[
            tuple[int, int], tuple[Hashable, tuple[object, ...], int]
        ] = {}

    @property
    def active_requests(self) -> int:
        return len(self._requests)

    @property
    def request_ids(self) -> tuple[Hashable, ...]:
        """Return registered request IDs without exposing mutable ownership state."""
        return tuple(self._requests)

    def can_append_dense_empty(self, token_count: int) -> bool:
        """Return whether every fixed slot/layer is active, empty and sub-page."""
        if token_count < 1:
            return False
        page_size = self.layer_caches[0].cold_cache.pool.page_size
        if token_count >= page_size or len(self._requests) != self.capacity:
            return False
        for slot in range(self.capacity):
            if self._slots[slot] is _EMPTY_SLOT:
                return False
        for cache in self.layer_caches:
            for slot in range(self.capacity):
                sequence_id = cache.cold_cache.sequence_id(slot)
                if cache.hot_length(slot) or cache.cold_cache.total_tokens(sequence_id):
                    return False
        return True

    def can_append_ragged_subpage(self, row_counts: Sequence[int]) -> bool:
        """Return whether fixed identity slots can append without filling a page."""
        counts = tuple(int(count) for count in row_counts)
        if len(counts) != self.capacity or not any(counts):
            return False
        page_size = self.layer_caches[0].cold_cache.pool.page_size
        if any(count < 0 or count >= page_size for count in counts):
            return False
        for cache in self.layer_caches:
            for slot, count in enumerate(counts):
                if not count:
                    continue
                try:
                    cache.cold_cache.sequence_id(slot)
                except KeyError:
                    return False
                if cache.hot_length(slot) + count >= page_size:
                    return False
        return True

    def ragged_single_page_boundary_slots(
        self, row_counts: Sequence[int]
    ) -> tuple[int, ...] | None:
        """Classify a ragged append that stays within one page per active row."""
        counts = tuple(int(count) for count in row_counts)
        if len(counts) != self.capacity or not any(counts):
            return None
        page_size = self.layer_caches[0].cold_cache.pool.page_size
        if any(count < 0 or count > page_size for count in counts):
            return None
        boundaries = []
        reference = self.layer_caches[0]
        for slot, count in enumerate(counts):
            if not count:
                continue
            try:
                reference.cold_cache.sequence_id(slot)
            except KeyError:
                return None
            target = reference.hot_length(slot) + count
            if target > page_size:
                return None
            if target == page_size:
                boundaries.append(slot)
        for cache in self.layer_caches[1:]:
            for slot, count in enumerate(counts):
                if count and cache.hot_length(slot) != reference.hot_length(slot):
                    return None
        return tuple(boundaries)

    def ragged_one_boundary_plan(
        self, row_counts: Sequence[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
        """Plan prefix rows that cross exactly one page boundary at most."""
        counts = tuple(int(count) for count in row_counts)
        if len(counts) != self.capacity or not any(counts):
            return None
        page_size = self.layer_caches[0].cold_cache.pool.page_size
        reference = self.layer_caches[0]
        first_counts = []
        second_counts = []
        final_lengths = []
        boundaries = []
        crossed = False
        for slot, count in enumerate(counts):
            if count < 0:
                return None
            if not count:
                first_counts.append(0)
                second_counts.append(0)
                final_lengths.append(reference.hot_length(slot) if self._slots[slot] is not _EMPTY_SLOT else 0)
                continue
            try:
                reference.cold_cache.sequence_id(slot)
            except KeyError:
                return None
            hot_length = reference.hot_length(slot)
            target = hot_length + count
            if target >= 2 * page_size:
                return None
            first = min(count, page_size - hot_length)
            second = count - first
            first_counts.append(first)
            second_counts.append(second)
            if target >= page_size:
                boundaries.append(slot)
                final_lengths.append(second)
                crossed = crossed or target > page_size
            else:
                final_lengths.append(target)
        if not crossed:
            return None
        for cache in self.layer_caches[1:]:
            for slot, count in enumerate(counts):
                if count and cache.hot_length(slot) != reference.hot_length(slot):
                    return None
        return (
            tuple(first_counts),
            tuple(second_counts),
            tuple(final_lengths),
            tuple(boundaries),
        )

    def plan_ragged_multi_page(
        self, row_counts: Sequence[int]
    ) -> NQKVRaggedAppendPlan:
        """Build a cache-state-checked immutable plan for every fixed row."""
        counts = tuple(int(count) for count in row_counts)
        if len(counts) != self.capacity:
            raise ValueError("row_counts must match fixed capacity")
        reference = self.layer_caches[0]
        hot_lengths = []
        page_counts = []
        for slot, count in enumerate(counts):
            if self._slots[slot] is _EMPTY_SLOT:
                if count:
                    raise RuntimeError("inactive rows cannot append NQKV tokens")
                hot_lengths.append(0)
                page_counts.append(0)
                continue
            hot_lengths.append(reference.hot_length(slot))
            page_counts.append(int(reference.cold_cache.page_counts[slot].item()))
        for cache in self.layer_caches[1:]:
            observed_hot = tuple(
                cache.hot_length(slot) if self._slots[slot] is not _EMPTY_SLOT else 0
                for slot in range(self.capacity)
            )
            observed_pages = tuple(
                int(cache.cold_cache.page_counts[slot].item())
                for slot in range(self.capacity)
            )
            if observed_hot != tuple(hot_lengths):
                raise RuntimeError("NQKV hot lengths diverged across layers")
            if observed_pages != tuple(page_counts):
                raise RuntimeError("NQKV page counts diverged across layers")
        return plan_ragged_multi_page_append(
            hot_lengths,
            counts,
            page_counts,
            page_size=reference.cold_cache.pool.page_size,
            max_pages_per_sequence=reference.cold_cache.max_pages_per_sequence,
        )

    def reserve_multi_page_append(self, plan: NQKVRaggedAppendPlan) -> None:
        """Atomically reserve every page required by a multi-row append plan."""
        if not isinstance(plan, NQKVRaggedAppendPlan):
            raise TypeError("plan must be an NQKVRaggedAppendPlan")
        if len(plan.rows) != self.capacity:
            raise ValueError("append plan must match fixed capacity")
        if self._pending_multi_pages:
            raise RuntimeError("a multi-page append reservation is already pending")
        required_per_layer = plan.total_required_pages
        if any(
            cache.cold_cache.pool.free_pages < required_per_layer
            for cache in self.layer_caches
        ):
            raise RuntimeError("insufficient NQKV physical pages for append plan")
        reservations = []
        try:
            for layer_idx, cache in enumerate(self.layer_caches):
                for row in plan.rows:
                    if not row.required_pages:
                        continue
                    sequence_id = cache.cold_cache.sequence_id(row.row)
                    handles = []
                    for page_offset in range(row.required_pages):
                        handle = cache.cold_cache.reserve_page(sequence_id)
                        page_index = row.initial_page_count + page_offset
                        cache.cold_cache.page_indices[row.row, page_index] = handle.page_id
                        handles.append(handle)
                        reservations.append(
                            (cache, sequence_id, row.row, page_index, handle)
                        )
                    self._pending_multi_pages[(layer_idx, row.row)] = (
                        sequence_id,
                        tuple(handles),
                        row.initial_page_count,
                    )
        except Exception:
            for cache, sequence_id, slot, page_index, handle in reversed(reservations):
                cache.cold_cache.page_indices[slot, page_index] = -1
                cache.cold_cache.cancel_reserved_page(sequence_id, handle)
            self._pending_multi_pages.clear()
            raise

    def cancel_multi_page_append(self) -> None:
        """Cancel all unpublished pages held for the current multi-page plan."""
        pending = tuple(self._pending_multi_pages.items())
        self._pending_multi_pages.clear()
        for (layer_idx, slot), (sequence_id, handles, first_page) in reversed(pending):
            cache = self.layer_caches[layer_idx]
            for offset, handle in reversed(tuple(enumerate(handles))):
                cache.cold_cache.page_indices[slot, first_page + offset] = -1
                cache.cold_cache.cancel_reserved_page(sequence_id, handle)

    def prepare_multi_page_round(
        self,
        layer_idx: int,
        segments: Sequence[object | None],
    ) -> None:
        """Point graph target IDs at the reserved page used by one round."""
        self._validate_layer(layer_idx)
        if len(segments) != self.capacity:
            raise ValueError("round segments must match fixed capacity")
        cache = self.layer_caches[layer_idx]
        cache.graph_target_page_ids.zero_()
        for slot, segment in enumerate(segments):
            if segment is None or not getattr(segment, "flush_after"):
                continue
            pending = self._pending_multi_pages.get((layer_idx, slot))
            if pending is None:
                raise RuntimeError("missing multi-page reservation for flush segment")
            ordinal = int(getattr(segment, "page_ordinal"))
            handles = pending[1]
            if not 0 <= ordinal < len(handles):
                raise RuntimeError("flush segment page ordinal is not reserved")
            cache.graph_target_page_ids[slot] = handles[ordinal].page_id

    def commit_multi_page_append(self, plan: NQKVRaggedAppendPlan) -> None:
        """Commit host ownership after every planned page was GPU-published."""
        if not isinstance(plan, NQKVRaggedAppendPlan):
            raise TypeError("plan must be an NQKVRaggedAppendPlan")
        pending = tuple(self._pending_multi_pages.items())
        for (layer_idx, slot), (sequence_id, handles, _) in pending:
            cache = self.layer_caches[layer_idx]
            for handle in handles:
                cache.cold_cache.commit_graph_page(sequence_id, handle)
            cache.synchronize_host_length(slot, plan.rows[slot].final_hot_length)
        self._pending_multi_pages.clear()

    def pending_multi_page_ids(
        self, layer_idx: int, slot: int
    ) -> tuple[int, ...]:
        """Return reserved physical IDs for one layer/slot without publishing them."""
        self._validate_layer(layer_idx)
        self._validate_slot(slot)
        pending = self._pending_multi_pages.get((layer_idx, slot))
        if pending is None:
            return ()
        return tuple(handle.page_id for handle in pending[1])

    def ragged_multi_page_plan(
        self, row_counts: Sequence[int]
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
        tuple[int, ...],
        tuple[int, ...],
    ] | None:
        """Plan prefix rows that flush two or more pages in one prefill.

        Each returned segment fits in the currently active hot page.  A row
        that fills its page appears in the matching ``boundary_slots`` entry;
        the following segment therefore starts at hot offset zero.  The plan
        is host-only and does not reserve or mutate any physical page.

        Returns ``(source_starts, segment_counts, boundary_slots,
        final_lengths, flush_counts)``.  ``None`` deliberately leaves zero- or
        one-boundary appends on the already validated fast paths.
        """
        try:
            plan = self.plan_ragged_multi_page(row_counts)
        except (TypeError, ValueError, RuntimeError, KeyError):
            return None
        if max(plan.required_pages_per_row, default=0) < 2:
            return None
        starts_by_segment = []
        counts_by_segment = []
        boundaries_by_segment = []
        for round_index in range(plan.max_rounds):
            segments = plan.round(round_index)
            starts_by_segment.append(tuple(
                segment.source_start if segment is not None else plan.rows[row].token_count
                for row, segment in enumerate(segments)
            ))
            counts_by_segment.append(tuple(
                segment.token_count if segment is not None else 0
                for segment in segments
            ))
            boundaries_by_segment.append(tuple(
                row for row, segment in enumerate(segments)
                if segment is not None and segment.flush_after
            ))
        return (
            tuple(starts_by_segment),
            tuple(counts_by_segment),
            tuple(boundaries_by_segment),
            plan.final_hot_lengths,
            plan.required_pages_per_row,
        )

    def register(self, sequence_id: Hashable, *, slot: int) -> None:
        if sequence_id in self._requests:
            raise ValueError("sequence is already registered in NQKV manager")
        self._validate_slot(slot)
        if self._slots[slot] is not _EMPTY_SLOT:
            raise RuntimeError("requested NQKV manager slot is already active")
        registered: list[NQKVHotColdCache] = []
        try:
            for layer_idx, cache in enumerate(self.layer_caches):
                cache.register(sequence_id, slot=slot)
                registered.append(cache)
        except Exception:
            for cache in reversed(registered):
                cache.release(sequence_id)
            raise
        self._requests[sequence_id] = slot
        self._slots[slot] = sequence_id

    def register_batch(self, assignments: Sequence[tuple[Hashable, int]]) -> None:
        """Register one admission microbatch atomically across requests/layers."""
        committed: list[Hashable] = []
        try:
            for sequence_id, slot in assignments:
                self.register(sequence_id, slot=slot)
                committed.append(sequence_id)
        except Exception:
            for sequence_id in reversed(committed):
                self.release(sequence_id)
            raise

    def release(self, sequence_id: Hashable) -> int:
        try:
            slot = self._requests[sequence_id]
        except KeyError as error:
            raise KeyError("sequence is not registered in NQKV manager") from error
        for layer_idx, cache in enumerate(self.layer_caches):
            cache.release(sequence_id)
            self._pending_graph_pages.pop((layer_idx, slot), None)
        del self._requests[sequence_id]
        self._slots[slot] = _EMPTY_SLOT
        return slot

    def release_slots(self, slots: Sequence[int]) -> tuple[Hashable, ...]:
        released = []
        for slot in slots:
            self._validate_slot(slot)
            sequence_id = self._slots[slot]
            if sequence_id is _EMPTY_SLOT:
                continue
            resolved = cast(Hashable, sequence_id)
            self.release(resolved)
            released.append(resolved)
        return tuple(released)

    def append_layer(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> BatchedNQKVWriteResult:
        self._validate_layer(layer_idx)
        return self.writers[layer_idx].append(
            key, value, slot_ids=slot_ids, write_mask=write_mask
        )

    def append_layer_dense_empty(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> BatchedNQKVWriteResult:
        """Bulk-fill an identity-slot dense prefill for one empty layer."""
        self._validate_layer(layer_idx)
        return self.writers[layer_idx].append_dense_empty(key, value)

    def append_layer_ragged_subpage(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        write_mask: torch.Tensor,
        row_counts: Sequence[int],
    ) -> BatchedNQKVWriteResult:
        """Launch a prevalidated GPU ragged append and advance its host mirror."""
        self._validate_layer(layer_idx)
        counts = tuple(int(count) for count in row_counts)
        if len(counts) != self.capacity:
            raise ValueError("row_counts must match fixed capacity")
        cache = self.layer_caches[layer_idx]
        _launch_append_hot_ragged_subpage_graph_safe(
            key,
            value,
            cache,
            slot_ids=slot_ids,
            write_mask=write_mask,
        )
        active_slots = []
        for slot, count in enumerate(counts):
            if count:
                cache.synchronize_host_length(
                    slot, cache.hot_length(slot) + count
                )
                active_slots.append(slot)
        return BatchedNQKVWriteResult(
            appended_tokens=sum(counts),
            flushed_pages=0,
            active_slots=tuple(active_slots),
        )

    def append_layer_ragged_segment(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        source_starts: torch.Tensor,
        segment_counts: torch.Tensor,
    ) -> None:
        """Launch one prevalidated compact source segment per fixed row."""
        self._validate_layer(layer_idx)
        _launch_append_hot_ragged_segment_graph_safe(
            key,
            value,
            self.layer_caches[layer_idx],
            slot_ids=slot_ids,
            source_starts=source_starts,
            segment_counts=segment_counts,
        )

    def attend_layer(
        self,
        layer_idx: int,
        query: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor:
        self._validate_layer(layer_idx)
        workspace = self._attention_workspace(
            layer_idx,
            batch_size=query.shape[0],
            output_dtype=query.dtype,
        )
        return nqkv_decode_attention_batched_k1(
            query,
            self.layer_caches[layer_idx],
            slot_ids=slot_ids,
            active_mask=active_mask,
            scale=scale,
            workspace=workspace,
        )

    def attend_layer_graph_safe(
        self,
        layer_idx: int,
        query: torch.Tensor,
        *,
        slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Launch fixed-address K1 attention after capture-time validation."""
        self._validate_layer(layer_idx)
        workspace = self._attention_workspace(
            layer_idx,
            batch_size=query.shape[0],
            output_dtype=query.dtype,
        )
        return _launch_nqkv_decode_attention_batched_k1(
            query,
            self.layer_caches[layer_idx],
            slot_ids=slot_ids,
            active_mask=active_mask,
            attention_scale=float(scale),
            workspace=workspace,
        )

    def _attention_workspace(
        self,
        layer_idx: int,
        *,
        batch_size: int,
        output_dtype: torch.dtype,
    ) -> BatchedNQKVAttentionWorkspace:
        """Return one stable-address attention workspace per execution shape."""
        if batch_size < 1 or batch_size > self.capacity:
            raise ValueError("attention batch_size must be within cache capacity")
        key = (layer_idx, batch_size, output_dtype)
        workspace = self._attention_workspaces.get(key)
        if workspace is None:
            workspace = BatchedNQKVAttentionWorkspace.create(
                self.layer_caches[layer_idx],
                batch_size=batch_size,
                query_heads=self.query_heads,
                output_dtype=output_dtype,
            )
            self._attention_workspaces[key] = workspace
        return workspace

    def slot(self, sequence_id: Hashable) -> int:
        try:
            return self._requests[sequence_id]
        except KeyError as error:
            raise KeyError("sequence is not registered in NQKV manager") from error

    def sequence_id(self, slot: int) -> Hashable:
        self._validate_slot(slot)
        sequence_id = self._slots[slot]
        if sequence_id is _EMPTY_SLOT:
            raise KeyError("NQKV manager slot is not active")
        return cast(Hashable, sequence_id)

    def assert_aligned(self) -> None:
        """Raise if any layer disagrees with manager request/slot ownership."""
        for sequence_id, slot in self._requests.items():
            for layer_idx, cache in enumerate(self.layer_caches):
                if cache.cold_cache.slot(sequence_id) != slot:
                    raise RuntimeError("NQKV layer slot ownership is misaligned")
        for slot, sequence_id in enumerate(self._slots):
            expected = sequence_id is not _EMPTY_SLOT
            for cache in self.layer_caches:
                actual = bool(cache.cold_cache.active[slot].item())
                if actual != expected:
                    raise RuntimeError("NQKV layer active masks are misaligned")

    def synchronize_graph_hot_lengths(self, slots: Sequence[int]) -> tuple[int, ...]:
        """Mirror graph-mutated GPU hot lengths into every layer's host state."""
        resolved = tuple(slots)
        for slot in resolved:
            self._validate_slot(slot)
        if not resolved:
            return ()
        indices = torch.tensor(resolved, device=self.device, dtype=torch.long)
        lengths = tuple(
            int(value)
            for value in self.layer_caches[0].hot_lengths[indices].cpu().tolist()
        )
        for cache in self.layer_caches:
            observed = tuple(
                int(value) for value in cache.hot_lengths[indices].cpu().tolist()
            )
            if observed != lengths:
                raise RuntimeError("NQKV graph hot lengths diverged across layers")
            for slot, length in zip(resolved, lengths, strict=True):
                cache.synchronize_host_length(slot, length)
        return lengths

    def reserve_graph_transition_pages(self, slots: Sequence[int]) -> None:
        """Reserve one unpublished physical page per slot and layer atomically."""
        resolved = tuple(slots)
        if len(set(resolved)) != len(resolved):
            raise ValueError("graph transition slots must be unique")
        for slot in resolved:
            self._validate_slot(slot)
        if not resolved:
            return
        indices = torch.tensor(resolved, device=self.device, dtype=torch.long)
        page_counts = tuple(
            int(value)
            for value in self.layer_caches[0].cold_cache.page_counts[indices]
            .cpu().tolist()
        )
        reservations = []
        try:
            for layer_idx, cache in enumerate(self.layer_caches):
                observed = tuple(
                    int(value)
                    for value in cache.cold_cache.page_counts[indices].cpu().tolist()
                )
                if observed != page_counts:
                    raise RuntimeError("NQKV graph page counts diverged across layers")
                for slot, page_index in zip(resolved, page_counts, strict=True):
                    sequence_id = cache.cold_cache.sequence_id(slot)
                    handle = cache.cold_cache.reserve_page(sequence_id)
                    cache.cold_cache.page_indices[slot, page_index] = handle.page_id
                    cache.graph_target_page_ids[slot] = handle.page_id
                    reservations.append((cache, sequence_id, slot, page_index, handle))
                    self._pending_graph_pages[(layer_idx, slot)] = (
                        sequence_id, handle, page_index
                    )
        except Exception:
            for cache, sequence_id, slot, page_index, handle in reversed(reservations):
                cache.cold_cache.page_indices[slot, page_index] = -1
                cache.graph_target_page_ids[slot] = 0
                cache.cold_cache.cancel_reserved_page(sequence_id, handle)
            for key in tuple(self._pending_graph_pages):
                if key[1] in resolved:
                    del self._pending_graph_pages[key]
            raise

    def commit_graph_transition_pages(self, slots: Sequence[int]) -> None:
        """Commit host handle ownership after the boundary Graph replay."""
        resolved = tuple(slots)
        for slot in resolved:
            self._validate_slot(slot)
        for layer_idx, cache in enumerate(self.layer_caches):
            for slot in resolved:
                try:
                    sequence_id, handle, _ = self._pending_graph_pages.pop(
                        (layer_idx, slot)
                    )
                except KeyError as error:
                    raise RuntimeError(
                        "missing pending NQKV graph page reservation"
                    ) from error
                cache.cold_cache.commit_graph_page(sequence_id, handle)
                cache.synchronize_host_length(slot, 0)

    def cancel_graph_transition_pages(self, slots: Sequence[int]) -> None:
        """Roll back unpublished cross-layer graph page reservations."""
        resolved = tuple(slots)
        for slot in resolved:
            self._validate_slot(slot)
        for layer_idx, cache in enumerate(self.layer_caches):
            for slot in resolved:
                pending = self._pending_graph_pages.pop((layer_idx, slot), None)
                if pending is None:
                    continue
                sequence_id, handle, page_index = pending
                published_count = int(cache.cold_cache.page_counts[slot].item())
                cache.cold_cache.page_indices[slot, page_index] = -1
                if published_count > page_index:
                    cache.cold_cache.page_counts[slot] = page_index
                    cache.cold_cache.last_page_len[slot] = (
                        cache.cold_cache.pool.page_size if page_index else 0
                    )
                    cache.hot_lengths[slot] = cache.cold_cache.pool.page_size
                    cache.synchronize_host_length(
                        slot, cache.cold_cache.pool.page_size
                    )
                cache.graph_target_page_ids[slot] = 0
                cache.cold_cache.cancel_reserved_page(sequence_id, handle)

    def _validate_slot(self, slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        if not 0 <= slot < self.capacity:
            raise ValueError("slot is outside NQKV manager capacity")

    def _validate_layer(self, layer_idx: int) -> None:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError("layer_idx must be an integer")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is outside NQKV manager layers")

    @staticmethod
    def _configuration(cache: NQKVHotColdCache) -> tuple:
        cold = cache.cold_cache
        pool = cold.pool
        return (
            cold.max_sequences,
            cold.max_pages_per_sequence,
            pool.page_size,
            pool.num_kv_heads,
            pool.head_dim,
            pool.q,
            pool.betas,
            pool.value_beta_values,
            (pool.output_dtype, pool.device),
        )
