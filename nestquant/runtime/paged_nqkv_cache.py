"""Graph-stable logical sequence tables over the CUDA NQKV page pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, cast

import torch

from ..core.kv_format import PackedNQKVPage
from .nqkv_pool import NQKVPageHandle, NQKVPagePool
from .nqkv_page import PreparedNQKVPage


_EMPTY_SLOT = object()


@dataclass
class _SequenceState:
    slot: int
    pages: list[NQKVPageHandle] = field(default_factory=list)
    reserved: list[NQKVPageHandle] = field(default_factory=list)


@dataclass(frozen=True)
class BatchedNQKVMetadata:
    """Fixed-address GPU metadata consumed by batched paged NQKV kernels.

    ``page_indices`` keeps a dense ``[capacity, max_pages]`` view for simple
    graph-stable indexing. ``page_indptr`` exposes the equivalent flattened
    CSR boundaries, so later kernels can consume the same allocation without
    rebuilding metadata when active sequence lengths differ.
    """

    slot_ids: torch.Tensor
    page_indptr: torch.Tensor
    page_indices: torch.Tensor
    page_counts: torch.Tensor
    last_page_lens: torch.Tensor
    hot_lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    active_mask: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        capacity: int,
        max_pages_per_sequence: int,
        device: torch.device | str,
    ) -> "BatchedNQKVMetadata":
        for name, value in (
            ("capacity", capacity),
            ("max_pages_per_sequence", max_pages_per_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        target = torch.device(device)
        return cls(
            slot_ids=torch.arange(capacity, dtype=torch.int32, device=target),
            page_indptr=torch.arange(
                0,
                (capacity + 1) * max_pages_per_sequence,
                max_pages_per_sequence,
                dtype=torch.int32,
                device=target,
            ),
            page_indices=torch.full(
                (capacity, max_pages_per_sequence),
                -1,
                dtype=torch.int32,
                device=target,
            ),
            page_counts=torch.zeros(capacity, dtype=torch.int32, device=target),
            last_page_lens=torch.zeros(capacity, dtype=torch.int32, device=target),
            hot_lengths=torch.zeros(capacity, dtype=torch.int32, device=target),
            sequence_lengths=torch.zeros(capacity, dtype=torch.int32, device=target),
            active_mask=torch.zeros(capacity, dtype=torch.bool, device=target),
        )

    def pointers(self) -> tuple[int, ...]:
        """Return every backing address for CUDA Graph stability checks."""
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.slot_ids,
                self.page_indptr,
                self.page_indices,
                self.page_counts,
                self.last_page_lens,
                self.hot_lengths,
                self.sequence_lengths,
                self.active_mask,
            )
        )


class PagedNQKVCache:
    """Own logical sequences and fixed-address CUDA page-table tensors."""

    def __init__(
        self,
        pool: NQKVPagePool,
        *,
        max_sequences: int,
        max_pages_per_sequence: int,
    ) -> None:
        if not isinstance(pool, NQKVPagePool):
            raise TypeError("pool must be an NQKVPagePool")
        for name, value in (
            ("max_sequences", max_sequences),
            ("max_pages_per_sequence", max_pages_per_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.pool = pool
        self.max_sequences = max_sequences
        self.max_pages_per_sequence = max_pages_per_sequence
        self.metadata = BatchedNQKVMetadata.create(
            capacity=max_sequences,
            max_pages_per_sequence=max_pages_per_sequence,
            device=pool.device,
        )
        # Compatibility aliases keep the existing single-sequence K1 path on
        # the same tensors while the batched kernels adopt ``metadata``.
        self.slot_ids = self.metadata.slot_ids
        self.page_indptr = self.metadata.page_indptr
        self.page_indices = self.metadata.page_indices
        self.page_counts = self.metadata.page_counts
        self.last_page_len = self.metadata.last_page_lens
        self.hot_lengths = self.metadata.hot_lengths
        self.sequence_lengths = self.metadata.sequence_lengths
        self.active = self.metadata.active_mask
        self._free_slots = list(range(max_sequences - 1, -1, -1))
        self._sequences: dict[Hashable, _SequenceState] = {}
        self._slot_sequences: list[object] = [_EMPTY_SLOT] * max_sequences

    @property
    def active_sequences(self) -> int:
        return len(self._sequences)

    def register(self, sequence_id: Hashable, *, slot: int | None = None) -> int:
        """Register a sequence, optionally binding it to a scheduler slot."""
        _validate_sequence_id(sequence_id)
        if sequence_id in self._sequences:
            raise ValueError("sequence is already registered")
        if not self._free_slots:
            raise RuntimeError("NQKV sequence table is full")
        if slot is None:
            slot = self._free_slots.pop()
        else:
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise TypeError("slot must be an integer or None")
            if not 0 <= slot < self.max_sequences:
                raise ValueError("slot is outside NQKV sequence capacity")
            if slot not in self._free_slots:
                raise RuntimeError("requested NQKV slot is already active")
            self._free_slots.remove(slot)
        self._sequences[sequence_id] = _SequenceState(slot=slot)
        self._slot_sequences[slot] = sequence_id
        self.page_indices[slot].fill_(-1)
        self.page_counts[slot] = 0
        self.last_page_len[slot] = 0
        self.hot_lengths[slot] = 0
        self.sequence_lengths[slot] = 0
        self.active[slot] = True
        return slot

    def append_page(self, sequence_id: Hashable, page: PackedNQKVPage) -> NQKVPageHandle:
        handle = self.reserve_page(sequence_id)
        try:
            self.pool.store(handle, page)
        except Exception:
            self.cancel_reserved_page(sequence_id, handle)
            raise
        return self._publish_reserved_page(sequence_id, handle, page.valid_tokens)

    def append_prepared_page(
        self, sequence_id: Hashable, page: PreparedNQKVPage
    ) -> NQKVPageHandle:
        """Append a page produced entirely on GPU."""
        handle = self.reserve_page(sequence_id)
        return self.commit_prepared_page(sequence_id, handle, page)

    def reserve_page(self, sequence_id: Hashable) -> NQKVPageHandle:
        """Reserve one physical page without publishing it to GPU page tables."""
        state = self._state(sequence_id)
        if len(state.pages) + len(state.reserved) >= self.max_pages_per_sequence:
            raise RuntimeError("sequence exceeds max_pages_per_sequence")
        if state.pages and int(self.last_page_len[state.slot].item()) != self.pool.page_size:
            raise RuntimeError("replace the partial last page before appending another page")
        handle = self.pool.allocate()
        state.reserved.append(handle)
        return handle

    def commit_prepared_page(
        self,
        sequence_id: Hashable,
        handle: NQKVPageHandle,
        page: PreparedNQKVPage,
    ) -> NQKVPageHandle:
        """Store and publish a page previously reserved by the control plane."""
        state = self._state(sequence_id)
        if handle not in state.reserved:
            raise RuntimeError("NQKV page handle is not reserved for this sequence")
        try:
            self.pool.store_prepared(handle, page)
        except Exception:
            self.cancel_reserved_page(sequence_id, handle)
            raise
        return self._publish_reserved_page(sequence_id, handle, page.valid_tokens)

    def commit_graph_page(
        self, sequence_id: Hashable, handle: NQKVPageHandle
    ) -> NQKVPageHandle:
        """Commit host ownership after a Graph published pool/table tensors."""
        state = self._state(sequence_id)
        try:
            state.reserved.remove(handle)
        except ValueError as error:
            raise RuntimeError(
                "NQKV graph page handle is not reserved for this sequence"
            ) from error
        state.pages.append(handle)
        self.pool.mark_graph_stored(handle, valid_tokens=self.pool.page_size)
        return handle

    def cancel_reserved_page(
        self, sequence_id: Hashable, handle: NQKVPageHandle
    ) -> None:
        """Return an unpublished reservation to the physical page pool."""
        state = self._state(sequence_id)
        try:
            state.reserved.remove(handle)
        except ValueError as error:
            raise RuntimeError(
                "NQKV page handle is not reserved for this sequence"
            ) from error
        self.pool.release(handle)

    def _publish_reserved_page(
        self, sequence_id: Hashable, handle: NQKVPageHandle, valid_tokens: int
    ) -> NQKVPageHandle:
        state = self._state(sequence_id)
        try:
            state.reserved.remove(handle)
        except ValueError as error:
            raise RuntimeError(
                "NQKV page handle is not reserved for this sequence"
            ) from error
        state.pages.append(handle)
        page_index = len(state.pages) - 1
        self.page_indices[state.slot, page_index] = handle.page_id
        self.page_counts[state.slot] = len(state.pages)
        self.last_page_len[state.slot] = valid_tokens
        self.sequence_lengths[state.slot] = self._cold_token_count(state)
        return handle

    def replace_last_page(self, sequence_id: Hashable, page: PackedNQKVPage) -> NQKVPageHandle:
        state = self._state(sequence_id)
        if not state.pages:
            raise RuntimeError("sequence has no page to replace")
        previous = int(self.last_page_len[state.slot].item())
        if page.valid_tokens < previous:
            raise ValueError("replacement page cannot shrink the valid prefix")
        handle = state.pages[-1]
        self.pool.store(handle, page)
        self.last_page_len[state.slot] = page.valid_tokens
        self.sequence_lengths[state.slot] = self._cold_token_count(state)
        return handle

    def release(self, sequence_id: Hashable) -> None:
        state = self._state(sequence_id)
        for handle in (*state.pages, *state.reserved):
            self.pool.release(handle)
        slot = state.slot
        self.page_indices[slot].fill_(-1)
        self.page_counts[slot] = 0
        self.last_page_len[slot] = 0
        self.hot_lengths[slot] = 0
        self.sequence_lengths[slot] = 0
        self.active[slot] = False
        del self._sequences[sequence_id]
        self._slot_sequences[slot] = _EMPTY_SLOT
        self._free_slots.append(slot)

    def slot(self, sequence_id: Hashable) -> int:
        return self._state(sequence_id).slot

    def handles(self, sequence_id: Hashable) -> tuple[NQKVPageHandle, ...]:
        return tuple(self._state(sequence_id).pages)

    def reserved_handles(self, sequence_id: Hashable) -> tuple[NQKVPageHandle, ...]:
        return tuple(self._state(sequence_id).reserved)

    def sequence_id(self, slot: int) -> Hashable:
        """Resolve an active scheduler slot back to its logical sequence ID."""
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        if not 0 <= slot < self.max_sequences:
            raise ValueError("slot is outside NQKV sequence capacity")
        sequence_id = self._slot_sequences[slot]
        if sequence_id is _EMPTY_SLOT:
            raise KeyError("NQKV slot is not active")
        return cast(Hashable, sequence_id)

    def total_tokens(self, sequence_id: Hashable) -> int:
        state = self._state(sequence_id)
        if not state.pages:
            return 0
        return (len(state.pages) - 1) * self.pool.page_size + int(
            self.last_page_len[state.slot].item()
        )

    def table_pointers(self) -> tuple[int, ...]:
        """Expose addresses for tests and future CUDA Graph capture checks."""
        return self.metadata.pointers()

    def _cold_token_count(self, state: _SequenceState) -> int:
        if not state.pages:
            return 0
        return (len(state.pages) - 1) * self.pool.page_size + int(
            self.last_page_len[state.slot].item()
        )

    def _state(self, sequence_id: Hashable) -> _SequenceState:
        _validate_sequence_id(sequence_id)
        try:
            return self._sequences[sequence_id]
        except KeyError as error:
            raise KeyError("sequence is not registered") from error


def _validate_sequence_id(sequence_id: Hashable) -> None:
    try:
        hash(sequence_id)
    except TypeError as error:
        raise TypeError("sequence_id must be hashable") from error
