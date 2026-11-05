"""Fixed-capacity Transformers Llama integration for batched NQKV K1."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
import math

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..core.rotation import hadamard
from .nqkv_batch_manager import BatchedNQKVCacheManager
from .nqkv_hot_cache import NQKVHotColdCache
from .nqkv_graph_writer import _launch_append_hot_k1_graph_safe
from .nqkv_graph_pack import launch_pack_full_hot_pages_graph_safe
from .nqkv_segment_plan import NQKVRaggedAppendPlan
from .nqkv_pool import NQKVPagePool
from .paged_nqkv_cache import PagedNQKVCache


BATCHED_NQKV_ATTENTION_NAME = "nestquant_batched_nqkv_k1"


class BatchedNQKVLlamaCache(Cache):
    """Multi-layer fixed-capacity NQKV cache driven by scheduler slot state."""

    def __init__(
        self,
        *,
        capacity: int,
        num_layers: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_cache_len: int,
        page_size: int,
        q: int,
        key_betas: Sequence[float],
        value_betas: Sequence[float],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        super().__init__(layers=[])
        for name, value in (
            ("capacity", capacity),
            ("num_layers", num_layers),
            ("num_query_heads", num_query_heads),
            ("num_kv_heads", num_kv_heads),
            ("head_dim", head_dim),
            ("max_cache_len", max_cache_len),
            ("page_size", page_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_query_heads % num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        target = torch.device(device)
        if target.type == "cuda" and target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        max_pages = math.ceil(max_cache_len / page_size)
        layer_caches = []
        for _ in range(num_layers):
            pool = NQKVPagePool(
                capacity=capacity * max_pages,
                page_size=page_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q=q,
                key_betas=key_betas,
                value_betas=value_betas,
                output_dtype=dtype,
                device=target,
            )
            layer_caches.append(
                NQKVHotColdCache(
                    PagedNQKVCache(
                        pool,
                        max_sequences=capacity,
                        max_pages_per_sequence=max_pages,
                    )
                )
            )
        self.manager = BatchedNQKVCacheManager(
            layer_caches, query_heads=num_query_heads
        )
        self.capacity = capacity
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # Transformers 5.17+ exposes ``Cache.max_cache_len`` as a read-only
        # compatibility property that delegates to get_max_length(). Store
        # our backing value privately so this backend works on both sides of
        # that upstream API transition.
        self._max_cache_len = max_cache_len
        self.page_size = page_size
        self.dtype = dtype
        self.device = target
        self.slot_ids = torch.arange(capacity, device=target, dtype=torch.int32)
        self.active_mask = torch.zeros(capacity, device=target, dtype=torch.bool)
        self.execution_slot_ids = self.slot_ids
        self.execution_active_mask = self.active_mask
        self._write_mask: torch.Tensor | None = None
        self._dense_empty_prefill = False
        self._ragged_subpage_counts: tuple[int, ...] | None = None
        self.dense_prefill_layer_launch_count = 0
        self.ragged_subpage_layer_launch_count = 0
        self.ragged_subpage_enabled = True
        self.ragged_full_page_pack_enabled = True
        self.ragged_cross_page_enabled = False
        self.ragged_multi_page_enabled = False
        self.multi_page_batched_pack_enabled = True
        self._ragged_boundary_slots: tuple[int, ...] = ()
        self._ragged_cross_page_plan: tuple[
            tuple[int, ...], tuple[int, ...], tuple[int, ...]
        ] | None = None
        self._ragged_first_counts_gpu: torch.Tensor | None = None
        self._ragged_second_counts_gpu: torch.Tensor | None = None
        self._ragged_zero_starts_gpu = torch.zeros(
            capacity, device=target, dtype=torch.int32
        )
        self._ragged_multi_page_plan: NQKVRaggedAppendPlan | None = None
        self._ragged_multi_round_gpu: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool], ...
        ] = ()
        self.ragged_full_page_layer_pack_count = 0
        self.ragged_cross_page_layer_launch_count = 0
        self.ragged_multi_page_layer_round_count = 0
        self.multi_page_batched_pack_count = 0
        self._graph_decode_enabled = False
        self._graph_boundary_pack_enabled = False
        self.boundary_mask = torch.zeros(
            capacity, device=target, dtype=torch.bool
        )

    def register(self, sequence_id: Hashable, *, slot: int) -> None:
        self.manager.register(sequence_id, slot=slot)

    def register_batch(self, assignments: Sequence[tuple[Hashable, int]]) -> None:
        self.manager.register_batch(assignments)

    def release_slots(self, slots: Sequence[int]) -> tuple[Hashable, ...]:
        return self.manager.release_slots(slots)

    def set_batch_state(
        self,
        *,
        active_mask: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> None:
        if active_mask.dtype != torch.bool or active_mask.shape != (self.capacity,):
            raise ValueError("active_mask must be bool with shape [capacity]")
        if write_mask.dtype != torch.bool or write_mask.ndim != 2:
            raise ValueError("write_mask must be bool with shape [capacity, tokens]")
        if write_mask.shape[0] != self.capacity:
            raise ValueError("write_mask batch must equal cache capacity")
        if active_mask.device != self.device or write_mask.device != self.device:
            raise ValueError("batch state tensors must be on the cache device")
        if bool((write_mask.any(dim=1) & ~active_mask).any().item()):
            raise ValueError("inactive rows cannot request KV writes")
        self.active_mask.copy_(active_mask)
        self.execution_slot_ids = self.slot_ids
        self.execution_active_mask = self.active_mask
        self._write_mask = write_mask
        masks = write_mask.detach().cpu()
        row_counts = tuple(int(row.sum().item()) for row in masks)
        prefix_mask = True
        for row, count in enumerate(row_counts):
            if count and not bool(masks[row, :count].all().item()):
                prefix_mask = False
                break
            if bool(masks[row, count:].any().item()):
                prefix_mask = False
                break
        self._dense_empty_prefill = bool(
            prefix_mask
            and all(count == write_mask.shape[1] for count in row_counts)
            and self.manager.can_append_dense_empty(write_mask.shape[1])
        )
        self._ragged_subpage_counts = (
            row_counts
            if (
                prefix_mask
                and not self._dense_empty_prefill
                and self.ragged_subpage_enabled
                and self.manager.can_append_ragged_subpage(row_counts)
            )
            else None
        )
        self._ragged_boundary_slots = ()
        self._ragged_cross_page_plan = None
        self._ragged_first_counts_gpu = None
        self._ragged_second_counts_gpu = None
        self._ragged_multi_page_plan = None
        self._ragged_multi_round_gpu = ()
        if (
            prefix_mask
            and not self._dense_empty_prefill
            and self.ragged_subpage_enabled
            and self.ragged_full_page_pack_enabled
        ):
            multi_plan = (
                self.manager.plan_ragged_multi_page(row_counts)
                if self.ragged_multi_page_enabled else None
            )
            use_multi = bool(
                multi_plan is not None
                and max(multi_plan.required_pages_per_row, default=0) >= 2
            )
            if use_multi:
                assert multi_plan is not None
                self._ragged_subpage_counts = None
                self._ragged_multi_page_plan = multi_plan
                rounds = []
                for round_index in range(multi_plan.max_rounds):
                    segments = multi_plan.round(round_index)
                    starts = torch.tensor(
                        [segment.source_start if segment is not None else 0 for segment in segments],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    counts = torch.tensor(
                        [segment.token_count if segment is not None else 0 for segment in segments],
                        device=self.device,
                        dtype=torch.int32,
                    )
                    flush_mask = torch.tensor(
                        [bool(segment is not None and segment.flush_after) for segment in segments],
                        device=self.device,
                        dtype=torch.bool,
                    )
                    has_flush = any(
                        segment is not None and segment.flush_after
                        for segment in segments
                    )
                    rounds.append((starts, counts, flush_mask, has_flush))
                self._ragged_multi_round_gpu = tuple(rounds)
                self.manager.reserve_multi_page_append(multi_plan)
                cross_plan = None
                boundary_slots = None
            else:
                cross_plan = (
                    self.manager.ragged_one_boundary_plan(row_counts)
                    if self.ragged_cross_page_enabled else None
                )
            if not use_multi and cross_plan is not None:
                first_counts, second_counts, final_lengths, boundary_slots = cross_plan
                self._ragged_subpage_counts = None
                self._ragged_cross_page_plan = (
                    first_counts, second_counts, final_lengths
                )
                self._ragged_boundary_slots = boundary_slots
                self._ragged_first_counts_gpu = torch.tensor(
                    first_counts, device=self.device, dtype=torch.int32
                )
                self._ragged_second_counts_gpu = torch.tensor(
                    second_counts, device=self.device, dtype=torch.int32
                )
                self.manager.reserve_graph_transition_pages(boundary_slots)
                self.boundary_mask.zero_()
                self.boundary_mask[
                    torch.tensor(boundary_slots, device=self.device, dtype=torch.long)
                ] = True
            elif not use_multi:
                boundary_slots = self.manager.ragged_single_page_boundary_slots(
                    row_counts
                )
            if not use_multi and cross_plan is None and boundary_slots is not None:
                self._ragged_subpage_counts = row_counts
                self._ragged_boundary_slots = boundary_slots
                if boundary_slots:
                    self.manager.reserve_graph_transition_pages(boundary_slots)
                    self.boundary_mask.zero_()
                    self.boundary_mask[
                        torch.tensor(
                            boundary_slots, device=self.device, dtype=torch.long
                        )
                    ] = True

    def set_decode_execution_state(
        self,
        *,
        slot_ids: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """Map compact K1 execution rows to stable logical cache slots.

        This is the decode-only foundation for a multi-capacity Graph Pool.
        Prefill continues to use :meth:`set_batch_state`, whose row count equals
        logical capacity and owns the existing transactional page planners.
        """
        if slot_ids.dtype != torch.int32 or slot_ids.ndim != 1:
            raise ValueError("execution slot_ids must be rank-1 torch.int32")
        if active_mask.dtype != torch.bool or active_mask.shape != slot_ids.shape:
            raise ValueError("execution active_mask must be bool and match slot_ids")
        if slot_ids.device != self.device or active_mask.device != self.device:
            raise ValueError("execution state tensors must be on the cache device")
        if not 1 <= slot_ids.numel() <= self.capacity:
            raise ValueError("execution row count must be within cache capacity")
        if bool(((slot_ids < 0) | (slot_ids >= self.capacity)).any().item()):
            raise ValueError("execution slot_ids are outside logical capacity")
        selected = slot_ids[active_mask]
        if selected.numel() != torch.unique(selected).numel():
            raise ValueError("active execution rows must map to unique logical slots")
        self.execution_slot_ids = slot_ids
        self.execution_active_mask = active_mask
        self._write_mask = active_mask[:, None]
        self._dense_empty_prefill = False
        self._ragged_subpage_counts = None
        self._ragged_boundary_slots = ()
        self._ragged_cross_page_plan = None
        self._ragged_first_counts_gpu = None
        self._ragged_second_counts_gpu = None
        self._ragged_multi_page_plan = None
        self._ragged_multi_round_gpu = ()

    def set_prefill_execution_state(
        self,
        *,
        slot_ids: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> None:
        """Map compact prefill rows onto logical slots via the transactional writer."""
        if write_mask.dtype != torch.bool or write_mask.ndim != 2:
            raise ValueError("execution write_mask must be rank-2 bool")
        active_mask = write_mask.any(dim=1)
        self.set_decode_execution_state(
            slot_ids=slot_ids,
            active_mask=active_mask,
        )
        self._write_mask = write_mask

    def enable_graph_decode(self, *, boundary_pack: bool = False) -> None:
        """Route subsequent K1 updates through the host-free GPU writer."""
        self._graph_decode_enabled = True
        self._graph_boundary_pack_enabled = boundary_pack
        self._write_mask = self.execution_active_mask[:, None]
        self._dense_empty_prefill = False
        self._ragged_subpage_counts = None
        self._ragged_boundary_slots = ()
        self._ragged_cross_page_plan = None
        self._ragged_first_counts_gpu = None
        self._ragged_second_counts_gpu = None
        self._ragged_multi_page_plan = None
        self._ragged_multi_round_gpu = ()

    def disable_graph_decode(self) -> None:
        self._graph_decode_enabled = False
        self._graph_boundary_pack_enabled = False

    @property
    def graph_decode_enabled(self) -> bool:
        return self._graph_decode_enabled

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        self._validate_layer(layer_idx)
        if self._write_mask is None:
            raise RuntimeError("set_batch_state must run before model forward")
        if key_states.ndim != 4 or key_states.shape != value_states.shape:
            raise ValueError("batched NQKV update requires matching BHTD Key/Value")
        execution_rows = self.execution_slot_ids.numel()
        if key_states.shape != (
            execution_rows,
            self.num_kv_heads,
            self._write_mask.shape[1],
            self.head_dim,
        ):
            raise ValueError("Key/Value shape disagrees with batched NQKV state")
        if key_states.device != self.device or key_states.dtype != self.dtype:
            raise ValueError("Key/Value must match the NQKV cache device and dtype")
        rotated_key = hadamard(key_states.float()).to(self.dtype)
        if self._graph_decode_enabled:
            if key_states.shape[2] != 1:
                raise RuntimeError("NQKV graph mode supports K1 decode only")
            _launch_append_hot_k1_graph_safe(
                rotated_key[:, :, 0].contiguous(),
                value_states[:, :, 0].contiguous(),
                self.manager.layer_caches[layer_idx],
                slot_ids=self.execution_slot_ids,
                active_mask=self.execution_active_mask,
            )
            if self._graph_boundary_pack_enabled:
                launch_pack_full_hot_pages_graph_safe(
                    self.manager.layer_caches[layer_idx],
                    boundary_mask=self.boundary_mask,
                )
        else:
            dense_key = rotated_key.permute(0, 2, 1, 3).contiguous()
            dense_value = value_states.permute(0, 2, 1, 3).contiguous()
            if self._dense_empty_prefill:
                self.manager.append_layer_dense_empty(
                    layer_idx, dense_key, dense_value
                )
                self.dense_prefill_layer_launch_count += 1
            elif self._ragged_multi_page_plan is not None:
                plan = self._ragged_multi_page_plan
                for round_index, (starts, counts, flush_mask, has_flush) in enumerate(
                    self._ragged_multi_round_gpu
                ):
                    segments = plan.round(round_index)
                    self.manager.prepare_multi_page_round(
                        layer_idx, segments
                    )
                    self.manager.append_layer_ragged_segment(
                        layer_idx,
                        dense_key,
                        dense_value,
                        slot_ids=self.slot_ids,
                        source_starts=starts,
                        segment_counts=counts,
                    )
                    if has_flush:
                        launch_pack_full_hot_pages_graph_safe(
                            self.manager.layer_caches[layer_idx],
                            boundary_mask=flush_mask,
                            batched_encode=self.multi_page_batched_pack_enabled,
                        )
                        if self.multi_page_batched_pack_enabled:
                            self.multi_page_batched_pack_count += 1
                    self.ragged_multi_page_layer_round_count += 1
                cache = self.manager.layer_caches[layer_idx]
                for row in plan.rows:
                    if row.token_count:
                        cache.synchronize_host_length(
                            row.row, row.final_hot_length
                        )
            elif self._ragged_cross_page_plan is not None:
                first_counts, second_counts, final_lengths = self._ragged_cross_page_plan
                assert self._ragged_first_counts_gpu is not None
                assert self._ragged_second_counts_gpu is not None
                self.manager.append_layer_ragged_segment(
                    layer_idx,
                    dense_key,
                    dense_value,
                    slot_ids=self.slot_ids,
                    source_starts=self._ragged_zero_starts_gpu,
                    segment_counts=self._ragged_first_counts_gpu,
                )
                launch_pack_full_hot_pages_graph_safe(
                    self.manager.layer_caches[layer_idx],
                    boundary_mask=self.boundary_mask,
                )
                self.manager.append_layer_ragged_segment(
                    layer_idx,
                    dense_key,
                    dense_value,
                    slot_ids=self.slot_ids,
                    source_starts=self._ragged_first_counts_gpu,
                    segment_counts=self._ragged_second_counts_gpu,
                )
                cache = self.manager.layer_caches[layer_idx]
                for slot, final_length in enumerate(final_lengths):
                    if first_counts[slot] or second_counts[slot]:
                        cache.synchronize_host_length(slot, final_length)
                self.ragged_cross_page_layer_launch_count += 1
                self.ragged_full_page_layer_pack_count += 1
            elif self._ragged_subpage_counts is not None:
                self.manager.append_layer_ragged_subpage(
                    layer_idx,
                    dense_key,
                    dense_value,
                    slot_ids=self.slot_ids,
                    write_mask=self._write_mask,
                    row_counts=self._ragged_subpage_counts,
                )
                self.ragged_subpage_layer_launch_count += 1
                if self._ragged_boundary_slots:
                    launch_pack_full_hot_pages_graph_safe(
                        self.manager.layer_caches[layer_idx],
                        boundary_mask=self.boundary_mask,
                    )
                    self.ragged_full_page_layer_pack_count += 1
            else:
                self.manager.append_layer(
                    layer_idx,
                    dense_key,
                    dense_value,
                    slot_ids=self.execution_slot_ids,
                    write_mask=self._write_mask,
                )
        return rotated_key, value_states

    def finish_prefill_page_transaction(self, *, success: bool) -> None:
        """Commit or cancel pages reserved by one eager ragged prefill."""
        if self._ragged_multi_page_plan is not None:
            plan = self._ragged_multi_page_plan
            affected = tuple(row.row for row in plan.rows if row.token_count)
            if success:
                self.manager.commit_multi_page_append(plan)
            else:
                self.manager.cancel_multi_page_append()
                self.manager.release_slots(affected)
            self.boundary_mask.zero_()
            self._ragged_multi_page_plan = None
            self._ragged_multi_round_gpu = ()
            return
        slots = self._ragged_boundary_slots
        if not slots:
            return
        if success:
            self.manager.commit_graph_transition_pages(slots)
            if self._ragged_cross_page_plan is not None:
                final_lengths = self._ragged_cross_page_plan[2]
                for cache in self.manager.layer_caches:
                    for slot in slots:
                        cache.synchronize_host_length(slot, final_lengths[slot])
        else:
            self.manager.cancel_graph_transition_pages(slots)
            if self._ragged_cross_page_plan is not None:
                first_counts, second_counts, _ = self._ragged_cross_page_plan
                affected = tuple(
                    slot for slot, (first, second) in enumerate(
                        zip(first_counts, second_counts, strict=True)
                    ) if first or second
                )
                self.manager.release_slots(affected)
        self.boundary_mask.zero_()
        self._ragged_boundary_slots = ()
        self._ragged_cross_page_plan = None
        self._ragged_first_counts_gpu = None
        self._ragged_second_counts_gpu = None

    def attend(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        *,
        scale: float,
    ) -> torch.Tensor:
        self._validate_layer(layer_idx)
        expected = (
            self.execution_slot_ids.numel(),
            self.num_query_heads,
            1,
            self.head_dim,
        )
        if query_states.shape != expected:
            raise ValueError("batched NQKV decode requires [capacity, heads, 1, dim]")
        rotated_query = hadamard(query_states.float()).to(self.dtype)
        query = rotated_query[:, :, 0].contiguous()
        if self._graph_decode_enabled:
            output = self.manager.attend_layer_graph_safe(
                layer_idx,
                query,
                slot_ids=self.execution_slot_ids,
                active_mask=self.execution_active_mask,
                scale=scale,
            )
        else:
            output = self.manager.attend_layer(
                layer_idx,
                query,
                slot_ids=self.execution_slot_ids,
                active_mask=self.execution_active_mask,
                scale=scale,
            )
        return output.unsqueeze(2)

    def prefill(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        scale: float,
    ) -> torch.Tensor:
        """Use current BF16 K/V for prefill while NQKV stores valid rows."""
        rotated_query = hadamard(query_states.float()).to(self.dtype)
        return F.scaled_dot_product_attention(
            rotated_query,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None,
            scale=scale,
            enable_gqa=self.num_query_heads != self.num_kv_heads,
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        self._validate_layer(layer_idx)
        cache = self.manager.layer_caches[layer_idx]
        lengths = [
            cache.total_tokens(sequence_id)
            for sequence_id in self.manager.request_ids
        ]
        return max(lengths, default=0)

    def get_max_length(self, layer_idx: int | None = None) -> int:
        if layer_idx is not None:
            self._validate_layer(layer_idx)
        return self._max_cache_len

    @property
    def max_cache_len(self) -> int:
        """Stable public capacity across Transformers cache API versions."""
        return self._max_cache_len

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        self._validate_layer(layer_idx)
        return self.get_seq_length(layer_idx) + query_length, 0

    @property
    def batch_size(self) -> int:
        return self.capacity

    @property
    def is_compileable(self) -> bool:
        return False

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        del beam_idx
        raise NotImplementedError("batched NQKV backend does not support beam search")

    def _validate_layer(self, layer_idx: int) -> None:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError("layer_idx must be an integer")
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("layer_idx is outside the batched NQKV layer range")


def _batched_nqkv_attention_interface(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    cache = getattr(module, "_nestquant_batched_nqkv_cache", None)
    if not isinstance(cache, BatchedNQKVLlamaCache):
        raise RuntimeError("Llama attention module has no batched NQKV cache")
    if kwargs.get("dropout", 0.0) != 0.0:
        raise ValueError("batched NQKV backend supports inference dropout=0 only")
    scale = float(kwargs.get("scaling", module.scaling))
    if query.shape[2] == 1:
        return cache.attend(module.layer_idx, query, scale=scale), None
    return cache.prefill(
        query, key, value, attention_mask, scale=scale
    ), None


def install_batched_nqkv_llama_backend(
    model: torch.nn.Module,
    cache: BatchedNQKVLlamaCache,
) -> str:
    """Attach fixed-capacity batched NQKV attention to every Llama layer."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(cache, BatchedNQKVLlamaCache):
        raise TypeError("cache must be a BatchedNQKVLlamaCache")
    ALL_ATTENTION_FUNCTIONS.register(
        BATCHED_NQKV_ATTENTION_NAME, _batched_nqkv_attention_interface
    )
    attached = 0
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            if not 0 <= module.layer_idx < cache.num_layers:
                raise ValueError("Llama layer index disagrees with batched NQKV cache")
            module._nestquant_batched_nqkv_cache = cache
            module.config._attn_implementation = BATCHED_NQKV_ATTENTION_NAME
            attached += 1
    if attached != cache.num_layers:
        raise ValueError(
            f"expected {cache.num_layers} LlamaAttention modules, found {attached}"
        )
    if hasattr(model, "config"):
        model.config._attn_implementation = BATCHED_NQKV_ATTENTION_NAME
    return BATCHED_NQKV_ATTENTION_NAME
