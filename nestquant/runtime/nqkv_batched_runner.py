"""Eager full-model runner for continuous batching over batched NQKV."""

from __future__ import annotations

import torch

from .nqkv_batched_hf_backend import BatchedNQKVLlamaCache


class BatchedNQKVLlamaRunner:
    """Adapt a causal Llama model and NQKV cache to the scheduler protocol.

    Decode is deliberately eager: page allocation and hot-page mutation are
    host-controlled today, so claiming full-model CUDA Graph capture here
    would be incorrect. ``capture`` performs a shape-checked warmup only.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cache: BatchedNQKVLlamaCache,
        *,
        execution_policy: str = "full",
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch module")
        if not isinstance(cache, BatchedNQKVLlamaCache):
            raise TypeError("cache must be a BatchedNQKVLlamaCache")
        if model.training:
            raise ValueError("model must be in eval mode")
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("model must contain parameters") from error
        if device.type != "cuda" or device != cache.device:
            raise ValueError("model and NQKV cache must share one CUDA device")
        if not hasattr(model, "config") or not hasattr(model, "lm_head"):
            raise ValueError("runner requires a causal language model with lm_head")
        if execution_policy not in {"fixed", "decode", "full"}:
            raise ValueError("execution_policy must be fixed, decode, or full")
        self.model = model
        self.cache = cache
        self.execution_policy = execution_policy
        # Cross/multi-page ragged prefill reserves pages before the model call
        # and requires the runner's post-forward commit/rollback boundary.
        self.cache.ragged_cross_page_enabled = True
        self.cache.ragged_multi_page_enabled = True
        self.batch_size = cache.capacity
        self.max_cache_len = cache.max_cache_len
        self.device = device
        self._generation = [0] * self.batch_size
        self._prepared = False
        self._warm = False
        self._graphed_decode = None
        self._graphed_boundary_decode = None
        self._graph_pool: dict[int, object] = {}
        self._boundary_graph_pool: dict[int, object] = {}
        self._graph_slot_ids: dict[int, torch.Tensor] = {}
        self._graph_active_masks: dict[int, torch.Tensor] = {}
        self.graph_replay_counts: dict[int, int] = {}
        self.compact_prefill_calls = 0
        self.compact_prefill_rows = 0
        self._graph_token: torch.Tensor | None = None
        self._graph_positions: torch.Tensor | None = None
        self.graph_boundary_fallback_count = 0
        self.graph_boundary_replay_count = 0

    @property
    def graph_captured(self) -> bool:
        return self._graphed_decode is not None

    def _validate_common(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        write_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> None:
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 torch.long")
        if input_ids.shape[0] != self.batch_size:
            raise ValueError("input batch must equal fixed runner capacity")
        if positions.dtype != torch.long or positions.shape != input_ids.shape:
            raise ValueError("positions must be torch.long and match input_ids")
        if write_mask.dtype != torch.bool or write_mask.shape != input_ids.shape:
            raise ValueError("write_mask must be bool and match input_ids")
        if valid_lengths.dtype != torch.long or valid_lengths.shape != (self.batch_size,):
            raise ValueError("valid_lengths must have shape [batch_size]")
        if any(
            tensor.device != self.device
            for tensor in (input_ids, positions, write_mask, valid_lengths)
        ):
            raise ValueError("runner inputs must be on the model CUDA device")
        if bool(write_mask.any().item()):
            written_positions = positions[write_mask]
            if int(written_positions.min().item()) < 0:
                raise ValueError("written positions must be non-negative")
            if int(written_positions.max().item()) >= self.max_cache_len:
                raise ValueError("written positions exceed the NQKV cache")

    def _register_new_rows(self, row_mask: torch.Tensor) -> None:
        assignments = []
        for slot in row_mask.nonzero(as_tuple=False).flatten().cpu().tolist():
            try:
                self.cache.manager.sequence_id(slot)
            except KeyError:
                self._generation[slot] += 1
                assignments.append((f"slot-{slot}-generation-{self._generation[slot]}", slot))
        if assignments:
            self.cache.register_batch(assignments)

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> torch.Tensor:
        active = write_mask.any(dim=1)
        self.cache.set_batch_state(active_mask=active, write_mask=write_mask)
        try:
            output = self.model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=write_mask,
                past_key_values=self.cache,
                use_cache=True,
            )
        except Exception:
            self.cache.finish_prefill_page_transaction(success=False)
            raise
        self.cache.finish_prefill_page_transaction(success=True)
        return output.logits

    def _forward_compact_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        write_mask: torch.Tensor,
        slot_ids: torch.Tensor,
    ) -> torch.Tensor:
        self.cache.set_prefill_execution_state(
            slot_ids=slot_ids,
            write_mask=write_mask,
        )
        try:
            output = self.model(
                input_ids=input_ids,
                position_ids=positions,
                attention_mask=write_mask,
                past_key_values=self.cache,
                use_cache=True,
            )
        except Exception:
            self.cache.finish_prefill_page_transaction(success=False)
            raise
        self.cache.finish_prefill_page_transaction(success=True)
        return output.logits

    def _forward_graph(
        self, token: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        output = self.model(
            input_ids=token,
            position_ids=positions,
            attention_mask=self.cache.execution_active_mask[:, None],
            past_key_values=self.cache,
            use_cache=True,
        )
        return output.logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Warm all fixed rows; the engine clears them immediately afterward."""
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be rank-2 torch.long")
        if input_ids.shape[0] != self.batch_size or input_ids.device != self.device:
            raise ValueError("input_ids batch or device does not match the runner")
        width = input_ids.shape[1]
        if width < 1 or width > self.max_cache_len:
            raise ValueError("prefill width is outside the NQKV cache")
        positions = torch.arange(width, device=self.device).expand(self.batch_size, -1)
        write_mask = torch.ones_like(input_ids, dtype=torch.bool)
        self._register_new_rows(write_mask.any(dim=1))
        self.cache.disable_graph_decode()
        logits = self._forward(input_ids, positions, write_mask)
        self._prepared = True
        return logits

    @torch.inference_mode()
    def prefill_slots(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor,
        write_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not self._prepared:
            raise RuntimeError("initial prefill must prepare the runner")
        self._validate_common(input_ids, positions, write_mask, valid_lengths)
        row_active = write_mask.any(dim=1)
        self._register_new_rows(row_active)
        self.cache.disable_graph_decode()
        active_count = int(row_active.sum().item())
        if (
            self.execution_policy != "full"
            or active_count == 0
            or active_count == self.batch_size
        ):
            return self._forward(input_ids, positions, write_mask)
        self.compact_prefill_calls += 1
        self.compact_prefill_rows += active_count
        compact_logits = self._forward_compact_prefill(
            input_ids[row_active],
            positions[row_active],
            write_mask[row_active],
            self.cache.slot_ids[row_active],
        )
        logits = torch.zeros(
            (self.batch_size, compact_logits.shape[1], compact_logits.shape[2]),
            device=compact_logits.device,
            dtype=compact_logits.dtype,
        )
        logits[row_active] = compact_logits
        return logits

    @torch.inference_mode()
    def capture(
        self,
        sample_token: torch.Tensor,
        *,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> None:
        """Capture fixed-capacity K1 decode while NQKV pages remain hot."""
        if not self._prepared:
            raise RuntimeError("prefill must prepare the runner before warmup")
        if self._graphed_decode is not None:
            return
        self._validate_common(sample_token, positions, active_mask, valid_lengths)
        widths = []
        if self.execution_policy == "fixed":
            widths.append(self.batch_size)
        else:
            width = 1
            while width < self.batch_size:
                widths.append(width)
                width *= 2
            widths.append(self.batch_size)
        for width in widths:
            slot_ids = torch.arange(width, device=self.device, dtype=torch.int32)
            execution_active = torch.ones(width, device=self.device, dtype=torch.bool)
            self._graph_slot_ids[width] = slot_ids
            self._graph_active_masks[width] = execution_active
            self.cache.set_decode_execution_state(
                slot_ids=slot_ids,
                active_mask=execution_active,
            )
            self.cache.enable_graph_decode(boundary_pack=False)
            graph_token = sample_token[:width].clone()
            graph_positions = positions[:width].clone()
            self._graph_pool[width] = torch.cuda.make_graphed_callables(
                self._forward_graph,
                (graph_token, graph_positions),
                num_warmup_iters=3,
            )
            if width == self.batch_size:
                self._graph_token = graph_token
                self._graph_positions = graph_positions
                self._graphed_decode = self._graph_pool[width]
        # Capture a second graph whose per-layer update packs only rows selected
        # by boundary_mask. Dummy reservations provide valid target addresses
        # during capture; request initialization releases them afterward.
        self.cache.manager.reserve_graph_transition_pages(tuple(range(self.batch_size)))
        for width in widths:
            self.cache.set_decode_execution_state(
                slot_ids=self._graph_slot_ids[width],
                active_mask=self._graph_active_masks[width],
            )
            self.cache.boundary_mask.zero_()
            self.cache.boundary_mask[:width] = True
            self.cache.enable_graph_decode(boundary_pack=True)
            boundary_token = sample_token[:width].clone()
            boundary_positions = positions[:width].clone()
            self._boundary_graph_pool[width] = torch.cuda.make_graphed_callables(
                self._forward_graph,
                (boundary_token, boundary_positions),
                num_warmup_iters=3,
            )
            if width == self.batch_size:
                self._graphed_boundary_decode = self._boundary_graph_pool[width]
        self.cache.boundary_mask.zero_()
        self.cache.set_batch_state(
            active_mask=active_mask[:, 0],
            write_mask=active_mask,
        )
        self.cache.enable_graph_decode(boundary_pack=False)
        self._warm = True

    @torch.inference_mode()
    def decode(
        self,
        token: torch.Tensor,
        *,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self._warm or self._graphed_decode is None
            or self._graphed_boundary_decode is None
        ):
            raise RuntimeError("capture must warm the runner before decode")
        self._validate_common(token, positions, active_mask, valid_lengths)
        row_active = active_mask[:, 0]
        active_slots = self.cache.slot_ids[row_active].long()
        active_count = int(active_slots.numel())
        if active_count == 0:
            return torch.zeros(
                (*token.shape, self.model.config.vocab_size),
                device=self.device,
                dtype=next(self.model.parameters()).dtype,
            )
        graph_width = next(
            width for width in sorted(self._graph_pool) if width >= active_count
        )
        self.graph_replay_counts[graph_width] = (
            self.graph_replay_counts.get(graph_width, 0) + 1
        )
        graph_slots = self._graph_slot_ids[graph_width]
        graph_active = self._graph_active_masks[graph_width]
        graph_slots.fill_(0)
        graph_slots[:active_count].copy_(active_slots.to(torch.int32))
        graph_active.zero_()
        graph_active[:active_count] = True
        compact_token = torch.zeros(
            (graph_width, token.shape[1]), device=self.device, dtype=token.dtype
        )
        compact_positions = torch.zeros_like(compact_token)
        compact_token[:active_count].copy_(token[row_active])
        compact_positions[:active_count].copy_(positions[row_active])
        graph = self._graph_pool[graph_width]
        if active_slots.numel():
            lengths = self.cache.manager.layer_caches[0].hot_lengths[active_slots]
            if bool((lengths >= self.cache.page_size - 1).any().item()):
                boundary_slots_tensor = active_slots[
                    lengths >= self.cache.page_size - 1
                ]
                boundary_slots = tuple(
                    int(slot) for slot in boundary_slots_tensor.cpu().tolist()
                )
                self.cache.manager.reserve_graph_transition_pages(boundary_slots)
                self.cache.boundary_mask.zero_()
                self.cache.boundary_mask[boundary_slots_tensor] = True
                graph = self._boundary_graph_pool[graph_width]
                output = graph(compact_token, compact_positions)
                self.cache.manager.commit_graph_transition_pages(boundary_slots)
                self.graph_boundary_replay_count += 1
            else:
                output = graph(compact_token, compact_positions)
        else:
            output = graph(compact_token, compact_positions)
        full_output = torch.zeros(
            (self.batch_size, output.shape[1], output.shape[2]),
            device=output.device,
            dtype=output.dtype,
        )
        full_output[row_active] = output[:active_count]
        return full_output

    def clear_slots(self, slot_mask: torch.Tensor) -> None:
        if slot_mask.dtype != torch.bool or slot_mask.shape != (self.batch_size,):
            raise ValueError("slot_mask must be bool with shape [batch_size]")
        if slot_mask.device != self.device:
            raise ValueError("slot_mask must be on the model device")
        slots = slot_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        self.cache.release_slots(slots)

