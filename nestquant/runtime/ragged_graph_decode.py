"""CUDA Graph decode with independent cache positions for each batch row."""

from __future__ import annotations

from typing import Any

import torch

from .ragged_cache import RaggedCacheController


class NQRTRaggedGraphDecodeRunner:
    """Replay fixed-shape decode while positions and active rows vary per request."""

    def __init__(self, model: torch.nn.Module, *, batch_size: int, max_cache_len: int) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch module")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_cache_len, bool) or not isinstance(max_cache_len, int) or max_cache_len < 2:
            raise ValueError("max_cache_len must be at least two")
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("model must contain parameters") from error
        if device.type != "cuda":
            raise ValueError("ragged CUDA Graph decode requires a CUDA model")
        if model.training:
            raise ValueError("model must be in eval mode")
        from transformers import StaticCache

        self.model = model
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = device
        self.cache = StaticCache(model.config, max_cache_len=max_cache_len)
        self.controller = RaggedCacheController(
            self.cache, batch_size=batch_size, max_cache_len=max_cache_len
        )
        self._graphed_decode: Any | None = None
        self._prepared = False
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def _validate_decode_inputs(
        self,
        token: torch.Tensor,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> None:
        if token.dtype != torch.long or token.shape != (self.batch_size, 1):
            raise ValueError("token must be torch.long with shape [batch_size, 1]")
        if positions.dtype != torch.long or positions.shape != (self.batch_size, 1):
            raise ValueError("positions must be torch.long with shape [batch_size, 1]")
        if active_mask.dtype != torch.bool or active_mask.shape != (self.batch_size, 1):
            raise ValueError("active_mask must be torch.bool with shape [batch_size, 1]")
        if valid_lengths.dtype != torch.long or valid_lengths.shape != (self.batch_size,):
            raise ValueError("valid_lengths must be torch.long with shape [batch_size]")
        tensors = (token, positions, active_mask, valid_lengths)
        if any(tensor.device != self.device for tensor in tensors):
            raise ValueError("all decode inputs must be on the model device")
        if positions.min().item() < 0 or positions.max().item() >= self.max_cache_len:
            raise ValueError("positions are outside the cache")
        if valid_lengths.min().item() < 1 or valid_lengths.max().item() > self.max_cache_len:
            raise ValueError("valid_lengths are outside the cache")
        expected = torch.where(active_mask[:, 0], positions[:, 0] + 1, valid_lengths)
        if torch.any(expected != valid_lengths).item():
            raise ValueError("active rows must have valid_lengths == positions + 1")

    def _forward_token(
        self,
        token: torch.Tensor,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        self.controller.set_write_state(positions, active_mask, validate_bounds=False)
        attention_mask = self.controller.attention_mask(
            positions, valid_lengths=valid_lengths,
            dtype=next(self.model.parameters()).dtype,
        )
        return self.model(
            input_ids=token,
            past_key_values=self.cache,
            position_ids=positions,
            attention_mask=attention_mask,
            use_cache=True,
        ).logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reset and prefill equal-length prompts for every physical slot."""
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] != self.batch_size or input_ids.device != self.device:
            raise ValueError("input_ids batch or device does not match the runner")
        prompt_length = input_ids.shape[1]
        if prompt_length < 1 or prompt_length >= self.max_cache_len:
            raise ValueError("prompt length must be in [1, max_cache_len - 1]")
        self.cache.reset()
        positions = torch.arange(prompt_length, device=self.device).view(1, -1).expand(
            self.batch_size, -1
        )
        if self.controller.installed:
            active = torch.ones_like(positions, dtype=torch.bool)
            self.controller.set_write_state(positions, active)
            lengths = torch.full(
                (self.batch_size,), prompt_length, dtype=torch.long, device=self.device
            )
            attention_mask = self.controller.attention_mask(
                positions, valid_lengths=lengths,
                dtype=next(self.model.parameters()).dtype,
            )
            output = self.model(
                input_ids=input_ids, past_key_values=self.cache,
                position_ids=positions, attention_mask=attention_mask, use_cache=True,
            )
        else:
            output = self.model(
                input_ids=input_ids, past_key_values=self.cache,
                position_ids=positions, use_cache=True,
            )
            self.controller.install()
        self._prepared = True
        return output.logits[:, -1:, :]

    @torch.inference_mode()
    def prefill_slots(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor,
        write_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill selected slots while preserving every inactive cache row."""
        if not self._prepared or not self.controller.installed:
            raise RuntimeError("an initial prefill must initialize the cache first")
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        if input_ids.shape[0] != self.batch_size or input_ids.device != self.device:
            raise ValueError("input_ids batch or device does not match the runner")
        if positions.dtype != torch.long or positions.shape != input_ids.shape:
            raise ValueError("positions must be torch.long and match input_ids")
        if write_mask.dtype != torch.bool or write_mask.shape != input_ids.shape:
            raise ValueError("write_mask must be torch.bool and match input_ids")
        if valid_lengths.dtype != torch.long or valid_lengths.shape != (self.batch_size,):
            raise ValueError("valid_lengths must be torch.long with shape [batch_size]")
        if any(tensor.device != self.device for tensor in (positions, write_mask, valid_lengths)):
            raise ValueError("all prefill inputs must be on the model device")
        self.controller.set_write_state(positions, write_mask)
        attention_mask = self.controller.attention_mask(
            positions, valid_lengths=valid_lengths,
            dtype=next(self.model.parameters()).dtype,
        )
        return self.model(
            input_ids=input_ids,
            past_key_values=self.cache,
            position_ids=positions,
            attention_mask=attention_mask,
            use_cache=True,
        ).logits

    @torch.inference_mode()
    def capture(
        self,
        sample_token: torch.Tensor,
        *,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> None:
        if not self._prepared:
            raise RuntimeError("prefill must run before capture")
        if self._captured:
            return
        self._validate_decode_inputs(sample_token, positions, active_mask, valid_lengths)
        # Sample tensors become replay's static input buffers. Keep them
        # private so later copy_ operations cannot mutate caller-owned views.
        graph_inputs = tuple(
            tensor.clone() for tensor in (sample_token, positions, active_mask, valid_lengths)
        )
        self._graphed_decode = torch.cuda.make_graphed_callables(
            self._forward_token,
            graph_inputs,
            num_warmup_iters=3,
        )
        self._captured = True

    @torch.inference_mode()
    def decode_eager(
        self,
        token: torch.Tensor,
        *,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not self._prepared:
            raise RuntimeError("prefill must run before decode")
        self._validate_decode_inputs(token, positions, active_mask, valid_lengths)
        return self._forward_token(token, positions, active_mask, valid_lengths)

    @torch.inference_mode()
    def decode(
        self,
        token: torch.Tensor,
        *,
        positions: torch.Tensor,
        active_mask: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not self._captured or self._graphed_decode is None:
            raise RuntimeError("capture must run before graph decode")
        self._validate_decode_inputs(token, positions, active_mask, valid_lengths)
        return self._graphed_decode(token, positions, active_mask, valid_lengths)

    def clear_slots(self, slot_mask: torch.Tensor) -> None:
        if slot_mask.device != self.device:
            raise ValueError("slot_mask must be on the model device")
        self.controller.clear_slots(slot_mask)
