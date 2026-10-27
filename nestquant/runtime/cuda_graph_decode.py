"""Reusable CUDA Graph runner for fixed-batch autoregressive decode."""

from __future__ import annotations

import types
from collections.abc import Callable, Iterable
from typing import Any

import torch


class NQRTGraphDecodeRunner:
    """Capture and replay one-token decode with a fixed-address StaticCache.

    The model and batch size are fixed for the runner lifetime. Prompt length,
    token values and cache positions may change as long as they fit inside
    ``max_cache_len``. Greedy argmax remains outside the captured graph.
    """

    def __init__(self, model: torch.nn.Module, *, batch_size: int,
                 max_cache_len: int) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch module")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(max_cache_len, bool) or not isinstance(max_cache_len, int) or max_cache_len < 2:
            raise ValueError("max_cache_len must be an integer of at least two")
        try:
            device = next(model.parameters()).device
        except StopIteration as error:
            raise ValueError("model must contain parameters") from error
        if device.type != "cuda":
            raise ValueError("CUDA Graph decode requires a CUDA model")
        if model.training:
            raise ValueError("model must be in eval mode")
        from transformers import StaticCache

        self.model = model
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = device
        self.cache = StaticCache(model.config, max_cache_len=max_cache_len)
        self._position_holder: list[torch.Tensor | None] = [None]
        self._graphed_decode: Any | None = None
        self._captured = False
        self._prepared = False
        self.prompt_length = 0
        self._install_fixed_updates_after_initialization = False

    @property
    def captured(self) -> bool:
        return self._captured

    def _validate_input_ids(self, input_ids: torch.Tensor, *, one_token: bool) -> None:
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a torch.long tensor")
        if input_ids.device != self.device or input_ids.ndim != 2:
            raise ValueError("input_ids must be a rank-2 tensor on the model device")
        if input_ids.shape[0] != self.batch_size:
            raise ValueError("input batch size does not match the runner")
        if one_token and input_ids.shape[1] != 1:
            raise ValueError("decode input must contain exactly one token per request")

    def _fixed_update(self, layer, key_states, value_states, *args, **kwargs):
        del args, kwargs
        position = self._position_holder[0]
        if position is None:
            raise RuntimeError("cache position was not set")
        layer.keys.index_copy_(2, position, key_states)
        layer.values.index_copy_(2, position, value_states)
        return layer.keys, layer.values

    def _install_fixed_cache_updates(self) -> None:
        for layer in self.cache.layers:
            layer.update = types.MethodType(self._fixed_update, layer)
        self._install_fixed_updates_after_initialization = True

    def _forward_token(self, token: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        self._position_holder[0] = position
        position_ids = position.view(1, 1).expand(self.batch_size, 1)
        return self.model(
            input_ids=token,
            past_key_values=self.cache,
            cache_position=position,
            position_ids=position_ids,
            use_cache=True,
        ).logits

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reset the cache, run prompt prefill and return final-position logits."""
        self._validate_input_ids(input_ids, one_token=False)
        prompt_length = int(input_ids.shape[1])
        if prompt_length < 1 or prompt_length >= self.max_cache_len:
            raise ValueError("prompt length must be in [1, max_cache_len - 1]")
        self.cache.reset()
        positions = torch.arange(prompt_length, device=self.device, dtype=torch.long)
        if self._install_fixed_updates_after_initialization:
            self._position_holder[0] = positions
            position_ids = positions.view(1, -1).expand(self.batch_size, -1)
            output = self.model(
                input_ids=input_ids, past_key_values=self.cache,
                cache_position=positions, position_ids=position_ids, use_cache=True,
            )
        else:
            # The first official update lazily creates the fixed-address K/V tensors.
            output = self.model(
                input_ids=input_ids, past_key_values=self.cache,
                cache_position=positions, position_ids=positions.view(1, -1),
                use_cache=True,
            )
            self._install_fixed_cache_updates()
        self.prompt_length = prompt_length
        self._prepared = True
        return output.logits[:, -1:, :]

    @torch.inference_mode()
    def capture(self, sample_token: torch.Tensor, *, position: int | None = None) -> None:
        """Capture one-token decode; safe to call once per runner."""
        if not self._prepared:
            raise RuntimeError("prefill must run before capture")
        if self._captured:
            return
        self._validate_input_ids(sample_token, one_token=True)
        position_value = self.prompt_length if position is None else position
        if position_value < 0 or position_value >= self.max_cache_len:
            raise ValueError("capture position is outside the cache")
        sample_position = torch.tensor([position_value], device=self.device, dtype=torch.long)
        # Graphed callables reuse and overwrite their sample tensors as static
        # input buffers. Clone here so replay cannot mutate caller-owned data.
        graph_token = sample_token.clone()
        graph_position = sample_position.clone()
        self._graphed_decode = torch.cuda.make_graphed_callables(
            self._forward_token, (graph_token, graph_position), num_warmup_iters=3,
        )
        self._captured = True

    @torch.inference_mode()
    def decode_eager(self, token: torch.Tensor, *, position: int) -> torch.Tensor:
        """Run the graph-compatible decode body without replay."""
        if not self._prepared:
            raise RuntimeError("prefill must run before decode")
        self._validate_input_ids(token, one_token=True)
        if position < 0 or position >= self.max_cache_len:
            raise ValueError("decode position is outside the cache")
        position_tensor = torch.tensor([position], device=self.device, dtype=torch.long)
        return self._forward_token(token, position_tensor)

    @torch.inference_mode()
    def decode(self, token: torch.Tensor, *, position: int) -> torch.Tensor:
        """Replay captured decode for one token and return the static logits tensor."""
        if not self._captured or self._graphed_decode is None:
            raise RuntimeError("capture must run before graph decode")
        self._validate_input_ids(token, one_token=True)
        if position < 0 or position >= self.max_cache_len:
            raise ValueError("decode position is outside the cache")
        position_tensor = torch.tensor([position], device=self.device, dtype=torch.long)
        return self._graphed_decode(token, position_tensor)

    @torch.inference_mode()
    def greedy_generate(self, input_ids: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens greedily, capturing on the first call if necessary."""
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if input_ids.shape[1] + max_new_tokens > self.max_cache_len:
            raise ValueError("prompt and generated tokens exceed max_cache_len")
        logits = self.prefill(input_ids)
        token = logits.argmax(dim=-1).contiguous()
        generated = [token.clone()]
        if max_new_tokens == 1:
            return token
        if not self._captured:
            self.capture(token, position=int(input_ids.shape[1]))
        for position in range(int(input_ids.shape[1]),
                              int(input_ids.shape[1]) + max_new_tokens - 1):
            logits = self.decode(token, position=position)
            token = logits[:, -1:, :].argmax(dim=-1).contiguous()
            generated.append(token.clone())
        return torch.cat(generated, dim=1)


class NQRTGraphDecodePool:
    """Own fixed-batch graph runners and route a synchronized decode cohort.

    ``active_count`` requests are padded to the smallest configured capacity.
    Batch rows are independent, so inactive rows cannot modify active rows;
    returned logits are sliced back to the active request count. All active
    requests currently share prompt length and decode position.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        batch_sizes: Iterable[int] = (1, 2, 4, 8),
        max_cache_len: int,
        runner_factory: Callable[..., NQRTGraphDecodeRunner] = NQRTGraphDecodeRunner,
    ) -> None:
        sizes = tuple(batch_sizes)
        if not sizes:
            raise ValueError("batch_sizes must not be empty")
        if any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes):
            raise ValueError("batch_sizes must contain positive integers")
        if tuple(sorted(set(sizes))) != sizes:
            raise ValueError("batch_sizes must be strictly increasing and unique")
        self.model = model
        self.batch_sizes = sizes
        self.max_cache_len = max_cache_len
        self.runners = {
            size: runner_factory(model, batch_size=size, max_cache_len=max_cache_len)
            for size in sizes
        }
        self._active_count: int | None = None
        self._capacity: int | None = None

    @property
    def max_batch_size(self) -> int:
        return self.batch_sizes[-1]

    @property
    def active_count(self) -> int | None:
        return self._active_count

    @property
    def capacity(self) -> int | None:
        return self._capacity

    def select_capacity(self, active_count: int) -> int:
        if isinstance(active_count, bool) or not isinstance(active_count, int):
            raise TypeError("active_count must be an integer")
        if active_count < 1:
            raise ValueError("active_count must be positive")
        for size in self.batch_sizes:
            if active_count <= size:
                return size
        raise ValueError("active_count exceeds the largest graph capacity")

    @staticmethod
    def _pad_rows(tensor: torch.Tensor, capacity: int, pad_value: int) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError("pool inputs must be rank-2 tensors")
        missing = capacity - tensor.shape[0]
        if missing < 0:
            raise ValueError("input batch exceeds selected capacity")
        if missing == 0:
            return tensor.contiguous()
        padding = torch.full(
            (missing, tensor.shape[1]), pad_value,
            dtype=tensor.dtype, device=tensor.device,
        )
        return torch.cat((tensor, padding), dim=0).contiguous()

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, *, pad_token_id: int = 0) -> torch.Tensor:
        """Prefill a same-length request cohort and return active-row logits."""
        active_count = int(input_ids.shape[0]) if input_ids.ndim >= 1 else 0
        capacity = self.select_capacity(active_count)
        padded = self._pad_rows(input_ids, capacity, pad_token_id)
        logits = self.runners[capacity].prefill(padded)
        self._active_count = active_count
        self._capacity = capacity
        return logits[:active_count]

    @torch.inference_mode()
    def capture(self, sample_token: torch.Tensor, *, position: int | None = None,
                pad_token_id: int = 0) -> None:
        if self._active_count is None or self._capacity is None:
            raise RuntimeError("prefill must run before capture")
        if sample_token.shape[0] != self._active_count:
            raise ValueError("sample token batch does not match active_count")
        padded = self._pad_rows(sample_token, self._capacity, pad_token_id)
        self.runners[self._capacity].capture(padded, position=position)

    @torch.inference_mode()
    def decode(self, token: torch.Tensor, *, position: int,
               pad_token_id: int = 0) -> torch.Tensor:
        """Replay the selected graph and remove inactive padding rows."""
        if self._active_count is None or self._capacity is None:
            raise RuntimeError("prefill must run before decode")
        if token.shape[0] != self._active_count:
            raise ValueError("token batch does not match active_count")
        padded = self._pad_rows(token, self._capacity, pad_token_id)
        logits = self.runners[self._capacity].decode(padded, position=position)
        return logits[:self._active_count]

    @torch.inference_mode()
    def decode_eager(self, token: torch.Tensor, *, position: int,
                     pad_token_id: int = 0) -> torch.Tensor:
        """Run the selected runner eagerly with the same padding contract."""
        if self._active_count is None or self._capacity is None:
            raise RuntimeError("prefill must run before decode")
        if token.shape[0] != self._active_count:
            raise ValueError("token batch does not match active_count")
        padded = self._pad_rows(token, self._capacity, pad_token_id)
        logits = self.runners[self._capacity].decode_eager(padded, position=position)
        return logits[:self._active_count]
