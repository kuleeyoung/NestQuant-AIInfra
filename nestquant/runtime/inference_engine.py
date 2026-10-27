"""Host-driven continuous-batching inference engine for ragged CUDA Graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .continuous_batching import ContinuousBatchScheduler, DecodeRequest


class RaggedDecodeBackend(Protocol):
    batch_size: int
    max_cache_len: int
    device: torch.device

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def prefill_slots(
        self, input_ids: torch.Tensor, *, positions: torch.Tensor,
        write_mask: torch.Tensor, valid_lengths: torch.Tensor,
    ) -> torch.Tensor: ...
    def capture(
        self, sample_token: torch.Tensor, *, positions: torch.Tensor,
        active_mask: torch.Tensor, valid_lengths: torch.Tensor,
    ) -> None: ...
    def decode(
        self, token: torch.Tensor, *, positions: torch.Tensor,
        active_mask: torch.Tensor, valid_lengths: torch.Tensor,
    ) -> torch.Tensor: ...
    def clear_slots(self, slot_mask: torch.Tensor) -> None: ...


@dataclass(frozen=True)
class EngineStep:
    admitted_request_ids: tuple[str, ...]
    generated_tokens: dict[str, int]
    finished_request_ids: tuple[str, ...]
    active_request_ids: tuple[str | None, ...]


@dataclass(frozen=True)
class AdmissionPolicy:
    """Wait briefly for a useful prompt microbatch before occupying free slots."""

    min_prefill_batch: int = 4
    max_wait_steps: int = 2

    def __post_init__(self) -> None:
        if (isinstance(self.min_prefill_batch, bool)
                or not isinstance(self.min_prefill_batch, int)
                or self.min_prefill_batch < 1):
            raise ValueError("min_prefill_batch must be a positive integer")
        if (isinstance(self.max_wait_steps, bool)
                or not isinstance(self.max_wait_steps, int)
                or self.max_wait_steps < 0):
            raise ValueError("max_wait_steps must be a non-negative integer")


class GreedyHostSampler:
    """Select argmax tokens outside the captured CUDA Graph."""

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("sampler logits must have shape [requests, vocabulary]")
        return logits.argmax(dim=-1)


class ContinuousBatchInferenceEngine:
    """Combine request admission, ragged KV state and host-side sampling."""

    def __init__(
        self,
        backend: RaggedDecodeBackend,
        *,
        pad_token_id: int = 0,
        sampler: GreedyHostSampler | None = None,
        admission_policy: AdmissionPolicy | None = None,
    ) -> None:
        self.backend = backend
        self.capacity = backend.batch_size
        self.max_cache_len = backend.max_cache_len
        self.device = backend.device
        self.pad_token_id = int(pad_token_id)
        self.sampler = sampler or GreedyHostSampler()
        self.admission_policy = admission_policy or AdmissionPolicy()
        self.scheduler = ContinuousBatchScheduler(self.capacity)
        self._initialized = False
        self._step_index = 0
        self._free_since_step: int | None = None

    def submit(self, request: DecodeRequest) -> None:
        if len(request.prompt_token_ids) + request.max_new_tokens > self.max_cache_len:
            raise ValueError("prompt plus max_new_tokens exceeds the KV cache")
        self.scheduler.submit(request)

    @torch.inference_mode()
    def initialize(self) -> None:
        """Allocate fixed cache tensors, capture decode, then clear dummy rows."""
        if self._initialized:
            return
        dummy = torch.full(
            (self.capacity, 1), self.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        self.backend.prefill(dummy)
        positions = torch.zeros((self.capacity, 1), dtype=torch.long, device=self.device)
        active = torch.ones((self.capacity, 1), dtype=torch.bool, device=self.device)
        lengths = torch.ones((self.capacity,), dtype=torch.long, device=self.device)
        self.backend.capture(
            dummy, positions=positions, active_mask=active, valid_lengths=lengths
        )
        self.backend.clear_slots(torch.ones(self.capacity, dtype=torch.bool, device=self.device))
        self._initialized = True

    def _cache_length(self, request: DecodeRequest | None) -> int:
        if request is None:
            return 1
        generated = len(request.generated_token_ids)
        return len(request.prompt_token_ids) + max(generated - 1, 0)

    def _admission_limit(self) -> int:
        waiting = self.scheduler.pending_count()
        free = self.scheduler.free_slot_count()
        running = len(self.scheduler.running)
        if waiting == 0 or free == 0:
            self._free_since_step = None
            return 0
        if running == 0:
            self._free_since_step = None
            return min(waiting, free)
        if self._free_since_step is None:
            self._free_since_step = self._step_index
        candidate_batch = min(waiting, free)
        waited_steps = self._step_index - self._free_since_step
        if (candidate_batch >= self.admission_policy.min_prefill_batch
                or waited_steps >= self.admission_policy.max_wait_steps):
            self._free_since_step = None
            return candidate_batch
        return 0

    @torch.inference_mode()
    def _prefill_admitted(self, admitted: tuple[DecodeRequest, ...]) -> tuple[dict[str, int], tuple[str, ...]]:
        if not admitted:
            return {}, ()
        width = max(len(request.prompt_token_ids) for request in admitted)
        input_ids = torch.full(
            (self.capacity, width), self.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        # Every query column gets a unique cache index, including padding.
        # Duplicate scatter indices have undefined last-writer behavior and an
        # inactive padded write could otherwise overwrite the final real KV.
        positions = torch.arange(width, dtype=torch.long, device=self.device).view(1, -1).expand(
            self.capacity, -1
        ).clone()
        write_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        valid_lengths = torch.tensor(
            [self._cache_length(request) for request in self.scheduler.slots],
            dtype=torch.long, device=self.device,
        )
        for request in admitted:
            assert request.slot is not None
            length = len(request.prompt_token_ids)
            input_ids[request.slot, :length] = torch.tensor(
                request.prompt_token_ids, dtype=torch.long, device=self.device
            )
            write_mask[request.slot, :length] = True
            valid_lengths[request.slot] = length

        logits = self.backend.prefill_slots(
            input_ids, positions=positions,
            write_mask=write_mask, valid_lengths=valid_lengths,
        )
        rows = torch.tensor([request.slot for request in admitted], device=self.device)
        columns = torch.tensor(
            [len(request.prompt_token_ids) - 1 for request in admitted], device=self.device
        )
        tokens = self.sampler(logits[rows, columns]).tolist()
        tokens_by_slot = {request.slot: int(token) for request, token in zip(admitted, tokens)}
        generated = {request.request_id: token for request, token in zip(admitted, tokens)}
        finished = self.scheduler.record_tokens(tokens_by_slot)
        return generated, tuple(request.request_id for request in finished)

    @torch.inference_mode()
    def _decode_existing(
        self, existing_ids: set[str]
    ) -> tuple[dict[str, int], tuple[str, ...]]:
        slots = self.scheduler.slots
        selected = [
            request for request in slots
            if request is not None and request.request_id in existing_ids
        ]
        if not selected:
            return {}, ()
        token = torch.full(
            (self.capacity, 1), self.pad_token_id,
            dtype=torch.long, device=self.device,
        )
        positions = torch.zeros((self.capacity, 1), dtype=torch.long, device=self.device)
        active_mask = torch.zeros((self.capacity, 1), dtype=torch.bool, device=self.device)
        valid_lengths = torch.tensor(
            [self._cache_length(request) for request in slots],
            dtype=torch.long, device=self.device,
        )
        for request in selected:
            assert request.slot is not None and request.generated_token_ids
            position = len(request.prompt_token_ids) + len(request.generated_token_ids) - 1
            token[request.slot, 0] = request.generated_token_ids[-1]
            positions[request.slot, 0] = position
            active_mask[request.slot, 0] = True
            valid_lengths[request.slot] = position + 1

        logits = self.backend.decode(
            token, positions=positions,
            active_mask=active_mask, valid_lengths=valid_lengths,
        )
        rows = torch.tensor([request.slot for request in selected], device=self.device)
        next_tokens = self.sampler(logits[rows, -1]).tolist()
        tokens_by_slot = {request.slot: int(value) for request, value in zip(selected, next_tokens)}
        generated = {request.request_id: value for request, value in zip(selected, next_tokens)}
        finished = self.scheduler.record_tokens(tokens_by_slot)
        return generated, tuple(request.request_id for request in finished)

    @torch.inference_mode()
    def step(self) -> EngineStep:
        if not self._initialized:
            self.initialize()
        released = self.scheduler.consume_released_slots()
        if released:
            clear_mask = torch.zeros(self.capacity, dtype=torch.bool, device=self.device)
            clear_mask[list(released)] = True
            self.backend.clear_slots(clear_mask)

        existing_ids = {request.request_id for request in self.scheduler.running}
        admission_limit = self._admission_limit()
        admitted = self.scheduler.admit(limit=admission_limit)
        prefill_generated, prefill_finished = self._prefill_admitted(admitted)
        decode_generated, decode_finished = self._decode_existing(existing_ids)
        generated = {**decode_generated, **prefill_generated}
        finished = decode_finished + prefill_finished
        active_ids = tuple(
            None if request is None else request.request_id
            for request in self.scheduler.slots
        )
        result = EngineStep(
            admitted_request_ids=tuple(request.request_id for request in admitted),
            generated_tokens=generated,
            finished_request_ids=finished,
            active_request_ids=active_ids,
        )
        self._step_index += 1
        return result

    def has_work(self) -> bool:
        return bool(self.scheduler.waiting or self.scheduler.running)

    def forget_terminal(self, request_id: str) -> DecodeRequest:
        return self.scheduler.forget_terminal(request_id)
