"""Deterministic slot lifecycle for a continuous-batching prototype."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class DecodeRequest:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None = None
    status: RequestStatus = RequestStatus.WAITING
    slot: int | None = None
    generated_token_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")


class ContinuousBatchScheduler:
    """Manage waiting/running/terminal requests and reusable physical slots."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._waiting: deque[DecodeRequest] = deque()
        self._slots: list[DecodeRequest | None] = [None] * capacity
        self._requests: dict[str, DecodeRequest] = {}
        self._released_slots: set[int] = set()

    @property
    def slots(self) -> tuple[DecodeRequest | None, ...]:
        return tuple(self._slots)

    @property
    def waiting(self) -> tuple[DecodeRequest, ...]:
        return tuple(self._waiting)

    @property
    def running(self) -> tuple[DecodeRequest, ...]:
        return tuple(request for request in self._slots if request is not None)

    def submit(self, request: DecodeRequest) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"duplicate request_id: {request.request_id}")
        self._requests[request.request_id] = request
        self._waiting.append(request)

    def admit(self, limit: int | None = None) -> tuple[DecodeRequest, ...]:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        admitted = []
        for slot, current in enumerate(self._slots):
            if not self._waiting or (limit is not None and len(admitted) >= limit):
                break
            if current is not None:
                continue
            request = self._waiting.popleft()
            request.status = RequestStatus.RUNNING
            request.slot = slot
            self._slots[slot] = request
            self._released_slots.discard(slot)
            admitted.append(request)
        return tuple(admitted)

    def record_tokens(self, tokens_by_slot: dict[int, int]) -> tuple[DecodeRequest, ...]:
        finished = []
        for slot, token in tokens_by_slot.items():
            if slot < 0 or slot >= self.capacity or self._slots[slot] is None:
                raise ValueError(f"slot {slot} is not running")
            request = self._slots[slot]
            assert request is not None
            request.generated_token_ids.append(int(token))
            hit_eos = request.eos_token_id is not None and token == request.eos_token_id
            hit_limit = len(request.generated_token_ids) >= request.max_new_tokens
            if hit_eos or hit_limit:
                self._finish_slot(slot, RequestStatus.FINISHED)
                finished.append(request)
        return tuple(finished)

    def cancel(self, request_id: str) -> DecodeRequest:
        request = self._requests[request_id]
        if request.status is RequestStatus.WAITING:
            self._waiting.remove(request)
            request.status = RequestStatus.CANCELLED
            return request
        if request.status is RequestStatus.RUNNING:
            assert request.slot is not None
            self._finish_slot(request.slot, RequestStatus.CANCELLED)
            return request
        return request

    def _finish_slot(self, slot: int, status: RequestStatus) -> None:
        request = self._slots[slot]
        if request is None:
            raise ValueError(f"slot {slot} is already free")
        request.status = status
        request.slot = None
        self._slots[slot] = None
        self._released_slots.add(slot)

    def consume_released_slots(self) -> tuple[int, ...]:
        released = tuple(sorted(self._released_slots))
        self._released_slots.clear()
        return released

    def active_mask(self) -> tuple[bool, ...]:
        return tuple(request is not None for request in self._slots)

    def positions(self) -> tuple[int, ...]:
        """Return the position where each running request's next input is written."""
        return tuple(
            0 if request is None else (
                len(request.prompt_token_ids) + max(len(request.generated_token_ids) - 1, 0)
            )
            for request in self._slots
        )

    def pending_count(self) -> int:
        return len(self._waiting)

    def free_slot_count(self) -> int:
        return sum(request is None for request in self._slots)

    def get(self, request_id: str) -> DecodeRequest:
        return self._requests[request_id]

    def forget_terminal(self, request_id: str) -> DecodeRequest:
        """Remove a completed/cancelled request from the long-lived registry."""
        request = self._requests[request_id]
        if request.status not in (RequestStatus.FINISHED, RequestStatus.CANCELLED):
            raise ValueError("only terminal requests can be forgotten")
        del self._requests[request_id]
        return request

    def registered_count(self) -> int:
        return len(self._requests)
