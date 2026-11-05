"""Thread-owned online serving loop for the continuous-batching engine."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
import queue
from pathlib import Path
import threading
import time
from typing import Callable, Iterator

import torch

from .continuous_batching import DecodeRequest
from .inference_engine import ContinuousBatchInferenceEngine


class OnlineServiceStatus(str, Enum):
    FINISHED = "finished"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


class ServiceOverloadedError(RuntimeError):
    """Raised when bounded in-flight capacity is exhausted."""


@dataclass(frozen=True)
class OnlineServiceResult:
    request_id: str
    status: OnlineServiceStatus
    generated_token_ids: tuple[int, ...]
    submitted_at: float
    finished_at: float
    timings_ms: dict[str, float | None]
    error: str | None = None


_TOKEN_END = object()


class OnlineRequestHandle:
    """Thread-safe result and token stream returned to a network adapter."""

    def __init__(
        self,
        request_id: str,
        cancel_callback: Callable[[str], None],
    ) -> None:
        self.request_id = request_id
        self._cancel_callback = cancel_callback
        self._future: Future[OnlineServiceResult] = Future()
        self._tokens: queue.Queue[int | object] = queue.Queue()

    def cancel(self) -> None:
        self._cancel_callback(self.request_id)

    def result(self, timeout: float | None = None) -> OnlineServiceResult:
        return self._future.result(timeout=timeout)

    def iter_tokens(self, timeout: float | None = None) -> Iterator[int]:
        while True:
            item = self._tokens.get(timeout=timeout)
            if item is _TOKEN_END:
                return
            yield int(item)


@dataclass
class _RequestContext:
    request: DecodeRequest
    handle: OnlineRequestHandle
    submitted_at: float
    deadline: float | None
    registered_at: float | None = None
    admitted_at: float | None = None
    first_token_at: float | None = None
    prefill_compute_seconds: float | None = None


@dataclass(frozen=True)
class _SubmitCommand:
    context: _RequestContext


@dataclass(frozen=True)
class _CancelCommand:
    request_id: str


class _StopCommand:
    pass


class OnlineInferenceService:
    """Run one inference engine on a dedicated owner thread.

    The worker is the only thread allowed to mutate the scheduler, runner or
    CUDA-backed KV cache. Network adapters may submit/cancel concurrently.
    """

    def __init__(
        self,
        engine: ContinuousBatchInferenceEngine,
        *,
        max_inflight_requests: int,
        default_timeout_seconds: float | None = None,
        idle_wait_seconds: float = 0.001,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_inflight_requests < 1:
            raise ValueError("max_inflight_requests must be positive")
        if default_timeout_seconds is not None and default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be positive")
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be positive")
        self.engine = engine
        self.max_inflight_requests = int(max_inflight_requests)
        self.default_timeout_seconds = default_timeout_seconds
        self.idle_wait_seconds = float(idle_wait_seconds)
        self._clock = clock
        self._commands: queue.Queue[object] = queue.Queue()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._reserved_ids: set[str] = set()
        self._contexts: dict[str, _RequestContext] = {}
        self._ready: Future[None] = Future()
        self._thread: threading.Thread | None = None
        self._accepting = False
        self._diagnostics: dict[str, int | float | None] = {}
        self._diagnostic_high_water = {
            "registered_requests_peak": 0,
            "nqkv_allocated_pages_peak": 0,
        }

    def start(self, timeout: float | None = None) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("service has already been started")
            self._accepting = True
            self._thread = threading.Thread(
                target=self._run, name="nestquant-online-engine", daemon=True
            )
            self._thread.start()
        self._ready.result(timeout=timeout)

    def submit(
        self,
        request: DecodeRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> OnlineRequestHandle:
        if timeout_seconds is None:
            timeout_seconds = self.default_timeout_seconds
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        with self._lock:
            if not self._accepting:
                raise RuntimeError("service is not accepting requests")
            if request.request_id in self._reserved_ids:
                raise ValueError(f"duplicate request_id: {request.request_id}")
            if len(self._reserved_ids) >= self.max_inflight_requests:
                raise ServiceOverloadedError(
                    "maximum in-flight request capacity reached"
                )
            self._reserved_ids.add(request.request_id)
        handle = OnlineRequestHandle(request.request_id, self.cancel)
        submitted_at = self._clock()
        deadline = (
            None if timeout_seconds is None else submitted_at + timeout_seconds
        )
        self._commands.put(_SubmitCommand(_RequestContext(
            request=request,
            handle=handle,
            submitted_at=submitted_at,
            deadline=deadline,
        )))
        self._wake.set()
        return handle

    def cancel(self, request_id: str) -> None:
        with self._lock:
            if request_id not in self._reserved_ids:
                return
        self._commands.put(_CancelCommand(request_id))
        self._wake.set()

    def stop(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._accepting = False
        self._commands.put(_StopCommand())
        self._wake.set()
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("online service did not stop before timeout")

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return len(self._reserved_ids)

    @property
    def diagnostics_snapshot(self) -> dict[str, int | float | None]:
        """Return the last worker-owned resource snapshot without touching CUDA."""
        with self._lock:
            return dict(self._diagnostics)

    def _update_diagnostics(self) -> None:
        scheduler = self.engine.scheduler
        snapshot: dict[str, int | float | None] = {
            "registered_requests": scheduler.registered_count(),
            "waiting_requests": scheduler.pending_count(),
            "running_requests": len(scheduler.running),
            "cpu_rss_bytes": _current_rss_bytes(),
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "nqkv_allocated_pages": None,
            "nqkv_free_pages": None,
        }
        if self.engine.device.type == "cuda":
            snapshot["cuda_allocated_bytes"] = torch.cuda.memory_allocated(
                self.engine.device
            )
            snapshot["cuda_reserved_bytes"] = torch.cuda.memory_reserved(
                self.engine.device
            )
        cache = getattr(self.engine.backend, "cache", None)
        manager = getattr(cache, "manager", None)
        layer_caches = getattr(manager, "layer_caches", ())
        if layer_caches:
            pools = [layer.cold_cache.pool for layer in layer_caches]
            snapshot["nqkv_allocated_pages"] = sum(
                pool.allocated_pages for pool in pools
            )
            snapshot["nqkv_free_pages"] = sum(pool.free_pages for pool in pools)
        replay_counts = getattr(self.engine.backend, "graph_replay_counts", {})
        for width, count in sorted(replay_counts.items()):
            snapshot[f"graph_m{int(width)}_replays"] = int(count)
        snapshot["compact_prefill_calls"] = int(
            getattr(self.engine.backend, "compact_prefill_calls", 0)
        )
        snapshot["compact_prefill_rows"] = int(
            getattr(self.engine.backend, "compact_prefill_rows", 0)
        )
        snapshot["graph_boundary_replays"] = int(
            getattr(self.engine.backend, "graph_boundary_replay_count", 0)
        )
        snapshot["graph_boundary_fallbacks"] = int(
            getattr(self.engine.backend, "graph_boundary_fallback_count", 0)
        )
        self._diagnostic_high_water["registered_requests_peak"] = max(
            self._diagnostic_high_water["registered_requests_peak"],
            int(snapshot["registered_requests"] or 0),
        )
        self._diagnostic_high_water["nqkv_allocated_pages_peak"] = max(
            self._diagnostic_high_water["nqkv_allocated_pages_peak"],
            int(snapshot["nqkv_allocated_pages"] or 0),
        )
        snapshot.update(self._diagnostic_high_water)
        with self._lock:
            self._diagnostics = snapshot

    def _finish(
        self,
        context: _RequestContext,
        status: OnlineServiceStatus,
        *,
        error: str | None = None,
    ) -> None:
        finished_at = self._clock()
        registered_at = context.registered_at
        admitted_at = context.admitted_at
        first_token_at = context.first_token_at
        prefill_ms = (
            None if context.prefill_compute_seconds is None
            else context.prefill_compute_seconds * 1000.0
        )
        admission_to_first_ms = (
            None if admitted_at is None or first_token_at is None
            else (first_token_at - admitted_at) * 1000.0
        )
        result = OnlineServiceResult(
            request_id=context.request.request_id,
            status=status,
            generated_token_ids=tuple(context.request.generated_token_ids),
            submitted_at=context.submitted_at,
            finished_at=finished_at,
            timings_ms={
                "command_queue": (
                    None if registered_at is None
                    else (registered_at - context.submitted_at) * 1000.0
                ),
                "waiting_queue": (
                    None if registered_at is None or admitted_at is None
                    else (admitted_at - registered_at) * 1000.0
                ),
                "prefill_compute": prefill_ms,
                "post_prefill_to_first_token": (
                    None if admission_to_first_ms is None or prefill_ms is None
                    else max(admission_to_first_ms - prefill_ms, 0.0)
                ),
                "admission_to_first_token": admission_to_first_ms,
                "server_ttft": (
                    None if first_token_at is None
                    else (first_token_at - context.submitted_at) * 1000.0
                ),
                "server_e2e": (finished_at - context.submitted_at) * 1000.0,
            },
            error=error,
        )
        self._contexts.pop(context.request.request_id, None)
        with self._lock:
            self._reserved_ids.discard(context.request.request_id)
        self._update_diagnostics()
        if not context.handle._future.done():
            context.handle._future.set_result(result)
            context.handle._tokens.put(_TOKEN_END)

    def _cancel_context(
        self,
        context: _RequestContext,
        status: OnlineServiceStatus,
    ) -> None:
        request_id = context.request.request_id
        self.engine.cancel(request_id)
        self.engine.forget_terminal(request_id)
        self._finish(context, status)

    def _drain_commands(self) -> bool:
        stopping = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return stopping
            if isinstance(command, _SubmitCommand):
                context = command.context
                try:
                    self.engine.submit(context.request)
                except Exception as error:
                    self._finish(
                        context, OnlineServiceStatus.FAILED, error=str(error)
                    )
                else:
                    context.registered_at = self._clock()
                    self._contexts[context.request.request_id] = context
            elif isinstance(command, _CancelCommand):
                context = self._contexts.get(command.request_id)
                if context is not None:
                    self._cancel_context(context, OnlineServiceStatus.CANCELLED)
            elif isinstance(command, _StopCommand):
                stopping = True
            else:
                raise TypeError(f"unknown service command: {type(command)!r}")

    def _expire_requests(self) -> None:
        now = self._clock()
        expired = [
            context for context in self._contexts.values()
            if context.deadline is not None and now >= context.deadline
        ]
        for context in expired:
            self._cancel_context(context, OnlineServiceStatus.TIMED_OUT)

    def _publish_step(self) -> None:
        step_started = self._clock()
        step = self.engine.step()
        step_finished = self._clock()
        for request_id in step.admitted_request_ids:
            context = self._contexts.get(request_id)
            if context is not None:
                context.admitted_at = step_started
                context.prefill_compute_seconds = step.prefill_seconds
        for request_id, token in step.generated_tokens.items():
            context = self._contexts.get(request_id)
            if context is not None:
                if context.first_token_at is None:
                    context.first_token_at = step_finished
                context.handle._tokens.put(int(token))
        for request_id in step.finished_request_ids:
            context = self._contexts.get(request_id)
            if context is None:
                continue
            self.engine.forget_terminal(request_id)
            self._finish(context, OnlineServiceStatus.FINISHED)

    def _fail_all(self, error: BaseException) -> None:
        for context in tuple(self._contexts.values()):
            try:
                self.engine.cancel(context.request.request_id)
                self.engine.forget_terminal(context.request.request_id)
            except Exception:
                pass
            self._finish(context, OnlineServiceStatus.FAILED, error=str(error))

    def _run(self) -> None:
        try:
            self.engine.initialize()
        except BaseException as error:
            with self._lock:
                self._accepting = False
            self._ready.set_exception(error)
            return
        self._ready.set_result(None)
        self._update_diagnostics()
        try:
            stopping = False
            while not stopping:
                self._wake.clear()
                stopping = self._drain_commands()
                self._expire_requests()
                if stopping:
                    break
                if self.engine.has_work():
                    self._publish_step()
                else:
                    self._wake.wait(self.idle_wait_seconds)
                self._update_diagnostics()
        except BaseException as error:
            self._fail_all(error)
        finally:
            for context in tuple(self._contexts.values()):
                self._cancel_context(context, OnlineServiceStatus.CANCELLED)
            with self._lock:
                self._accepting = False
                self._thread = None


def _current_rss_bytes() -> int | None:
    """Read current Linux RSS without adding a psutil runtime dependency."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None
