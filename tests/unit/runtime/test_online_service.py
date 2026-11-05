import time

import pytest
import torch

from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.inference_engine import ContinuousBatchInferenceEngine
from nestquant.runtime.online_service import (
    OnlineInferenceService,
    OnlineServiceStatus,
    ServiceOverloadedError,
)


class FakeBackend:
    def __init__(self, *, batch_size=1, decode_delay=0.0):
        self.batch_size = batch_size
        self.max_cache_len = 4096
        self.device = torch.device("cpu")
        self.decode_delay = decode_delay
        self.clear_calls = []

    @staticmethod
    def _logits(input_ids):
        logits = torch.zeros(*input_ids.shape, 32)
        next_ids = (input_ids + 1).clamp_max(31)
        return logits.scatter(-1, next_ids.unsqueeze(-1), 1.0)

    def prefill(self, input_ids):
        return self._logits(input_ids)

    def prefill_slots(self, input_ids, **kwargs):
        return self._logits(input_ids)

    def capture(self, sample_token, **kwargs):
        return None

    def decode(self, token, **kwargs):
        if self.decode_delay:
            time.sleep(self.decode_delay)
        return self._logits(token)

    def clear_slots(self, slot_mask):
        self.clear_calls.append(slot_mask.clone())


def _service(*, max_inflight=4, decode_delay=0.0):
    backend = FakeBackend(decode_delay=decode_delay)
    engine = ContinuousBatchInferenceEngine(backend)
    service = OnlineInferenceService(
        engine, max_inflight_requests=max_inflight, idle_wait_seconds=0.001
    )
    service.start(timeout=2)
    return service, backend


def test_online_service_streams_tokens_and_finishes() -> None:
    service, _ = _service()
    try:
        handle = service.submit(DecodeRequest("finished", (1, 2), 3))
        assert list(handle.iter_tokens(timeout=2)) == [3, 4, 5]
        result = handle.result(timeout=2)
        assert result.status is OnlineServiceStatus.FINISHED
        assert result.generated_token_ids == (3, 4, 5)
        assert result.timings_ms["command_queue"] is not None
        assert result.timings_ms["waiting_queue"] is not None
        assert result.timings_ms["prefill_compute"] is not None
        assert result.timings_ms["server_ttft"] is not None
        assert result.timings_ms["server_e2e"] >= result.timings_ms["server_ttft"]
        assert service.inflight_count == 0
        snapshot = service.diagnostics_snapshot
        assert snapshot["registered_requests"] == 0
        assert snapshot["running_requests"] == 0
        assert "cuda_allocated_bytes" in snapshot
        assert snapshot["registered_requests_peak"] >= 1
        assert snapshot["nqkv_allocated_pages_peak"] == 0
    finally:
        service.stop(timeout=2)


def test_online_service_cancel_releases_running_slot() -> None:
    service, backend = _service(decode_delay=0.01)
    try:
        handle = service.submit(DecodeRequest("cancelled", (1,), 100))
        assert next(handle.iter_tokens(timeout=2)) == 2
        handle.cancel()
        result = handle.result(timeout=2)
        assert result.status is OnlineServiceStatus.CANCELLED
        assert service.inflight_count == 0
        assert any(mask.tolist() == [True] for mask in backend.clear_calls)
    finally:
        service.stop(timeout=2)


def test_online_service_times_out_waiting_request() -> None:
    service, _ = _service(decode_delay=0.03)
    try:
        first = service.submit(DecodeRequest("occupy", (1,), 5))
        second = service.submit(
            DecodeRequest("timeout", (2,), 1), timeout_seconds=0.005
        )
        timed_out = second.result(timeout=2)
        assert timed_out.status is OnlineServiceStatus.TIMED_OUT
        first.cancel()
        assert first.result(timeout=2).status is OnlineServiceStatus.CANCELLED
    finally:
        service.stop(timeout=2)


def test_online_service_applies_backpressure_before_queue_growth() -> None:
    service, _ = _service(max_inflight=1, decode_delay=0.01)
    try:
        first = service.submit(DecodeRequest("first", (1,), 100))
        with pytest.raises(ServiceOverloadedError, match="capacity"):
            service.submit(DecodeRequest("rejected", (2,), 1))
        first.cancel()
        assert first.result(timeout=2).status is OnlineServiceStatus.CANCELLED
    finally:
        service.stop(timeout=2)


def test_engine_cancel_clears_backend_without_an_extra_step() -> None:
    backend = FakeBackend()
    engine = ContinuousBatchInferenceEngine(backend)
    request = DecodeRequest("running", (1,), 5)
    engine.submit(request)
    engine.step()
    engine.cancel("running")
    assert backend.clear_calls[-1].tolist() == [True]
    assert not engine.has_work()
