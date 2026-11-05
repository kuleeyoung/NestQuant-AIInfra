import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from nestquant.runtime.continuous_batching import DecodeRequest
from nestquant.runtime.http_service import NestQuantHTTPServer
from nestquant.runtime.inference_engine import ContinuousBatchInferenceEngine
from nestquant.runtime.online_service import OnlineInferenceService


class FakeTokenizer:
    eos_token_id = None

    def encode(self, text, *, add_special_tokens):
        return [int(value) for value in text.split()]

    def decode(self, token_ids, *, skip_special_tokens):
        return " ".join(str(value) for value in token_ids)


class FakeBackend:
    batch_size = 1
    max_cache_len = 1024
    device = torch.device("cpu")

    def __init__(self, decode_delay=0.0):
        self.decode_delay = decode_delay

    @staticmethod
    def _logits(input_ids):
        logits = torch.zeros(*input_ids.shape, 64)
        return logits.scatter(
            -1, (input_ids + 1).clamp_max(63).unsqueeze(-1), 1.0
        )

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
        return None


def _server(*, delay=0.0, max_inflight=4):
    engine = ContinuousBatchInferenceEngine(FakeBackend(delay))
    service = OnlineInferenceService(
        engine, max_inflight_requests=max_inflight, idle_wait_seconds=0.001
    )
    server = NestQuantHTTPServer(service, FakeTokenizer())
    server.start(timeout=2)
    host, port = server.address
    return server, f"http://{host}:{port}"


def _json_request(url, *, method="GET", payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=3) as response:
        return response.status, json.loads(response.read())


def test_http_health_generate_and_metrics() -> None:
    server, base = _server()
    try:
        status, health = _json_request(base + "/health")
        assert status == 200 and health["status"] == "ok"
        status, generated = _json_request(
            base + "/v1/generate",
            method="POST",
            payload={"request_id": "basic", "prompt": "1 2", "max_new_tokens": 3},
        )
        assert status == 200
        assert generated["generated_token_ids"] == [3, 4, 5]
        assert generated["text"] == "3 4 5"
        assert generated["timings_ms"]["server_ttft"] >= 0
        assert generated["timings_ms"]["prefill_compute"] >= 0
        _, metrics = _json_request(base + "/metrics")
        assert metrics["submitted"] == 1
        assert metrics["completed"] == 1
        assert metrics["inflight"] == 0
        assert metrics["runtime"]["registered_requests"] == 0
    finally:
        server.stop(timeout=2)


def test_http_rejects_invalid_request() -> None:
    server, base = _server()
    try:
        try:
            _json_request(
                base + "/v1/generate", method="POST",
                payload={"prompt": "", "max_new_tokens": 0},
            )
        except HTTPError as error:
            assert error.code == 400
            payload = json.loads(error.read())
            assert payload["error"] == "invalid_request"
        else:
            raise AssertionError("invalid request was accepted")
    finally:
        server.stop(timeout=2)


def test_http_cancel_endpoint_terminates_live_request() -> None:
    server, base = _server(delay=0.02)
    response = {}

    def generate():
        try:
            _json_request(
                base + "/v1/generate", method="POST",
                payload={
                    "request_id": "cancel-me",
                    "prompt": "1",
                    "max_new_tokens": 100,
                },
            )
        except HTTPError as error:
            response["code"] = error.code
            response["payload"] = json.loads(error.read())

    thread = threading.Thread(target=generate)
    try:
        thread.start()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            _, metrics = _json_request(base + "/metrics")
            if metrics["inflight"] == 1:
                break
            time.sleep(0.005)
        status, cancelling = _json_request(
            base + "/v1/requests/cancel-me", method="DELETE"
        )
        assert status == 202 and cancelling["status"] == "cancelling"
        thread.join(timeout=2)
        assert response["code"] == 409
        assert response["payload"]["status"] == "cancelled"
    finally:
        server.stop(timeout=2)


def test_http_backpressure_maps_to_429() -> None:
    server, base = _server(delay=0.02, max_inflight=1)
    first = server.service.submit(DecodeRequest("occupy", (1,), 100))
    try:
        try:
            _json_request(
                base + "/v1/generate", method="POST",
                payload={"prompt": "2", "max_new_tokens": 1},
            )
        except HTTPError as error:
            assert error.code == 429
            assert json.loads(error.read())["error"] == "overloaded"
        else:
            raise AssertionError("overloaded request was accepted")
    finally:
        first.cancel()
        first.result(timeout=2)
        server.stop(timeout=2)
