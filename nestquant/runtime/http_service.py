"""Minimal HTTP/JSON adapter for :mod:`nestquant.runtime.online_service`."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from typing import Protocol
from urllib.parse import unquote
import uuid

from .continuous_batching import DecodeRequest
from .online_service import (
    OnlineInferenceService,
    OnlineRequestHandle,
    OnlineServiceStatus,
    ServiceOverloadedError,
)


class TokenizerProtocol(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...


class NestQuantHTTPServer:
    """Expose health, metrics, generation and cancellation over HTTP/JSON.

    This is a dependency-free demonstration server. ``ThreadingHTTPServer``
    handles network connections, while ``OnlineInferenceService`` remains the
    sole owner of model, scheduler and CUDA state.
    """

    def __init__(
        self,
        service: OnlineInferenceService,
        tokenizer: TokenizerProtocol,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.service = service
        self.tokenizer = tokenizer
        self._lock = threading.Lock()
        self._handles: dict[str, OnlineRequestHandle] = {}
        self._metrics = {
            "submitted": 0,
            "completed": 0,
            "cancelled": 0,
            "timed_out": 0,
            "failed": 0,
            "overloaded": 0,
        }
        self._httpd = ThreadingHTTPServer((host, port), self._handler_type())
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    def start(self, timeout: float | None = None) -> None:
        self.service.start(timeout=timeout)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="nestquant-http-server",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.service.stop(timeout=timeout)

    def _record_result(self, status: OnlineServiceStatus) -> None:
        key = {
            OnlineServiceStatus.FINISHED: "completed",
            OnlineServiceStatus.CANCELLED: "cancelled",
            OnlineServiceStatus.TIMED_OUT: "timed_out",
            OnlineServiceStatus.FAILED: "failed",
        }[status]
        with self._lock:
            self._metrics[key] += 1

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "NestQuantHTTP/0.1"

            def log_message(self, format, *args):
                return

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 1_048_576:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("request JSON must be an object")
                return payload

            def do_GET(self):
                if self.path == "/health":
                    self._json(HTTPStatus.OK, {
                        "status": "ok",
                        "inflight": owner.service.inflight_count,
                    })
                    return
                if self.path == "/metrics":
                    with owner._lock:
                        metrics = dict(owner._metrics)
                    metrics["inflight"] = owner.service.inflight_count
                    metrics["runtime"] = owner.service.diagnostics_snapshot
                    self._json(HTTPStatus.OK, metrics)
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

            def do_DELETE(self):
                prefix = "/v1/requests/"
                if not self.path.startswith(prefix):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                request_id = unquote(self.path[len(prefix):])
                with owner._lock:
                    handle = owner._handles.get(request_id)
                if handle is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown_request"})
                    return
                handle.cancel()
                self._json(HTTPStatus.ACCEPTED, {
                    "request_id": request_id, "status": "cancelling"
                })

            def do_POST(self):
                if self.path not in ("/v1/generate", "/v1/generate/stream"):
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                try:
                    payload = self._read_json()
                    prompt = payload["prompt"]
                    max_new_tokens = payload.get("max_new_tokens", 32)
                    timeout_seconds = payload.get("timeout_seconds")
                    request_id = payload.get("request_id") or uuid.uuid4().hex
                    if not isinstance(prompt, str) or not prompt:
                        raise ValueError("prompt must be a non-empty string")
                    if (isinstance(max_new_tokens, bool)
                            or not isinstance(max_new_tokens, int)
                            or max_new_tokens < 1):
                        raise ValueError("max_new_tokens must be a positive integer")
                    if not isinstance(request_id, str) or not request_id:
                        raise ValueError("request_id must be a non-empty string")
                    token_ids = tuple(int(token) for token in owner.tokenizer.encode(
                        prompt, add_special_tokens=True
                    ))
                    request = DecodeRequest(
                        request_id,
                        token_ids,
                        max_new_tokens,
                        eos_token_id=owner.tokenizer.eos_token_id,
                    )
                    handle = owner.service.submit(
                        request, timeout_seconds=timeout_seconds
                    )
                except ServiceOverloadedError as error:
                    with owner._lock:
                        owner._metrics["overloaded"] += 1
                    self._json(HTTPStatus.TOO_MANY_REQUESTS, {
                        "error": "overloaded", "detail": str(error)
                    })
                    return
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {
                        "error": "invalid_request", "detail": str(error)
                    })
                    return
                with owner._lock:
                    owner._handles[request_id] = handle
                    owner._metrics["submitted"] += 1
                try:
                    if self.path.endswith("/stream"):
                        self._stream(handle)
                    else:
                        self._complete(handle)
                finally:
                    with owner._lock:
                        owner._handles.pop(request_id, None)

            def _complete(self, handle: OnlineRequestHandle) -> None:
                result = handle.result()
                owner._record_result(result.status)
                status = {
                    OnlineServiceStatus.FINISHED: HTTPStatus.OK,
                    OnlineServiceStatus.CANCELLED: HTTPStatus.CONFLICT,
                    OnlineServiceStatus.TIMED_OUT: HTTPStatus.GATEWAY_TIMEOUT,
                    OnlineServiceStatus.FAILED: HTTPStatus.INTERNAL_SERVER_ERROR,
                }[result.status]
                text = owner.tokenizer.decode(
                    list(result.generated_token_ids), skip_special_tokens=True
                )
                self._json(status, {
                    "request_id": result.request_id,
                    "status": result.status.value,
                    "generated_token_ids": list(result.generated_token_ids),
                    "text": text,
                    "timings_ms": result.timings_ms,
                    "error": result.error,
                })

            def _stream(self, handle: OnlineRequestHandle) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    for token in handle.iter_tokens():
                        event = {
                            "request_id": handle.request_id,
                            "token_id": token,
                            "text": owner.tokenizer.decode(
                                [token], skip_special_tokens=True
                            ),
                        }
                        self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
                        self.wfile.flush()
                    result = handle.result()
                    owner._record_result(result.status)
                    self.wfile.write(json.dumps({
                        "request_id": result.request_id,
                        "status": result.status.value,
                        "timings_ms": result.timings_ms,
                        "done": True,
                    }).encode("utf-8") + b"\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    handle.cancel()

        return Handler
