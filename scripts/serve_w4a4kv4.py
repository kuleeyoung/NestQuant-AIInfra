"""Serve one packed W4A4KV4 checkpoint over the minimal HTTP/JSON adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading
import time

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.models.fused_linear import convert_linears_to_fused_nqrt_
from nestquant.runtime.http_service import NestQuantHTTPServer
from nestquant.runtime.inference_engine import (
    AdmissionPolicy,
    ContinuousBatchInferenceEngine,
)
from nestquant.runtime.nqkv_batched_hf_backend import (
    BatchedNQKVLlamaCache,
    install_batched_nqkv_llama_backend,
)
from nestquant.runtime.nqkv_batched_runner import BatchedNQKVLlamaRunner
from nestquant.runtime.online_service import OnlineInferenceService
from scripts.benchmark_full_nqrt_model import (
    ACTIVATION_BETAS,
    _load_model,
    _projection_names,
)
from scripts.benchmark_service_workloads import KEY_BETAS, VALUE_BETAS


DEFAULT_ENABLE_M1_DOWN_SPLITK = True


def add_m1_down_splitk_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the quality-gated service policy and its explicit rollback flag."""
    splitk_group = parser.add_mutually_exclusive_group()
    splitk_group.add_argument(
        "--enable-m1-down-splitk",
        dest="enable_m1_down_splitk",
        action="store_true",
        help="Use the quality-gated split-8 M1 K8192/N2048 down-projection path (default).",
    )
    splitk_group.add_argument(
        "--disable-m1-down-splitk",
        dest="enable_m1_down_splitk",
        action="store_false",
        help="Roll back to the unsplit M1 path for output-stability comparisons.",
    )
    parser.set_defaults(enable_m1_down_splitk=DEFAULT_ENABLE_M1_DOWN_SPLITK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--max-cache-len", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--min-prefill-batch", type=int, default=2)
    parser.add_argument("--max-wait-steps", type=int, default=2)
    parser.add_argument(
        "--profile-control-file",
        type=Path,
        help=(
            "Optional local control file accepting start, stop and shutdown. "
            "start/stop call the CUDA Profiler API for an external Nsight capture."
        ),
    )
    parser.add_argument("--torch-profiler-output", type=Path)
    parser.add_argument("--torch-profiler-summary", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--execution-policy",
        choices=("fixed", "decode", "full"),
        default="full",
        help="Select fixed-capacity, compact-decode, or compact-prefill+decode execution.",
    )
    parser.add_argument(
        "--compute-backend",
        choices=("packed_w4a4_tiled", "packed_w4a4_cuda_mma"),
        default="packed_w4a4_cuda_mma",
    )
    add_m1_down_splitk_arguments(parser)
    parser.add_argument(
        "--m1-down-splitk-fallback-name",
        action="append",
        default=[],
        help=(
            "Experimental fused-linear module kept on unsplit M1 while split-K "
            "is enabled elsewhere; may be repeated."
        ),
    )
    args = parser.parse_args()
    if args.capacity < 1 or args.max_inflight < args.capacity:
        parser.error("--max-inflight must be at least --capacity, both positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in [1, 65535]")
    if (args.torch_profiler_output is None) != (args.torch_profiler_summary is None):
        parser.error("torch profiler output and summary must be provided together")
    if args.torch_profiler_output is not None and args.profile_control_file is None:
        parser.error("torch profiler requires --profile-control-file")
    if args.m1_down_splitk_fallback_name and not args.enable_m1_down_splitk:
        parser.error("split-K fallback names require --enable-m1-down-splitk")

    reader = NQMCFileReader(args.checkpoint, verify_checksum=False)
    model_id = reader.model_metadata["model_id"]
    revision = reader.model_metadata["model_revision"]
    model = _load_model(model_id, revision, args.offline)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=args.offline, use_fast=True
    )
    names = _projection_names(model, reader)
    convert_linears_to_fused_nqrt_(
        model,
        reader,
        names,
        output_dtype=torch.bfloat16,
        group_qkv=False,
        compute_backend=args.compute_backend,
        activation_q=14,
        activation_betas=ACTIVATION_BETAS,
        enable_m1_down_splitk=args.enable_m1_down_splitk,
        m1_down_splitk_fallback_names=args.m1_down_splitk_fallback_name,
    )
    head_dim = int(getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    ))
    cache = BatchedNQKVLlamaCache(
        capacity=args.capacity,
        num_layers=int(model.config.num_hidden_layers),
        num_query_heads=int(model.config.num_attention_heads),
        num_kv_heads=int(model.config.num_key_value_heads),
        head_dim=head_dim,
        max_cache_len=args.max_cache_len,
        page_size=args.page_size,
        q=14,
        key_betas=KEY_BETAS,
        value_betas=VALUE_BETAS,
        dtype=torch.bfloat16,
        device="cuda",
    )
    install_batched_nqkv_llama_backend(model, cache)
    runner = BatchedNQKVLlamaRunner(
        model,
        cache,
        execution_policy=args.execution_policy,
    )
    engine = ContinuousBatchInferenceEngine(
        runner,
        pad_token_id=int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0),
        admission_policy=AdmissionPolicy(
            args.min_prefill_batch, args.max_wait_steps
        ),
    )
    service = OnlineInferenceService(
        engine,
        max_inflight_requests=args.max_inflight,
        default_timeout_seconds=args.request_timeout,
    )
    server = NestQuantHTTPServer(
        service, tokenizer, host=args.host, port=args.port
    )
    stopped = threading.Event()
    profiler_thread = None

    def stop_handler(signum, frame):
        stopped.set()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    if args.profile_control_file is not None:
        control_path = args.profile_control_file.resolve()
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.unlink(missing_ok=True)

        def profile_control_loop() -> None:
            last_command = None
            profiler_active = False
            torch_profiler = None

            def stop_torch_profiler() -> None:
                nonlocal torch_profiler
                if torch_profiler is None:
                    return
                torch_profiler.stop()
                assert args.torch_profiler_output is not None
                assert args.torch_profiler_summary is not None
                args.torch_profiler_output.parent.mkdir(parents=True, exist_ok=True)
                args.torch_profiler_summary.parent.mkdir(parents=True, exist_ok=True)
                torch_profiler.export_chrome_trace(str(args.torch_profiler_output))
                events = []
                for event in torch_profiler.key_averages():
                    device_total = float(getattr(event, "device_time_total", 0.0))
                    self_device = float(
                        getattr(event, "self_device_time_total", 0.0)
                    )
                    if device_total <= 0.0 and self_device <= 0.0:
                        continue
                    events.append({
                        "name": event.key,
                        "count": int(event.count),
                        "device_time_total_us": device_total,
                        "self_device_time_total_us": self_device,
                        "cpu_time_total_us": float(event.cpu_time_total),
                    })
                events.sort(
                    key=lambda item: item["self_device_time_total_us"],
                    reverse=True,
                )
                args.torch_profiler_summary.write_text(
                    json.dumps({
                        "status": "passed",
                        "event_count": len(events),
                        "events": events,
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                torch_profiler = None
                print("PyTorch CUDA profiler capture stopped", flush=True)

            while not stopped.is_set():
                try:
                    command = control_path.read_text(encoding="utf-8").strip().lower()
                except FileNotFoundError:
                    command = ""
                if command and command != last_command:
                    if command == "start" and not profiler_active:
                        if args.torch_profiler_output is not None:
                            torch_profiler = torch.profiler.profile(
                                activities=(
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ),
                                record_shapes=False,
                                profile_memory=False,
                            )
                            torch_profiler.start()
                        torch.cuda.cudart().cudaProfilerStart()
                        profiler_active = True
                        print("CUDA profiler capture started", flush=True)
                    elif command == "stop" and profiler_active:
                        torch.cuda.cudart().cudaProfilerStop()
                        stop_torch_profiler()
                        profiler_active = False
                        print("CUDA profiler capture stopped", flush=True)
                    elif command == "shutdown":
                        if profiler_active:
                            torch.cuda.cudart().cudaProfilerStop()
                            stop_torch_profiler()
                        stopped.set()
                        break
                    last_command = command
                time.sleep(0.05)

        profiler_thread = threading.Thread(
            target=profile_control_loop,
            name="cuda-profiler-control",
            daemon=True,
        )
        profiler_thread.start()
    server.start()
    host, port = server.address
    print(f"NestQuant W4A4KV4 server listening on http://{host}:{port}", flush=True)
    print("POST /v1/generate or /v1/generate/stream; Ctrl+C to stop", flush=True)
    try:
        stopped.wait()
    finally:
        server.stop(timeout=30)
        if profiler_thread is not None:
            profiler_thread.join(timeout=5)


if __name__ == "__main__":
    main()
