"""Measure NQMC validation and record-wise model reload without running evaluation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import threading
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.core.model_checkpoint_file import NQMCFileReader
from nestquant.core.model_checkpoint import PackedNestQuantModel


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable; run this benchmark on Linux/WSL2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--mode", choices=("stream", "legacy"), default="stream")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        local_files_only=args.offline,
    ).to(args.device)
    model_load_seconds = time.perf_counter() - started
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    baseline_rss = _rss_bytes()
    peak_rss = baseline_rss
    stop = threading.Event()

    def sample_rss() -> None:
        nonlocal peak_rss
        while not stop.wait(0.01):
            peak_rss = max(peak_rss, _rss_bytes())

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    validation_started = time.perf_counter()
    if args.mode == "stream":
        checkpoint = NQMCFileReader(args.checkpoint)
        validation_seconds = time.perf_counter() - validation_started
        load_started = time.perf_counter()
        checkpoint.load_into_model(model, device=args.device)
        tensor_count = checkpoint.tensor_count
        quantized_count = sum(e.kind == "nestquant" for e in checkpoint.entries)
    else:
        blob = args.checkpoint.read_bytes()
        checkpoint = PackedNestQuantModel.from_bytes(blob)
        validation_seconds = time.perf_counter() - validation_started
        load_started = time.perf_counter()
        restored = checkpoint.state_dict(device=args.device)
        model.load_state_dict(restored, strict=True)
        tensor_count = checkpoint.tensor_count
        quantized_count = len(checkpoint.quantized_tensors)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    stop.set()
    sampler.join()
    peak_rss = max(peak_rss, _rss_bytes())

    result = {
        "scope": "NQMC reload benchmark; no PPL or downstream evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "model_id": args.model,
        "model_revision": args.revision,
        "device": args.device,
        "load_mode": args.mode,
        "torch_version": torch.__version__,
        "model_load_seconds": model_load_seconds,
        "nqmc_checksum_and_manifest_seconds": validation_seconds,
        "nqmc_record_validation_and_load_seconds": load_seconds,
        "nqmc_total_seconds": validation_seconds + load_seconds,
        "rss_before_nqmc_bytes": baseline_rss,
        "rss_peak_during_nqmc_bytes": peak_rss,
        "rss_peak_increment_bytes": peak_rss - baseline_rss,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None
        ),
        "tensor_count": tensor_count,
        "unique_quantized_tensors": quantized_count,
        "execution": (
            "checkpoint records dequantized one at a time into floating-point model tensors"
            if args.mode == "stream" else
            "whole file, parsed container and full restored state dict retained together"
        ),
        "claim_boundary": "measures checkpoint reload only; does not measure inference speed or PPL",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
