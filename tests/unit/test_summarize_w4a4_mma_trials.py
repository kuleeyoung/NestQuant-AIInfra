import json
from pathlib import Path

import pytest

from scripts.summarize_w4a4_mma_trials import summarize_trials


def _report(tiled: float, cuda: float, *, seed: int = 7) -> dict[str, object]:
    environment = {"pytorch": "test", "cuda_runtime": "test"}
    common = {
        "model_id": "model", "model_revision": "revision",
        "device_name": "gpu", "environment": environment,
        "input_seed": seed, "sequence_lengths": [16], "batch_sizes": [1],
        "decode_tokens": 4, "repeats": 7,
        "modes": ["packed_w4a4_tiled", "packed_w4a4_cuda_mma"],
        "activation_q": 14, "activation_betas": [0.25, 0.5, 0.75, 1.0],
    }
    def path(mode: str, latency: float, mma: int, fallback: int):
        return {
            "mode": mode,
            "steady_prefill_seconds": {"median": latency},
            "ttft_seconds": latency + 0.01,
            "tpot_seconds": 0.2,
            "w4a4_dispatch_counts": {
                "cuda_mma_m16": mma, "tiled_fallback": fallback,
            },
            "max_abs_logit_error_vs_first_mode": 1.0 if mma else 0.0,
            "relative_l2_logit_error_vs_first_mode": 0.1 if mma else 0.0,
        }
    common["results"] = [{
        "input_ids_sha256": "tokens",
        "results": [
            path("packed_w4a4_tiled", tiled, 0, 0),
            path("packed_w4a4_cuda_mma", cuda, 1232, 448),
        ],
    }]
    return common


def _write(path: Path, report: dict[str, object]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_summarizes_independent_trial_medians(tmp_path: Path):
    paths = [
        _write(tmp_path / f"trial{index}.json", _report(tiled, cuda))
        for index, (tiled, cuda) in enumerate(
            [(0.18, 0.12), (0.17, 0.11), (0.19, 0.13)], start=1
        )
    ]
    summary = summarize_trials(paths)
    assert summary["trial_count"] == 3
    assert summary["tiled_median_seconds_across_trials"]["median"] == 0.18
    assert summary["cuda_mma_median_seconds_across_trials"]["median"] == 0.12
    assert summary["speedup_across_trials"]["median"] == pytest.approx(1.5)


def test_rejects_protocol_drift(tmp_path: Path):
    paths = [
        _write(tmp_path / "trial1.json", _report(0.18, 0.12)),
        _write(tmp_path / "trial2.json", _report(0.18, 0.12, seed=8)),
        _write(tmp_path / "trial3.json", _report(0.18, 0.12)),
    ]
    with pytest.raises(ValueError, match="input_seed"):
        summarize_trials(paths)
