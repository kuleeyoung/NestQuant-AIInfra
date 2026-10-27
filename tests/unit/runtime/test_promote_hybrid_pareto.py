from nestquant.cli.promote_hybrid_pareto import build_summary


def test_build_summary_keeps_only_auditable_pareto_fields() -> None:
    report = {
        "device": "test-gpu",
        "torch_version": "test-torch",
        "model_id": "test-model",
        "model_revision": "test-revision",
        "pareto_budget_mib": [0],
        "claim_boundary": "finite measured set only",
        "results": [{
            "budget_mib": 0,
            "resident_projection_count": 0,
            "packed_projection_count": 2,
            "cuda_allocated_bytes_after_warmup": 1024,
            "actual_packed_device_bytes": 512,
            "steady_prefill_seconds": {"median": 0.003},
            "ttft": {"median_ms": 4.0, "p95_ms": 5.0},
            "tpot": {"median_ms": 2.0, "p95_ms": 2.5},
            "decode_tokens_per_second_per_request": 500.0,
            "relative_l2_logit_error_vs_bf16": 0.1,
            "large_unpublished_samples": [1, 2, 3],
        }],
    }
    validation = {"status": "passed", "budget_points": 1}
    hashes = {
        "raw": "raw-hash",
        "profile": "profile-hash",
        "config": "config-hash",
        "checkpoint": "checkpoint-hash",
    }

    summary = build_summary(report, validation, hashes=hashes)

    assert summary["status"] == "passed"
    assert summary["source_raw_sha256"] == "raw-hash"
    assert summary["checkpoint_sha256"] == "checkpoint-hash"
    assert summary["points"][0]["steady_prefill_median_ms"] == 3.0
    assert "large_unpublished_samples" not in summary["points"][0]
