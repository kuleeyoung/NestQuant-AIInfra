import copy

import pytest

from nestquant.cli.validate_service_benchmark import validate_report


def valid_report() -> dict:
    run = {
        "run_id": "p-a-r0",
        "profile": "p",
        "policy": "a",
        "scheduled_requests": 2,
        "completed_requests": 2,
        "failed_requests": 0,
        "rejected_requests": {},
        "completion_rate": 1.0,
        "final_registered_requests": 0,
        "repetition": 0,
    }
    for prefix in ("ttft_ms", "tpot_ms", "e2e_ms", "step_latency_ms"):
        run.update({f"{prefix}_p50": 1.0, f"{prefix}_p95": 2.0, f"{prefix}_p99": 3.0})
    return {
        "status": "passed",
        "profiles": [{
            "name": "p", "length_mode": "uniform", "prompt_length": [4, 5]
        }],
        "policies": [{"name": "a"}],
        "warmup_prompt_lengths": [4, 5],
        "runs": [run],
        "summaries": [{"profile": "p", "policy": "a", "repetitions": 1}],
    }


def test_validate_complete_report() -> None:
    assert validate_report(valid_report()) == {
        "status": "passed", "profiles": 1, "policies": 1,
        "repetitions": 1, "runs": 1, "scheduled_requests": 2,
        "completed_requests": 2, "failed_requests": 0,
        "rejected_requests": 0, "final_registered_requests": 0,
    }


@pytest.mark.parametrize(
    "mutation", ["failed", "rejected", "retained", "unordered", "cold"]
)
def test_validate_rejects_broken_evidence(mutation: str) -> None:
    report = copy.deepcopy(valid_report())
    if mutation == "failed":
        report["runs"][0]["failed_requests"] = 1
    elif mutation == "rejected":
        report["runs"][0]["rejected_requests"] = {"request-1": "capacity"}
    elif mutation == "retained":
        report["runs"][0]["final_registered_requests"] = 1
    elif mutation == "unordered":
        report["runs"][0]["ttft_ms_p95"] = 0.5
    else:
        report["warmup_prompt_lengths"] = [4]
    with pytest.raises(ValueError):
        validate_report(report)


def test_validate_rejects_duplicate_repetition() -> None:
    report = valid_report()
    duplicate = copy.deepcopy(report["runs"][0])
    duplicate["run_id"] = "p-a-r1"
    report["runs"].append(duplicate)
    report["summaries"][0]["repetitions"] = 2
    with pytest.raises(ValueError, match="repetitions"):
        validate_report(report)
