"""Run the auditable NestQuant CPU/CUDA runtime verification stages."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
STAGE_ORDER = ("cpu", "cuda_e8", "j2", "graph")


@dataclass(frozen=True)
class VerificationStage:
    name: str
    command: tuple[str, ...]


def build_stages(*, python: str, checkpoint: Path | None,
                 graph_output: Path) -> list[VerificationStage]:
    stages = [
        VerificationStage("cpu", (
            python, "-m", "pytest", "-q",
        )),
        VerificationStage("cuda_e8", (
            python, "scripts/build_test_cuda_e8.py",
        )),
        VerificationStage("j2", (
            python, "scripts/build_test_nqrt4_mma_j2.py",
        )),
    ]
    if checkpoint is not None:
        stages.append(VerificationStage("graph", (
            python, "scripts/benchmark_nqrt_graph_runner.py", str(checkpoint),
            "--batch-size", "1", "--sequence-length", "128",
            "--new-tokens", "8", "--validation-steps", "4",
            "--iterations", "5", "--repeats", "2", "--offline",
            "--output", str(graph_output),
        )))
    return stages


def select_stages(stages: Sequence[VerificationStage],
                  requested: Sequence[str]) -> list[VerificationStage]:
    names = list(STAGE_ORDER) if "all" in requested else list(dict.fromkeys(requested))
    available = {stage.name: stage for stage in stages}
    missing = [name for name in names if name not in available]
    if missing:
        suffix = " (pass --checkpoint to enable graph verification)" if "graph" in missing else ""
        raise ValueError(f"unavailable verification stages: {', '.join(missing)}{suffix}")
    return [available[name] for name in names]


def execute_stage(command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    """Execute one stage; kept separate so dry-run tests do not mock platform probes."""
    return subprocess.run(command, cwd=ROOT, check=False)


def run_verification(stages: Sequence[VerificationStage], *, dry_run: bool,
                     report_path: Path | None) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "dry_run": dry_run,
        "status": "planned" if dry_run else "running",
        "stages": [],
    }
    stage_reports: list[dict[str, object]] = report["stages"]  # type: ignore[assignment]
    for stage in stages:
        command = list(stage.command)
        item: dict[str, object] = {"name": stage.name, "command": command}
        print(f"[{stage.name}] {' '.join(command)}", flush=True)
        if dry_run:
            item["status"] = "planned"
            stage_reports.append(item)
            continue
        started = time.perf_counter()
        completed = execute_stage(command)
        item["elapsed_seconds"] = time.perf_counter() - started
        item["returncode"] = completed.returncode
        item["status"] = "passed" if completed.returncode == 0 else "failed"
        stage_reports.append(item)
        if completed.returncode != 0:
            report["status"] = "failed"
            _write_report(report_path, report)
            raise subprocess.CalledProcessError(completed.returncode, command)
    report["status"] = "planned" if dry_run else "passed"
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_report(report_path, report)
    return report


def _write_report(path: Path | None, report: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", choices=("all",) + STAGE_ORDER,
                        default=None, help="stage to run; repeat the option to select several")
    parser.add_argument("--checkpoint", type=Path,
                        help="NQMC checkpoint required by the graph stage")
    parser.add_argument("--graph-output", type=Path,
                        default=Path("results/raw/runtime_verification_graph.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/raw/runtime_verification.json"))
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for child stages")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    requested = args.stage or (["all"] if args.checkpoint is not None else ["cpu", "cuda_e8", "j2"])
    stages = build_stages(
        python=args.python, checkpoint=args.checkpoint, graph_output=args.graph_output)
    try:
        chosen = select_stages(stages, requested)
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run and args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    run_verification(chosen, dry_run=args.dry_run, report_path=args.report)


if __name__ == "__main__":
    main()
