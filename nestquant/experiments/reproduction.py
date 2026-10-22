"""Config-driven orchestration for the auditable NestQuant reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reproduction.v1.json"
FULL_CONFIRMATION = "RUN_FULL_LLAMA_REPRODUCTION"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_reproduction_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("schema_version") != 1:
        raise ValueError("reproduction config schema_version must be 1")
    stages = config.get("stages")
    if not isinstance(stages, dict) or not all(name in stages for name in ("smoke", "full")):
        raise ValueError("reproduction config must define smoke and full stages")
    for stage_name in ("smoke", "full"):
        if not isinstance(stages[stage_name], list):
            raise ValueError(f"stage {stage_name!r} must be a list")
        for step in stages[stage_name]:
            if not isinstance(step, dict) or not step.get("name") or not isinstance(step.get("command"), list):
                raise ValueError(f"invalid step in stage {stage_name!r}")
    return config


def _resolve_path(value: str, project_root: Path) -> Path:
    rendered = value.replace("{project_root}", str(project_root))
    path = Path(rendered)
    return path if path.is_absolute() else project_root / path


def render_command(command: Sequence[str], project_root: Path) -> list[str]:
    replacements = {"{python}": sys.executable, "{project_root}": str(project_root)}
    return [replacements.get(part, part.replace("{project_root}", str(project_root))) for part in command]


def preflight(config: dict[str, Any], project_root: Path, *, require_cuda: bool) -> dict[str, Any]:
    import torch

    required = [_resolve_path(item, project_root) for item in config.get("required_files", [])]
    missing = [str(path) for path in required if not path.exists()]
    cuda_available = torch.cuda.is_available()
    checks = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "required_file_count": len(required),
        "missing_files": missing,
    }
    checks["status"] = "failed" if missing or (require_cuda and not cuda_available) else "passed"
    if require_cuda and not cuda_available:
        checks["cuda_error"] = "full reproduction requires a CUDA device"
    return checks


def plan_stage(config: dict[str, Any], stage: str, project_root: Path) -> list[dict[str, Any]]:
    plan = []
    for step in config["stages"][stage]:
        expected = [_resolve_path(item, project_root) for item in step.get("expected_outputs", [])]
        plan.append({
            "name": step["name"],
            "command": render_command(step["command"], project_root),
            "expected_outputs": [str(path) for path in expected],
            "outputs_complete": bool(expected) and all(path.exists() for path in expected),
        })
    return plan


def execute_stage(
    config: dict[str, Any], stage: str, project_root: Path, *, dry_run: bool, force: bool,
) -> list[dict[str, Any]]:
    records = []
    for step in plan_stage(config, stage, project_root):
        record = dict(step)
        if step["outputs_complete"] and not force:
            record["status"] = "skipped_existing"
        elif dry_run:
            record["status"] = "planned"
        else:
            print(f"[{stage}] {step['name']}", flush=True)
            subprocess.run(step["command"], cwd=project_root, check=True)
            missing = [path for path in step["expected_outputs"] if not Path(path).exists()]
            if missing:
                raise RuntimeError(f"step {step['name']!r} did not create: {missing}")
            record["status"] = "completed"
        records.append(record)
    return records


def _nested_value(payload: dict[str, Any], path: Sequence[str]) -> Any:
    cursor: Any = payload
    for component in path:
        cursor = cursor[component]
    return cursor


def validate_reviewed_result(
    config: dict[str, Any], project_root: Path, *, require_raw: bool,
    evidence_field: str = "raw_evidence",
) -> dict[str, Any]:
    spec = config["reviewed_result"]
    summary_path = _resolve_path(spec["summary"], project_root)
    summary = _read_json(summary_path)
    errors: list[str] = []
    for key, expected in spec["expected"].items():
        if summary.get(key) != expected:
            errors.append(f"summary {key!r} does not match reproduction config")
    methods = summary.get("nestquant", {})
    for method in ("w4", "w4kv4", "w4a4", "w4a4kv4"):
        ppl = methods.get(method, {}).get("perplexity")
        if not isinstance(ppl, (int, float)) or ppl <= 0:
            errors.append(f"missing positive NestQuant PPL for {method}")

    raw_records = []
    for item in spec.get(evidence_field, []):
        path = _resolve_path(item["path"], project_root)
        if not path.exists():
            raw_records.append({"path": str(path), "status": "missing"})
            if require_raw:
                errors.append(f"required raw evidence is missing: {path}")
            continue
        raw = _read_json(path)
        cursor = _nested_value(raw, item["value_path"])
        expected = methods[item["method"]]["perplexity"]
        if cursor != expected:
            errors.append(f"raw PPL mismatch: {path}")
        for check in item.get("checks", []):
            if _nested_value(raw, check["value_path"]) != check["expected"]:
                errors.append(f"raw protocol mismatch for {check['name']}: {path}")
        raw_records.append({"path": str(path), "status": "matched"})
    return {
        "status": "passed" if not errors else "failed",
        "summary": str(summary_path),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "raw_evidence": raw_records,
        "errors": errors,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("plan", "preflight", "smoke", "full", "validate", "all"), default="plan")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="rerun steps even when expected outputs exist")
    parser.add_argument("--confirm-full", default="", help=f"full-stage guard; exact value: {FULL_CONFIRMATION}")
    parser.add_argument("--require-raw-evidence", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("results/raw/reproduction_run.json"))
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else PROJECT_ROOT
    config = load_reproduction_config(config_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "stage": args.stage,
        "dry_run": args.dry_run,
    }

    if args.stage == "plan":
        payload["preflight"] = preflight(config, project_root, require_cuda=False)
        payload["smoke"] = plan_stage(config, "smoke", project_root)
        payload["full"] = plan_stage(config, "full", project_root)
    else:
        full_requested = args.stage in ("full", "all")
        if full_requested and not args.dry_run and args.confirm_full != FULL_CONFIRMATION:
            parser.error(f"full execution requires --confirm-full {FULL_CONFIRMATION}")
        payload["preflight"] = preflight(
            config, project_root, require_cuda=full_requested and not args.dry_run
        )
        if payload["preflight"]["status"] != "passed":
            _write_report(_resolve_path(str(args.report), project_root), payload)
            return 2
        try:
            if args.stage in ("smoke", "all"):
                payload["smoke"] = execute_stage(
                    config, "smoke", project_root, dry_run=args.dry_run, force=args.force
                )
            if full_requested:
                payload["full"] = execute_stage(
                    config, "full", project_root, dry_run=args.dry_run, force=args.force
                )
        except Exception as error:
            payload["status"] = "failed"
            payload["error"] = {"type": type(error).__name__, "message": str(error)}
            _write_report(_resolve_path(str(args.report), project_root), payload)
            return 4
        if args.stage in ("validate", "all"):
            validate_new_run = args.stage == "all" and not args.dry_run
            payload["validation"] = validate_reviewed_result(
                config, project_root,
                require_raw=args.require_raw_evidence or validate_new_run,
                evidence_field="full_evidence" if validate_new_run else "raw_evidence",
            )
            if payload["validation"]["status"] != "passed":
                _write_report(_resolve_path(str(args.report), project_root), payload)
                return 3
    report = _resolve_path(str(args.report), project_root)
    _write_report(report, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
