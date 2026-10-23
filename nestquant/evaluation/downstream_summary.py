"""Validate and summarize comparable lm-eval downstream runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


METHOD_ORDER = ("bf16", "w4", "w4a4kv4")
METRICS = ("acc,none", "acc_norm,none")


def _load_completed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"result is not completed: {path}")
    if payload.get("method") not in METHOD_ORDER:
        raise ValueError(f"unsupported method in {path}: {payload.get('method')!r}")
    return payload


def _protocol_signature(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model", {})
    return {
        "tasks": payload.get("tasks"),
        "limit_per_task": payload.get("limit_per_task"),
        "zero_shot": payload.get("zero_shot"),
        "model_id": model.get("id"),
        "model_revision": model.get("revision"),
        "lm_eval": payload.get("environment", {}).get("lm_eval"),
        "task_provenance": payload.get("task_provenance"),
    }


def _sample_map(payload: dict[str, Any], task: str) -> dict[tuple[int, str], dict]:
    samples = payload["lm_eval_results"]["samples"][task]
    mapped = {(sample["doc_id"], sample["doc_hash"]): sample for sample in samples}
    if len(mapped) != len(samples):
        raise ValueError(f"duplicate sample identity for {payload['method']} / {task}")
    return mapped


def _correct(sample: dict[str, Any], metric: str) -> bool:
    sample_metric = metric.split(",", 1)[0]
    value = sample.get(sample_metric)
    if value not in (0, 0.0, 1, 1.0, False, True):
        raise ValueError(f"sample metric {sample_metric!r} is not binary: {value!r}")
    return bool(value)


def summarize_downstream_results(paths: list[Path]) -> dict[str, Any]:
    """Return a comparison only when all runs use the same protocol and samples."""
    payloads = [_load_completed(path) for path in paths]
    by_method = {payload["method"]: payload for payload in payloads}
    if len(by_method) != len(payloads):
        raise ValueError("duplicate method result")
    missing = [method for method in METHOD_ORDER if method not in by_method]
    if missing:
        raise ValueError(f"missing method results: {', '.join(missing)}")

    baseline = by_method["bf16"]
    signature = _protocol_signature(baseline)
    for method in METHOD_ORDER[1:]:
        candidate_signature = _protocol_signature(by_method[method])
        if candidate_signature != signature:
            raise ValueError(f"protocol mismatch between bf16 and {method}")

    tasks = list(signature["tasks"])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "protocol": signature,
        "methods": {},
        "correctness_transitions_vs_bf16": {},
        "evidence_boundary": baseline.get("evidence_boundary"),
    }
    for method in METHOD_ORDER:
        payload = by_method[method]
        task_results = payload["lm_eval_results"]["results"]
        summary["methods"][method] = {
            "elapsed_seconds": payload.get("elapsed_seconds"),
            "tasks": {
                task: {
                    "sample_len": task_results[task]["sample_len"],
                    **{
                        metric: task_results[task][metric]
                        for metric in METRICS
                    },
                    **{
                        f"delta_{metric}_pp_vs_bf16": 100.0 * (
                            task_results[task][metric]
                            - baseline["lm_eval_results"]["results"][task][metric]
                        )
                        for metric in METRICS
                    },
                }
                for task in tasks
            },
        }

    for method in METHOD_ORDER[1:]:
        summary["correctness_transitions_vs_bf16"][method] = {}
        for task in tasks:
            base_samples = _sample_map(baseline, task)
            candidate_samples = _sample_map(by_method[method], task)
            if base_samples.keys() != candidate_samples.keys():
                raise ValueError(f"sample identity mismatch for {method} / {task}")
            metric_transitions = {}
            for metric in METRICS:
                counts = {
                    "both_correct": 0,
                    "bf16_correct_quantized_wrong": 0,
                    "bf16_wrong_quantized_correct": 0,
                    "both_wrong": 0,
                }
                for identity, base_sample in base_samples.items():
                    base_correct = _correct(base_sample, metric)
                    candidate_correct = _correct(candidate_samples[identity], metric)
                    if base_correct and candidate_correct:
                        counts["both_correct"] += 1
                    elif base_correct:
                        counts["bf16_correct_quantized_wrong"] += 1
                    elif candidate_correct:
                        counts["bf16_wrong_quantized_correct"] += 1
                    else:
                        counts["both_wrong"] += 1
                counts["net_correct_change"] = (
                    counts["bf16_wrong_quantized_correct"]
                    - counts["bf16_correct_quantized_wrong"]
                )
                metric_transitions[metric] = counts
            summary["correctness_transitions_vs_bf16"][method][task] = metric_transitions
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Llama-3.2-1B PIQA 与 HellaSwag 结果",
        "",
        "> 本报告由原始 JSON 自动生成。只有通过协议与逐样本身份校验的结果才会进入表格。",
        "",
        "## 汇总指标",
        "",
        "| 任务 | 方法 | 样本数 | acc | 相对 BF16 | acc_norm | 相对 BF16 | 耗时（分钟） |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    tasks = summary["protocol"]["tasks"]
    for task in tasks:
        for method in METHOD_ORDER:
            method_data = summary["methods"][method]
            values = method_data["tasks"][task]
            elapsed = method_data["elapsed_seconds"]
            lines.append(
                f"| {task} | {method} | {values['sample_len']} | "
                f"{values['acc,none']:.4f} | {values['delta_acc,none_pp_vs_bf16']:+.2f} pp | "
                f"{values['acc_norm,none']:.4f} | "
                f"{values['delta_acc_norm,none_pp_vs_bf16']:+.2f} pp | "
                f"{elapsed / 60:.1f} |"
            )

    lines.extend(["", "## 相对 BF16 的逐题正确性变化", ""])
    for method in METHOD_ORDER[1:]:
        lines.extend([f"### {method}", ""])
        for task in tasks:
            lines.append(f"- **{task}**")
            for metric in METRICS:
                counts = summary["correctness_transitions_vs_bf16"][method][task][metric]
                lines.append(
                    f"  - `{metric}`：BF16 正确但量化后错误 "
                    f"{counts['bf16_correct_quantized_wrong']}；BF16 错误但量化后正确 "
                    f"{counts['bf16_wrong_quantized_correct']}；净正确数变化 "
                    f"{counts['net_correct_change']:+d}。"
                )
        lines.append("")

    lines.extend([
        "## 证据边界",
        "",
        summary["evidence_boundary"],
        "",
        "这里的分数用于判断量化质量。它们不构成真实低比特 kernel 的显存、吞吐或延迟证据。",
        "",
    ])
    return "\n".join(lines)
