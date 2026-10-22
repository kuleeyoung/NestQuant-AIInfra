"""Validate and summarize the batch-4 W4 -> W4KV4 -> W4A4KV4 progression."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "w4": ROOT / "results/raw/llama32_1b_w4_ldlq_cache_control_batch4_full_20261021.json",
    "w4kv4": ROOT / "results/raw/llama32_1b_w4kv4_upstream_betas_batch4_full_20261021.json",
    "w4a4kv4": ROOT / "results/raw/llama32_1b_w4a4kv4_upstream_betas_batch4_full_20261021.json",
}


def main() -> None:
    manifests = {name: json.loads(path.read_text("utf-8")) for name, path in RUNS.items()}
    for name, manifest in manifests.items():
        if manifest["status"] != "completed" or manifest["config"]["evaluation_batch_size"] != 4:
            raise ValueError(f"{name} must be a completed batch-4 full run")
    getters = {
        "model_revision": lambda d: d["config"]["model_revision"],
        "dataset_revision": lambda d: d["config"]["dataset_revision"],
        "token_ids_sha256": lambda d: d["data"]["token_ids_sha256"],
        "predicted_token_count": lambda d: d["data"]["predicted_token_count"],
        "source_sha256": lambda d: d["source"]["nestquant_python_source_sha256"],
        "checkpoint_sha256": lambda d: d["results"]["nestquant_w4"]["checkpoint_round_trip"]["file_sha256"],
    }
    comparable = {}
    for field, getter in getters.items():
        values = {getter(manifest) for manifest in manifests.values()}
        if len(values) != 1:
            raise ValueError(f"runs disagree on {field}: {values}")
        comparable[field] = values.pop()

    rows = {}
    baseline = manifests["w4"]["results"]["nestquant_w4"]["evaluation"]["perplexity"]
    kv4 = manifests["w4kv4"]["results"]["nestquant_w4"]["evaluation"]["perplexity"]
    labels = {"w4": "W4 LDLQ", "w4kv4": "W4KV4", "w4a4kv4": "W4A4KV4"}
    for name, manifest in manifests.items():
        method = manifest["results"]["nestquant_w4"]
        ppl = method["evaluation"]["perplexity"]
        activation = method.get("activation_quantization")
        kv = method.get("kv_cache_quantization")
        rows[name] = {
            "label": labels[name],
            "perplexity": ppl,
            "relative_change_vs_w4_percent": (ppl / baseline - 1) * 100,
            "relative_change_vs_w4kv4_percent": (ppl / kv4 - 1) * 100,
            "activation_relative_squared_error": None if activation is None else activation["relative_squared_error"],
            "activation_overload_fraction": None if activation is None else activation["overload_fraction"],
            "kv_relative_squared_error": None if kv is None else kv["relative_squared_error"],
            "manifest": str(RUNS[name].resolve()),
            "manifest_sha256": hashlib.sha256(RUNS[name].read_bytes()).hexdigest(),
            "peak_cuda_memory_allocated_bytes": method["evaluation"]["peak_cuda_memory_allocated_bytes"],
            "elapsed_seconds": method["elapsed_seconds"],
        }
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "validated",
        "evaluation_batch_size": 4,
        "comparability": comparable,
        "results": rows,
        "evidence_boundary": {
            "weights": "packed NQMC W4 is reloaded, then dequantized for BF16 compute",
            "activations": "A4 is floating reconstruction quality simulation",
            "kv_cache": "KV4 is floating reconstruction quality simulation",
            "speed": "elapsed time is reference-code overhead, not low-bit kernel speed",
        },
    }
    output_json = ROOT / "results/llama32_1b_w4a4kv4_progression.json"
    output_md = ROOT / "results/llama32_1b_w4a4kv4_progression.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    output_md.write_text(render_markdown(payload), "utf-8")
    print(f"Saved {output_json}")
    print(f"Saved {output_md}")


def render_markdown(payload: dict) -> str:
    rows = payload["results"]
    lines = [
        "# Llama-3.2-1B W4 → W4KV4 → W4A4KV4",
        "",
        "三组完整 WikiText-2 test 运行均使用 batch=4，并通过模型、数据、token、源码与 NQMC 哈希一致性检查。",
        "",
        "| 设置 | PPL | 相对 W4 | 相对 W4KV4 | 峰值 CUDA allocated |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("w4", "w4kv4", "w4a4kv4"):
        row = rows[name]
        lines.append(
            f"| {row['label']} | {row['perplexity']:.6f} | "
            f"{row['relative_change_vs_w4_percent']:+.3f}% | "
            f"{row['relative_change_vs_w4kv4_percent']:+.3f}% | "
            f"{row['peak_cuda_memory_allocated_bytes'] / 2**30:.3f} GiB |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- KV4 在 W4 上增加约 4.43% PPL；继续加入 A4 后，相对 W4KV4 再增加约 7.88%。当前配置中 A4 是更主要的新增质量损失来源。",
        "- A4 覆盖每层 q/k/v/o 与 gate/up/down 七类投影输入，使用作者公开默认 activation beta；KV4 使用作者公开默认 key/value beta。",
        "- batch=4 将本机 RTX 5070 利用率提高到接近满载，但不会形成推理加速证据，因为 A4/KV4 都在 Python/PyTorch 中重构为浮点张量。",
        "",
        "## 证据边界",
        "",
        "权重来自真实 packed NQMC 文件，但载入后以 BF16 执行；A4 和 KV4 都是质量仿真，没有 packed activation/KV buffer 或低比特 GEMM/attention kernel。因此这里只能陈述 PPL 与参考实现运行资源，不能陈述真实低比特延迟、吞吐或显存收益。",
        "",
        "## 可复核哈希",
        "",
        f"- token：`{payload['comparability']['token_ids_sha256']}`",
        f"- NQMC：`{payload['comparability']['checkpoint_sha256']}`",
        f"- Python 源码：`{payload['comparability']['source_sha256']}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
