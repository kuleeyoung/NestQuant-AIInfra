"""Validate BF16/W4/W4A4KV4 outputs and write JSON plus Markdown summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.evaluation.downstream_summary import render_markdown, summarize_downstream_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run_dir / f"{method}.json" for method in ("bf16", "w4", "w4a4kv4")]
    summary = summarize_downstream_results(paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
