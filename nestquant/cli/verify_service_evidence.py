"""Verify a promoted service benchmark against its raw report and config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .promote_service_benchmark import sha256_file, verify_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="raw benchmark JSON")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = verify_summary(
        report,
        config,
        summary,
        raw_sha256=sha256_file(args.report),
        config_sha256=sha256_file(args.config),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
