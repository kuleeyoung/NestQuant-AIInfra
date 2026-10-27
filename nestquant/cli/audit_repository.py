"""Read-only checks for publishing a portable NestQuant repository."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from urllib.parse import unquote
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


DEFAULT_MAX_BYTES = 10 * 1024 * 1024
TEXT_SUFFIXES = {
    ".bash",
    ".cfg",
    ".cu",
    ".cuh",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SCANNED_PREFIXES = ("configs/", "docs/", "nestquant/", "scripts/", "tests/")
SCANNED_ROOT_FILES = {"README.md", "pyproject.toml"}
SENSITIVE_NAMES = re.compile(
    r"(^|/)(\.env($|\.)|id_(rsa|dsa|ecdsa|ed25519)(\.|$)|.*\.(pem|key|p12|pfx)$)",
    re.IGNORECASE,
)
ABSOLUTE_PATHS = (
    re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'`]+"),
    re.compile(r"(?<![A-Za-z0-9_])/(?:home|Users|mnt/[A-Za-z])/[A-Za-z0-9_.@+%/\\-]+"),
)
SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|hf[_-]?token|secret)\b"
    r"\s*[:=]\s*[\"']([^\"']+)[\"']"
)
PLACEHOLDER_MARKERS = ("example", "placeholder", "replace", "your_", "<", "${")
ALLOW_MARKER = "repository-audit: allow"
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int | None
    message: str


def publishable_files(repo_root: Path) -> list[Path]:
    """Return tracked and non-ignored untracked files that a commit may publish."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [repo_root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _is_scanned_text(relative: str) -> bool:
    return relative in SCANNED_ROOT_FILES or relative.startswith(SCANNED_PREFIXES)


def audit_paths(
    repo_root: Path,
    paths: Iterable[Path],
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Finding]:
    findings: list[Finding] = []
    for path in paths:
        relative = _relative(path, repo_root)
        normalized = PurePosixPath(relative)
        if SENSITIVE_NAMES.search(relative):
            findings.append(Finding("sensitive_filename", relative, None, "sensitive file is tracked"))
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > max_bytes:
            findings.append(
                Finding("large_file", relative, None, f"tracked file is {size} bytes (limit {max_bytes})")
            )
        if path.suffix.lower() not in TEXT_SUFFIXES or not _is_scanned_text(relative):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            findings.append(Finding("invalid_utf8", relative, None, "text file is not valid UTF-8"))
            continue
        for line_number, line in enumerate(lines, start=1):
            if ALLOW_MARKER in line:
                continue
            if any(pattern.search(line) for pattern in ABSOLUTE_PATHS):
                findings.append(
                    Finding("absolute_path", relative, line_number, "machine-specific absolute path")
                )
            match = SECRET_ASSIGNMENT.search(line)
            if match and not any(marker in match.group(2).lower() for marker in PLACEHOLDER_MARKERS):
                findings.append(
                    Finding("possible_secret", relative, line_number, "possible embedded credential")
                )
            if path.suffix.lower() == ".md":
                for raw_target in MARKDOWN_LINK.findall(line):
                    target = raw_target.strip()
                    if target.startswith("<") and ">" in target:
                        target = target[1:target.index(">")]
                    else:
                        target = target.split(maxsplit=1)[0]
                    target = unquote(target.split("#", 1)[0])
                    if (not target or "://" in target or target.startswith(("#", "mailto:", "data:"))
                            or any(marker in target for marker in ("{", "}", "*"))):
                        continue
                    candidate = (path.parent / target).resolve()
                    try:
                        candidate.relative_to(repo_root.resolve())
                    except ValueError:
                        findings.append(Finding(
                            "link_outside_repository", relative, line_number,
                            "local Markdown link points outside the repository",
                        ))
                        continue
                    if not candidate.exists():
                        findings.append(Finding(
                            "broken_local_link", relative, line_number,
                            f"local Markdown link target does not exist: {target}",
                        ))
    return findings


def audit_repository(repo_root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> list[Finding]:
    return audit_paths(repo_root, publishable_files(repo_root), max_bytes=max_bytes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-mib", type=float, default=DEFAULT_MAX_BYTES / 1024**2)
    parser.add_argument("--json-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    candidates = publishable_files(repo_root)
    findings = audit_paths(repo_root, candidates, max_bytes=int(args.max_mib * 1024**2))
    payload = {
        "status": "passed" if not findings else "failed",
        "repo_root": str(repo_root),
        "publishable_files": len(candidates),
        "findings": [asdict(finding) for finding in findings],
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
