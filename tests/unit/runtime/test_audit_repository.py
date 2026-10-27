from pathlib import Path

from nestquant.cli.audit_repository import audit_paths


def _audit(tmp_path: Path, relative: str, content: str):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return audit_paths(tmp_path, [path], max_bytes=1024)


def test_audit_accepts_portable_placeholders(tmp_path: Path) -> None:
    findings = _audit(
        tmp_path,
        "docs/example.md",
        "cd /path/to/NestQuant-AIInfra\napi_key = '<your_api_key>'\n",
    )
    assert findings == []


def test_audit_rejects_machine_specific_absolute_path(tmp_path: Path) -> None:
    sample = 'ROOT="/home/alice/NestQuant"\n'  # repository-audit: allow
    findings = _audit(tmp_path, "scripts/run.sh", sample)
    assert [(item.kind, item.line) for item in findings] == [("absolute_path", 1)]


def test_audit_reports_secret_without_exposing_value(tmp_path: Path) -> None:
    secret = "hf_this_must_not_appear_in_output"  # repository-audit: allow
    sample = f'hf_token = "{secret}"\n'  # repository-audit: allow
    findings = _audit(tmp_path, "configs/private.toml", sample)
    assert [item.kind for item in findings] == ["possible_secret"]
    assert secret not in findings[0].message


def test_audit_rejects_sensitive_filename_and_large_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("x" * 10, encoding="utf-8")
    findings = audit_paths(tmp_path, [path], max_bytes=4)
    assert {item.kind for item in findings} == {"sensitive_filename", "large_file"}


def test_audit_checks_local_markdown_links(tmp_path: Path) -> None:
    target = tmp_path / "results" / "ok.md"
    target.parent.mkdir(parents=True)
    target.write_text("ok\n", encoding="utf-8")
    findings = _audit(
        tmp_path,
        "docs/index.md",
        "[valid](../results/ok.md) [missing](../results/missing.md) "
        "[web](https://example.com/page)\n",
    )
    assert [(item.kind, item.line) for item in findings] == [("broken_local_link", 1)]


def test_audit_rejects_markdown_link_outside_repository(tmp_path: Path) -> None:
    findings = _audit(tmp_path, "docs/index.md", "[outside](../../outside.md)\n")
    assert [item.kind for item in findings] == ["link_outside_repository"]
