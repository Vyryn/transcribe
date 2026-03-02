from __future__ import annotations

from pathlib import Path

from transcribe.compliance import run_url_literal_check


def test_url_literal_check_detects_violation(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text('ENDPOINT = "https://example.org"\n', encoding="utf-8")

    violations = run_url_literal_check(tmp_path)
    assert len(violations) == 1
    assert violations[0][0] == source


def test_url_literal_check_ignores_safe_source(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text('value = "offline_only"\n', encoding="utf-8")

    violations = run_url_literal_check(tmp_path)
    assert violations == []
