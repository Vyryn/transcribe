from __future__ import annotations

import re
import subprocess
from pathlib import Path

from transcribe.network_guard import run_network_guard_self_test

_SCAN_EXTENSIONS = {".py", ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg", ".sh"}
_EXCLUDED_PREFIXES = ("tests/",)
_EXCLUDED_FILENAMES = {
    "README.md",
    "REGULATORY_CHECKLIST.md",
    "REGULATOR_COMPLIANCE_POLICY.md",
    "clinical note synthesis llm prompt.md",
    "uv.lock",
}
# Build without embedding literal URL tokens in source.
_URL_LITERAL_PATTERN = re.compile(r"(?:http|https)" + r"://")


def tracked_files(repo_root: Path) -> list[Path]:
    """Return tracked files from git, with filesystem fallback.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    list[Path]
        Candidate files to scan.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return [path for path in repo_root.rglob("*") if path.is_file()]

    files: list[Path] = []
    for raw_path in completed.stdout.splitlines():
        rel = raw_path.strip()
        if not rel:
            continue
        files.append(repo_root / rel)
    return files


def run_url_literal_check(repo_root: Path) -> list[tuple[Path, int, str]]:
    """Scan source files for URL literals.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    list[tuple[Path, int, str]]
        Violations as ``(file_path, line_number, line_text)`` tuples.
    """
    violations: list[tuple[Path, int, str]] = []
    for path in tracked_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        if rel in _EXCLUDED_FILENAMES:
            continue
        if rel.startswith(_EXCLUDED_PREFIXES):
            continue
        if path.suffix.lower() not in _SCAN_EXTENSIONS:
            continue

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(lines, start=1):
            if _URL_LITERAL_PATTERN.search(line):
                violations.append((path, line_number, line.strip()))
    return violations


def enforce_no_url_literals(repo_root: Path) -> int:
    """Run URL-literal check and print CLI-friendly output.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.

    Returns
    -------
    int
        Exit code where ``0`` means pass.
    """
    violations = run_url_literal_check(repo_root)
    if not violations:
        print("PASS: no URL literals found in runtime source files")
        return 0

    print("FAIL: URL literals found in runtime source files:")
    for path, line_number, line in violations:
        print(f"- {path.relative_to(repo_root)}:{line_number}: {line}")
    return 1


def run_network_compliance_check() -> int:
    """Run outbound-network compliance checks and print results.

    Returns
    -------
    int
        Exit code where ``0`` means pass.
    """
    results = run_network_guard_self_test()
    if results["outbound_blocked"] and results["loopback_allowed"]:
        print("PASS: outbound network blocked and loopback allowed")
        return 0

    print("FAIL: network guard self-test did not pass")
    print(results)
    return 1
