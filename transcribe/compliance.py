from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from transcribe.network_guard import run_network_guard_self_test

_SCAN_EXTENSIONS = {".py", ".toml", ".json", ".yaml", ".yml", ".ini", ".cfg", ".sh"}
_EXCLUDED_PREFIXES = ("tests/",)
_EXCLUDED_PATHS = {
    "scripts/build_windows_standalone.py",
}
_EXCLUDED_FILENAMES = {
    "README.md",
    "REGULATORY_CHECKLIST.md",
    "REGULATOR_COMPLIANCE_POLICY.md",
    "clinical note synthesis llm prompt.md",
    "uv.lock",
}
# Build without embedding literal URL tokens in source.
_URL_LITERAL_PATTERN = re.compile(r"(?:http|https)" + r"://")

@dataclass(frozen=True, slots=True)
class ComplianceCheckReport:
    """Structured compliance-check result for CLI and UI consumers."""

    passed: bool
    exit_code: int
    summary: str
    details: tuple[str, ...] = ()



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
        if rel in _EXCLUDED_PATHS:
            continue
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


def evaluate_url_literal_compliance(repo_root: Path) -> ComplianceCheckReport:
    """Evaluate URL-literal compliance and return a structured report."""
    violations = run_url_literal_check(repo_root)
    if not violations:
        return ComplianceCheckReport(
            passed=True,
            exit_code=0,
            summary="No URL literals found in runtime source files.",
        )

    details = tuple(
        f"{path.relative_to(repo_root)}:{line_number}: {line}"
        for path, line_number, line in violations
    )
    return ComplianceCheckReport(
        passed=False,
        exit_code=1,
        summary=f"Found {len(violations)} URL literal violation(s) in runtime source files.",
        details=details,
    )


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
    report = evaluate_url_literal_compliance(repo_root)
    if report.passed:
        print(f"PASS: {report.summary}")
        return report.exit_code

    print(f"FAIL: {report.summary}")
    for detail in report.details:
        print(f"- {detail}")
    return report.exit_code


def evaluate_network_compliance() -> ComplianceCheckReport:
    """Evaluate outbound-network compliance for the current process."""
    results = run_network_guard_self_test()
    if results["outbound_blocked"] and results["loopback_allowed"]:
        return ComplianceCheckReport(
            passed=True,
            exit_code=0,
            summary="Outbound network is blocked and loopback is allowed.",
        )

    observed = (
        f"Observed outbound_blocked={results['outbound_blocked']}, "
        f"loopback_allowed={results['loopback_allowed']}."
    )
    if not results["outbound_blocked"]:
        return ComplianceCheckReport(
            passed=False,
            exit_code=1,
            summary="Outbound network is currently allowed in this process.",
            details=(
                observed,
                "If you enabled network in the UI, disable 'Allow Network Access' and restart before rerunning this check.",
            ),
        )

    return ComplianceCheckReport(
        passed=False,
        exit_code=1,
        summary="Loopback connections are unexpectedly blocked in this process.",
        details=(observed,),
    )


def run_network_compliance_check() -> int:
    """Run outbound-network compliance checks and print results.

    Returns
    -------
    int
        Exit code where ``0`` means pass.
    """
    report = evaluate_network_compliance()
    if report.passed:
        print(f"PASS: {report.summary}")
        return report.exit_code

    print(f"FAIL: {report.summary}")
    for detail in report.details:
        print(detail)
    return report.exit_code
