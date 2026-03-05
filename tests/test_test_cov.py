from __future__ import annotations

import sys
import types

import transcribe.test_cov as test_cov


def test_main_invokes_pytest_with_coverage_defaults(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_pytest_main(args: list[str]) -> int:
        captured["args"] = args
        return 0

    fake_pytest_module = types.SimpleNamespace(main=fake_pytest_main)
    monkeypatch.setitem(sys.modules, "pytest", fake_pytest_module)

    rc = test_cov.main([])

    assert rc == 0
    assert captured["args"] == [
        "tests",
        "--cov=transcribe",
        "--cov-report=term-missing",
    ]


def test_main_forwards_user_args(monkeypatch) -> None:
    captured: dict[str, list[str]] = {}

    def fake_pytest_main(args: list[str]) -> int:
        captured["args"] = args
        return 0

    fake_pytest_module = types.SimpleNamespace(main=fake_pytest_main)
    monkeypatch.setitem(sys.modules, "pytest", fake_pytest_module)

    rc = test_cov.main(["-k", "live_session"])

    assert rc == 0
    assert captured["args"] == [
        "tests",
        "--cov=transcribe",
        "--cov-report=term-missing",
        "-k",
        "live_session",
    ]
