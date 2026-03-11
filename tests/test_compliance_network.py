from __future__ import annotations

import pytest

from transcribe.compliance import evaluate_network_compliance, run_network_compliance_check


def test_evaluate_network_compliance_includes_user_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.compliance as compliance_module

    monkeypatch.setattr(
        compliance_module,
        "run_network_guard_self_test",
        lambda: {"outbound_blocked": False, "loopback_allowed": True},
    )

    report = evaluate_network_compliance()

    assert report.passed is False
    assert report.summary == "Outbound network is currently allowed in this process."
    assert any("disable 'Allow Network Access' and restart" in detail for detail in report.details)


def test_run_network_compliance_check_fails_when_outbound_not_blocked(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import transcribe.compliance as compliance_module

    monkeypatch.setattr(
        compliance_module,
        "run_network_guard_self_test",
        lambda: {"outbound_blocked": False, "loopback_allowed": True},
    )

    exit_code = run_network_compliance_check()

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "Outbound network is currently allowed in this process." in output
    assert "disable 'Allow Network Access' and restart" in output
