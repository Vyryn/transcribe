from __future__ import annotations

from types import SimpleNamespace

import pytest

import transcribe.cli as cli_module
import transcribe.runtime_env as runtime_env


def test_load_and_configure_logging_skips_network_guard_when_network_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    args = SimpleNamespace(config=None, log_level=None, debug=False)

    monkeypatch.setenv(runtime_env.ALLOW_NETWORK_ENV, "1")
    monkeypatch.setattr(
        cli_module,
        "load_app_config",
        lambda **kwargs: SimpleNamespace(log_level="INFO", redact_logs=True, offline_only=True),
    )
    monkeypatch.setattr(
        cli_module,
        "configure_logging",
        lambda level, redact_logs: observed.setdefault("logging", (level, redact_logs)),
    )
    monkeypatch.setattr(
        cli_module,
        "install_outbound_network_guard",
        lambda: observed.setdefault("guard_installed", True),
    )

    cli_module.load_and_configure_logging(args)

    assert observed["logging"] == ("INFO", True)
    assert "guard_installed" not in observed


def test_load_and_configure_logging_installs_network_guard_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    args = SimpleNamespace(config=None, log_level=None, debug=False)

    monkeypatch.delenv(runtime_env.ALLOW_NETWORK_ENV, raising=False)
    monkeypatch.setattr(
        cli_module,
        "load_app_config",
        lambda **kwargs: SimpleNamespace(log_level="INFO", redact_logs=True, offline_only=True),
    )
    monkeypatch.setattr(cli_module, "configure_logging", lambda level, redact_logs: None)
    monkeypatch.setattr(
        cli_module,
        "install_outbound_network_guard",
        lambda: observed.setdefault("guard_installed", True),
    )

    cli_module.load_and_configure_logging(args)

    assert observed["guard_installed"] is True
