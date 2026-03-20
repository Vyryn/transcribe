from __future__ import annotations

import sys
from pathlib import Path
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


def test_run_models_install_handles_missing_console_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transcribe.packaged_assets as packaged_assets_module

    target_path = tmp_path / "models" / "notes" / "model.gguf"

    monkeypatch.setattr(
        cli_module,
        "resolve_app_runtime_paths",
        lambda: SimpleNamespace(
            models_root=tmp_path / "models",
            installed_assets_state_path=tmp_path / "installed-assets.json",
        ),
    )

    def fake_install(
        manifest,
        *,
        models_root: Path,
        installed_state_path: Path,
        model_ids: list[str] | None = None,
        default_only: bool = False,
        progress_callback=None,
        hf_cache_dir: Path | None = None,
    ):
        _ = (manifest, models_root, installed_state_path, model_ids, default_only, hf_cache_dir)
        asset = SimpleNamespace(model_id="qwen3.5:4b-q4_K_M")
        if progress_callback is not None:
            progress_callback("installing", asset, target_path)
            progress_callback("installed", asset, target_path)
        return (SimpleNamespace(model_id=asset.model_id, skipped=False),)

    monkeypatch.setattr(packaged_assets_module, "install_packaged_model_assets", fake_install)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stderr__", None)

    exit_code = cli_module.run_models(
        SimpleNamespace(
            models_command="install",
            default=False,
            model_ids=["qwen3.5:4b-q4_K_M"],
            quiet=False,
        )
    )

    assert exit_code == 0


def test_run_models_list_uses_default_manifest_without_file_in_development(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[tuple[object, bool]] = []

    monkeypatch.setattr(
        cli_module,
        "resolve_app_runtime_paths",
        lambda: SimpleNamespace(
            models_root=tmp_path / "models",
            installed_assets_state_path=tmp_path / "installed-assets.json",
        ),
    )
    monkeypatch.setattr(cli_module, "write_console_line", lambda message, error=False: captured.append((message, error)))

    exit_code = cli_module.run_models(SimpleNamespace(models_command="list"))

    assert exit_code == 0
    assert any("ibm-granite/granite-4.0-1b-speech" in str(message) for message, _ in captured)
    assert all(error is False for _, error in captured)
