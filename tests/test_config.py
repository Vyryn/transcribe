from __future__ import annotations

from pathlib import Path

import pytest

from transcribe.config import load_app_config


def test_load_app_config_defaults(tmp_path: Path) -> None:
    config = load_app_config(config_path=tmp_path / "missing.toml")
    assert config.offline_only is True
    assert config.network_block_mode == "deny_outbound"


def test_load_app_config_blocks_offline_disable_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIBE_OFFLINE_ONLY", "false")
    with pytest.raises(ValueError, match="offline_only"):
        load_app_config(config_path=Path("does-not-exist.toml"))


def test_load_app_config_supports_overrides(tmp_path: Path) -> None:
    config_file = tmp_path / "transcribe.toml"
    config_file.write_text(
        """
[transcribe]
log_level = "DEBUG"
redact_logs = false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_app_config(config_path=config_file, overrides={"redact_logs": True})
    assert config.log_level == "DEBUG"
    assert config.redact_logs is True
