from __future__ import annotations

import os
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from transcribe.models import AppConfig

DEFAULT_CONFIG_FILE = "transcribe.toml"
ENV_PREFIX = "TRANSCRIBE_"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def parse_bool(value: str) -> bool:
    """Parse a string into a boolean.

    Parameters
    ----------
    value : str
        Text value from environment variables or config overrides.

    Returns
    -------
    bool
        Parsed boolean value.
    """
    normalized = value.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def load_toml(path: Path) -> dict[str, Any]:
    """Load application configuration values from a TOML file.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed config values or an empty dictionary when the file is absent.
    """
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    section = raw.get("transcribe", raw)
    if not isinstance(section, dict):
        raise ValueError(f"Config file {path} must contain a dictionary-like structure")
    return dict(section)


def load_env() -> dict[str, Any]:
    """Load application configuration overrides from environment variables.

    Returns
    -------
    dict[str, Any]
        Parsed environment-backed configuration values.
    """
    values: dict[str, Any] = {}
    if f"{ENV_PREFIX}OFFLINE_ONLY" in os.environ:
        values["offline_only"] = parse_bool(os.environ[f"{ENV_PREFIX}OFFLINE_ONLY"])
    if f"{ENV_PREFIX}NETWORK_BLOCK_MODE" in os.environ:
        values["network_block_mode"] = os.environ[f"{ENV_PREFIX}NETWORK_BLOCK_MODE"]
    if f"{ENV_PREFIX}LOG_LEVEL" in os.environ:
        values["log_level"] = os.environ[f"{ENV_PREFIX}LOG_LEVEL"]
    if f"{ENV_PREFIX}REDACT_LOGS" in os.environ:
        values["redact_logs"] = parse_bool(os.environ[f"{ENV_PREFIX}REDACT_LOGS"])
    if f"{ENV_PREFIX}DATA_DIR" in os.environ:
        values["data_dir"] = Path(os.environ[f"{ENV_PREFIX}DATA_DIR"])
    return values


def load_app_config(
    *,
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Load and validate the application configuration.

    Parameters
    ----------
    config_path : Path | None, optional
        Optional TOML file path. Defaults to ``transcribe.toml``.
    overrides : Mapping[str, Any] | None, optional
        Highest-priority configuration overrides.

    Returns
    -------
    AppConfig
        Validated application configuration.

    Raises
    ------
    ValueError
        Raised when ``offline_only`` is disabled.
    """
    path = config_path or Path(DEFAULT_CONFIG_FILE)
    merged: dict[str, Any] = asdict(AppConfig())
    merged.update(load_toml(path))
    merged.update(load_env())
    if overrides:
        merged.update(dict(overrides))

    if "data_dir" in merged and not isinstance(merged["data_dir"], Path):
        merged["data_dir"] = Path(merged["data_dir"])

    config = AppConfig(**merged)
    if not config.offline_only:
        raise ValueError("Phase 0 requires offline_only=true. Disable attempts are blocked by design.")
    return config
