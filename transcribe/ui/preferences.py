from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from transcribe.runtime_env import resolve_app_runtime_paths

UI_PREFERENCES_FILENAME = "ui-preferences.json"


@dataclass(frozen=True, slots=True)
class UiPreferences:
    """Persisted UI-level preferences that should survive restarts."""

    advanced_ui: bool = False
    allow_network: bool = False


def preferences_path() -> Path:
    """Return the on-disk preferences path for the current runtime."""
    return resolve_app_runtime_paths().data_root / UI_PREFERENCES_FILENAME


def load_ui_preferences() -> UiPreferences:
    """Load persisted UI preferences, falling back to safe defaults."""
    path = preferences_path()
    if not path.is_file():
        return UiPreferences()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return UiPreferences()
    if not isinstance(payload, dict):
        return UiPreferences()
    allow_network = bool(payload.get("allow_network", False))
    advanced_ui = bool(payload.get("advanced_ui", False))
    if allow_network:
        advanced_ui = True
    return UiPreferences(
        advanced_ui=advanced_ui,
        allow_network=allow_network,
    )


def save_ui_preferences(preferences: UiPreferences) -> None:
    """Persist UI preferences to the runtime data directory."""
    path = preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "advanced_ui": preferences.advanced_ui,
        "allow_network": preferences.allow_network,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
