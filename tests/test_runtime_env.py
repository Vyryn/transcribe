from __future__ import annotations

from pathlib import Path

import pytest

import transcribe.runtime_env as runtime_env
from transcribe.runtime_defaults import DEFAULT_LIVE_TRANSCRIPTION_MODEL, DEFAULT_SESSION_NOTES_MODEL


def _resolved(path: str) -> Path:
    return Path(path).expanduser().resolve()


def test_detect_runtime_mode_defaults_to_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime_env.PACKAGED_RUNTIME_ENV, raising=False)
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: False)

    assert runtime_env.detect_runtime_mode() == runtime_env.RuntimeMode.DEVELOPMENT


def test_detect_runtime_mode_honors_packaged_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")

    assert runtime_env.detect_runtime_mode() == runtime_env.RuntimeMode.PACKAGED


def test_resolve_app_runtime_paths_uses_local_app_data_for_packaged_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, "/tmp/transcribe-app")
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, "/tmp/localappdata/Transcribe")

    paths = runtime_env.resolve_app_runtime_paths()
    install_root = _resolved("/tmp/transcribe-app")
    data_root = _resolved("/tmp/localappdata/Transcribe")

    assert paths.mode == runtime_env.RuntimeMode.PACKAGED
    assert paths.install_root == install_root
    assert paths.data_root == data_root
    assert paths.notes_runtime_binary.parent == install_root / "runtime/llm"
    assert paths.notes_runtime_binary.name in {"llama-server", "llama-server.exe"}
    assert paths.notes_prompt_path == install_root / "prompts/clinical_note_synthesis_llm_prompt.md"


def test_validate_transcription_model_for_packaged_runtime_rejects_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")

    with pytest.raises(ValueError, match="supports only bundled ASR models"):
        runtime_env.validate_transcription_model_for_runtime("faster-whisper-medium")


def test_validate_transcription_model_for_packaged_runtime_allows_parakeet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")

    assert runtime_env.validate_transcription_model_for_runtime(DEFAULT_LIVE_TRANSCRIPTION_MODEL) == (
        DEFAULT_LIVE_TRANSCRIPTION_MODEL
    )


def test_resolve_bundled_notes_model_path_uses_known_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, "/tmp/transcribe-app")

    model_path = runtime_env.resolve_bundled_notes_model_path(DEFAULT_SESSION_NOTES_MODEL)

    assert model_path == _resolved("/tmp/transcribe-app") / "models/notes/qwen3.5-4b-q4_k_m.gguf"
