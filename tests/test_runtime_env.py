from __future__ import annotations

from pathlib import Path

import pytest

import transcribe.runtime_env as runtime_env
from transcribe.runtime_defaults import DEFAULT_LIVE_TRANSCRIPTION_MODEL, DEFAULT_SESSION_NOTES_MODEL


def test_detect_runtime_mode_defaults_to_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: False)

    assert runtime_env.detect_runtime_mode() == runtime_env.RuntimeMode.DEVELOPMENT


def test_detect_runtime_mode_uses_frozen_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)

    assert runtime_env.detect_runtime_mode() == runtime_env.RuntimeMode.PACKAGED


def test_bundled_transcription_model_specs_include_granite() -> None:
    specs = runtime_env.bundled_transcription_model_specs()

    assert runtime_env.PACKAGED_GRANITE_TRANSCRIPTION_MODEL in {spec.model_id for spec in specs}


def test_resolve_app_runtime_paths_uses_data_root_models_for_packaged_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    data_root = tmp_path / "data"
    app_root.mkdir()
    data_root.mkdir()

    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(data_root))

    paths = runtime_env.resolve_app_runtime_paths()

    assert paths.mode == runtime_env.RuntimeMode.PACKAGED
    assert paths.install_root == app_root.resolve()
    assert paths.data_root == data_root.resolve()
    assert paths.models_root == data_root.resolve() / "models"
    assert paths.notes_runtime_binary.parent == app_root.resolve() / "runtime/llm"
    assert paths.notes_prompt_path == app_root.resolve() / "clinical_note_synthesis_llm_prompt.md"
    assert paths.notes_models[DEFAULT_SESSION_NOTES_MODEL] == data_root.resolve() / "models/notes/qwen3.5-4b-q4_k_m.gguf"
    assert paths.transcription_models[DEFAULT_LIVE_TRANSCRIPTION_MODEL] == (
        data_root.resolve() / "models/asr/nvidia/parakeet-tdt-0.6b-v3"
    )
    assert paths.transcription_models[runtime_env.PACKAGED_GRANITE_TRANSCRIPTION_MODEL] == (
        data_root.resolve() / "models/asr/ibm-granite/granite-4.0-1b-speech"
    )


def test_resolve_app_runtime_paths_uses_repo_models_for_development(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: False)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(repo_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(tmp_path / "data"))

    paths = runtime_env.resolve_app_runtime_paths()

    assert paths.mode == runtime_env.RuntimeMode.DEVELOPMENT
    assert paths.models_root == repo_root.resolve() / "models"


def test_validate_transcription_model_for_packaged_runtime_rejects_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    app_root.mkdir()
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(tmp_path / "data"))

    with pytest.raises(ValueError, match="supports only packaged ASR models"):
        runtime_env.validate_transcription_model_for_runtime("faster-whisper-medium")


def test_validate_transcription_model_for_packaged_runtime_requires_installed_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    app_root.mkdir()
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(tmp_path / "data"))

    with pytest.raises(ValueError, match="Run `transcribe models install --model"):
        runtime_env.validate_transcription_model_for_runtime(DEFAULT_LIVE_TRANSCRIPTION_MODEL)


def test_validate_transcription_model_for_packaged_runtime_allows_installed_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    data_root = tmp_path / "data"
    app_root.mkdir()
    data_root.mkdir()
    (data_root / "models/asr/nvidia/parakeet-tdt-0.6b-v3").mkdir(parents=True)

    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(data_root))

    assert runtime_env.validate_transcription_model_for_runtime(DEFAULT_LIVE_TRANSCRIPTION_MODEL) == (
        DEFAULT_LIVE_TRANSCRIPTION_MODEL
    )


def test_resolve_bundled_notes_model_path_uses_packaged_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app_root = tmp_path / "app"
    data_root = tmp_path / "data"
    app_root.mkdir()
    data_root.mkdir()
    note_path = data_root / "models/notes/qwen3.5-4b-q4_k_m.gguf"
    note_path.parent.mkdir(parents=True)
    note_path.write_bytes(b"x")

    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(data_root))

    model_path = runtime_env.resolve_bundled_notes_model_path(DEFAULT_SESSION_NOTES_MODEL)

    assert model_path == note_path.resolve()


def test_network_access_allowed_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime_env.ALLOW_NETWORK_ENV, raising=False)

    assert runtime_env.network_access_allowed() is False



def test_set_network_access_allowed_updates_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime_env.ALLOW_NETWORK_ENV, raising=False)

    runtime_env.set_network_access_allowed(True)
    assert runtime_env.network_access_allowed() is True

    runtime_env.set_network_access_allowed(False)
    assert runtime_env.network_access_allowed() is False
