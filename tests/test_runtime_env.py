from __future__ import annotations

import json
from pathlib import Path

import pytest

import transcribe.runtime_env as runtime_env
from transcribe.runtime_defaults import DEFAULT_LIVE_TRANSCRIPTION_MODEL, DEFAULT_SESSION_NOTES_MODEL


def _resolved(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _write_manifest(app_root: Path) -> None:
    payload = {
        "schema_version": "transcribe-packaged-assets-v1",
        "assets": [
            {
                "model_id": DEFAULT_SESSION_NOTES_MODEL,
                "kind": "notes",
                "relative_path": "notes/qwen3.5-4b-q4_k_m.gguf",
                "source_type": "huggingface_file",
                "repo_id": "repo/notes-4b",
                "revision": "rev-notes-4b",
                "filename": "Qwen3.5-4B-Q4_K_M.gguf",
                "required_files": [],
                "sha256": "0" * 64,
                "size_bytes": 1,
                "default_install": True,
            },
            {
                "model_id": "qwen3.5:2b-q4_K_M",
                "kind": "notes",
                "relative_path": "notes/qwen3.5-2b-q4_k_m.gguf",
                "source_type": "huggingface_file",
                "repo_id": "repo/notes-2b",
                "revision": "rev-notes-2b",
                "filename": "Qwen3.5-2B-Q4_K_M.gguf",
                "required_files": [],
                "sha256": "1" * 64,
                "size_bytes": 1,
                "default_install": False,
            },
            {
                "model_id": DEFAULT_LIVE_TRANSCRIPTION_MODEL,
                "kind": "transcription",
                "relative_path": "asr/nvidia/parakeet-tdt-0.6b-v3",
                "source_type": "huggingface_snapshot",
                "repo_id": DEFAULT_LIVE_TRANSCRIPTION_MODEL,
                "revision": "rev-parakeet",
                "filename": None,
                "required_files": [
                    {
                        "path": "parakeet-tdt-0.6b-v3.nemo",
                        "sha256": "2" * 64,
                        "size_bytes": 1,
                    }
                ],
                "sha256": "3" * 64,
                "size_bytes": 1,
                "default_install": True,
            },
            {
                "model_id": "nvidia/canary-qwen-2.5b",
                "kind": "transcription",
                "relative_path": "asr/nvidia/canary-qwen-2.5b",
                "source_type": "huggingface_snapshot",
                "repo_id": "nvidia/canary-qwen-2.5b",
                "revision": "rev-canary",
                "filename": None,
                "required_files": [
                    {
                        "path": "config.json",
                        "sha256": "4" * 64,
                        "size_bytes": 1,
                    }
                ],
                "sha256": "5" * 64,
                "size_bytes": 1,
                "default_install": False,
            },
        ],
    }
    (app_root / runtime_env.PACKAGED_ASSET_MANIFEST_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def test_detect_runtime_mode_defaults_to_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(runtime_env.PACKAGED_RUNTIME_ENV, raising=False)
    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: False)

    assert runtime_env.detect_runtime_mode() == runtime_env.RuntimeMode.DEVELOPMENT


def test_detect_runtime_mode_honors_packaged_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")

    assert runtime_env.detect_runtime_mode() == runtime_env.RuntimeMode.PACKAGED


def test_resolve_app_runtime_paths_uses_data_root_models_for_packaged_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    data_root = tmp_path / "data"
    app_root.mkdir()
    data_root.mkdir()
    _write_manifest(app_root)

    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(data_root))

    paths = runtime_env.resolve_app_runtime_paths()

    assert paths.mode == runtime_env.RuntimeMode.PACKAGED
    assert paths.install_root == app_root.resolve()
    assert paths.data_root == data_root.resolve()
    assert paths.models_root == data_root.resolve() / "models"
    assert paths.notes_runtime_binary.parent == app_root.resolve() / "runtime/llm"
    assert paths.notes_prompt_path == app_root.resolve() / "prompts/clinical_note_synthesis_llm_prompt.md"
    assert paths.notes_models[DEFAULT_SESSION_NOTES_MODEL] == data_root.resolve() / "models/notes/qwen3.5-4b-q4_k_m.gguf"
    assert paths.transcription_models[DEFAULT_LIVE_TRANSCRIPTION_MODEL] == (
        data_root.resolve() / "models/asr/nvidia/parakeet-tdt-0.6b-v3"
    )


def test_validate_transcription_model_for_packaged_runtime_rejects_unsupported_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_root = tmp_path / "app"
    app_root.mkdir()
    _write_manifest(app_root)
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")
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
    _write_manifest(app_root)
    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")
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
    _write_manifest(app_root)
    (data_root / "models/asr/nvidia/parakeet-tdt-0.6b-v3").mkdir(parents=True)

    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")
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
    _write_manifest(app_root)
    note_path = data_root / "models/notes/qwen3.5-4b-q4_k_m.gguf"
    note_path.parent.mkdir(parents=True)
    note_path.write_bytes(b"x")

    monkeypatch.setenv(runtime_env.PACKAGED_RUNTIME_ENV, "1")
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(app_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(data_root))

    model_path = runtime_env.resolve_bundled_notes_model_path(DEFAULT_SESSION_NOTES_MODEL)

    assert model_path == note_path.resolve()
