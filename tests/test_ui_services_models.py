from __future__ import annotations

from pathlib import Path

import pytest

import transcribe.runtime_env as runtime_env
import transcribe.ui.services as services_module
from transcribe.ui.types import ModelsInstallRequest, UiCommonOptions


def test_list_models_uses_shared_default_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: False)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(repo_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(tmp_path / "data"))

    result = services_module.list_models()

    assert result.error is None
    assert result.manifest_path is None
    assert [item.model_id for item in result.items] == [
        "qwen3.5:4b-q4_K_M",
        "qwen3.5:2b-q4_K_M",
        "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/canary-qwen-2.5b",
        "ibm-granite/granite-4.0-1b-speech",
    ]


def test_install_models_uses_shared_default_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transcribe.packaged_assets as packaged_assets_module

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_path = repo_root / "models" / "notes" / "qwen3.5-4b-q4_k_m.gguf"
    observed: dict[str, object] = {}

    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: False)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(repo_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(services_module, "ensure_network_downloads_available", lambda task_name, common: None)

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
        _ = (installed_state_path, progress_callback, hf_cache_dir)
        observed["model_ids"] = model_ids
        observed["default_only"] = default_only
        observed["models_root"] = models_root
        observed["manifest_model_ids"] = [asset.model_id for asset in manifest.assets]
        return (
            packaged_assets_module.PackagedAssetInstallResult(
                model_id="qwen3.5:4b-q4_K_M",
                target_path=target_path,
                skipped=False,
            ),
        )

    monkeypatch.setattr(packaged_assets_module, "install_packaged_model_assets", fake_install)

    result = services_module.install_models(
        ModelsInstallRequest(
            common=UiCommonOptions(allow_network=True),
            model_ids=("qwen3.5:4b-q4_K_M",),
        )
    )

    assert result.installed_model_ids == ("qwen3.5:4b-q4_K_M",)
    assert observed["model_ids"] == ["qwen3.5:4b-q4_K_M"]
    assert observed["default_only"] is False
    assert observed["models_root"] == repo_root.resolve() / "models"
    assert "ibm-granite/granite-4.0-1b-speech" in observed["manifest_model_ids"]


def test_install_models_survives_missing_console_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import sys
    import transcribe.packaged_assets as packaged_assets_module

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_path = repo_root / "models" / "notes" / "qwen3.5-4b-q4_k_m.gguf"
    observed_progress: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(runtime_env, "is_frozen_app", lambda: True)
    monkeypatch.setenv(runtime_env.APP_ROOT_ENV, str(repo_root))
    monkeypatch.setenv(runtime_env.DATA_ROOT_ENV, str(tmp_path / "data"))
    monkeypatch.setattr(services_module, "ensure_network_downloads_available", lambda task_name, common: None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stderr__", None)

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
        assert sys.stdout is None
        assert sys.stderr is None
        if progress_callback is not None:
            progress_callback(
                "installing",
                packaged_assets_module.PackagedModelAsset(
                    model_id="qwen3.5:4b-q4_K_M",
                    kind="notes",
                    relative_path="notes/qwen3.5-4b-q4_k_m.gguf",
                    source_type="huggingface_file",
                    repo_id="repo/notes-4b",
                    revision="rev-notes-4b",
                    filename="Qwen3.5-4B-Q4_K_M.gguf",
                    required_files=(),
                    sha256="0" * 64,
                    size_bytes=0,
                    default_install=True,
                ),
                target_path,
            )
        return (
            packaged_assets_module.PackagedAssetInstallResult(
                model_id="qwen3.5:4b-q4_K_M",
                target_path=target_path,
                skipped=False,
            ),
        )

    monkeypatch.setattr(packaged_assets_module, "install_packaged_model_assets", fake_install)

    result = services_module.install_models(
        ModelsInstallRequest(
            common=UiCommonOptions(allow_network=True),
            model_ids=("qwen3.5:4b-q4_K_M",),
        ),
        progress_callback=lambda event, fields: observed_progress.append((event, fields)),
    )

    assert result.installed_model_ids == ("qwen3.5:4b-q4_K_M",)
    assert observed_progress == [
        (
            "models_installing",
            {
                "model_id": "qwen3.5:4b-q4_K_M",
                "kind": "notes",
                "target_path": str(target_path),
            },
        )
    ]
