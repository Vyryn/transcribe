from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

import transcribe.packaged_assets as packaged_assets
from transcribe.packaged_assets import (
    PACKAGED_ASSET_SCHEMA_VERSION,
    PackagedAssetFile,
    PackagedAssetsManifest,
    PackagedModelAsset,
    build_directory_asset,
    build_single_file_asset,
    install_packaged_model_assets,
    load_installed_asset_state,
    load_packaged_asset_manifest,
    select_packaged_model_assets,
    verify_installed_asset,
    write_packaged_asset_manifest,
)


def _build_manifest(tmp_path: Path) -> tuple[PackagedAssetsManifest, dict[tuple[str, str, str], Path]]:
    source_root = tmp_path / "sources"
    source_root.mkdir()

    notes_4b = source_root / "Qwen3.5-4B-Q4_K_M.gguf"
    notes_4b.write_bytes(b"notes-4b")
    notes_2b = source_root / "Qwen3.5-2B-Q4_K_M.gguf"
    notes_2b.write_bytes(b"notes-2b")

    parakeet_dir = source_root / "parakeet"
    parakeet_dir.mkdir()
    (parakeet_dir / "parakeet-tdt-0.6b-v3.nemo").write_bytes(b"parakeet")

    canary_dir = source_root / "canary"
    canary_dir.mkdir()
    (canary_dir / "config.json").write_text("{}", encoding="utf-8")
    (canary_dir / "LICENSES").write_text("license", encoding="utf-8")
    (canary_dir / "model.safetensors").write_bytes(b"canary")

    granite_dir = source_root / "granite"
    granite_dir.mkdir()
    (granite_dir / "config.json").write_text("{}", encoding="utf-8")
    (granite_dir / "processor_config.json").write_text("{}", encoding="utf-8")
    (granite_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (granite_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (granite_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (granite_dir / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (granite_dir / "added_tokens.json").write_text("{}", encoding="utf-8")
    (granite_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (granite_dir / "merges.txt").write_text("merge", encoding="utf-8")
    (granite_dir / "chat_template.jinja").write_text("USER: {{ message }}", encoding="utf-8")
    (granite_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (granite_dir / "model-00001-of-00003.safetensors").write_bytes(b"granite-1")
    (granite_dir / "model-00002-of-00003.safetensors").write_bytes(b"granite-2")
    (granite_dir / "model-00003-of-00003.safetensors").write_bytes(b"granite-3")

    manifest = PackagedAssetsManifest(
        schema_version=PACKAGED_ASSET_SCHEMA_VERSION,
        assets=(
            build_single_file_asset(
                model_id="qwen3.5:4b-q4_K_M",
                kind="notes",
                relative_path="notes/qwen3.5-4b-q4_k_m.gguf",
                repo_id="repo/notes-4b",
                revision="rev-notes-4b",
                filename="Qwen3.5-4B-Q4_K_M.gguf",
                source_path=notes_4b,
                default_install=True,
            ),
            build_single_file_asset(
                model_id="qwen3.5:2b-q4_K_M",
                kind="notes",
                relative_path="notes/qwen3.5-2b-q4_k_m.gguf",
                repo_id="repo/notes-2b",
                revision="rev-notes-2b",
                filename="Qwen3.5-2B-Q4_K_M.gguf",
                source_path=notes_2b,
                default_install=False,
            ),
            build_directory_asset(
                model_id="nvidia/parakeet-tdt-0.6b-v3",
                kind="transcription",
                relative_path="asr/nvidia/parakeet-tdt-0.6b-v3",
                repo_id="nvidia/parakeet-tdt-0.6b-v3",
                revision="rev-parakeet",
                source_root=parakeet_dir,
                required_files=("parakeet-tdt-0.6b-v3.nemo",),
                default_install=True,
            ),
            build_directory_asset(
                model_id="nvidia/canary-qwen-2.5b",
                kind="transcription",
                relative_path="asr/nvidia/canary-qwen-2.5b",
                repo_id="nvidia/canary-qwen-2.5b",
                revision="rev-canary",
                source_root=canary_dir,
                required_files=("config.json", "LICENSES", "model.safetensors"),
                default_install=False,
            ),
            build_directory_asset(
                model_id="ibm-granite/granite-4.0-1b-speech",
                kind="transcription",
                relative_path="asr/ibm-granite/granite-4.0-1b-speech",
                repo_id="ibm-granite/granite-4.0-1b-speech",
                revision="rev-granite",
                source_root=granite_dir,
                required_files=(
                    "added_tokens.json",
                    "chat_template.jinja",
                    "config.json",
                    "merges.txt",
                    "model-00001-of-00003.safetensors",
                    "model-00002-of-00003.safetensors",
                    "model-00003-of-00003.safetensors",
                    "model.safetensors.index.json",
                    "preprocessor_config.json",
                    "processor_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
                default_install=False,
            ),
        ),
    )
    repo_files = {
        ("repo/notes-4b", "rev-notes-4b", "Qwen3.5-4B-Q4_K_M.gguf"): notes_4b,
        ("repo/notes-2b", "rev-notes-2b", "Qwen3.5-2B-Q4_K_M.gguf"): notes_2b,
        ("nvidia/parakeet-tdt-0.6b-v3", "rev-parakeet", "parakeet-tdt-0.6b-v3.nemo"): parakeet_dir / "parakeet-tdt-0.6b-v3.nemo",
        ("nvidia/canary-qwen-2.5b", "rev-canary", "config.json"): canary_dir / "config.json",
        ("nvidia/canary-qwen-2.5b", "rev-canary", "LICENSES"): canary_dir / "LICENSES",
        ("nvidia/canary-qwen-2.5b", "rev-canary", "model.safetensors"): canary_dir / "model.safetensors",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "added_tokens.json"): granite_dir / "added_tokens.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "chat_template.jinja"): granite_dir / "chat_template.jinja",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "config.json"): granite_dir / "config.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "merges.txt"): granite_dir / "merges.txt",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "model-00001-of-00003.safetensors"): granite_dir / "model-00001-of-00003.safetensors",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "model-00002-of-00003.safetensors"): granite_dir / "model-00002-of-00003.safetensors",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "model-00003-of-00003.safetensors"): granite_dir / "model-00003-of-00003.safetensors",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "model.safetensors.index.json"): granite_dir / "model.safetensors.index.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "preprocessor_config.json"): granite_dir / "preprocessor_config.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "processor_config.json"): granite_dir / "processor_config.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "special_tokens_map.json"): granite_dir / "special_tokens_map.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "tokenizer.json"): granite_dir / "tokenizer.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "tokenizer_config.json"): granite_dir / "tokenizer_config.json",
        ("ibm-granite/granite-4.0-1b-speech", "rev-granite", "vocab.json"): granite_dir / "vocab.json",
    }
    return manifest, repo_files


def test_manifest_round_trip_and_selection(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    manifest_path = tmp_path / "packaged-assets.json"

    write_packaged_asset_manifest(manifest, manifest_path)
    loaded = load_packaged_asset_manifest(manifest_path)

    assert [asset.model_id for asset in select_packaged_model_assets(loaded, default_only=True)] == [
        "qwen3.5:4b-q4_K_M",
        "nvidia/parakeet-tdt-0.6b-v3",
    ]
    assert [asset.model_id for asset in select_packaged_model_assets(loaded, model_ids=["ibm-granite/granite-4.0-1b-speech"])] == [
        "ibm-granite/granite-4.0-1b-speech"
    ]
    with pytest.raises(ValueError, match="unknown model ids"):
        select_packaged_model_assets(loaded, model_ids=["missing-model"])


def test_verify_installed_asset_detects_corruption(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    asset = manifest.assets[0]
    models_root = tmp_path / "models"
    target_path = models_root / asset.relative_path
    target_path.parent.mkdir(parents=True)
    target_path.write_bytes(b"notes-4b")

    assert verify_installed_asset(asset, models_root=models_root) is True

    target_path.write_bytes(b"corrupt")
    assert verify_installed_asset(asset, models_root=models_root) is False


def test_install_packaged_model_assets_installs_and_skips_existing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, repo_files = _build_manifest(tmp_path)
    models_root = tmp_path / "installed-models"
    state_path = tmp_path / "installed-assets.json"
    events: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, revision: str, filename: str, cache_dir: Path) -> Path:
        _ = cache_dir
        return repo_files[(repo_id, revision, filename)]

    monkeypatch.setattr(packaged_assets, "_hf_download_file", fake_download)

    first_results = install_packaged_model_assets(
        manifest,
        models_root=models_root,
        installed_state_path=state_path,
        default_only=True,
        progress_callback=lambda event, asset, target_path: events.append((event, asset.model_id)),
    )

    assert [result.model_id for result in first_results] == [
        "qwen3.5:4b-q4_K_M",
        "nvidia/parakeet-tdt-0.6b-v3",
    ]
    assert all(result.skipped is False for result in first_results)
    assert (models_root / "notes/qwen3.5-4b-q4_k_m.gguf").exists()
    assert (models_root / "asr/nvidia/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3.nemo").exists()
    assert load_installed_asset_state(state_path)["qwen3.5:4b-q4_K_M"]["revision"] == "rev-notes-4b"

    second_results = install_packaged_model_assets(
        manifest,
        models_root=models_root,
        installed_state_path=state_path,
        default_only=True,
    )

    assert all(result.skipped is True for result in second_results)
    assert ("installed", "qwen3.5:4b-q4_K_M") in events
    assert ("installed", "nvidia/parakeet-tdt-0.6b-v3") in events


def test_install_packaged_model_assets_raises_on_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, repo_files = _build_manifest(tmp_path)
    wrong_file = tmp_path / "wrong.gguf"
    wrong_file.write_bytes(b"badnotes")

    def fake_download(*, repo_id: str, revision: str, filename: str, cache_dir: Path) -> Path:
        _ = (cache_dir, revision)
        if repo_id == "repo/notes-4b":
            return wrong_file
        return repo_files[(repo_id, revision, filename)]

    monkeypatch.setattr(packaged_assets, "_hf_download_file", fake_download)

    with pytest.raises(RuntimeError, match="hash mismatch"):
        install_packaged_model_assets(
            manifest,
            models_root=tmp_path / "installed-models",
            installed_state_path=tmp_path / "installed-assets.json",
            model_ids=["qwen3.5:4b-q4_K_M"],
        )


def test_install_packaged_model_assets_raises_on_size_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, repo_files = _build_manifest(tmp_path)
    wrong_file = tmp_path / "partial.gguf"
    wrong_file.write_bytes(b"short")

    def fake_download(*, repo_id: str, revision: str, filename: str, cache_dir: Path) -> Path:
        _ = (cache_dir, revision)
        if repo_id == "repo/notes-4b":
            return wrong_file
        return repo_files[(repo_id, revision, filename)]

    monkeypatch.setattr(packaged_assets, "_hf_download_file", fake_download)

    with pytest.raises(RuntimeError, match="size mismatch"):
        install_packaged_model_assets(
            manifest,
            models_root=tmp_path / "installed-models",
            installed_state_path=tmp_path / "installed-assets.json",
            model_ids=["qwen3.5:4b-q4_K_M"],
        )


def test_install_packaged_model_assets_supports_remote_only_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_source = tmp_path / "remote"
    remote_source.mkdir()
    note_file = remote_source / "Qwen3.5-4B-Q4_K_M.gguf"
    note_file.write_bytes(b"notes-4b")
    parakeet_file = remote_source / "parakeet-tdt-0.6b-v3.nemo"
    parakeet_file.write_bytes(b"parakeet")

    manifest = PackagedAssetsManifest(
        schema_version=PACKAGED_ASSET_SCHEMA_VERSION,
        assets=(
            PackagedModelAsset(
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
            PackagedModelAsset(
                model_id="nvidia/parakeet-tdt-0.6b-v3",
                kind="transcription",
                relative_path="asr/nvidia/parakeet-tdt-0.6b-v3",
                source_type="huggingface_snapshot",
                repo_id="nvidia/parakeet-tdt-0.6b-v3",
                revision="rev-parakeet",
                filename=None,
                required_files=(
                    PackagedAssetFile(path="parakeet-tdt-0.6b-v3.nemo", sha256="0" * 64, size_bytes=0),
                ),
                sha256="0" * 64,
                size_bytes=0,
                default_install=True,
            ),
        ),
    )

    repo_files = {
        ("repo/notes-4b", "rev-notes-4b", "Qwen3.5-4B-Q4_K_M.gguf"): note_file,
        ("nvidia/parakeet-tdt-0.6b-v3", "rev-parakeet", "parakeet-tdt-0.6b-v3.nemo"): parakeet_file,
    }

    def fake_download(*, repo_id: str, revision: str, filename: str, cache_dir: Path) -> Path:
        _ = cache_dir
        return repo_files[(repo_id, revision, filename)]

    monkeypatch.setattr(packaged_assets, "_hf_download_file", fake_download)

    results = install_packaged_model_assets(
        manifest,
        models_root=tmp_path / "installed-models",
        installed_state_path=tmp_path / "installed-assets.json",
        default_only=True,
    )

    assert [result.model_id for result in results] == ["qwen3.5:4b-q4_K_M", "nvidia/parakeet-tdt-0.6b-v3"]
    assert verify_installed_asset(manifest.assets[0], models_root=tmp_path / "installed-models") is True
    assert verify_installed_asset(manifest.assets[1], models_root=tmp_path / "installed-models") is True


def test_hf_download_file_disables_progress_bars_for_packaged_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, str | None] = {}
    downloaded_path = tmp_path / "hf-cache" / "downloaded.bin"

    def fake_hf_hub_download(*, repo_id: str, filename: str, revision: str, local_files_only: bool, cache_dir: str) -> str:
        _ = (repo_id, filename, revision, local_files_only, cache_dir)
        observed["HF_HUB_DISABLE_PROGRESS_BARS"] = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.write_bytes(b"downloaded")
        return str(downloaded_path)

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    resolved = packaged_assets._hf_download_file(
        repo_id="repo/model",
        revision="rev-1",
        filename="model.bin",
        cache_dir=tmp_path / "cache",
    )

    assert observed["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert resolved == downloaded_path.resolve()


def test_hf_download_file_patches_missing_console_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    downloaded_path = tmp_path / "hf-cache" / "downloaded.bin"

    def fake_hf_hub_download(*, repo_id: str, filename: str, revision: str, local_files_only: bool, cache_dir: str) -> str:
        _ = (repo_id, filename, revision, local_files_only, cache_dir)
        observed["stdout_has_write"] = callable(getattr(sys.stdout, "write", None))
        observed["stderr_has_write"] = callable(getattr(sys.stderr, "write", None))
        assert sys.stdout is not None
        assert sys.stderr is not None
        print("download progress")
        sys.stderr.write("download warning\n")
        downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded_path.write_bytes(b"downloaded")
        return str(downloaded_path)

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.hf_hub_download = fake_hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "__stdout__", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "__stderr__", None)

    resolved = packaged_assets._hf_download_file(
        repo_id="repo/model",
        revision="rev-1",
        filename="model.bin",
        cache_dir=tmp_path / "cache",
    )

    assert observed == {
        "stdout_has_write": True,
        "stderr_has_write": True,
    }
    assert resolved == downloaded_path.resolve()
    assert sys.stdout is None
    assert sys.stderr is None
