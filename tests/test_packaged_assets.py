from __future__ import annotations

from pathlib import Path

import pytest

import transcribe.packaged_assets as packaged_assets
from transcribe.packaged_assets import (
    PACKAGED_ASSET_SCHEMA_VERSION,
    PackagedAssetsManifest,
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
        ),
    )
    repo_files = {
        ("repo/notes-4b", "rev-notes-4b", "Qwen3.5-4B-Q4_K_M.gguf"): notes_4b,
        ("repo/notes-2b", "rev-notes-2b", "Qwen3.5-2B-Q4_K_M.gguf"): notes_2b,
        ("nvidia/parakeet-tdt-0.6b-v3", "rev-parakeet", "parakeet-tdt-0.6b-v3.nemo"): parakeet_dir / "parakeet-tdt-0.6b-v3.nemo",
        ("nvidia/canary-qwen-2.5b", "rev-canary", "config.json"): canary_dir / "config.json",
        ("nvidia/canary-qwen-2.5b", "rev-canary", "LICENSES"): canary_dir / "LICENSES",
        ("nvidia/canary-qwen-2.5b", "rev-canary", "model.safetensors"): canary_dir / "model.safetensors",
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


