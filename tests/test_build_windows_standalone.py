from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_build_module() -> ModuleType:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "build_windows_standalone.py"
    spec = importlib.util.spec_from_file_location("build_windows_standalone", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sanitize_version_accepts_semver() -> None:
    module = _load_build_module()

    assert module.sanitize_version("1.2.3") == "1.2.3"
    assert module.sanitize_version("1.2.3-rc.1") == "1.2.3-rc.1"
    assert module.sanitize_version("1.2.3+build.5") == "1.2.3+build.5"


def test_sanitize_version_rejects_invalid_semver() -> None:
    module = _load_build_module()

    with pytest.raises(ValueError, match="semver"):
        module.sanitize_version("1.2")

    with pytest.raises(ValueError, match="semver"):
        module.sanitize_version("version-1.2.3")


def test_build_packaged_assets_manifest_defaults() -> None:
    module = _load_build_module()
    manifest = module.build_packaged_assets_manifest()
    defaults = {asset.model_id for asset in manifest.assets if asset.default_install}
    canary_asset = next(asset for asset in manifest.assets if asset.model_id == "nvidia/canary-qwen-2.5b")

    assert manifest.schema_version == "transcribe-packaged-assets-v1"
    assert defaults == {
        "nvidia/parakeet-tdt-0.6b-v3",
        "qwen3.5:4b-q4_K_M",
    }
    assert tuple(file_entry.path for file_entry in canary_asset.required_files) == (
        "config.json",
        "generation_config.json",
        "LICENSES",
        "model.safetensors",
        "tokenizer.model",
    )


def test_safe_filename_component_normalizes_build_metadata() -> None:
    module = _load_build_module()

    assert module.safe_filename_component("1.2.3+build.5") == "1.2.3-build.5"


def test_distribution_top_level_packages_uses_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_build_module()

    class FakeDistribution:
        files = (
            Path("faster_whisper/__init__.py"),
            Path("faster_whisper/transcribe.py"),
            Path("faster_whisper-1.2.1.dist-info/METADATA"),
        )

        def read_text(self, filename: str) -> str | None:
            assert filename == "top_level.txt"
            return None

    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name: FakeDistribution())
    monkeypatch.setattr(
        module,
        "find_module_spec",
        lambda name: object() if name == "faster_whisper" else None,
    )

    assert module.distribution_top_level_packages("faster-whisper") == ("faster_whisper",)


def test_run_command_handles_missing_text_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_build_module()

    def fake_run(*args: object, **kwargs: object) -> object:
        return module.subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout=None, stderr=None)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    completed = module.run_command(["cmd"])

    assert completed.returncode == 0


def test_run_command_uses_available_stream_text_for_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_build_module()

    def fake_run(*args: object, **kwargs: object) -> object:
        return module.subprocess.CompletedProcess(
            args=["cmd"],
            returncode=1,
            stdout=None,
            stderr="build failed",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="build failed"):
        module.run_command(["cmd"])


def test_require_inno_setup_returns_existing_compiler(tmp_path: Path) -> None:
    module = _load_build_module()
    iscc_path = tmp_path / "ISCC.exe"
    iscc_path.write_text("", encoding="utf-8")

    resolved = module.require_inno_setup(explicit_path=iscc_path, downloads_dir=tmp_path / "downloads")

    assert resolved == iscc_path.resolve()


def test_require_inno_setup_bootstraps_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _load_build_module()
    downloads_dir = tmp_path / "downloads"
    install_dir = tmp_path / "Inno Setup 6"
    installed_iscc = install_dir / "ISCC.exe"

    monkeypatch.setattr(module, "find_inno_setup_compiler", lambda explicit_path: None)
    monkeypatch.setattr(module, "inno_setup_install_dir", lambda: install_dir)

    def fake_download_file(*, url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("installer", encoding="utf-8")
        return destination

    def fake_run_command(command: list[object]) -> object:
        installed_iscc.parent.mkdir(parents=True, exist_ok=True)
        installed_iscc.write_text("", encoding="utf-8")
        return module.subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "download_file", fake_download_file)
    monkeypatch.setattr(module, "run_command", fake_run_command)

    resolved = module.require_inno_setup(explicit_path=None, downloads_dir=downloads_dir)

    assert resolved == installed_iscc.resolve()


def test_write_pyinstaller_spec_builds_windowed_launcher(tmp_path: Path) -> None:
    module = _load_build_module()
    spec_path = tmp_path / "transcribe.spec"

    module.write_pyinstaller_spec(
        destination=spec_path,
        package_roots=("transcribe",),
        distribution_names=("transcribe",),
        icon_path=tmp_path / "transcribe.ico",
    )

    spec_text = spec_path.read_text(encoding="utf-8")

    assert "console=False" in spec_text


def test_select_github_asset_falls_back_to_windows_x64_runtime_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_build_module()

    monkeypatch.setattr(
        module,
        "fetch_json",
        lambda url: {
            "assets": [
                {
                    "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.invalid/cudart-llama-bin-win-cuda-12.4-x64.zip",
                },
                {
                    "name": "llama-b8361-bin-macos-arm64.tar.gz",
                    "browser_download_url": "https://example.invalid/llama-b8361-bin-macos-arm64.tar.gz",
                },
            ]
        },
    )

    asset = module.select_github_asset(
        repo=module.DEFAULT_LLAMA_CPP_REPO,
        release=module.DEFAULT_LLAMA_CPP_RELEASE,
        patterns=module.LLAMA_RUNTIME_ARCHIVE_PATTERNS,
    )

    assert asset.name == "cudart-llama-bin-win-cuda-12.4-x64.zip"
