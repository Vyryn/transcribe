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
    granite_asset = next(asset for asset in manifest.assets if asset.model_id == "ibm-granite/granite-4.0-1b-speech")

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
    assert tuple(file_entry.path for file_entry in granite_asset.required_files) == (
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
    )


def test_project_runtime_distribution_names_include_hf_xet() -> None:
    module = _load_build_module()

    distribution_names = module.project_runtime_distribution_names(module.load_pyproject())

    assert "hf-xet" in distribution_names


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
    assert "exclude_binaries=True" in spec_text
    assert "COLLECT(" in spec_text


def test_stage_built_app_copies_onedir_bundle_contents(tmp_path: Path) -> None:
    module = _load_build_module()
    bundle_dir = tmp_path / "dist" / module.APP_NAME
    bundle_internal_dir = bundle_dir / "_internal"
    stage_dir = tmp_path / "stage"
    expected_executable = bundle_dir / f"{module.APP_NAME}.exe"
    expected_runtime_file = bundle_internal_dir / "python313.dll"

    bundle_internal_dir.mkdir(parents=True)
    expected_executable.write_text("exe", encoding="utf-8")
    expected_runtime_file.write_text("dll", encoding="utf-8")

    staged_executable = module.stage_built_app(bundle_dir, stage_dir)

    assert staged_executable == (stage_dir / f"{module.APP_NAME}.exe").resolve()
    assert (stage_dir / f"{module.APP_NAME}.exe").read_text(encoding="utf-8") == "exe"
    assert (stage_dir / "_internal" / "python313.dll").read_text(encoding="utf-8") == "dll"


def test_assemble_staged_app_overlays_llama_runtime_after_bundle_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_build_module()
    bundle_dir = tmp_path / "dist" / module.APP_NAME
    bundle_runtime_dir = bundle_dir / "runtime" / "llm"
    bundle_runtime_dir.mkdir(parents=True)
    (bundle_dir / f"{module.APP_NAME}.exe").write_text("exe", encoding="utf-8")
    (bundle_runtime_dir / "llama-server.exe").write_text("stale", encoding="utf-8")

    prompt_path = tmp_path / "clinical_note_synthesis_llm_prompt.md"
    prompt_path.write_text("prompt", encoding="utf-8")
    icon_path = tmp_path / "transcribe.ico"
    icon_path.write_text("icon", encoding="utf-8")

    layout = module.BuildLayout(
        root=tmp_path / "build",
        downloads_dir=tmp_path / "build" / "downloads",
        pyinstaller_work_dir=tmp_path / "build" / "work",
        pyinstaller_dist_dir=tmp_path / "build" / "dist",
        stage_dir=tmp_path / "stage",
        stage_prompts_dir=tmp_path / "stage" / "prompts",
        stage_runtime_dir=tmp_path / "stage" / "runtime" / "llm",
        installer_dir=tmp_path / "build" / "installer",
        spec_path=tmp_path / "build" / "transcribe.spec",
    )

    def fake_stage_llama_runtime(*, llama_runtime_dir, llama_cpp_release, downloads_dir, destination_dir) -> None:
        del llama_runtime_dir, llama_cpp_release, downloads_dir
        destination_dir.mkdir(parents=True, exist_ok=True)
        (destination_dir / "llama-server.exe").write_text("fresh", encoding="utf-8")

    monkeypatch.setattr(module, "PROMPT_SOURCE_PATH", prompt_path)
    monkeypatch.setattr(module, "stage_llama_runtime", fake_stage_llama_runtime)

    staged_executable = module.assemble_staged_app(
        bundle_dir=bundle_dir,
        layout=layout,
        args=module.argparse.Namespace(llama_runtime_dir=None, llama_cpp_release="latest"),
        icon_path=icon_path,
    )

    assert staged_executable == (layout.stage_dir / f"{module.APP_NAME}.exe").resolve()
    assert (layout.stage_runtime_dir / "llama-server.exe").read_text(encoding="utf-8") == "fresh"
    assert (layout.stage_dir / f"{module.APP_NAME}.exe").read_text(encoding="utf-8") == "exe"
    assert (layout.stage_prompts_dir / "clinical_note_synthesis_llm_prompt.md").read_text(encoding="utf-8") == "prompt"
    assert (layout.stage_dir / "transcribe.ico").read_text(encoding="utf-8") == "icon"


def test_verify_staged_llama_runtime_requires_binary(tmp_path: Path) -> None:
    module = _load_build_module()
    layout = module.BuildLayout(
        root=tmp_path / "build",
        downloads_dir=tmp_path / "build" / "downloads",
        pyinstaller_work_dir=tmp_path / "build" / "work",
        pyinstaller_dist_dir=tmp_path / "build" / "dist",
        stage_dir=tmp_path / "stage",
        stage_prompts_dir=tmp_path / "stage" / "prompts",
        stage_runtime_dir=tmp_path / "stage" / "runtime" / "llm",
        installer_dir=tmp_path / "build" / "installer",
        spec_path=tmp_path / "build" / "transcribe.spec",
    )

    with pytest.raises(RuntimeError, match="missing the bundled llama.cpp runtime"):
        module.verify_staged_llama_runtime(layout)


def test_installer_template_hides_shell_wrapper_and_shows_model_progress() -> None:
    template_path = Path(__file__).resolve().parent.parent / "packaging" / "windows" / "transcribe_installer.iss"
    template_text = template_path.read_text(encoding="utf-8")

    assert "ExpandConstant('{cmd}')" not in template_text
    assert 'set "TRANSCRIBE_ALLOW_NETWORK=1"' not in template_text
    assert "SetEnvironmentVariableW@kernel32.dll" in template_text
    assert "CreateOutputProgressPage" in template_text
    assert "Installing model " in template_text
    assert "ExpandConstant('{app}\\Transcribe.exe')" in template_text
    assert "Voice: IBM Granite 4.0 1B Speech" in template_text
    assert "ibm-granite/granite-4.0-1b-speech" in template_text
    assert "models install --quiet --model " in template_text
    assert "CheckBox.Height := ScaleY(28);" in template_text
    assert "PreviousCheckBox.Top + PreviousCheckBox.Height + ScaleY(8)" in template_text
    assert 'Source: "{#SourceDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs' in template_text


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
