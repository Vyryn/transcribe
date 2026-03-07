from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from transcribe.packaged_assets import load_packaged_asset_manifest
from transcribe.runtime_defaults import DEFAULT_LIVE_TRANSCRIPTION_MODEL, DEFAULT_SESSION_NOTES_MODEL

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "build_windows_standalone.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_windows_standalone", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_script = _load_module()


def test_build_parser_uses_bootstrap_defaults() -> None:
    args = build_script.build_parser().parse_args([])

    assert args.bootstrap_missing is True
    assert args.notes_model_4b is None
    assert args.notes_model_2b is None
    assert args.notes_model_4b_repo == build_script.DEFAULT_NOTES_MODEL_4B_REPO
    assert args.notes_model_2b_file == build_script.DEFAULT_NOTES_MODEL_2B_FILE
    assert args.notes_model_4b_revision == build_script.DEFAULT_NOTES_MODEL_4B_REVISION
    assert args.parakeet_model_repo == build_script.DEFAULT_PARAKEET_MODEL_REPO


def test_select_llama_cpp_windows_asset_prefers_cpu_x64_archive() -> None:
    asset = build_script._select_llama_cpp_windows_asset(
        [
            {
                "name": "llama-b6123-bin-win-avx2-x64.zip",
                "browser_download_url": "https://example.invalid/avx2.zip",
            },
            {
                "name": "llama-b7472-bin-win-cpu-x64.zip",
                "browser_download_url": "https://example.invalid/cpu.zip",
            },
        ]
    )

    assert asset.name == "llama-b7472-bin-win-cpu-x64.zip"
    assert asset.download_url == "https://example.invalid/cpu.zip"


def test_select_inno_setup_asset_prefers_installer_exe() -> None:
    asset = build_script._select_inno_setup_asset(
        [
            {
                "name": "innosetup-6.5.0-unicode.exe.sig",
                "browser_download_url": "https://example.invalid/installer.sig",
            },
            {
                "name": "innosetup-6.5.0.exe",
                "browser_download_url": "https://example.invalid/installer.exe",
            },
        ]
    )

    assert asset.name == "innosetup-6.5.0.exe"
    assert asset.download_url == "https://example.invalid/installer.exe"


def test_stage_runtime_assets_copies_prompt_runtime_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt body\n", encoding="utf-8")
    monkeypatch.setattr(build_script, "PROMPT_SOURCE_PATH", prompt_path)

    llama_runtime_dir = tmp_path / "llama-runtime"
    llama_runtime_dir.mkdir()
    (llama_runtime_dir / "llama-server.exe").write_bytes(b"exe")
    (llama_runtime_dir / "ggml-base.dll").write_bytes(b"dll")
    (llama_runtime_dir / "llama-cli.exe").write_bytes(b"skip")

    notes_model_4b = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    notes_model_2b = tmp_path / "Qwen3.5-2B-Q4_K_M.gguf"
    notes_model_4b.write_bytes(b"4b")
    notes_model_2b.write_bytes(b"2b")

    parakeet_model_dir = tmp_path / "parakeet"
    canary_model_dir = tmp_path / "canary"
    parakeet_model_dir.mkdir()
    canary_model_dir.mkdir()
    (parakeet_model_dir / "parakeet-tdt-0.6b-v3.nemo").write_bytes(b"nemo")
    (parakeet_model_dir / "README.md").write_text("skip", encoding="utf-8")
    (canary_model_dir / "config.json").write_text("{}", encoding="utf-8")
    (canary_model_dir / "LICENSES").write_text("license", encoding="utf-8")
    (canary_model_dir / "model.safetensors").write_bytes(b"canary")
    (canary_model_dir / "README.md").write_text("skip", encoding="utf-8")

    stage_dir = tmp_path / "stage"
    build_script._stage_runtime_assets(
        stage_dir=stage_dir,
        llama_runtime_dir=llama_runtime_dir,
        notes_model_4b=notes_model_4b,
        notes_model_4b_repo=build_script.DEFAULT_NOTES_MODEL_4B_REPO,
        notes_model_4b_revision=build_script.DEFAULT_NOTES_MODEL_4B_REVISION,
        notes_model_4b_file=build_script.DEFAULT_NOTES_MODEL_4B_FILE,
        notes_model_2b=notes_model_2b,
        notes_model_2b_repo=build_script.DEFAULT_NOTES_MODEL_2B_REPO,
        notes_model_2b_revision=build_script.DEFAULT_NOTES_MODEL_2B_REVISION,
        notes_model_2b_file=build_script.DEFAULT_NOTES_MODEL_2B_FILE,
        parakeet_model_dir=parakeet_model_dir,
        parakeet_model_repo=build_script.DEFAULT_PARAKEET_MODEL_REPO,
        parakeet_model_revision=build_script.DEFAULT_PARAKEET_MODEL_REVISION,
        canary_model_dir=canary_model_dir,
        canary_model_repo=build_script.DEFAULT_CANARY_MODEL_REPO,
        canary_model_revision=build_script.DEFAULT_CANARY_MODEL_REVISION,
    )

    manifest = load_packaged_asset_manifest(stage_dir / build_script.PACKAGED_ASSET_MANIFEST_FILENAME)

    assert (stage_dir / "prompts" / "clinical_note_synthesis_llm_prompt.md").read_text(encoding="utf-8") == "prompt body\n"
    assert (stage_dir / "runtime" / "llm" / "llama-server.exe").read_bytes() == b"exe"
    assert (stage_dir / "runtime" / "llm" / "ggml-base.dll").read_bytes() == b"dll"
    assert not (stage_dir / "runtime" / "llm" / "llama-cli.exe").exists()
    assert not (stage_dir / "models").exists()
    assert [asset.model_id for asset in manifest.assets] == [
        DEFAULT_SESSION_NOTES_MODEL,
        "qwen3.5:2b-q4_K_M",
        DEFAULT_LIVE_TRANSCRIPTION_MODEL,
        "nvidia/canary-qwen-2.5b",
    ]
    assert manifest.assets[0].default_install is True
    assert manifest.assets[2].required_files[0].path == "parakeet-tdt-0.6b-v3.nemo"


def test_build_pyinstaller_command_targets_packaged_entrypoint_and_excludes_dev_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_script,
        "_module_available",
        lambda module_name: module_name == "nemo.collections.asr",
    )

    command = build_script._build_pyinstaller_command(
        pyinstaller_command=("python", "-m", "PyInstaller"),
        stage_dir=tmp_path / "dist" / "transcribe",
        build_dir=tmp_path / "build",
    )

    assert str(build_script.REPO_ROOT / "packaged_main.py") == command[-1]
    assert "huggingface_hub" in command
    assert "transcribe.packaged_assets" in command
    assert "datasets" in command
    assert "transcribe.bench" in command
    assert "nemo.collections.asr" in command
    assert "--collect-all" in command
    assert "sounddevice" in command
    assert "nemo" not in command[command.index("--collect-all") + 1 :]


def test_ensure_inno_setup_bootstraps_local_install_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tools_dir = tmp_path / "tools"
    installer_path = tools_dir / "downloads" / "innosetup.exe"
    installer_path.parent.mkdir(parents=True, exist_ok=True)
    installer_path.write_bytes(b"installer")
    observed: dict[str, list[str]] = {}

    monkeypatch.setattr(build_script, "_resolve_executable", lambda candidate: None)
    monkeypatch.setattr(build_script, "_resolve_known_executable", lambda paths: None)
    monkeypatch.setattr(
        build_script,
        "_download_inno_setup_installer",
        lambda *, tools_dir, release: installer_path,
    )

    def fake_run(command, **kwargs):
        del kwargs
        observed["command"] = list(command)
        iscc_path = tools_dir / "inno-setup" / "ISCC.exe"
        iscc_path.parent.mkdir(parents=True, exist_ok=True)
        iscc_path.write_bytes(b"iscc")

    monkeypatch.setattr(build_script, "_run", fake_run)

    resolved = build_script._ensure_inno_setup(
        inno_setup_exe="iscc",
        bootstrap_missing=True,
        tools_dir=tools_dir,
    )

    assert Path(resolved) == (tools_dir / "inno-setup" / "ISCC.exe").resolve()
    assert observed["command"][0] == str(installer_path)
    assert "/VERYSILENT" in observed["command"]


def test_ensure_windows_native_build_environment_captures_vcvars_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    vcvars_path = tmp_path / "vs-buildtools" / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    vcvars_path.parent.mkdir(parents=True, exist_ok=True)
    vcvars_path.write_text("", encoding="utf-8")
    scripts_dir = tmp_path / "venv" / "Scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    observed: dict[str, Path] = {}

    monkeypatch.setattr(build_script, "_python_scripts_dir", lambda: scripts_dir)
    monkeypatch.setattr(
        build_script,
        "_ensure_windows_python_build_helpers",
        lambda **kwargs: {"PATH": "C:\\toolchain"},
    )
    generator_root = tmp_path / "generator-root"
    monkeypatch.setattr(build_script, "_find_vcvars64_bat", lambda **kwargs: vcvars_path)
    monkeypatch.setattr(
        build_script,
        "_visual_studio_build_tools_install_root",
        lambda *, tools_dir: generator_root,
    )

    def fake_capture_batch_environment(batch_path: Path, *, env=None, args=()):
        del env, args
        observed["batch_path"] = batch_path
        return {"PATH": "C:\\captured"}

    def fake_resolve_executable(candidate: str, env: dict[str, str]) -> str | None:
        del env
        if candidate == "cmake":
            return "C:\\captured\\cmake.exe"
        if candidate == "ninja":
            return "C:\\captured\\ninja.exe"
        if candidate == "cl":
            return "C:\\captured\\cl.exe"
        return None

    monkeypatch.setattr(build_script, "_capture_batch_environment", fake_capture_batch_environment)
    monkeypatch.setattr(build_script, "_resolve_executable_in_environment", fake_resolve_executable)

    env = build_script._ensure_windows_native_build_environment(
        bootstrap_missing=True,
        bootstrap_dir=tmp_path / "bootstrap",
    )

    assert observed["batch_path"] == vcvars_path
    assert env["CMAKE_GENERATOR"] == "Visual Studio 17 2022"
    assert env["CMAKE_GENERATOR_INSTANCE"] == str(generator_root)
    assert str(scripts_dir) in env["PATH"]


def test_module_available_returns_false_when_parent_package_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_script.importlib.util, "find_spec", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))

    assert build_script._module_available("nemo.collections.asr") is False
