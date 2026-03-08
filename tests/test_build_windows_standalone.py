from __future__ import annotations

import importlib.util
import json
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


def _make_build_inputs(tmp_path: Path, *, backend: str, include_canary_model: bool) -> object:
    llama_runtime_dir = tmp_path / "llama-runtime"
    llama_runtime_dir.mkdir()
    (llama_runtime_dir / "llama-server.exe").write_bytes(b"exe")
    (llama_runtime_dir / "ggml-base.dll").write_bytes(b"dll")

    notes_model_4b = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    notes_model_2b = tmp_path / "Qwen3.5-2B-Q4_K_M.gguf"
    notes_model_4b.write_bytes(b"4b")
    notes_model_2b.write_bytes(b"2b")

    parakeet_model_dir = tmp_path / "parakeet"
    parakeet_model_dir.mkdir()
    (parakeet_model_dir / "parakeet-tdt-0.6b-v3.nemo").write_bytes(b"nemo")

    canary_model_dir = None
    if include_canary_model:
        canary_model_dir = tmp_path / "canary"
        canary_model_dir.mkdir()
        (canary_model_dir / "config.json").write_text("{}", encoding="utf-8")
        (canary_model_dir / "LICENSES").write_text("license", encoding="utf-8")
        (canary_model_dir / "model.safetensors").write_bytes(b"canary")

    return build_script.ResolvedBuildInputs(
        backend=backend,
        package_command=("python", "-m", backend),
        inno_setup_exe="iscc.exe",
        llama_runtime_dir=llama_runtime_dir,
        notes_model_4b=notes_model_4b,
        notes_model_2b=notes_model_2b,
        parakeet_model_dir=parakeet_model_dir,
        canary_model_dir=canary_model_dir,
    )


def test_build_parser_uses_bootstrap_defaults_and_new_flags() -> None:
    args = build_script.build_parser().parse_args([])

    assert args.backend == "nuitka"
    assert args.phase == "all"
    assert args.clean is False
    assert args.report_path == build_script.DEFAULT_BUILD_REPORT_PATH
    assert args.bootstrap_missing is True
    assert args.notes_model_4b is None
    assert args.notes_model_2b is None
    assert args.notes_model_4b_repo == build_script.DEFAULT_NOTES_MODEL_4B_REPO
    assert args.notes_model_2b_file == build_script.DEFAULT_NOTES_MODEL_2B_FILE
    assert args.notes_model_4b_revision == build_script.DEFAULT_NOTES_MODEL_4B_REVISION
    assert args.parakeet_model_repo == build_script.DEFAULT_PARAKEET_MODEL_REPO
    assert args.include_canary_model is False


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


@pytest.mark.parametrize("include_canary_model,expected_models", [
    (False, [DEFAULT_SESSION_NOTES_MODEL, "qwen3.5:2b-q4_K_M", DEFAULT_LIVE_TRANSCRIPTION_MODEL]),
    (True, [DEFAULT_SESSION_NOTES_MODEL, "qwen3.5:2b-q4_K_M", DEFAULT_LIVE_TRANSCRIPTION_MODEL, "nvidia/canary-qwen-2.5b"]),
])
def test_stage_runtime_assets_copies_prompt_runtime_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_canary_model: bool,
    expected_models: list[str],
) -> None:
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
    parakeet_model_dir.mkdir()
    (parakeet_model_dir / "parakeet-tdt-0.6b-v3.nemo").write_bytes(b"nemo")
    (parakeet_model_dir / "README.md").write_text("skip", encoding="utf-8")

    canary_model_dir = None
    if include_canary_model:
        canary_model_dir = tmp_path / "canary"
        canary_model_dir.mkdir()
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
    assert [asset.model_id for asset in manifest.assets] == expected_models
    assert manifest.assets[0].default_install is True
    assert manifest.assets[2].required_files[0].path == "parakeet-tdt-0.6b-v3.nemo"


@pytest.mark.parametrize("include_canary_model", [False, True])
def test_build_pyinstaller_command_targets_packaged_entrypoint_and_excludes_dev_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_canary_model: bool,
) -> None:
    monkeypatch.setattr(
        build_script,
        "_module_available",
        lambda module_name: module_name == "nemo.collections.asr",
    )

    command = build_script._build_pyinstaller_command(
        package_command=("python", "-m", "PyInstaller"),
        stage_dir=tmp_path / "dist" / "transcribe",
        build_dir=tmp_path / "build",
        clean=False,
        include_canary_model=include_canary_model,
    )

    assert str(build_script.REPO_ROOT / "packaged_main.py") == command[-1]
    assert "huggingface_hub" in command
    assert "transcribe.packaged_assets" in command
    assert "datasets" in command
    assert "transcribe.bench" in command
    assert "nemo.collections.asr" in command
    assert "--collect-all" in command
    assert "sounddevice" in command
    assert "--clean" not in command
    assert ("nemo.collections.speechlm2.models" in command) is include_canary_model


def test_build_pyinstaller_command_adds_clean_flag(tmp_path: Path) -> None:
    command = build_script._build_pyinstaller_command(
        package_command=("python", "-m", "PyInstaller"),
        stage_dir=tmp_path / "dist" / "transcribe",
        build_dir=tmp_path / "build",
        clean=True,
        include_canary_model=False,
    )

    assert "--clean" in command


@pytest.mark.parametrize("include_canary_model", [False, True])
def test_build_nuitka_command_targets_packaged_entrypoint_and_excludes_dev_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_canary_model: bool,
) -> None:
    monkeypatch.setattr(
        build_script,
        "_module_available",
        lambda module_name: module_name == "nemo.collections.asr",
    )

    command = build_script._build_nuitka_command(
        package_command=("python", "-m", "nuitka"),
        build_dir=tmp_path / "build",
        clean=False,
        include_canary_model=include_canary_model,
    )

    assert str(build_script.REPO_ROOT / "packaged_main.py") == command[-1]
    assert "--standalone" in command
    assert "--output-filename=transcribe.exe" in command
    assert "--include-package=transcribe" in command
    assert "--include-module=sounddevice" in command
    assert "--noinclude-numba-mode=nofollow" in command
    assert "--module-parameter=numba-disable-jit=yes" in command
    assert f"--user-package-configuration-file={build_script.NUITKA_PACKAGE_CONFIG_PATH}" in command
    assert any(item.startswith("--nofollow-import-to=") and "transcribe.bench" in item for item in command)
    assert "--include-module=nemo.collections.asr" in command
    assert ("--include-module=nemo.collections.speechlm2.models" in command) is include_canary_model


def test_resolve_or_bootstrap_build_inputs_keeps_nuitka_backend_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_inputs = _make_build_inputs(tmp_path, backend="nuitka", include_canary_model=False)
    monkeypatch.setattr(
        build_script,
        "_resolve_packaging_command",
        lambda *, backend, bootstrap_missing, bootstrap_dir: ("python", "-m", backend),
    )
    monkeypatch.setattr(build_script, "_ensure_inno_setup", lambda **kwargs: "iscc.exe")

    args = build_script.build_parser().parse_args(
        [
            "--backend",
            "nuitka",
            "--skip-installer",
            "--llama-runtime-dir",
            str(build_inputs.llama_runtime_dir),
            "--notes-model-4b",
            str(build_inputs.notes_model_4b),
            "--notes-model-2b",
            str(build_inputs.notes_model_2b),
            "--parakeet-model-dir",
            str(build_inputs.parakeet_model_dir),
        ]
    )

    resolved = build_script._resolve_or_bootstrap_build_inputs(args)

    assert resolved.backend == "nuitka"
    assert resolved.package_command == ("python", "-m", "nuitka")
    assert resolved.canary_model_dir is None


def test_resolve_or_bootstrap_build_inputs_includes_canary_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_inputs = _make_build_inputs(tmp_path, backend="nuitka", include_canary_model=True)
    monkeypatch.setattr(
        build_script,
        "_resolve_packaging_command",
        lambda *, backend, bootstrap_missing, bootstrap_dir: ("python", "-m", backend),
    )
    monkeypatch.setattr(build_script, "_ensure_inno_setup", lambda **kwargs: "iscc.exe")

    args = build_script.build_parser().parse_args(
        [
            "--backend",
            "nuitka",
            "--include-canary-model",
            "--skip-installer",
            "--llama-runtime-dir",
            str(build_inputs.llama_runtime_dir),
            "--notes-model-4b",
            str(build_inputs.notes_model_4b),
            "--notes-model-2b",
            str(build_inputs.notes_model_2b),
            "--parakeet-model-dir",
            str(build_inputs.parakeet_model_dir),
            "--canary-model-dir",
            str(build_inputs.canary_model_dir),
        ]
    )

    resolved = build_script._resolve_or_bootstrap_build_inputs(args)

    assert resolved.canary_model_dir == build_inputs.canary_model_dir


def test_validate_nuitka_python_runtime_allows_python_313(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_script.sys, "version_info", (3, 13, 2))

    build_script._validate_nuitka_python_runtime()


def test_validate_nuitka_python_runtime_rejects_unsupported_python(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_script.sys, "version_info", (3, 14, 3))

    with pytest.raises(RuntimeError, match="currently require Python 3.13"):
        build_script._validate_nuitka_python_runtime()


def test_resolve_packaging_command_allows_pyinstaller_under_unsupported_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_script.sys, "version_info", (3, 14, 3))
    monkeypatch.setattr(
        build_script,
        "_ensure_python_build_dependencies",
        lambda *, bootstrap_missing, bootstrap_dir: ("python", "-m", "PyInstaller"),
    )

    command = build_script._resolve_packaging_command(
        backend="pyinstaller",
        bootstrap_missing=False,
        bootstrap_dir=Path("bootstrap"),
    )

    assert command == ("python", "-m", "PyInstaller")


def test_write_build_report_includes_phase_durations_and_inventory(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stage"
    build_dir = tmp_path / "build"
    bootstrap_dir = tmp_path / "bootstrap"
    installer_dir = tmp_path / "installer"
    report_path = tmp_path / "report.json"
    (stage_dir / "_internal" / "torch").mkdir(parents=True)
    (stage_dir / "_internal" / "torch" / "big.bin").write_bytes(b"x" * 32)
    build_dir.mkdir()
    installer_dir.mkdir()
    bootstrap_dir.mkdir()
    installer_path = installer_dir / "transcribe.exe"
    installer_path.write_bytes(b"installer")

    build_script._write_build_report(
        report_path,
        backend="pyinstaller",
        requested_backend="pyinstaller",
        selected_phase="all",
        clean=False,
        phase_results=[build_script.PhaseResult(name="prepare", status="completed", duration_sec=1.25, details={"backend": "pyinstaller"})],
        stage_dir=stage_dir,
        build_dir=build_dir,
        bootstrap_dir=bootstrap_dir,
        installer_dir=installer_dir,
        installer_path=installer_path,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "pyinstaller"
    assert payload["requested_backend"] == "pyinstaller"
    assert payload["phases"][0]["name"] == "prepare"
    assert payload["phases"][0]["duration_sec"] == 1.25
    assert payload["artifacts"]["stage"]["exists"] is True
    assert payload["artifacts"]["stage"]["internal_top_directories"][0]["path"].endswith("torch")
    assert payload["artifacts"]["installer_file"]["size_bytes"] == len(b"installer")


@pytest.mark.parametrize("backend", ["pyinstaller", "nuitka"])
def test_main_smoke_acceptance_stages_packaged_output_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    report_path = tmp_path / "build-report.json"
    stage_dir = tmp_path / "stage"
    build_dir = tmp_path / "build"
    bootstrap_dir = tmp_path / "bootstrap"
    installer_dir = tmp_path / "installer"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt body\n", encoding="utf-8")
    monkeypatch.setattr(build_script, "PROMPT_SOURCE_PATH", prompt_path)

    monkeypatch.setattr(
        build_script,
        "_resolve_or_bootstrap_build_inputs",
        lambda args: _make_build_inputs(tmp_path, backend=backend, include_canary_model=args.include_canary_model),
    )

    def fake_build_package_bundle(*, backend: str, package_command, stage_dir: Path, build_dir: Path, clean: bool, include_canary_model: bool) -> Path:
        _ = (backend, package_command, build_dir, clean, include_canary_model)
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "transcribe.exe").write_bytes(b"exe")
        return stage_dir

    monkeypatch.setattr(build_script, "_build_package_bundle", fake_build_package_bundle)

    rc = build_script.main(
        [
            "--backend",
            backend,
            "--skip-installer",
            "--stage-dir",
            str(stage_dir),
            "--build-dir",
            str(build_dir),
            "--bootstrap-dir",
            str(bootstrap_dir),
            "--installer-dir",
            str(installer_dir),
            "--report-path",
            str(report_path),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert (stage_dir / "transcribe.exe").exists()
    assert (stage_dir / "packaged-assets.json").exists()
    assert (stage_dir / "prompts" / "clinical_note_synthesis_llm_prompt.md").exists()
    assert (stage_dir / "runtime" / "llm" / "llama-server.exe").exists()
    assert payload["backend"] == backend
    assert payload["requested_backend"] == backend
    assert payload["selected_phase"] == "all"
    assert [phase["name"] for phase in payload["phases"]] == ["prepare", "package", "stage-assets", "installer"]
    assert payload["phases"][-1]["status"] == "skipped"


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
        lambda **kwargs: {"PATH": "C:\toolchain"},
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
        return {"PATH": r"C:\captured"}

    def fake_resolve_executable(candidate: str, env: dict[str, str]) -> str | None:
        del env
        if candidate == "cmake":
            return r"C:\captured\cmake.exe"
        if candidate == "ninja":
            return r"C:\captured\ninja.exe"
        if candidate == "cl":
            return r"C:\captured\cl.exe"
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
