from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcribe.packaged_assets import (
    PACKAGED_ASSET_MANIFEST_FILENAME,
    PACKAGED_ASSET_SCHEMA_VERSION,
    PackagedAssetsManifest,
    build_directory_asset,
    build_single_file_asset,
    write_packaged_asset_manifest,
)
from transcribe.runtime_env import bundled_notes_model_specs, bundled_transcription_model_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE_DIR = REPO_ROOT / "dist" / "windows" / "transcribe"
DEFAULT_BUILD_DIR = REPO_ROOT / "dist" / "windows" / "build"
DEFAULT_INSTALLER_DIR = REPO_ROOT / "dist" / "windows" / "installer"
DEFAULT_BOOTSTRAP_DIR = REPO_ROOT / "dist" / "windows" / "_bootstrap"
DEFAULT_TOOLS_DIR = DEFAULT_BOOTSTRAP_DIR / "tools"
DEFAULT_ASSETS_DIR = DEFAULT_BOOTSTRAP_DIR / "assets"
DEFAULT_HF_CACHE_DIR = DEFAULT_BOOTSTRAP_DIR / "hf-cache"
DEFAULT_LLAMA_CPP_RELEASE = "latest"
DEFAULT_LLAMA_CPP_REPO = "ggml-org/llama.cpp"
DEFAULT_INNO_SETUP_RELEASE = "latest"
DEFAULT_INNO_SETUP_REPO = "jrsoftware/issrc"
DEFAULT_DOWNLOAD_USER_AGENT = "transcribe-windows-build-bootstrap"
DEFAULT_MODEL_BOOTSTRAP_RAM_GB = 64.0
DEFAULT_NOTES_MODEL_4B_REPO = "unsloth/Qwen3.5-4B-GGUF"
DEFAULT_NOTES_MODEL_4B_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_NOTES_MODEL_2B_REPO = "unsloth/Qwen3.5-2B-GGUF"
DEFAULT_NOTES_MODEL_2B_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_NOTES_MODEL_4B_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
DEFAULT_NOTES_MODEL_2B_REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
DEFAULT_PARAKEET_MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PARAKEET_MODEL_REVISION = "6d590f77001d318fb17a0b5bf7ee329a91b52598"
DEFAULT_PARAKEET_REQUIRED_FILES = ("parakeet-tdt-0.6b-v3.nemo",)
DEFAULT_CANARY_MODEL_REPO = "nvidia/canary-qwen-2.5b"
DEFAULT_CANARY_MODEL_REVISION = "6cfc37ec7edc35a0545c403f551ecdfa28133d72"
DEFAULT_CANARY_REQUIRED_FILES = ("config.json", "LICENSES", "model.safetensors")
DEFAULT_VS_BUILDTOOLS_BOOTSTRAPPER_URL = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
DEFAULT_VS_BUILDTOOLS_WORKLOAD = "Microsoft.VisualStudio.Workload.VCTools"
DEFAULT_CMAKE_PYTHON_PACKAGE = "cmake<4"
DEFAULT_NINJA_PYTHON_PACKAGE = "ninja"
PROMPT_SOURCE_PATH = REPO_ROOT / "clinical note synthesis llm prompt.md"
INNO_SCRIPT_PATH = REPO_ROOT / "packaging" / "windows" / "transcribe.iss"

_LLAMA_CPP_WINDOWS_ASSET_PATTERNS = (
    re.compile(r"llama-b\d+-bin-win-cpu-x64\.zip$", re.IGNORECASE),
    re.compile(r"llama-b\d+-bin-win-avx2-x64\.zip$", re.IGNORECASE),
)
_INNO_SETUP_ASSET_PATTERNS = (
    re.compile(r"innosetup-.*\.exe$", re.IGNORECASE),
    re.compile(r"isetup-.*\.exe$", re.IGNORECASE),
)
_VS_EDITIONS = ("BuildTools", "Community", "Professional", "Enterprise")
_VS_YEARS = ("2022", "2019", "2017")


@dataclass(frozen=True, slots=True)
class ResolvedBuildInputs:
    """Resolved toolchain and asset paths required for a Windows package build."""

    pyinstaller_command: tuple[str, ...]
    inno_setup_exe: str | None
    llama_runtime_dir: Path
    notes_model_4b: Path
    notes_model_2b: Path
    parakeet_model_dir: Path
    canary_model_dir: Path


@dataclass(frozen=True, slots=True)
class GitHubReleaseAsset:
    """Minimal GitHub release asset metadata needed for runtime downloads."""

    name: str
    download_url: str


def _copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


@contextlib.contextmanager
def _temporary_environment(overrides: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, original in previous.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _merged_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if overrides is not None:
        env.update(overrides)
    return env


def _prepend_path(env: Mapping[str, str] | None, *paths: Path) -> dict[str, str]:
    merged = _merged_environment(env)
    current_parts = [part for part in merged.get("PATH", "").split(os.pathsep) if part]
    additions = [str(path) for path in paths if str(path)]
    merged["PATH"] = os.pathsep.join([*additions, *current_parts])
    return merged


def _run(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    allowed_returncodes: Iterable[int] = (0,),
) -> None:
    print(f"Running: {' '.join(command)}")
    completed = subprocess.run(list(command), env=dict(env) if env is not None else None, cwd=cwd)
    if completed.returncode not in set(allowed_returncodes):
        raise subprocess.CalledProcessError(completed.returncode, completed.args)


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _resolve_executable(candidate: str) -> str | None:
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return str(candidate_path.resolve())
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    return None


def _resolve_executable_in_environment(candidate: str, env: Mapping[str, str]) -> str | None:
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return str(candidate_path.resolve())
    resolved = shutil.which(candidate, path=env.get("PATH"))
    if resolved is not None:
        return resolved
    return None


def _resolve_manual_path(path: Path | None, *, label: str) -> Path | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _python_scripts_dir() -> Path:
    return Path(sys.executable).resolve().parent


def _ensure_pip_available() -> None:
    if _module_available("pip"):
        return
    import ensurepip

    ensurepip.bootstrap(upgrade=True)


def _short_windows_work_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / "tb"
    return DEFAULT_BOOTSTRAP_DIR / "tb"


def _visual_studio_build_tools_install_root(*, tools_dir: Path) -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / "Transcribe" / "vsbt"
    return tools_dir / "vs-buildtools"


def _known_vswhere_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for env_var in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_var, "").strip()
        if not base:
            continue
        candidates.append(Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    return tuple(candidates)


def _query_vs_installation_paths() -> tuple[Path, ...]:
    vswhere = _resolve_executable("vswhere") or _resolve_known_executable(_known_vswhere_paths())
    if vswhere is None:
        return ()

    completed = subprocess.run(
        [
            vswhere,
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return ()

    paths: list[Path] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        paths.append(Path(value))
    return tuple(paths)


def _candidate_vcvars64_paths(*, extra_roots: Sequence[Path] = ()) -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in extra_roots:
        candidate = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for install_path in _query_vs_installation_paths():
        candidate = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for env_var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_var, "").strip()
        if not base:
            continue
        for year in _VS_YEARS:
            for edition in _VS_EDITIONS:
                candidate = Path(base) / "Microsoft Visual Studio" / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate not in seen:
                    candidates.append(candidate)
                    seen.add(candidate)

    return tuple(candidates)


def _find_vcvars64_bat(*, extra_roots: Sequence[Path] = ()) -> Path | None:
    for path in _candidate_vcvars64_paths(extra_roots=extra_roots):
        if path.exists():
            return path.resolve()
    return None


def _capture_batch_environment(
    batch_path: Path,
    *,
    args: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    command = f'cmd.exe /d /s /c call "{batch_path}"'
    if args:
        command = f"{command} {subprocess.list2cmdline(list(args))}"
    command = f"{command} && set"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        env=dict(env) if env is not None else None,
    )
    captured: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if not line or line.startswith("=") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        captured[key] = value
    return captured
def _cmake_requires_legacy_series(*, env: Mapping[str, str]) -> bool:
    cmake_exe = _resolve_executable_in_environment("cmake", env)
    if cmake_exe is None:
        return True
    completed = subprocess.run(
        [cmake_exe, "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(env),
    )
    if completed.returncode != 0:
        return True
    match = re.search(r"cmake version (\d+)\.(\d+)(?:\.(\d+))?", completed.stdout)
    if match is None:
        return True
    return int(match.group(1)) >= 4


def _ensure_windows_python_build_helpers(
    *,
    bootstrap_missing: bool,
    bootstrap_dir: Path,
    env: Mapping[str, str],
) -> dict[str, str]:
    del bootstrap_dir
    helper_env = _prepend_path(env, _python_scripts_dir())
    missing_packages: list[str] = []
    if _cmake_requires_legacy_series(env=helper_env):
        missing_packages.append(DEFAULT_CMAKE_PYTHON_PACKAGE)
    if _resolve_executable_in_environment("ninja", helper_env) is None:
        missing_packages.append(DEFAULT_NINJA_PYTHON_PACKAGE)

    if not missing_packages:
        return helper_env
    if not bootstrap_missing:
        raise FileNotFoundError(
            "Missing Windows native build helpers for NeMo bootstrap: "
            + ", ".join(missing_packages)
        )

    _ensure_pip_available()
    pip_env = _prepend_path(helper_env, _python_scripts_dir())
    _run([sys.executable, "-m", "pip", "install", *missing_packages], env=pip_env, cwd=REPO_ROOT)
    return _prepend_path(pip_env, _python_scripts_dir())


def _install_visual_studio_build_tools(*, tools_dir: Path) -> Path:
    install_root = _visual_studio_build_tools_install_root(tools_dir=tools_dir)
    expected_vcvars = install_root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    if expected_vcvars.exists():
        return expected_vcvars.resolve()

    installer_path = _download_file(
        DEFAULT_VS_BUILDTOOLS_BOOTSTRAPPER_URL,
        tools_dir / "downloads" / "vs_BuildTools.exe",
    )
    _run(
        [
            str(installer_path),
            "--quiet",
            "--wait",
            "--norestart",
            "--nocache",
            "--installPath",
            str(install_root),
            "--add",
            DEFAULT_VS_BUILDTOOLS_WORKLOAD,
            "--includeRecommended",
        ],
        allowed_returncodes=(0, 3010),
    )

    resolved = _find_vcvars64_bat(extra_roots=(install_root,))
    if resolved is None:
        raise FileNotFoundError(
            "Visual Studio Build Tools installation completed, but vcvars64.bat could not be located."
        )
    return resolved


def _ensure_windows_native_build_environment(*, bootstrap_missing: bool, bootstrap_dir: Path) -> dict[str, str]:
    tools_dir = (bootstrap_dir / "tools").resolve()
    env = _prepend_path(os.environ.copy(), _python_scripts_dir())
    env = _ensure_windows_python_build_helpers(
        bootstrap_missing=bootstrap_missing,
        bootstrap_dir=bootstrap_dir,
        env=env,
    )

    vcvars_path = _find_vcvars64_bat(extra_roots=(_visual_studio_build_tools_install_root(tools_dir=tools_dir),))
    if vcvars_path is None and _resolve_executable_in_environment("cl", env) is None:
        if not bootstrap_missing:
            raise FileNotFoundError(
                "MSVC build tools are required for `nemo_toolkit[asr]` on Windows. Install Visual Studio Build Tools or enable bootstrap downloads."
            )
        vcvars_path = _install_visual_studio_build_tools(tools_dir=tools_dir)

    if vcvars_path is not None:
        env = _capture_batch_environment(vcvars_path, env=env)
        env = _prepend_path(env, _python_scripts_dir())

    cmake_exe = _resolve_executable_in_environment("cmake", env)
    if cmake_exe is None:
        raise FileNotFoundError("`cmake` is still unavailable after Windows build-tool bootstrap.")

    env.setdefault("CMAKE_GENERATOR", "Visual Studio 17 2022")
    env.setdefault(
        "CMAKE_GENERATOR_INSTANCE",
        str(_visual_studio_build_tools_install_root(tools_dir=tools_dir)),
    )

    if _resolve_executable_in_environment("cl", env) is None:
        raise FileNotFoundError(
            "`cl.exe` is still unavailable after Windows build-tool bootstrap. "
            "Visual Studio Build Tools may require elevation or a reboot to complete installation."
        )
    return env


def _clear_uv_sdist_cache(uv_cache_dir: Path, *, package_name: str) -> None:
    package_cache_dir = uv_cache_dir / "sdists-v9" / "pypi" / package_name
    if package_cache_dir.exists():
        shutil.rmtree(package_cache_dir)


def _ensure_python_build_dependencies(*, bootstrap_missing: bool, bootstrap_dir: Path) -> tuple[str, ...]:
    pyinstaller_ready = _module_available("PyInstaller")
    nemo_ready = _module_available("nemo.collections.asr")
    if pyinstaller_ready and nemo_ready:
        return (sys.executable, "-m", "PyInstaller")
    if not bootstrap_missing:
        missing = []
        if not pyinstaller_ready:
            missing.append("PyInstaller")
        if not nemo_ready:
            missing.append("nemo_toolkit[asr]")
        raise RuntimeError(
            "Missing Python build dependencies for the Windows standalone package: "
            + ", ".join(missing)
        )

    dependency_env = _prepend_path(os.environ.copy(), _python_scripts_dir())
    if os.name == "nt" and not nemo_ready:
        dependency_env = _ensure_windows_native_build_environment(
            bootstrap_missing=bootstrap_missing,
            bootstrap_dir=bootstrap_dir,
        )

    uv_exe = _resolve_executable("uv")
    if uv_exe is not None:
        uv_cache_dir = bootstrap_dir / "uv-cache"
        if os.name == "nt":
            uv_cache_dir = _short_windows_work_root() / "uv"
        uv_cache_dir.mkdir(parents=True, exist_ok=True)
        env = _merged_environment(dependency_env)
        env["UV_CACHE_DIR"] = str(uv_cache_dir)
        if os.name == "nt":
            temp_dir = _short_windows_work_root() / "tmp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            env["TMP"] = str(temp_dir)
            env["TEMP"] = str(temp_dir)
            if not nemo_ready:
                _clear_uv_sdist_cache(uv_cache_dir, package_name="kaldialign")
        if not nemo_ready:
            _run([uv_exe, "sync", "--extra", "nemo-asr", "--inexact"], env=env, cwd=REPO_ROOT)
        if not pyinstaller_ready:
            _run([uv_exe, "pip", "install", "--python", sys.executable, "pyinstaller"], env=env, cwd=REPO_ROOT)
        return (sys.executable, "-m", "PyInstaller")

    _ensure_pip_available()
    requirements: list[str] = []
    if not nemo_ready:
        requirements.append("nemo_toolkit[asr]")
    if not pyinstaller_ready:
        requirements.append("pyinstaller")
    if requirements:
        _run([sys.executable, "-m", "pip", "install", *requirements], env=dependency_env, cwd=REPO_ROOT)
    return (sys.executable, "-m", "PyInstaller")


def _build_pyinstaller_command(
    *,
    pyinstaller_command: Sequence[str],
    stage_dir: Path,
    build_dir: Path,
) -> list[str]:
    command = [
        *pyinstaller_command,
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "transcribe",
        "--distpath",
        str(stage_dir.parent),
        "--workpath",
        str(build_dir / "work"),
        "--specpath",
        str(build_dir / "spec"),
        "--paths",
        str(REPO_ROOT),
        "--hidden-import",
        "transcribe.audio.windows_capture",
        "--hidden-import",
        "transcribe.audio.backend_loader",
        "--hidden-import",
        "transcribe.notes",
        "--hidden-import",
        "transcribe.packaged_assets",
        "--hidden-import",
        "transcribe.packaged_cli",
        "--hidden-import",
        "transcribe.transcription_runtime",
        "--hidden-import",
        "huggingface_hub",
        "--hidden-import",
        "huggingface_hub.file_download",
        "--hidden-import",
        "sounddevice",
        "--collect-all",
        "sounddevice",
    ]
    if _module_available("nemo.collections.asr"):
        command.extend(
            [
                "--hidden-import",
                "nemo.collections.asr",
                "--hidden-import",
                "nemo.collections.speechlm2.models",
                "--hidden-import",
                "omegaconf",
            ]
        )
    for excluded_module in (
        "datasets",
        "pyarrow",
        "matplotlib",
        "_pytest",
        "coverage",
        "mypy",
        "IPython",
        "pytest",
        "transcribe.bench",
        "transcribe.test_cov",
    ):
        command.extend(["--exclude-module", excluded_module])
    command.append(str(REPO_ROOT / "packaged_main.py"))
    return command


def _build_pyinstaller_bundle(*, pyinstaller_command: Sequence[str], stage_dir: Path, build_dir: Path) -> Path:
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    _run(_build_pyinstaller_command(pyinstaller_command=pyinstaller_command, stage_dir=stage_dir, build_dir=build_dir), cwd=REPO_ROOT)
    bundle_dir = stage_dir.parent / "transcribe"
    if bundle_dir != stage_dir:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        shutil.move(str(bundle_dir), str(stage_dir))
    return stage_dir


def _copy_llama_runtime_files(*, llama_runtime_dir: Path, stage_dir: Path) -> None:
    runtime_target_dir = stage_dir / "runtime" / "llm"
    if runtime_target_dir.exists():
        shutil.rmtree(runtime_target_dir)
    runtime_target_dir.mkdir(parents=True, exist_ok=True)

    llama_server = llama_runtime_dir / "llama-server.exe"
    if not llama_server.exists():
        raise FileNotFoundError(f"llama-server.exe was not found in runtime directory {llama_runtime_dir}")
    _copy_file(llama_server, runtime_target_dir / llama_server.name)

    for runtime_dependency in sorted(llama_runtime_dir.iterdir()):
        if not runtime_dependency.is_file():
            continue
        if runtime_dependency.name == llama_server.name:
            continue
        if runtime_dependency.suffix.lower() != ".dll":
            continue
        _copy_file(runtime_dependency, runtime_target_dir / runtime_dependency.name)


def _build_packaged_assets_manifest(
    *,
    notes_model_4b: Path,
    notes_model_4b_repo: str,
    notes_model_4b_revision: str,
    notes_model_4b_file: str,
    notes_model_2b: Path,
    notes_model_2b_repo: str,
    notes_model_2b_revision: str,
    notes_model_2b_file: str,
    parakeet_model_dir: Path,
    parakeet_model_repo: str,
    parakeet_model_revision: str,
    canary_model_dir: Path,
    canary_model_repo: str,
    canary_model_revision: str,
) -> PackagedAssetsManifest:
    notes_specs = bundled_notes_model_specs()
    transcription_specs = {spec.model_id: spec.relative_path.as_posix() for spec in bundled_transcription_model_specs()}

    return PackagedAssetsManifest(
        schema_version=PACKAGED_ASSET_SCHEMA_VERSION,
        assets=(
            build_single_file_asset(
                model_id=notes_specs[0].model_id,
                kind="notes",
                relative_path=notes_specs[0].relative_path.as_posix(),
                repo_id=notes_model_4b_repo,
                revision=notes_model_4b_revision,
                filename=notes_model_4b_file,
                source_path=notes_model_4b,
                default_install=True,
            ),
            build_single_file_asset(
                model_id=notes_specs[1].model_id,
                kind="notes",
                relative_path=notes_specs[1].relative_path.as_posix(),
                repo_id=notes_model_2b_repo,
                revision=notes_model_2b_revision,
                filename=notes_model_2b_file,
                source_path=notes_model_2b,
                default_install=False,
            ),
            build_directory_asset(
                model_id=DEFAULT_PARAKEET_MODEL_REPO,
                kind="transcription",
                relative_path=transcription_specs[DEFAULT_PARAKEET_MODEL_REPO],
                repo_id=parakeet_model_repo,
                revision=parakeet_model_revision,
                source_root=parakeet_model_dir,
                required_files=DEFAULT_PARAKEET_REQUIRED_FILES,
                default_install=True,
            ),
            build_directory_asset(
                model_id=DEFAULT_CANARY_MODEL_REPO,
                kind="transcription",
                relative_path=transcription_specs[DEFAULT_CANARY_MODEL_REPO],
                repo_id=canary_model_repo,
                revision=canary_model_revision,
                source_root=canary_model_dir,
                required_files=DEFAULT_CANARY_REQUIRED_FILES,
                default_install=False,
            ),
        ),
    )


def _stage_runtime_assets(
    *,
    stage_dir: Path,
    llama_runtime_dir: Path,
    notes_model_4b: Path,
    notes_model_4b_repo: str,
    notes_model_4b_revision: str,
    notes_model_4b_file: str,
    notes_model_2b: Path,
    notes_model_2b_repo: str,
    notes_model_2b_revision: str,
    notes_model_2b_file: str,
    parakeet_model_dir: Path,
    parakeet_model_repo: str,
    parakeet_model_revision: str,
    canary_model_dir: Path,
    canary_model_repo: str,
    canary_model_revision: str,
) -> None:
    stage_models_dir = stage_dir / "models"
    if stage_models_dir.exists():
        shutil.rmtree(stage_models_dir)

    _copy_file(PROMPT_SOURCE_PATH, stage_dir / "prompts" / "clinical_note_synthesis_llm_prompt.md")
    _copy_llama_runtime_files(llama_runtime_dir=llama_runtime_dir, stage_dir=stage_dir)
    manifest = _build_packaged_assets_manifest(
        notes_model_4b=notes_model_4b,
        notes_model_4b_repo=notes_model_4b_repo,
        notes_model_4b_revision=notes_model_4b_revision,
        notes_model_4b_file=notes_model_4b_file,
        notes_model_2b=notes_model_2b,
        notes_model_2b_repo=notes_model_2b_repo,
        notes_model_2b_revision=notes_model_2b_revision,
        notes_model_2b_file=notes_model_2b_file,
        parakeet_model_dir=parakeet_model_dir,
        parakeet_model_repo=parakeet_model_repo,
        parakeet_model_revision=parakeet_model_revision,
        canary_model_dir=canary_model_dir,
        canary_model_repo=canary_model_repo,
        canary_model_revision=canary_model_revision,
    )
    write_packaged_asset_manifest(manifest, stage_dir / PACKAGED_ASSET_MANIFEST_FILENAME)


def _build_inno_installer(*, inno_setup_exe: str, stage_dir: Path, installer_dir: Path) -> Path:
    installer_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            inno_setup_exe,
            f"/DSourceDir={stage_dir}",
            f"/DOutputDir={installer_dir}",
            str(INNO_SCRIPT_PATH),
        ],
        cwd=REPO_ROOT,
    )
    installer_candidates = sorted(installer_dir.glob("*.exe"))
    if not installer_candidates:
        raise FileNotFoundError(f"No installer executable was created in {installer_dir}")
    return installer_candidates[-1]


def _known_inno_setup_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for env_var in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_var, "").strip()
        if not base:
            continue
        candidates.append(Path(base) / "Inno Setup 6" / "ISCC.exe")
    return tuple(candidates)


def _resolve_known_executable(paths: Sequence[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path.resolve())
    return None


def _select_inno_setup_asset(assets: Sequence[Mapping[str, object]]) -> GitHubReleaseAsset:
    candidates: list[GitHubReleaseAsset] = []
    for raw_asset in assets:
        name = raw_asset.get("name")
        download_url = raw_asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(download_url, str):
            candidates.append(GitHubReleaseAsset(name=name, download_url=download_url))

    for pattern in _INNO_SETUP_ASSET_PATTERNS:
        for asset in candidates:
            if pattern.search(asset.name):
                return asset
    available = ", ".join(sorted(asset.name for asset in candidates))
    raise RuntimeError(
        "Unable to find a supported Inno Setup installer asset in the selected release. "
        f"Available assets: {available}"
    )


def _ensure_inno_setup(*, inno_setup_exe: str, bootstrap_missing: bool, tools_dir: Path) -> str:
    resolved = _resolve_executable(inno_setup_exe)
    if resolved is not None:
        return resolved
    local_bootstrap_install = tools_dir / "inno-setup" / "ISCC.exe"
    if local_bootstrap_install.exists():
        return str(local_bootstrap_install.resolve())
    resolved = _resolve_known_executable(_known_inno_setup_paths())
    if resolved is not None:
        return resolved
    if not bootstrap_missing:
        raise FileNotFoundError(
            f"Unable to find Inno Setup compiler executable {inno_setup_exe!r}. Install Inno Setup or pass --skip-installer."
        )

    installer_path = _download_inno_setup_installer(tools_dir=tools_dir, release=DEFAULT_INNO_SETUP_RELEASE)
    install_root = tools_dir / "inno-setup"
    _run(
        [
            str(installer_path),
            f"/DIR={install_root}",
            "/SP-",
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/NOICONS",
        ]
    )

    if local_bootstrap_install.exists():
        return str(local_bootstrap_install.resolve())
    resolved = _resolve_known_executable(_known_inno_setup_paths())
    if resolved is None:
        raise FileNotFoundError("Automatic Inno Setup installation completed, but ISCC.exe could not be located.")
    return resolved


def _github_release_api_url(*, repo: str, release: str) -> str:
    base = f"https://api.github.com/repos/{repo}/releases"
    if release == "latest":
        return f"{base}/latest"
    return f"{base}/tags/{release}"


def _fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": DEFAULT_DOWNLOAD_USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=120.0) as response:
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Unexpected JSON payload from {url!r}")
    return parsed


def _select_llama_cpp_windows_asset(assets: Sequence[Mapping[str, object]]) -> GitHubReleaseAsset:
    candidates: list[GitHubReleaseAsset] = []
    for raw_asset in assets:
        name = raw_asset.get("name")
        download_url = raw_asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(download_url, str):
            candidates.append(GitHubReleaseAsset(name=name, download_url=download_url))

    for pattern in _LLAMA_CPP_WINDOWS_ASSET_PATTERNS:
        for asset in candidates:
            if pattern.search(asset.name):
                return asset
    available = ", ".join(sorted(asset.name for asset in candidates))
    raise RuntimeError(
        "Unable to find a supported Windows llama.cpp runtime asset in the selected release. "
        f"Available assets: {available}"
    )


def _download_file(url: str, destination: Path) -> Path:
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request, timeout=300.0) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _find_named_file(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    if not matches:
        return None
    return matches[0]


def _download_inno_setup_installer(*, tools_dir: Path, release: str) -> Path:
    release_payload = _fetch_json(_github_release_api_url(repo=DEFAULT_INNO_SETUP_REPO, release=release))
    assets = release_payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("Inno Setup release payload did not include an asset list")
    asset = _select_inno_setup_asset(assets)
    return _download_file(asset.download_url, tools_dir / "downloads" / asset.name)


def _download_llama_cpp_runtime(*, tools_dir: Path, release: str) -> Path:
    release_payload = _fetch_json(_github_release_api_url(repo=DEFAULT_LLAMA_CPP_REPO, release=release))
    assets = release_payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("llama.cpp release payload did not include an asset list")
    asset = _select_llama_cpp_windows_asset(assets)
    extract_dir = tools_dir / "llama.cpp" / Path(asset.name).stem
    llama_server = _find_named_file(extract_dir, "llama-server.exe")
    if llama_server is not None:
        return llama_server.parent

    archive_path = _download_file(asset.download_url, tools_dir / "downloads" / asset.name)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)
    llama_server = _find_named_file(extract_dir, "llama-server.exe")
    if llama_server is None:
        raise FileNotFoundError(f"llama-server.exe was not present in extracted archive {archive_path}")
    return llama_server.parent


def _download_hf_file(*, repo_id: str, revision: str, filename: str, hf_cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Downloading bundled model assets requires `huggingface_hub`.") from exc

    hf_home = hf_cache_dir.resolve()
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    with _temporary_environment(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hub_cache),
            "HUGGINGFACE_HUB_CACHE": str(hub_cache),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        }
    ):
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=False,
            cache_dir=str(hub_cache),
        )
    path = Path(downloaded).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Downloaded Hugging Face asset does not exist: {path}")
    return path


def _bootstrap_notes_model(*, repo_id: str, revision: str, filename: str, hf_cache_dir: Path) -> Path:
    return _download_hf_file(repo_id=repo_id, revision=revision, filename=filename, hf_cache_dir=hf_cache_dir)


def _bootstrap_transcription_model(*, model_id: str, revision: str, hf_cache_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Downloading bundled transcription models requires `huggingface_hub`.") from exc

    hf_home = hf_cache_dir.resolve()
    hub_cache = hf_home / "hub"
    target_dir = hf_home / "model-snapshots" / model_id.replace("/", "--")
    hub_cache.mkdir(parents=True, exist_ok=True)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_environment(
        {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hub_cache),
            "HUGGINGFACE_HUB_CACHE": str(hub_cache),
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        }
    ):
        snapshot_dir = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=str(hub_cache),
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            local_files_only=False,
        )
    path = Path(snapshot_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resolved model cache directory does not exist for {model_id!r}: {path}")
    return path


def _resolve_or_bootstrap_build_inputs(args: argparse.Namespace) -> ResolvedBuildInputs:
    bootstrap_dir = args.bootstrap_dir.resolve()
    tools_dir = args.tools_dir.resolve()
    hf_cache_dir = args.hf_cache_dir.resolve()

    if args.pyinstaller_exe != "pyinstaller":
        resolved_pyinstaller = _resolve_executable(args.pyinstaller_exe)
        if resolved_pyinstaller is None:
            raise FileNotFoundError(f"PyInstaller executable does not exist: {args.pyinstaller_exe}")
        pyinstaller_command = (resolved_pyinstaller,)
    else:
        pyinstaller_command = _ensure_python_build_dependencies(
            bootstrap_missing=args.bootstrap_missing,
            bootstrap_dir=bootstrap_dir,
        )

    inno_setup_exe: str | None = None
    if not args.skip_installer:
        inno_setup_exe = _ensure_inno_setup(
            inno_setup_exe=args.inno_setup_exe,
            bootstrap_missing=args.bootstrap_missing,
            tools_dir=tools_dir,
        )

    llama_runtime_dir = _resolve_manual_path(args.llama_runtime_dir, label="llama runtime directory")
    llama_server = _resolve_manual_path(args.llama_server, label="llama-server executable")
    if llama_runtime_dir is None and llama_server is not None:
        llama_runtime_dir = llama_server.parent
    if llama_runtime_dir is None:
        if not args.bootstrap_missing:
            raise FileNotFoundError(
                "No llama.cpp runtime was provided. Pass --llama-runtime-dir or --llama-server, or enable bootstrap downloads."
            )
        llama_runtime_dir = _download_llama_cpp_runtime(tools_dir=tools_dir, release=args.llama_cpp_release)
    if not (llama_runtime_dir / "llama-server.exe").exists():
        raise FileNotFoundError(f"llama-server.exe was not found in runtime directory {llama_runtime_dir}")

    notes_model_4b = _resolve_manual_path(args.notes_model_4b, label="4B notes model")
    if notes_model_4b is None:
        if not args.bootstrap_missing:
            raise FileNotFoundError("Missing bundled 4B notes model. Pass --notes-model-4b.")
        notes_model_4b = _bootstrap_notes_model(
            repo_id=args.notes_model_4b_repo,
            revision=args.notes_model_4b_revision,
            filename=args.notes_model_4b_file,
            hf_cache_dir=hf_cache_dir,
        )

    notes_model_2b = _resolve_manual_path(args.notes_model_2b, label="2B notes model")
    if notes_model_2b is None:
        if not args.bootstrap_missing:
            raise FileNotFoundError("Missing bundled 2B notes model. Pass --notes-model-2b.")
        notes_model_2b = _bootstrap_notes_model(
            repo_id=args.notes_model_2b_repo,
            revision=args.notes_model_2b_revision,
            filename=args.notes_model_2b_file,
            hf_cache_dir=hf_cache_dir,
        )

    parakeet_model_dir = _resolve_manual_path(args.parakeet_model_dir, label="Parakeet ASR model directory")
    if parakeet_model_dir is None:
        if not args.bootstrap_missing:
            raise FileNotFoundError("Missing Parakeet ASR model directory. Pass --parakeet-model-dir.")
        parakeet_model_dir = _bootstrap_transcription_model(
            model_id=args.parakeet_model_repo,
            revision=args.parakeet_model_revision,
            hf_cache_dir=hf_cache_dir,
        )

    canary_model_dir = _resolve_manual_path(args.canary_model_dir, label="Canary ASR model directory")
    if canary_model_dir is None:
        if not args.bootstrap_missing:
            raise FileNotFoundError("Missing Canary ASR model directory. Pass --canary-model-dir.")
        canary_model_dir = _bootstrap_transcription_model(
            model_id=args.canary_model_repo,
            revision=args.canary_model_revision,
            hf_cache_dir=hf_cache_dir,
        )

    return ResolvedBuildInputs(
        pyinstaller_command=tuple(pyinstaller_command),
        inno_setup_exe=inno_setup_exe,
        llama_runtime_dir=llama_runtime_dir.resolve(),
        notes_model_4b=notes_model_4b.resolve(),
        notes_model_2b=notes_model_2b.resolve(),
        parakeet_model_dir=parakeet_model_dir.resolve(),
        canary_model_dir=canary_model_dir.resolve(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Windows standalone transcribe package")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--installer-dir", type=Path, default=DEFAULT_INSTALLER_DIR)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP_DIR)
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--hf-cache-dir", type=Path, default=DEFAULT_HF_CACHE_DIR)
    parser.add_argument("--pyinstaller-exe", default="pyinstaller")
    parser.add_argument("--inno-setup-exe", default="iscc")
    parser.add_argument("--llama-cpp-release", default=DEFAULT_LLAMA_CPP_RELEASE)
    parser.add_argument("--bootstrap-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-pyinstaller", action="store_true")
    parser.add_argument("--skip-installer", action="store_true")
    parser.add_argument("--llama-runtime-dir", type=Path, default=None)
    parser.add_argument("--llama-server", type=Path, default=None)
    parser.add_argument("--notes-model-4b", type=Path, default=None)
    parser.add_argument("--notes-model-2b", type=Path, default=None)
    parser.add_argument("--notes-model-4b-repo", default=DEFAULT_NOTES_MODEL_4B_REPO)
    parser.add_argument("--notes-model-4b-file", default=DEFAULT_NOTES_MODEL_4B_FILE)
    parser.add_argument("--notes-model-4b-revision", default=DEFAULT_NOTES_MODEL_4B_REVISION)
    parser.add_argument("--notes-model-2b-repo", default=DEFAULT_NOTES_MODEL_2B_REPO)
    parser.add_argument("--notes-model-2b-file", default=DEFAULT_NOTES_MODEL_2B_FILE)
    parser.add_argument("--notes-model-2b-revision", default=DEFAULT_NOTES_MODEL_2B_REVISION)
    parser.add_argument("--parakeet-model-dir", type=Path, default=None)
    parser.add_argument("--parakeet-model-repo", default=DEFAULT_PARAKEET_MODEL_REPO)
    parser.add_argument("--parakeet-model-revision", default=DEFAULT_PARAKEET_MODEL_REVISION)
    parser.add_argument("--canary-model-dir", type=Path, default=None)
    parser.add_argument("--canary-model-repo", default=DEFAULT_CANARY_MODEL_REPO)
    parser.add_argument("--canary-model-revision", default=DEFAULT_CANARY_MODEL_REVISION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_inputs = _resolve_or_bootstrap_build_inputs(args)

    stage_dir = args.stage_dir.resolve()
    if not args.skip_pyinstaller:
        _build_pyinstaller_bundle(
            pyinstaller_command=build_inputs.pyinstaller_command,
            stage_dir=stage_dir,
            build_dir=args.build_dir.resolve(),
        )
    elif not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory does not exist for --skip-pyinstaller: {stage_dir}")

    _stage_runtime_assets(
        stage_dir=stage_dir,
        llama_runtime_dir=build_inputs.llama_runtime_dir,
        notes_model_4b=build_inputs.notes_model_4b,
        notes_model_4b_repo=args.notes_model_4b_repo,
        notes_model_4b_revision=args.notes_model_4b_revision,
        notes_model_4b_file=args.notes_model_4b_file,
        notes_model_2b=build_inputs.notes_model_2b,
        notes_model_2b_repo=args.notes_model_2b_repo,
        notes_model_2b_revision=args.notes_model_2b_revision,
        notes_model_2b_file=args.notes_model_2b_file,
        parakeet_model_dir=build_inputs.parakeet_model_dir,
        parakeet_model_repo=args.parakeet_model_repo,
        parakeet_model_revision=args.parakeet_model_revision,
        canary_model_dir=build_inputs.canary_model_dir,
        canary_model_repo=args.canary_model_repo,
        canary_model_revision=args.canary_model_revision,
    )

    installer_path: Path | None = None
    if not args.skip_installer:
        assert build_inputs.inno_setup_exe is not None
        installer_path = _build_inno_installer(
            inno_setup_exe=build_inputs.inno_setup_exe,
            stage_dir=stage_dir,
            installer_dir=args.installer_dir.resolve(),
        )

    print(f"Windows stage: {stage_dir}")
    if installer_path is not None:
        print(f"Installer: {installer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())















