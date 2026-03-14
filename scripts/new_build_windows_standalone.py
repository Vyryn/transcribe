from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transcribe import __version__ as PACKAGE_VERSION
from transcribe.packaged_assets import (
    PACKAGED_ASSET_MANIFEST_FILENAME,
    PACKAGED_ASSET_SCHEMA_VERSION,
    PackagedAssetFile,
    PackagedAssetsManifest,
    PackagedModelAsset,
    write_packaged_asset_manifest,
)
from transcribe.runtime_env import bundled_notes_model_specs, bundled_transcription_model_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_ROOT = REPO_ROOT / "dist" / "wbuild"
STAGE_DIR = BUILD_ROOT / "stage"
NUITKA_BUILD_DIR = BUILD_ROOT / "nuitka"
INSTALLER_DIR = BUILD_ROOT / "installer"
CACHE_DIR = BUILD_ROOT / "_cache"
TOOLS_DIR = CACHE_DIR / "tools"
RELEASES_DIR = REPO_ROOT / "releases"
PROMPT_SOURCE_PATH = REPO_ROOT / "clinical note synthesis llm prompt.md"
INNO_SCRIPT_PATH = REPO_ROOT / "packaging" / "windows" / "new_transcribe_installer.iss"
NUITKA_PACKAGE_CONFIG_PATH = REPO_ROOT / "packaging" / "windows" / "nuitka-librosa-workaround.yml"
NUITKA_REPORT_FILENAME = "nuitka-report.xml"

APP_NAME = "Transcribe"
APP_PUBLISHER = "Transcribe"
DEFAULT_DOWNLOAD_USER_AGENT = "transcribe-new-windows-build"
DEFAULT_LLAMA_CPP_REPO = "ggml-org/llama.cpp"
DEFAULT_LLAMA_CPP_RELEASE = "latest"
DEFAULT_LLAMA_CPP_RELEASE_SCAN_COUNT = 12
DEFAULT_NOTES_MODEL_4B_REPO = "unsloth/Qwen3.5-4B-GGUF"
DEFAULT_NOTES_MODEL_4B_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_NOTES_MODEL_4B_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
DEFAULT_NOTES_MODEL_2B_REPO = "unsloth/Qwen3.5-2B-GGUF"
DEFAULT_NOTES_MODEL_2B_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_NOTES_MODEL_2B_REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
DEFAULT_PARAKEET_MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PARAKEET_MODEL_REVISION = "6d590f77001d318fb17a0b5bf7ee329a91b52598"
DEFAULT_PARAKEET_REQUIRED_FILES = ("parakeet-tdt-0.6b-v3.nemo",)
DEFAULT_CANARY_MODEL_REPO = "nvidia/canary-qwen-2.5b"
DEFAULT_CANARY_MODEL_REVISION = "6cfc37ec7edc35a0545c403f551ecdfa28133d72"
DEFAULT_CANARY_REQUIRED_FILES = ("config.json", "LICENSES", "model.safetensors")
UNKNOWN_PACKAGED_ASSET_SHA256 = "0" * 64
UNKNOWN_PACKAGED_ASSET_SIZE_BYTES = 0
REQUIRED_RUNTIME_MODULES = (
    "huggingface_hub",
    "nemo.collections.asr",
    "nemo.collections.speechlm2.models",
    "soundcard",
)
REQUIRED_STAGE_FILES = (
    Path("transcribe.exe"),
    Path("packaged-assets.json"),
    Path("prompts") / "clinical_note_synthesis_llm_prompt.md",
    Path("runtime") / "llm" / "llama-server.exe",
    Path("soundcard") / "mediafoundation.py.h",
)
SUPPORTED_PYTHON = (3, 13)
_LLAMA_CPP_WINDOWS_ASSET_PATTERNS = (
    re.compile(r"llama-b\d+-bin-win-cpu-x64\.zip$", re.IGNORECASE),
    re.compile(r"llama-b\d+-bin-win-avx2-x64\.zip$", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class GitHubReleaseAsset:
    """Minimal GitHub release asset metadata."""

    name: str
    download_url: str


@dataclass(frozen=True, slots=True)
class ResolvedBuildInputs:
    """All discovered artifacts required for the build."""

    version: str
    inno_setup_exe: str
    nuitka_command: tuple[str, ...]
    llama_runtime_dir: Path


def _module_available(module_name: str) -> bool:
    """Return whether one importable module is available."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def _ensure_supported_python() -> None:
    """Fail fast when the active interpreter is not the supported Windows build Python."""
    current = sys.version_info[:2]
    if current == SUPPORTED_PYTHON:
        return
    expected = ".".join(str(part) for part in SUPPORTED_PYTHON)
    observed = ".".join(str(part) for part in current)
    raise RuntimeError(
        f"This Windows builder requires Python {expected}. "
        f"Detected Python {observed} from {sys.executable}."
    )


def _resolve_executable(candidate: str) -> str | None:
    """Resolve one executable path from an explicit path or PATH lookup."""
    candidate_path = Path(candidate)
    if candidate_path.exists():
        return str(candidate_path.resolve())
    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved
    return None


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    allowed_returncodes: Iterable[int] = (0,),
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess and raise on unexpected exit codes."""
    print(f"Running: {' '.join(command)}")
    completed = subprocess.run(
        list(command),
        cwd=cwd or REPO_ROOT,
        env=dict(env) if env is not None else None,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if completed.returncode not in set(allowed_returncodes):
        if capture_output:
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _require_repo_runtime_dependencies() -> None:
    """Fail fast when the packaged app runtime dependencies are missing."""
    missing_modules = [module_name for module_name in REQUIRED_RUNTIME_MODULES if not _module_available(module_name)]
    if not missing_modules:
        return

    missing_list = ", ".join(missing_modules)
    raise RuntimeError(
        "The Windows build requires project dependencies that are not installed "
        f"({missing_list}). Run `uv sync --extra nemo-asr --inexact` and retry."
    )


def _require_nuitka() -> tuple[str, ...]:
    """Return the Nuitka command when the active environment already provides it."""
    if _module_available("nuitka"):
        return (sys.executable, "-m", "nuitka")

    raise RuntimeError(
        "Nuitka is required on the developer machine for Windows builds. "
        "Install it into the project environment and retry."
    )


def _known_inno_setup_paths() -> tuple[Path, ...]:
    """Return common Windows Inno Setup installation paths."""
    candidates: list[Path] = []
    for env_var in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(env_var, "").strip()
        if not base:
            continue
        candidates.append(Path(base) / "Inno Setup 6" / "ISCC.exe")
    return tuple(candidates)


def _resolve_known_executable(paths: Sequence[Path]) -> str | None:
    """Return the first existing executable path from a candidate list."""
    for path in paths:
        if path.exists():
            return str(path.resolve())
    return None


def _github_release_api_url(*, repo: str, release: str) -> str:
    """Build one GitHub release API URL."""
    base = f"https://api.github.com/repos/{repo}/releases"
    if release == "latest":
        return f"{base}/latest"
    return f"{base}/tags/{release}"


def _github_releases_api_url(*, repo: str, per_page: int) -> str:
    """Build one GitHub releases listing API URL."""
    return f"https://api.github.com/repos/{repo}/releases?per_page={per_page}"


def _fetch_json(url: str) -> dict[str, Any]:
    """Fetch one JSON object from the network."""
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


def _fetch_json_list(url: str) -> list[dict[str, Any]]:
    """Fetch one JSON list of objects from the network."""
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
    if not isinstance(parsed, list):
        raise RuntimeError(f"Unexpected JSON payload from {url!r}")

    items: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Unexpected JSON payload from {url!r}")
        items.append(entry)
    return items


def _download_file(url: str, destination: Path) -> Path:
    """Download one file when it is not already cached locally."""
    if destination.exists():
        return destination.resolve()

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_DOWNLOAD_USER_AGENT})
    with urllib.request.urlopen(request, timeout=300.0) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination.resolve()


def _select_github_asset(
    assets: Sequence[Mapping[str, object]],
    *,
    patterns: Sequence[re.Pattern[str]],
    label: str,
) -> GitHubReleaseAsset:
    """Select the first GitHub release asset that matches one preferred pattern."""
    candidates: list[GitHubReleaseAsset] = []
    for raw_asset in assets:
        name = raw_asset.get("name")
        download_url = raw_asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(download_url, str):
            candidates.append(GitHubReleaseAsset(name=name, download_url=download_url))

    for pattern in patterns:
        for asset in candidates:
            if pattern.search(asset.name):
                return asset

    available = ", ".join(sorted(asset.name for asset in candidates))
    raise RuntimeError(f"Unable to find a supported {label} asset. Available assets: {available}")


def _find_named_file(root: Path, filename: str) -> Path | None:
    """Return the first file with one exact filename under a root."""
    matches = sorted(root.rglob(filename))
    if not matches:
        return None
    return matches[0]


def _require_inno_setup() -> str:
    """Return the existing Inno Setup compiler path from the developer machine."""
    resolved = _resolve_executable("iscc")
    if resolved is not None:
        return resolved

    resolved = _resolve_known_executable(_known_inno_setup_paths())
    if resolved is not None:
        return resolved

    raise RuntimeError(
        "Inno Setup 6 is required on the developer machine for Windows builds. "
        "Install it and ensure `ISCC.exe` is on PATH or available in Program Files."
    )


def _download_llama_cpp_runtime() -> Path:
    """Download and extract the Windows llama.cpp runtime used for notes generation."""
    release_payload = _fetch_json(
        _github_release_api_url(repo=DEFAULT_LLAMA_CPP_REPO, release=DEFAULT_LLAMA_CPP_RELEASE)
    )
    assets = release_payload.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("llama.cpp release payload did not include an asset list.")

    try:
        asset = _select_github_asset(
            assets,
            patterns=_LLAMA_CPP_WINDOWS_ASSET_PATTERNS,
            label="Windows llama.cpp runtime",
        )
    except RuntimeError as latest_error:
        asset = None
        recent_releases = _fetch_json_list(
            _github_releases_api_url(
                repo=DEFAULT_LLAMA_CPP_REPO,
                per_page=DEFAULT_LLAMA_CPP_RELEASE_SCAN_COUNT,
            )
        )
        for release in recent_releases:
            release_assets = release.get("assets")
            if not isinstance(release_assets, list):
                continue
            try:
                asset = _select_github_asset(
                    release_assets,
                    patterns=_LLAMA_CPP_WINDOWS_ASSET_PATTERNS,
                    label="Windows llama.cpp runtime",
                )
            except RuntimeError:
                continue
            break
        if asset is None:
            raise latest_error

    extract_dir = TOOLS_DIR / "llama.cpp" / Path(asset.name).stem
    existing_server = _find_named_file(extract_dir, "llama-server.exe")
    if existing_server is not None:
        return existing_server.parent.resolve()

    archive_path = _download_file(asset.download_url, TOOLS_DIR / "downloads" / asset.name)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)

    llama_server = _find_named_file(extract_dir, "llama-server.exe")
    if llama_server is None:
        raise FileNotFoundError(f"llama-server.exe was not found in extracted archive {archive_path}")
    return llama_server.parent.resolve()


def _seed_distribution_metadata_names() -> tuple[str, ...]:
    """Return the first-pass Nuitka metadata distributions needed on a clean build."""
    packages_to_distributions = importlib.metadata.packages_distributions()
    names: dict[str, str] = {}
    for module_name in (
        "transcribe",
        "accelerate",
        "datasets",
        "filelock",
        "huggingface_hub",
        "libcst",
        "nemo",
        "numpy",
        "omegaconf",
        "packaging",
        "regex",
        "requests",
        "safetensors",
        "soundcard",
        "tokenizers",
        "torch",
        "tqdm",
        "transformers",
        "yaml",
    ):
        for distribution_name in packages_to_distributions.get(module_name, ()):
            if not _is_safe_metadata_distribution(distribution_name):
                continue
            names.setdefault(_normalize_distribution_name(distribution_name), distribution_name)
    for distribution_name in ("libcst", "transformers"):
        if not _is_safe_metadata_distribution(distribution_name):
            continue
        names.setdefault(_normalize_distribution_name(distribution_name), distribution_name)
    return tuple(names[key] for key in sorted(names))


def _normalize_distribution_name(name: str) -> str:
    """Normalize one distribution name for set comparisons."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _distribution_top_level_packages(distribution_name: str) -> tuple[str, ...]:
    """Return advertised top-level import roots for one installed distribution."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return ()

    top_level_text = distribution.read_text("top_level.txt")
    if top_level_text:
        top_level_names = [line.strip() for line in top_level_text.splitlines() if line.strip()]
        if top_level_names:
            return tuple(dict.fromkeys(top_level_names))

    file_paths = distribution.files or ()
    top_level_names = {
        file_path.parts[0]
        for file_path in file_paths
        if len(file_path.parts) > 1
        and file_path.parts[0]
        and not file_path.parts[0].endswith(".dist-info")
        and file_path.parts[0] != "__pycache__"
    }
    return tuple(sorted(top_level_names))


def _is_safe_metadata_distribution(distribution_name: str) -> bool:
    """Return whether one distribution is safe to include as Nuitka metadata."""
    normalized_distribution_name = _normalize_distribution_name(distribution_name)
    blocked_distribution_names = {
        "pyarrow",
        "matplotlib",
        "mako",
        "_pytest",
        "coverage",
        "mypy",
        "ipython",
        "pytest",
        "setuptools",
        "nemo-toolkit",
    }
    if normalized_distribution_name in blocked_distribution_names:
        return False

    blocked_top_level_packages = {"examples", "tests", "docs", "tools"}
    top_level_packages = tuple(
        _normalize_distribution_name(package_name)
        for package_name in _distribution_top_level_packages(distribution_name)
    )
    return not any(package_name in blocked_top_level_packages for package_name in top_level_packages)


def _default_nuitka_report_path(build_dir: Path) -> Path:
    """Return the report path written by the simplified Nuitka build."""
    return build_dir.resolve() / NUITKA_REPORT_FILENAME


def _reported_distribution_usage_names(build_dir: Path) -> tuple[str, ...]:
    """Return distribution names reported by the previous Nuitka build report."""
    report_path = _default_nuitka_report_path(build_dir)
    if not report_path.exists():
        return ()

    distribution_names: list[str] = []
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = re.search(r'<distribution-usage name="([^"]+)"', line)
        if match is not None:
            distribution_names.append(match.group(1))
    return tuple(dict.fromkeys(distribution_names))


def _reported_module_top_level_names(build_dir: Path) -> tuple[str, ...]:
    """Return top-level package names reported by the previous Nuitka report."""
    report_path = _default_nuitka_report_path(build_dir)
    if not report_path.exists():
        return ()

    module_names: set[str] = set()
    for line in report_path.read_text(encoding="utf-8").splitlines():
        match = re.search(r'module="([^"]+)"', line)
        if match is None:
            continue
        module_name = match.group(1).split(".", 1)[0].strip()
        if module_name:
            module_names.add(module_name)
    return tuple(sorted(module_names))


def _required_report_distribution_metadata_names(build_dir: Path) -> tuple[str, ...]:
    """Filter previous Nuitka report distributions down to safe metadata candidates."""
    reported_distribution_names = _reported_distribution_usage_names(build_dir)
    if not reported_distribution_names:
        return ()

    allowed_package_names = {module_name.lower() for module_name in _reported_module_top_level_names(build_dir)}
    allowed_package_names.update(
        _normalize_distribution_name(distribution_name)
        for distribution_name in _seed_distribution_metadata_names()
    )
    names: dict[str, str] = {}
    for distribution_name in reported_distribution_names:
        normalized_distribution_name = _normalize_distribution_name(distribution_name)
        if not _is_safe_metadata_distribution(distribution_name):
            continue
        top_level_packages = tuple(
            _normalize_distribution_name(package_name)
            for package_name in _distribution_top_level_packages(distribution_name)
        )
        candidate_package_names = {normalized_distribution_name, *top_level_packages}
        if candidate_package_names.isdisjoint(allowed_package_names):
            continue
        names.setdefault(normalized_distribution_name, distribution_name)
    return tuple(names[key] for key in sorted(names))


def _nuitka_distribution_metadata_names(build_dir: Path) -> tuple[str, ...]:
    """Return distribution metadata arguments for the next Nuitka build."""
    names: dict[str, str] = {}
    for distribution_name in _seed_distribution_metadata_names():
        names.setdefault(_normalize_distribution_name(distribution_name), distribution_name)
    for distribution_name in _required_report_distribution_metadata_names(build_dir):
        names.setdefault(_normalize_distribution_name(distribution_name), distribution_name)
    return tuple(names[key] for key in sorted(names))


def _supports_nuitka_option(nuitka_command: Sequence[str], option_fragment: str) -> bool:
    """Return whether the active Nuitka version advertises one CLI option."""
    try:
        completed = _run(
            [*nuitka_command, "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return False
    haystack = f"{completed.stdout}\n{completed.stderr}"
    return option_fragment in haystack


def _copy_file(source: Path, destination: Path) -> None:
    """Copy one file to the staged runtime tree."""
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_llama_runtime_files(*, llama_runtime_dir: Path, stage_dir: Path) -> None:
    """Copy the llama.cpp server runtime into the staged application bundle."""
    runtime_dir = stage_dir / "runtime" / "llm"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _copy_file(llama_runtime_dir / "llama-server.exe", runtime_dir / "llama-server.exe")
    for file_path in sorted(llama_runtime_dir.glob("*.dll")):
        _copy_file(file_path, runtime_dir / file_path.name)


def _unknown_packaged_asset_file(relative_path: str | Path) -> PackagedAssetFile:
    """Return one manifest file record with unknown size and hash metadata."""
    return PackagedAssetFile(
        path=str(relative_path).replace("\\", "/"),
        sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
        size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
    )


def _remote_single_file_asset(
    *,
    model_id: str,
    kind: str,
    relative_path: str | Path,
    repo_id: str,
    revision: str,
    filename: str,
    default_install: bool,
) -> PackagedModelAsset:
    """Build one remote-only file asset manifest entry."""
    return PackagedModelAsset(
        model_id=model_id,
        kind=kind,
        relative_path=str(relative_path).replace("\\", "/"),
        source_type="huggingface_file",
        repo_id=repo_id,
        revision=revision,
        filename=str(filename).replace("\\", "/"),
        required_files=(),
        sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
        size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
        default_install=default_install,
    )


def _remote_directory_asset(
    *,
    model_id: str,
    kind: str,
    relative_path: str | Path,
    repo_id: str,
    revision: str,
    required_files: tuple[str | Path, ...],
    default_install: bool,
) -> PackagedModelAsset:
    """Build one remote-only snapshot asset manifest entry."""
    return PackagedModelAsset(
        model_id=model_id,
        kind=kind,
        relative_path=str(relative_path).replace("\\", "/"),
        source_type="huggingface_snapshot",
        repo_id=repo_id,
        revision=revision,
        filename=None,
        required_files=tuple(_unknown_packaged_asset_file(path) for path in required_files),
        sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
        size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
        default_install=default_install,
    )


def _build_packaged_assets_manifest(
) -> PackagedAssetsManifest:
    """Build the packaged model manifest used by the installer and packaged UI."""
    notes_specs = bundled_notes_model_specs()
    transcription_specs = {spec.model_id: spec.relative_path.as_posix() for spec in bundled_transcription_model_specs()}

    assets = [
        _remote_single_file_asset(
            model_id=notes_specs[0].model_id,
            kind="notes",
            relative_path=notes_specs[0].relative_path.as_posix(),
            repo_id=DEFAULT_NOTES_MODEL_4B_REPO,
            revision=DEFAULT_NOTES_MODEL_4B_REVISION,
            filename=DEFAULT_NOTES_MODEL_4B_FILE,
            default_install=True,
        ),
        _remote_single_file_asset(
            model_id=notes_specs[1].model_id,
            kind="notes",
            relative_path=notes_specs[1].relative_path.as_posix(),
            repo_id=DEFAULT_NOTES_MODEL_2B_REPO,
            revision=DEFAULT_NOTES_MODEL_2B_REVISION,
            filename=DEFAULT_NOTES_MODEL_2B_FILE,
            default_install=False,
        ),
        _remote_directory_asset(
            model_id=DEFAULT_PARAKEET_MODEL_REPO,
            kind="transcription",
            relative_path=transcription_specs[DEFAULT_PARAKEET_MODEL_REPO],
            repo_id=DEFAULT_PARAKEET_MODEL_REPO,
            revision=DEFAULT_PARAKEET_MODEL_REVISION,
            required_files=DEFAULT_PARAKEET_REQUIRED_FILES,
            default_install=True,
        ),
        _remote_directory_asset(
            model_id=DEFAULT_CANARY_MODEL_REPO,
            kind="transcription",
            relative_path=transcription_specs[DEFAULT_CANARY_MODEL_REPO],
            repo_id=DEFAULT_CANARY_MODEL_REPO,
            revision=DEFAULT_CANARY_MODEL_REVISION,
            required_files=DEFAULT_CANARY_REQUIRED_FILES,
            default_install=False,
        ),
    ]
    return PackagedAssetsManifest(schema_version=PACKAGED_ASSET_SCHEMA_VERSION, assets=tuple(assets))


def _stage_runtime_assets(stage_dir: Path, build_inputs: ResolvedBuildInputs) -> None:
    """Write prompt, runtime helpers, and model manifest into the packaged bundle."""
    stage_models_dir = stage_dir / "models"
    if stage_models_dir.exists():
        shutil.rmtree(stage_models_dir)

    _copy_file(PROMPT_SOURCE_PATH, stage_dir / "prompts" / "clinical_note_synthesis_llm_prompt.md")
    _copy_llama_runtime_files(llama_runtime_dir=build_inputs.llama_runtime_dir, stage_dir=stage_dir)
    manifest = _build_packaged_assets_manifest()
    write_packaged_asset_manifest(manifest, stage_dir / PACKAGED_ASSET_MANIFEST_FILENAME)


def _validate_stage_bundle(stage_dir: Path) -> None:
    """Validate the staged Windows bundle before building the installer."""
    missing = [str(relative_path) for relative_path in REQUIRED_STAGE_FILES if not (stage_dir / relative_path).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Staged Windows bundle is missing required files: {missing_list}")

def _build_nuitka_command(*, nuitka_command: Sequence[str], build_dir: Path, version: str) -> list[str]:
    """Build the simplified Nuitka standalone command line."""
    build_dir.mkdir(parents=True, exist_ok=True)
    report_path = _default_nuitka_report_path(build_dir)
    metadata_names = _nuitka_distribution_metadata_names(build_dir)

    command = [
        *nuitka_command,
        "--standalone",
        "--assume-yes-for-downloads",
        f"--output-dir={build_dir}",
        "--output-filename=transcribe.exe",
        f"--company-name={APP_PUBLISHER}",
        f"--product-name={APP_NAME}",
        f"--file-version={_windows_file_version(version)}",
        f"--product-version={_windows_file_version(version)}",
        "--include-package=transcribe",
        "--include-module=transcribe.audio.windows_capture",
        "--include-module=transcribe.audio.backend_loader",
        "--include-module=transcribe.notes",
        "--include-module=transcribe.packaged_assets",
        "--include-module=transcribe.packaged_cli",
        "--include-module=transcribe.packaged_ui",
        "--include-module=transcribe.transcription_runtime",
        "--include-module=huggingface_hub",
        "--include-module=huggingface_hub.file_download",
        "--include-module=nemo.collections.asr",
        "--include-module=nemo.collections.speechlm2.models",
        "--include-module=omegaconf",
        "--include-package=soundcard",
        "--include-package-data=soundcard",
        "--enable-plugin=tk-inter",
        "--noinclude-numba-mode=nofollow",
        "--module-parameter=numba-disable-jit=yes",
        "--noinclude-setuptools-mode=nofollow",
        f"--user-package-configuration-file={NUITKA_PACKAGE_CONFIG_PATH}",
        f"--report={report_path}",
        "--nofollow-import-to=datasets,pyarrow,matplotlib,_pytest,coverage,mypy,IPython,pytest,transcribe.bench,transcribe.test_cov,torch.utils.cpp_extension,setuptools",
    ]
    if _supports_nuitka_option(nuitka_command, "--windows-console-mode="):
        command.append("--windows-console-mode=attach")
    for distribution_name in metadata_names:
        command.append(f"--include-distribution-metadata={distribution_name}")
    command.append(str(REPO_ROOT / "packaged_main.py"))
    return command


def _resolve_nuitka_bundle_dir(build_dir: Path) -> Path:
    """Resolve the standalone output directory produced by Nuitka."""
    candidates = sorted(path for path in build_dir.glob("*.dist") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No Nuitka standalone bundle directory was created in {build_dir}")
    return candidates[-1]


def _build_nuitka_bundle(stage_dir: Path, build_dir: Path, build_inputs: ResolvedBuildInputs) -> None:
    """Build the Windows standalone application with Nuitka."""
    if build_dir.exists():
        shutil.rmtree(build_dir)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    command = _build_nuitka_command(
        nuitka_command=build_inputs.nuitka_command,
        build_dir=build_dir,
        version=build_inputs.version,
    )
    _run(command, cwd=REPO_ROOT)
    bundle_dir = _resolve_nuitka_bundle_dir(build_dir)
    shutil.move(str(bundle_dir), str(stage_dir))


def _sanitize_version(value: str) -> str:
    """Normalize one user-facing version string."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("version must be a non-empty string")
    if normalized.startswith("v") and len(normalized) > 1:
        normalized = normalized[1:]
    return normalized


def _windows_file_version(value: str) -> str:
    """Convert one user-facing version string into a Windows file-version token."""
    numeric_parts = re.findall(r"\d+", value)
    if not numeric_parts:
        return "0.0.0.0"
    limited_parts = [str(int(part)) for part in numeric_parts[:4]]
    while len(limited_parts) < 4:
        limited_parts.append("0")
    return ".".join(limited_parts)


def _safe_filename_component(value: str) -> str:
    """Return one filesystem-safe token derived from user input."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-") or "build"


def _build_inno_installer(stage_dir: Path, build_inputs: ResolvedBuildInputs) -> Path:
    """Build the one-file Windows installer from the staged app directory."""
    INSTALLER_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in INSTALLER_DIR.iterdir():
        if candidate.is_dir():
            shutil.rmtree(candidate)
        else:
            candidate.unlink()

    output_base_filename = f"transcribe-{_safe_filename_component(build_inputs.version)}-setup"
    _run(
        [
            build_inputs.inno_setup_exe,
            f"/DSourceDir={stage_dir}",
            f"/DOutputDir={INSTALLER_DIR}",
            f"/DAppVersion={build_inputs.version}",
            f"/DOutputBaseFilename={output_base_filename}",
            str(INNO_SCRIPT_PATH),
        ],
        cwd=REPO_ROOT,
    )

    installer_candidates = sorted(INSTALLER_DIR.glob("*.exe"))
    if not installer_candidates:
        raise FileNotFoundError(f"No installer executable was created in {INSTALLER_DIR}")
    return installer_candidates[-1].resolve()


def _clear_directory_contents(directory: Path) -> None:
    """Delete all children inside one directory while keeping the directory itself."""
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _publish_release_installer(installer_path: Path, version: str) -> Path:
    """Replace the releases folder contents with the newly built installer."""
    _clear_directory_contents(RELEASES_DIR)
    destination = (RELEASES_DIR / f"Transcribe-{_safe_filename_component(version)}.exe").resolve()
    shutil.copy2(installer_path.resolve(), destination)
    return destination


def _resolve_build_inputs(version: str) -> ResolvedBuildInputs:
    """Resolve toolchain and runtime inputs for the Windows build."""
    _ensure_supported_python()
    _require_repo_runtime_dependencies()
    nuitka_command = _require_nuitka()
    inno_setup_exe = _require_inno_setup()
    llama_runtime_dir = _download_llama_cpp_runtime()
    return ResolvedBuildInputs(
        version=version,
        inno_setup_exe=inno_setup_exe,
        nuitka_command=nuitka_command,
        llama_runtime_dir=llama_runtime_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the simplified Windows installer flow."""
    parser = argparse.ArgumentParser(
        description=(
            "Build one Windows installer executable that installs Transcribe, "
            "with the customer installer downloading the default packaged models on first install."
        )
    )
    parser.add_argument(
        "--version",
        default=PACKAGE_VERSION,
        help="Version label used for the installer metadata and final release filename.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the simplified Windows standalone installer."""
    if os.name != "nt":
        raise RuntimeError("This builder only supports Windows hosts.")

    args = build_parser().parse_args(argv)
    version = _sanitize_version(args.version)
    build_inputs = _resolve_build_inputs(version)

    STAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _build_nuitka_bundle(STAGE_DIR, NUITKA_BUILD_DIR, build_inputs)
    _stage_runtime_assets(STAGE_DIR, build_inputs)
    _validate_stage_bundle(STAGE_DIR)
    installer_path = _build_inno_installer(STAGE_DIR, build_inputs)
    release_path = _publish_release_installer(installer_path, version)

    print(f"Staged app: {STAGE_DIR}")
    print(f"Installer: {installer_path}")
    print(f"Release: {release_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



