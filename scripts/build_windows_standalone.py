from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transcribe.packaged_assets import PackagedAssetsManifest  # noqa: E402
from transcribe.packaged_asset_defaults import build_default_packaged_assets_manifest  # noqa: E402

APP_NAME = "Transcribe"
MAIN_PATH = REPO_ROOT / "main.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
PROMPT_SOURCE_PATH = REPO_ROOT / "clinical_note_synthesis_llm_prompt.md"
RELEASES_DIR = REPO_ROOT / "releases"
WINDOWS_PACKAGING_DIR = REPO_ROOT / "packaging" / "windows"
INSTALLER_TEMPLATE_PATH = WINDOWS_PACKAGING_DIR / "transcribe_installer.iss"
ICON_PATH = WINDOWS_PACKAGING_DIR / "transcribe.ico"
BUILD_ROOT = REPO_ROOT / "build" / "windows_standalone"
DEFAULT_BUILD_SUBDIR = "pyinstaller"
DEFAULT_LLAMA_CPP_REPO = "ggml-org/llama.cpp"
DEFAULT_LLAMA_CPP_RELEASE = "latest"
DEFAULT_NOTES_MODEL_4B_REPO = "unsloth/Qwen3.5-4B-GGUF"
DEFAULT_NOTES_MODEL_4B_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
DEFAULT_NOTES_MODEL_4B_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_NOTES_MODEL_2B_REPO = "unsloth/Qwen3.5-2B-GGUF"
DEFAULT_NOTES_MODEL_2B_REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
DEFAULT_NOTES_MODEL_2B_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_PARAKEET_MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PARAKEET_MODEL_REVISION = "6d590f77001d318fb17a0b5bf7ee329a91b52598"
DEFAULT_PARAKEET_REQUIRED_FILES = ("parakeet-tdt-0.6b-v3.nemo",)
DEFAULT_CANARY_MODEL_REPO = "nvidia/canary-qwen-2.5b"
DEFAULT_CANARY_MODEL_REVISION = "6cfc37ec7edc35a0545c403f551ecdfa28133d72"
DEFAULT_CANARY_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "LICENSES",
    "model.safetensors",
    "tokenizer.model",
)
DEFAULT_GRANITE_MODEL_REPO = "ibm-granite/granite-4.0-1b-speech"
DEFAULT_GRANITE_MODEL_REVISION = "4eaf14d77837c989d00f59c26262b6b9d10a9091"
DEFAULT_GRANITE_REQUIRED_FILES = (
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
DEFAULT_GITHUB_API_ACCEPT = "application/vnd.github+json"
DEFAULT_GITHUB_USER_AGENT = "transcribe-windows-builder"
DEFAULT_INNO_SETUP_DOWNLOAD_URL = "https://jrsoftware.org/download.php/is.exe"
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
DIST_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
LLAMA_RUNTIME_ARCHIVE_PATTERNS = (
    re.compile(r"llama-.*-bin-win-cpu-x64\.zip$", re.IGNORECASE),
    re.compile(r".*win.*cpu.*x64.*\.zip$", re.IGNORECASE),
    re.compile(r".*(?:llama|cudart).*win.*x64.*\.zip$", re.IGNORECASE),
)
PYINSTALLER_EXTRA_MODULES = (
    "ctranslate2",
    "datasets",
    "editdistance",
    "faster_whisper",
    "hf_xet",
    "huggingface_hub",
    "hydra",
    "jiwer",
    "librosa",
    "lightning",
    "lightning_fabric",
    "lightning_utilities",
    "lhotse",
    "nemo",
    "omegaconf",
    "qwen_asr",
    "sentencepiece",
    "soundcard",
    "sounddevice",
    "tokenizers",
    "torch",
    "torchaudio",
    "torchcodec",
    "transformers",
    "transcribe",
    "yaml",
)
PYINSTALLER_HIDDEN_IMPORTS = (
    "transcribe.audio.backend_loader",
    "transcribe.audio.windows_capture",
    "transcribe.notes",
    "transcribe.packaged_assets",
    "transcribe.packaged_cli",
    "transcribe.packaged_ui",
    "transcribe.transcription_runtime",
    "huggingface_hub.file_download",
    "nemo.collections.asr",
    "nemo.collections.speechlm2.models",
    "omegaconf",
    "yaml",
)
PYINSTALLER_EXCLUDED_MODULES = (
    "_pytest",
    "coverage",
    "mypy",
    "pytest",
    "transcribe.bench",
    "transcribe.test_cov",
)


@dataclass(frozen=True, slots=True)
class GitHubReleaseAsset:
    """One downloadable asset from a GitHub release."""

    name: str
    download_url: str


@dataclass(frozen=True, slots=True)
class BuildLayout:
    """Filesystem layout for one Windows packaging run."""

    root: Path
    downloads_dir: Path
    pyinstaller_work_dir: Path
    pyinstaller_dist_dir: Path
    stage_dir: Path
    stage_prompts_dir: Path
    stage_runtime_dir: Path
    installer_dir: Path
    spec_path: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the Windows packaging CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build a Windows installer that packages the frozen Transcribe app, "
            "bundles llama.cpp runtime files, and downloads selected models during setup."
        )
    )
    parser.add_argument(
        "version",
        help="Semver version label used for the installer metadata and final release filename.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=BUILD_ROOT,
        help=f"Working build root (default: {BUILD_ROOT})",
    )
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=RELEASES_DIR,
        help=f"Final release output directory (default: {RELEASES_DIR})",
    )
    parser.add_argument(
        "--llama-runtime-dir",
        type=Path,
        default=None,
        help="Optional local directory containing llama.cpp runtime files to stage instead of downloading them.",
    )
    parser.add_argument(
        "--llama-cpp-release",
        default=DEFAULT_LLAMA_CPP_RELEASE,
        help=f"llama.cpp GitHub release tag to use when downloading runtime files (default: {DEFAULT_LLAMA_CPP_RELEASE})",
    )
    parser.add_argument(
        "--inno-setup-exe",
        type=Path,
        default=None,
        help="Optional explicit path to ISCC.exe. When omitted, common Inno Setup install paths are searched.",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip `uv sync` before building. Use only when the environment already has all direct and optional dependencies installed.",
    )
    parser.add_argument(
        "--skip-installer",
        action="store_true",
        help="Build and stage the frozen app but skip Inno Setup compilation.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete any existing build directory for this version before packaging.",
    )
    return parser


def sanitize_version(version: str) -> str:
    """Validate and normalize a semver version string.

    Parameters
    ----------
    version : str
        Candidate version label.

    Returns
    -------
    str
        Trimmed semver version.
    """
    normalized = version.strip()
    if not SEMVER_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"Version must be semver formatted like 1.2.3 or 1.2.3-rc.1. Received {version!r}."
        )
    return normalized


def safe_filename_component(value: str) -> str:
    """Return a Windows-safe filename fragment.

    Parameters
    ----------
    value : str
        Raw text to sanitize.

    Returns
    -------
    str
        Filename-safe fragment with separators normalized.
    """
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip(".-")
    if not normalized:
        raise ValueError("Filename component must not be empty after sanitization.")
    return normalized


def resolve_build_layout(*, build_root: Path, version: str) -> BuildLayout:
    """Resolve the working directory layout for one build.

    Parameters
    ----------
    build_root : Path
        Root directory for build artifacts.
    version : str
        Sanitized semver version.

    Returns
    -------
    BuildLayout
        Concrete build paths for this packaging run.
    """
    root = build_root.resolve() / safe_filename_component(version)
    pyinstaller_root = root / DEFAULT_BUILD_SUBDIR
    stage_dir = root / "stage"
    return BuildLayout(
        root=root,
        downloads_dir=root / "downloads",
        pyinstaller_work_dir=pyinstaller_root / "work",
        pyinstaller_dist_dir=pyinstaller_root / "dist",
        stage_dir=stage_dir,
        stage_prompts_dir=stage_dir / "prompts",
        stage_runtime_dir=stage_dir / "runtime" / "llm",
        installer_dir=root / "installer",
        spec_path=root / "transcribe_pyinstaller.spec",
    )


def clear_directory_contents(path: Path) -> None:
    """Delete one directory tree when it exists.

    Parameters
    ----------
    path : Path
        Directory to remove.
    """
    if path.exists():
        shutil.rmtree(path)


def ensure_directory(path: Path) -> Path:
    """Create one directory and return the resolved path.

    Parameters
    ----------
    path : Path
        Directory path to create.

    Returns
    -------
    Path
        Resolved directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess command with streaming-friendly logging.

    Parameters
    ----------
    command : Sequence[str | Path]
        Command and arguments to execute.
    cwd : Path, optional
        Working directory for the subprocess.
    env : Mapping[str, str] | None, optional
        Optional environment overrides.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed process result.
    """
    argv = [str(part) for part in command]
    print(f"Running: {' '.join(argv)}")
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=None if env is None else dict(env),
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if stdout.strip():
        print(stdout.rstrip())
    if completed.returncode == 0:
        if stderr.strip():
            print(stderr.rstrip())
        return completed
    detail = stderr.strip() or stdout.strip() or f"exit code {completed.returncode}"
    raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(argv)}\n{detail}")


def load_pyproject(path: Path = PYPROJECT_PATH) -> dict[str, object]:
    """Load the repository ``pyproject.toml``.

    Parameters
    ----------
    path : Path, optional
        TOML file to load.

    Returns
    -------
    dict[str, object]
        Parsed project metadata.
    """
    return tomllib.loads(path.read_text(encoding="utf-8"))


def parse_requirement_name(requirement: str) -> str:
    """Extract a distribution name from a PEP 508 requirement string.

    Parameters
    ----------
    requirement : str
        Raw requirement expression.

    Returns
    -------
    str
        Distribution name portion.
    """
    candidate = requirement.strip()
    if not candidate:
        raise ValueError("Requirement string must not be empty.")
    match = DIST_NAME_PATTERN.match(candidate)
    if match is None:
        raise ValueError(f"Unable to parse requirement name from {requirement!r}.")
    return match.group(0)


def project_runtime_distribution_names(pyproject: Mapping[str, object]) -> tuple[str, ...]:
    """Return direct runtime distribution names from dependencies and optional dependencies.

    Parameters
    ----------
    pyproject : Mapping[str, object]
        Parsed ``pyproject.toml`` structure.

    Returns
    -------
    tuple[str, ...]
        Sorted unique distribution names required for packaged runtime builds.
    """
    project = pyproject.get("project")
    if not isinstance(project, Mapping):
        raise ValueError("pyproject.toml is missing a [project] table.")
    raw_dependencies = project.get("dependencies", [])
    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(raw_dependencies, list):
        raise ValueError("[project].dependencies must be a list.")
    if not isinstance(optional_dependencies, Mapping):
        raise ValueError("[project.optional-dependencies] must be a table.")

    names: set[str] = set()
    for raw_requirement in raw_dependencies:
        if isinstance(raw_requirement, str):
            names.add(parse_requirement_name(raw_requirement))
    for raw_group in optional_dependencies.values():
        if not isinstance(raw_group, list):
            continue
        for raw_requirement in raw_group:
            if isinstance(raw_requirement, str):
                names.add(parse_requirement_name(raw_requirement))
    return tuple(sorted(names))


def project_optional_dependency_groups(pyproject: Mapping[str, object]) -> tuple[str, ...]:
    """Return all optional dependency group names from ``pyproject.toml``.

    Parameters
    ----------
    pyproject : Mapping[str, object]
        Parsed ``pyproject.toml`` structure.

    Returns
    -------
    tuple[str, ...]
        Sorted extra names.
    """
    project = pyproject.get("project")
    if not isinstance(project, Mapping):
        raise ValueError("pyproject.toml is missing a [project] table.")
    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, Mapping):
        raise ValueError("[project.optional-dependencies] must be a table.")
    return tuple(sorted(str(extra) for extra in optional_dependencies))


def sync_project_environment(pyproject: Mapping[str, object]) -> None:
    """Install direct and optional runtime dependencies into the project environment.

    Parameters
    ----------
    pyproject : Mapping[str, object]
        Parsed project metadata used to enumerate extras.
    """
    extras = project_optional_dependency_groups(pyproject)
    command: list[str | Path] = ["uv", "sync", "--frozen"]
    for extra in extras:
        command.extend(["--extra", extra])
    run_command(command)


def resolve_build_python() -> Path:
    """Resolve the Python interpreter that should run PyInstaller.

    Returns
    -------
    Path
        Preferred project environment Python interpreter.
    """
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python.resolve()
    return Path(sys.executable).resolve()


def ensure_pip_available(python_executable: Path) -> None:
    """Ensure ``pip`` is importable for the chosen build interpreter.

    Parameters
    ----------
    python_executable : Path
        Interpreter used for packaging tools.
    """
    try:
        run_command([python_executable, "-m", "pip", "--version"])
    except RuntimeError:
        run_command([python_executable, "-m", "ensurepip", "--upgrade"])


def ensure_pyinstaller_available(python_executable: Path) -> None:
    """Install PyInstaller into the build interpreter when needed.

    Parameters
    ----------
    python_executable : Path
        Interpreter used for packaging tools.
    """
    try:
        run_command([python_executable, "-m", "PyInstaller", "--version"])
        return
    except RuntimeError:
        pass
    ensure_pip_available(python_executable)
    run_command([python_executable, "-m", "pip", "install", "pyinstaller"])


def normalize_distribution_name(name: str) -> str:
    """Normalize a distribution name using PEP 503 rules.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str
        Normalized distribution name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def iter_distribution_top_level_candidates(
    distribution: importlib.metadata.Distribution,
) -> tuple[str, ...]:
    """Infer top-level import names from installed distribution metadata.

    Parameters
    ----------
    distribution : importlib.metadata.Distribution
        Installed distribution metadata handle.

    Returns
    -------
    tuple[str, ...]
        Unique candidate import roots discovered from the distribution metadata.
    """
    candidates: list[str] = []
    top_level_text = distribution.read_text("top_level.txt")
    if top_level_text:
        for line in top_level_text.splitlines():
            module_name = line.strip()
            if module_name:
                candidates.append(module_name)

    files = distribution.files or ()
    package_roots: set[str] = set()
    package_markers: set[str] = set()
    module_roots: set[str] = set()
    for package_file in files:
        parts = package_file.parts
        if not parts:
            continue
        top_level_name = parts[0]
        if top_level_name.endswith(".dist-info") or top_level_name.endswith(".data"):
            continue
        if len(parts) == 1 and top_level_name.endswith(".py"):
            module_roots.add(Path(top_level_name).stem)
            continue
        if len(parts) > 1 and parts[-1] == "__init__.py":
            package_markers.add(top_level_name)
        package_roots.add(top_level_name)

    candidates.extend(sorted(package_markers))
    candidates.extend(sorted(module_roots))
    candidates.extend(sorted(package_roots))
    return tuple(dict.fromkeys(candidates))


def find_module_spec(module_name: str) -> importlib.machinery.ModuleSpec | None:
    """Return a module spec when one import path is importable.

    Parameters
    ----------
    module_name : str
        Candidate module path to probe.

    Returns
    -------
    importlib.machinery.ModuleSpec | None
        Module spec when import resolution succeeds, otherwise ``None``.
    """
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def distribution_top_level_packages(distribution_name: str) -> tuple[str, ...]:
    """Return import roots associated with one installed distribution.

    Parameters
    ----------
    distribution_name : str
        Distribution name from dependency metadata.

    Returns
    -------
    tuple[str, ...]
        Importable top-level module names.
    """
    candidate_modules: list[str] = []
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        distribution = None

    if distribution is not None:
        candidate_modules.extend(iter_distribution_top_level_candidates(distribution))
    filtered: list[str] = []
    for module_name in candidate_modules:
        normalized_module = module_name.strip()
        if not normalized_module or normalized_module in filtered:
            continue
        if find_module_spec(normalized_module) is None:
            continue
        filtered.append(normalized_module)
    return tuple(filtered)


def build_pyinstaller_module_roots(distribution_names: Iterable[str]) -> tuple[str, ...]:
    """Resolve import roots to collect for the frozen application.

    Parameters
    ----------
    distribution_names : Iterable[str]
        Direct distribution names from the project runtime dependency graph.

    Returns
    -------
    tuple[str, ...]
        Sorted unique module roots that are importable in the current environment.
    """
    module_roots: set[str] = set(PYINSTALLER_EXTRA_MODULES)
    for distribution_name in distribution_names:
        module_roots.update(distribution_top_level_packages(distribution_name))
    return tuple(
        sorted(
            module_root
            for module_root in module_roots
            if find_module_spec(module_root) is not None
        )
    )


def missing_runtime_distributions(distribution_names: Iterable[str]) -> tuple[str, ...]:
    """Return direct distributions missing from the active environment.

    Parameters
    ----------
    distribution_names : Iterable[str]
        Direct distribution names that should be installed.

    Returns
    -------
    tuple[str, ...]
        Missing distribution names.
    """
    missing: list[str] = []
    for distribution_name in distribution_names:
        try:
            importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution_name)
    return tuple(sorted(missing))


def require_repo_runtime_dependencies(distribution_names: Iterable[str]) -> None:
    """Fail fast when required runtime distributions are missing.

    Parameters
    ----------
    distribution_names : Iterable[str]
        Direct runtime distributions expected in the build environment.
    """
    missing = missing_runtime_distributions(distribution_names)
    if not missing:
        return
    raise RuntimeError(
        "Missing Python runtime dependencies for the Windows standalone build: "
        + ", ".join(missing)
        + ". Run `uv sync --frozen` with the project extras first."
    )


def github_release_api_url(repo: str, release: str) -> str:
    """Build the GitHub releases API URL for one repo and release selector.

    Parameters
    ----------
    repo : str
        ``owner/name`` repository identifier.
    release : str
        Release selector. ``latest`` uses the latest-release endpoint.

    Returns
    -------
    str
        JSON API endpoint URL.
    """
    base = "https://api.github.com/repos/" + repo
    if release == "latest":
        return base + "/releases/latest"
    return base + "/releases/tags/" + urllib.parse.quote(release, safe="")


def fetch_json(url: str) -> dict[str, object]:
    """Fetch and decode one JSON document.

    Parameters
    ----------
    url : str
        Source URL.

    Returns
    -------
    dict[str, object]
        Parsed JSON payload.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Accept": DEFAULT_GITHUB_API_ACCEPT,
            "User-Agent": DEFAULT_GITHUB_USER_AGENT,
        },
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload from {url!r}.")
    return payload


def select_github_asset(
    *,
    repo: str,
    release: str,
    patterns: Sequence[re.Pattern[str]],
) -> GitHubReleaseAsset:
    """Select the first GitHub release asset whose name matches a preferred pattern.

    Parameters
    ----------
    repo : str
        ``owner/name`` repository identifier.
    release : str
        Release selector such as ``latest`` or a specific tag.
    patterns : Sequence[re.Pattern[str]]
        Asset-name patterns in preference order.

    Returns
    -------
    GitHubReleaseAsset
        Matching release asset metadata.
    """
    payload = fetch_json(github_release_api_url(repo, release))
    raw_assets = payload.get("assets", [])
    if not isinstance(raw_assets, list):
        raise RuntimeError(f"GitHub release payload for {repo!r} did not include an asset list.")
    assets: list[GitHubReleaseAsset] = []
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            continue
        name = raw_asset.get("name")
        download_url = raw_asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(download_url, str):
            assets.append(GitHubReleaseAsset(name=name, download_url=download_url))

    for pattern in patterns:
        for asset in assets:
            if pattern.search(asset.name):
                return asset
    available = ", ".join(asset.name for asset in assets)
    raise RuntimeError(
        f"Unable to find a matching GitHub release asset for {repo!r} release {release!r}. "
        f"Available assets: {available}"
    )


def download_file(*, url: str, destination: Path) -> Path:
    """Download one file to disk.

    Parameters
    ----------
    url : str
        Source URL.
    destination : Path
        Destination file path.

    Returns
    -------
    Path
        Resolved downloaded file path.
    """
    ensure_directory(destination.parent)
    request = urllib.request.Request(url, headers={"User-Agent": DEFAULT_GITHUB_USER_AGENT})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination.resolve()


def inno_setup_install_dir() -> Path:
    """Return the preferred per-user Inno Setup installation directory.

    Returns
    -------
    Path
        Preferred installation directory for bootstrap installs.
    """
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        return Path(local_app_data) / "Programs" / "Inno Setup 6"
    return REPO_ROOT / "build" / "tools" / "Inno Setup 6"


def copy_file(source: Path, destination: Path) -> None:
    """Copy one file and create parent directories first.

    Parameters
    ----------
    source : Path
        Source file path.
    destination : Path
        Destination file path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    """Copy a directory tree into one destination.

    Parameters
    ----------
    source : Path
        Source directory.
    destination : Path
        Destination directory.
    """
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def copy_directory_contents(source: Path, destination: Path) -> None:
    """Copy one directory's contents into an existing destination directory.

    Parameters
    ----------
    source : Path
        Source directory whose direct children should be copied.
    destination : Path
        Destination directory that receives the copied entries.
    """
    resolved_source = source.resolve()
    if not resolved_source.exists() or not resolved_source.is_dir():
        raise FileNotFoundError(resolved_source)
    ensure_directory(destination)
    for child in resolved_source.iterdir():
        target = destination / child.name
        if child.is_dir():
            copy_tree(child, target)
        else:
            copy_file(child, target)


def stage_llama_runtime_from_archive(archive_path: Path, destination_dir: Path) -> None:
    """Extract the required llama.cpp runtime files from a downloaded archive.

    Parameters
    ----------
    archive_path : Path
        Downloaded llama.cpp Windows zip archive.
    destination_dir : Path
        Runtime destination directory inside the staged app bundle.
    """
    clear_directory_contents(destination_dir)
    ensure_directory(destination_dir)
    extract_root = archive_path.parent / f"{archive_path.stem}_extract"
    clear_directory_contents(extract_root)
    ensure_directory(extract_root)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)

    runtime_binary = next(extract_root.rglob("llama-server.exe"), None)
    if runtime_binary is None:
        raise RuntimeError(f"llama.cpp archive did not contain llama-server.exe: {archive_path}")
    runtime_binary_dir = runtime_binary.parent

    copied_runtime_file = False
    for candidate in runtime_binary_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name.lower() == "llama-server.exe" or candidate.suffix.lower() == ".dll":
            copy_file(candidate, destination_dir / candidate.name)
            copied_runtime_file = True

    if not copied_runtime_file:
        raise RuntimeError(f"No usable llama.cpp runtime files were extracted from {archive_path}.")

    for candidate in extract_root.rglob("*"):
        if not candidate.is_file():
            continue
        lowered_name = candidate.name.lower()
        if lowered_name.startswith("license") or lowered_name.startswith("copying"):
            copy_file(candidate, destination_dir / candidate.name)

    clear_directory_contents(extract_root)


def stage_llama_runtime(
    *,
    llama_runtime_dir: Path | None,
    llama_cpp_release: str,
    downloads_dir: Path,
    destination_dir: Path,
) -> None:
    """Stage llama.cpp runtime files into the packaged application directory.

    Parameters
    ----------
    llama_runtime_dir : Path | None
        Optional local runtime directory to reuse instead of downloading from GitHub.
    llama_cpp_release : str
        GitHub release selector used when downloading the runtime archive.
    downloads_dir : Path
        Cache directory for downloaded archives.
    destination_dir : Path
        Runtime destination inside the staged bundle.
    """
    if llama_runtime_dir is not None:
        source_dir = llama_runtime_dir.expanduser().resolve()
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        if not (source_dir / "llama-server.exe").exists():
            raise RuntimeError(
                f"Local llama runtime directory is missing llama-server.exe: {source_dir}"
            )
        copy_tree(source_dir, destination_dir)
        return

    asset = select_github_asset(
        repo=DEFAULT_LLAMA_CPP_REPO,
        release=llama_cpp_release,
        patterns=LLAMA_RUNTIME_ARCHIVE_PATTERNS,
    )
    archive_path = downloads_dir / asset.name
    if not archive_path.exists():
        download_file(url=asset.download_url, destination=archive_path)
    stage_llama_runtime_from_archive(archive_path, destination_dir)


def build_packaged_assets_manifest() -> PackagedAssetsManifest:
    """Build the packaged model manifest used by the Windows installer.

    Returns
    -------
    PackagedAssetsManifest
        Manifest describing install-time model downloads.
    """
    return build_default_packaged_assets_manifest()


def write_pyinstaller_spec(
    *,
    destination: Path,
    package_roots: Sequence[str],
    distribution_names: Sequence[str],
    icon_path: Path,
) -> None:
    """Write the temporary PyInstaller spec file for this build.

    Parameters
    ----------
    destination : Path
        Spec file path to write.
    package_roots : Sequence[str]
        Import roots to collect recursively.
    distribution_names : Sequence[str]
        Distribution names whose metadata should be copied into the frozen build.
    icon_path : Path
        Application icon file.
    """
    destination.write_text(
        "\n".join(
            (
                "# -*- mode: python ; coding: utf-8 -*-",
                "from __future__ import annotations",
                "",
                "from pathlib import Path",
                "from PyInstaller.utils.hooks import collect_all, copy_metadata",
                "",
                f"repo_root = Path({str(REPO_ROOT)!r})",
                f"main_path = Path({str(MAIN_PATH)!r})",
                f"icon_path = Path({str(icon_path)!r})",
                f"package_roots = {list(package_roots)!r}",
                f"distribution_names = {list(distribution_names)!r}",
                f"hiddenimports = {list(PYINSTALLER_HIDDEN_IMPORTS)!r}",
                "",
                "datas = []",
                "binaries = []",
                "for package_root in package_roots:",
                "    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_root)",
                "    datas.extend(pkg_datas)",
                "    binaries.extend(pkg_binaries)",
                "    hiddenimports.extend(pkg_hiddenimports)",
                "for distribution_name in distribution_names:",
                "    try:",
                "        datas.extend(copy_metadata(distribution_name, recursive=True))",
                "    except Exception:",
                "        pass",
                "hiddenimports = sorted(set(hiddenimports))",
                "",
                "a = Analysis(",
                "    [str(main_path)],",
                "    pathex=[str(repo_root)],",
                "    binaries=binaries,",
                "    datas=datas,",
                "    hiddenimports=hiddenimports,",
                f"    excludes={list(PYINSTALLER_EXCLUDED_MODULES)!r},",
                "    noarchive=False,",
                ")",
                "pyz = PYZ(a.pure)",
                "exe = EXE(",
                "    pyz,",
                "    a.scripts,",
                "    [],",
                f"    name={APP_NAME!r},",
                "    exclude_binaries=True,",
                "    console=False,",
                "    icon=str(icon_path),",
                "    upx=False,",
                ")",
                "coll = COLLECT(",
                "    exe,",
                "    a.binaries,",
                "    a.datas,",
                f"    name={APP_NAME!r},",
                "    strip=False,",
                "    upx=False,",
                ")",
                "",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def build_pyinstaller_bundle(
    *,
    python_executable: Path,
    layout: BuildLayout,
    distribution_names: Sequence[str],
    icon_path: Path,
) -> Path:
    """Build the frozen application bundle directory with PyInstaller.

    Parameters
    ----------
    python_executable : Path
        Build interpreter that has PyInstaller installed.
    layout : BuildLayout
        Concrete build directories for this packaging run.
    distribution_names : Sequence[str]
        Direct runtime distribution names from the project.
    icon_path : Path
        Application icon file.

    Returns
    -------
    Path
        Frozen application bundle directory.
    """
    package_roots = build_pyinstaller_module_roots(distribution_names)
    write_pyinstaller_spec(
        destination=layout.spec_path,
        package_roots=package_roots,
        distribution_names=distribution_names,
        icon_path=icon_path,
    )
    clear_directory_contents(layout.pyinstaller_work_dir)
    clear_directory_contents(layout.pyinstaller_dist_dir)
    ensure_directory(layout.pyinstaller_work_dir)
    ensure_directory(layout.pyinstaller_dist_dir)
    run_command(
        [
            python_executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            layout.pyinstaller_dist_dir,
            "--workpath",
            layout.pyinstaller_work_dir,
            layout.spec_path,
        ]
    )
    executable_path = layout.pyinstaller_dist_dir / APP_NAME / f"{APP_NAME}.exe"
    if not executable_path.exists():
        raise RuntimeError(f"PyInstaller did not produce the expected executable: {executable_path}")
    return executable_path.parent.resolve()


def stage_runtime_assets(*, layout: BuildLayout, args: argparse.Namespace, icon_path: Path) -> None:
    """Stage prompt, icon, runtime files, and install-time manifest.

    Parameters
    ----------
    layout : BuildLayout
        Concrete build directories for this packaging run.
    args : argparse.Namespace
        Parsed build options.
    icon_path : Path
        Application icon file to stage alongside the bundle.
    """
    clear_directory_contents(layout.stage_dir)
    ensure_directory(layout.stage_dir)
    ensure_directory(layout.stage_prompts_dir)
    ensure_directory(layout.stage_runtime_dir)
    copy_file(PROMPT_SOURCE_PATH, layout.stage_prompts_dir / "clinical_note_synthesis_llm_prompt.md")
    copy_file(icon_path, layout.stage_dir / icon_path.name)
    stage_llama_runtime(
        llama_runtime_dir=args.llama_runtime_dir,
        llama_cpp_release=args.llama_cpp_release,
        downloads_dir=ensure_directory(layout.downloads_dir),
        destination_dir=layout.stage_runtime_dir,
    )


def stage_built_app(bundle_dir: Path, stage_dir: Path) -> Path:
    """Copy the frozen application bundle into the installer staging directory.

    Parameters
    ----------
    bundle_dir : Path
        PyInstaller-built application bundle directory.
    stage_dir : Path
        Installer staging directory.

    Returns
    -------
    Path
        Staged application executable path.
    """
    staged_executable = stage_dir / f"{APP_NAME}.exe"
    copy_directory_contents(bundle_dir, stage_dir)
    if not staged_executable.exists():
        raise RuntimeError(f"Staged app bundle is missing the expected executable: {staged_executable}")
    return staged_executable.resolve()


def known_inno_setup_paths() -> tuple[Path, ...]:
    """Return common Windows install paths for ``ISCC.exe``.

    Returns
    -------
    tuple[Path, ...]
        Candidate compiler paths in priority order.
    """
    candidates: list[Path] = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)", "").strip()
    program_files = os.environ.get("ProgramFiles", "").strip()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if program_files_x86:
        candidates.append(Path(program_files_x86) / "Inno Setup 6" / "ISCC.exe")
    if program_files:
        candidates.append(Path(program_files) / "Inno Setup 6" / "ISCC.exe")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Inno Setup 6" / "ISCC.exe")
    iscc_from_path = shutil.which("ISCC.exe")
    if iscc_from_path is not None:
        candidates.append(Path(iscc_from_path))
    deduplicated: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in deduplicated:
            continue
        deduplicated.append(resolved)
    return tuple(deduplicated)


def find_inno_setup_compiler(explicit_path: Path | None) -> Path | None:
    """Return the Inno Setup compiler path when one is available.

    Parameters
    ----------
    explicit_path : Path | None
        Optional explicit path to ``ISCC.exe``.

    Returns
    -------
    Path | None
        Resolved compiler path when available, otherwise ``None``.
    """
    if explicit_path is not None:
        candidate = explicit_path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    for candidate in known_inno_setup_paths():
        if candidate.exists():
            return candidate
    return None


def install_inno_setup(*, downloads_dir: Path) -> Path:
    """Download and install Inno Setup 6 for the current user.

    Parameters
    ----------
    downloads_dir : Path
        Directory used to cache the installer executable.

    Returns
    -------
    Path
        Resolved path to the installed ``ISCC.exe`` compiler.
    """
    installer_path = download_file(
        url=DEFAULT_INNO_SETUP_DOWNLOAD_URL,
        destination=downloads_dir / "innosetup-6-installer.exe",
    )
    install_dir = inno_setup_install_dir()
    run_command(
        [
            installer_path,
            "/SP-",
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            "/CURRENTUSER",
            f"/DIR={install_dir}",
        ]
    )
    iscc_path = install_dir / "ISCC.exe"
    if not iscc_path.exists():
        raise RuntimeError(f"Inno Setup bootstrap install did not create the expected compiler: {iscc_path}")
    return iscc_path.resolve()


def require_inno_setup(*, explicit_path: Path | None, downloads_dir: Path) -> Path:
    """Resolve the Inno Setup compiler, installing it when needed.

    Parameters
    ----------
    explicit_path : Path | None
        Optional explicit path to ``ISCC.exe``.
    downloads_dir : Path
        Directory used to cache the Inno Setup bootstrap installer.

    Returns
    -------
    Path
        Resolved compiler path.
    """
    compiler_path = find_inno_setup_compiler(explicit_path)
    if compiler_path is not None:
        return compiler_path
    if explicit_path is not None:
        raise RuntimeError(f"Unable to find the requested Inno Setup compiler: {explicit_path}")
    return install_inno_setup(downloads_dir=downloads_dir)


def build_inno_installer(
    *,
    iscc_path: Path,
    layout: BuildLayout,
    version: str,
) -> Path:
    """Compile the final single-file Windows installer with Inno Setup.

    Parameters
    ----------
    iscc_path : Path
        Path to ``ISCC.exe``.
    layout : BuildLayout
        Concrete build directories for this packaging run.
    version : str
        Semver version label.

    Returns
    -------
    Path
        Final installer executable path.
    """
    ensure_directory(layout.installer_dir)
    output_base_filename = f"{APP_NAME}-{safe_filename_component(version)}"
    run_command(
        [
            iscc_path,
            f"/DAppVersion={version}",
            f"/DSourceDir={layout.stage_dir}",
            f"/DOutputDir={layout.installer_dir}",
            f"/DOutputBaseFilename={output_base_filename}",
            INSTALLER_TEMPLATE_PATH,
        ]
    )
    installer_path = layout.installer_dir / f"{output_base_filename}.exe"
    if not installer_path.exists():
        raise RuntimeError(f"Inno Setup did not create the expected installer: {installer_path}")
    return installer_path.resolve()


def publish_release_installer(*, installer_path: Path, release_dir: Path) -> Path:
    """Copy the finished installer into the repository release directory.

    Parameters
    ----------
    installer_path : Path
        Built installer executable.
    release_dir : Path
        Final release output directory.

    Returns
    -------
    Path
        Published release artifact path.
    """
    ensure_directory(release_dir)
    published_path = release_dir.resolve() / installer_path.name
    copy_file(installer_path, published_path)
    return published_path


def main(argv: list[str] | None = None) -> int:
    """Run the Windows packaging workflow.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    version = sanitize_version(args.version)
    if os.name != "nt":
        raise SystemExit("Windows packaging is only supported on Windows hosts.")
    if not MAIN_PATH.exists():
        raise SystemExit(f"Missing app entrypoint: {MAIN_PATH}")
    if not PROMPT_SOURCE_PATH.exists():
        raise SystemExit(f"Missing notes prompt source file: {PROMPT_SOURCE_PATH}")
    if not INSTALLER_TEMPLATE_PATH.exists():
        raise SystemExit(f"Missing Inno Setup script template: {INSTALLER_TEMPLATE_PATH}")
    if not ICON_PATH.exists():
        raise SystemExit(f"Missing application icon: {ICON_PATH}")

    layout = resolve_build_layout(build_root=args.build_dir, version=version)
    if args.clean:
        clear_directory_contents(layout.root)
    ensure_directory(layout.root)
    pyproject = load_pyproject()
    distribution_names = project_runtime_distribution_names(pyproject)
    if not args.skip_sync:
        sync_project_environment(pyproject)
    require_repo_runtime_dependencies(distribution_names)

    python_executable = resolve_build_python()
    ensure_pyinstaller_available(python_executable)
    stage_runtime_assets(layout=layout, args=args, icon_path=ICON_PATH)
    bundle_dir = build_pyinstaller_bundle(
        python_executable=python_executable,
        layout=layout,
        distribution_names=distribution_names,
        icon_path=ICON_PATH,
    )
    staged_executable = stage_built_app(bundle_dir, layout.stage_dir)
    print(f"Staged app: {staged_executable}")

    if args.skip_installer:
        print(f"Windows stage: {layout.stage_dir}")
        return 0

    iscc_path = require_inno_setup(explicit_path=args.inno_setup_exe, downloads_dir=layout.downloads_dir)
    installer_path = build_inno_installer(iscc_path=iscc_path, layout=layout, version=version)
    release_path = publish_release_installer(installer_path=installer_path, release_dir=args.release_dir)
    print(f"Installer: {release_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
