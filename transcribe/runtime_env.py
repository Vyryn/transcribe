from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from transcribe.packaged_assets import (
    INSTALLED_ASSET_STATE_FILENAME,
    PACKAGED_ASSET_MANIFEST_FILENAME,
    load_packaged_asset_manifest,
)
from transcribe.runtime_defaults import (
    ALTERNATE_SESSION_NOTES_MODEL,
    DEFAULT_LIVE_TRANSCRIPTION_MODEL,
    DEFAULT_SESSION_NOTES_MODEL,
)

PACKAGED_RUNTIME_ENV = "TRANSCRIBE_PACKAGED"
APP_ROOT_ENV = "TRANSCRIBE_APP_ROOT"
DATA_ROOT_ENV = "TRANSCRIBE_DATA_DIR"
ALLOW_NETWORK_ENV = "TRANSCRIBE_ALLOW_NETWORK"

PACKAGED_ACCURACY_TRANSCRIPTION_MODEL = "nvidia/canary-qwen-2.5b"
PACKAGED_GRANITE_TRANSCRIPTION_MODEL = "ibm-granite/granite-4.0-1b-speech"


class RuntimeMode(StrEnum):
    """Execution-mode categories used to resolve bundled assets."""

    DEVELOPMENT = "development"
    PACKAGED = "packaged"


@dataclass(frozen=True, slots=True)
class BundledModelSpec:
    """One packaged model artifact resolved relative to the models root."""

    model_id: str
    relative_path: Path


@dataclass(frozen=True, slots=True)
class BundledBinarySpec:
    """One packaged runtime binary resolved relative to the install root."""

    logical_name: str
    relative_path: Path


@dataclass(frozen=True, slots=True)
class AppRuntimePaths:
    """Resolved filesystem layout for development or packaged execution."""

    mode: RuntimeMode
    install_root: Path
    data_root: Path
    runtime_root: Path
    models_root: Path
    prompt_root: Path
    packaged_assets_manifest_path: Path
    installed_assets_state_path: Path
    notes_runtime_binary: Path
    notes_prompt_path: Path
    notes_models: dict[str, Path]
    transcription_models: dict[str, Path]


def is_frozen_app() -> bool:
    """Return True when running from a frozen desktop bundle."""
    return bool(getattr(sys, "frozen", False))


def detect_runtime_mode() -> RuntimeMode:
    """Return the effective runtime mode for the current process."""
    raw = os.environ.get(PACKAGED_RUNTIME_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on", "packaged"}:
        return RuntimeMode.PACKAGED
    if is_frozen_app():
        return RuntimeMode.PACKAGED
    return RuntimeMode.DEVELOPMENT


def network_access_allowed() -> bool:
    """Return whether this process should permit outbound network access.

    Returns
    -------
    bool
        ``True`` when network-enabled UI flows should be allowed.
    """
    raw = os.environ.get(ALLOW_NETWORK_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def set_network_access_allowed(allowed: bool) -> None:
    """Persist the process-level network-access preference.

    Parameters
    ----------
    allowed : bool
        Desired outbound-network preference for subsequent runtime helpers.
    """
    os.environ[ALLOW_NETWORK_ENV] = "1" if allowed else "0"


def default_install_root() -> Path:
    """Return the install root used for bundled binaries and assets."""
    configured = os.environ.get(APP_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if is_frozen_app():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_data_root(*, mode: RuntimeMode, install_root: Path) -> Path:
    """Return the writable application data directory."""
    configured = os.environ.get(DATA_ROOT_ENV, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if mode == RuntimeMode.PACKAGED and os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data).expanduser().resolve() / "Transcribe"
    if mode == RuntimeMode.PACKAGED:
        return install_root / "data"
    return Path("data").resolve()


def _notes_binary_specs() -> tuple[BundledBinarySpec, ...]:
    executable_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    return (BundledBinarySpec("llama_server", Path("runtime/llm") / executable_name),)


def _resolve_notes_prompt_path(*, install_root: Path) -> Path:
    """Resolve the packaged notes prompt path, preferring the install root layout."""
    root_prompt_path = install_root / "clinical_note_synthesis_llm_prompt.md"
    prompts_dir_path = install_root / "prompts" / "clinical_note_synthesis_llm_prompt.md"
    if root_prompt_path.exists():
        return root_prompt_path
    if prompts_dir_path.exists():
        return prompts_dir_path
    return root_prompt_path


def bundled_notes_model_specs() -> tuple[BundledModelSpec, ...]:
    """Return the packaged note-model mapping relative to the models root."""
    return (
        BundledModelSpec(
            DEFAULT_SESSION_NOTES_MODEL,
            Path("notes/qwen3.5-4b-q4_k_m.gguf"),
        ),
        BundledModelSpec(
            ALTERNATE_SESSION_NOTES_MODEL,
            Path("notes/qwen3.5-2b-q4_k_m.gguf"),
        ),
    )


def bundled_transcription_model_specs() -> tuple[BundledModelSpec, ...]:
    """Return the packaged ASR model mapping relative to the models root."""
    return (
        BundledModelSpec(
            DEFAULT_LIVE_TRANSCRIPTION_MODEL,
            Path("asr/nvidia/parakeet-tdt-0.6b-v3"),
        ),
        BundledModelSpec(
            PACKAGED_ACCURACY_TRANSCRIPTION_MODEL,
            Path("asr/nvidia/canary-qwen-2.5b"),
        ),
        BundledModelSpec(
            PACKAGED_GRANITE_TRANSCRIPTION_MODEL,
            Path("asr/ibm-granite/granite-4.0-1b-speech"),
        ),
    )


def _manifest_model_specs(manifest_path: Path, *, kind: str) -> tuple[BundledModelSpec, ...]:
    manifest = load_packaged_asset_manifest(manifest_path)
    specs: list[BundledModelSpec] = []
    for asset in manifest.assets:
        if asset.kind != kind:
            continue
        specs.append(BundledModelSpec(asset.model_id, Path(asset.relative_path.replace("/", os.sep))))
    return tuple(specs)


def _resolve_model_specs(
    *,
    mode: RuntimeMode,
    manifest_path: Path,
) -> tuple[tuple[BundledModelSpec, ...], tuple[BundledModelSpec, ...]]:
    notes_specs = bundled_notes_model_specs()
    transcription_specs = bundled_transcription_model_specs()
    if mode != RuntimeMode.PACKAGED or not manifest_path.exists():
        return notes_specs, transcription_specs

    manifest_notes = _manifest_model_specs(manifest_path, kind="notes")
    manifest_transcription = _manifest_model_specs(manifest_path, kind="transcription")
    if manifest_notes:
        notes_specs = manifest_notes
    if manifest_transcription:
        transcription_specs = manifest_transcription
    return notes_specs, transcription_specs


def resolve_app_runtime_paths() -> AppRuntimePaths:
    """Resolve install, runtime, and bundled-model paths for the current process."""
    mode = detect_runtime_mode()
    install_root = default_install_root()
    data_root = default_data_root(mode=mode, install_root=install_root)
    runtime_root = install_root / "runtime"
    models_root = (data_root / "models") if mode == RuntimeMode.PACKAGED else (install_root / "models")
    prompt_root = install_root / "prompts"
    manifest_path = install_root / PACKAGED_ASSET_MANIFEST_FILENAME
    installed_assets_state_path = data_root / INSTALLED_ASSET_STATE_FILENAME
    notes_specs, transcription_specs = _resolve_model_specs(mode=mode, manifest_path=manifest_path)

    binary_map = {
        spec.logical_name: install_root / spec.relative_path
        for spec in _notes_binary_specs()
    }
    notes_models = {
        spec.model_id: models_root / spec.relative_path
        for spec in notes_specs
    }
    transcription_models = {
        spec.model_id: models_root / spec.relative_path
        for spec in transcription_specs
    }

    return AppRuntimePaths(
        mode=mode,
        install_root=install_root,
        data_root=data_root,
        runtime_root=runtime_root,
        models_root=models_root,
        prompt_root=prompt_root,
        packaged_assets_manifest_path=manifest_path,
        installed_assets_state_path=installed_assets_state_path,
        notes_runtime_binary=binary_map["llama_server"],
        notes_prompt_path=_resolve_notes_prompt_path(install_root=install_root),
        notes_models=notes_models,
        transcription_models=transcription_models,
    )


def default_notes_runtime() -> str:
    """Return the default notes runtime name for this execution mode."""
    if detect_runtime_mode() == RuntimeMode.PACKAGED:
        return "llama_cpp"
    return "ollama"


def validate_transcription_model_for_runtime(transcription_model: str) -> str:
    """Validate ASR model choice against packaged-build policy."""
    normalized = transcription_model.strip()
    if not normalized:
        raise ValueError("transcription_model must be non-empty")
    if detect_runtime_mode() != RuntimeMode.PACKAGED:
        return normalized

    runtime_paths = resolve_app_runtime_paths()
    supported = runtime_paths.transcription_models
    if normalized not in supported:
        supported_list = ", ".join(sorted(supported))
        raise ValueError(
            f"Packaged runtime supports only packaged ASR models: {supported_list}. "
            f"Received {normalized!r}."
        )
    if not supported[normalized].exists():
        raise ValueError(
            f"Packaged ASR model {normalized!r} is not installed at {supported[normalized]}. "
            f"Run `transcribe models install --model {normalized}`."
        )
    return normalized


def resolve_bundled_notes_model_path(model_id: str, *, runtime_paths: AppRuntimePaths | None = None) -> Path:
    """Return the packaged GGUF path for a notes model identifier."""
    resolved_paths = runtime_paths or resolve_app_runtime_paths()
    path = resolved_paths.notes_models.get(model_id)
    if path is None:
        supported = ", ".join(sorted(resolved_paths.notes_models))
        raise ValueError(f"Unsupported bundled notes model {model_id!r}. Supported: {supported}.")
    if resolved_paths.mode == RuntimeMode.PACKAGED and not path.exists():
        raise ValueError(
            f"Packaged notes model {model_id!r} is not installed at {path}. "
            f"Run `transcribe models install --model {model_id}`."
        )
    return path


def resolve_bundled_transcription_model_path(
    model_id: str,
    *,
    runtime_paths: AppRuntimePaths | None = None,
) -> Path:
    """Return the packaged ASR model path for a model identifier."""
    resolved_paths = runtime_paths or resolve_app_runtime_paths()
    path = resolved_paths.transcription_models.get(model_id)
    if path is None:
        supported = ", ".join(sorted(resolved_paths.transcription_models))
        raise ValueError(f"Unsupported bundled transcription model {model_id!r}. Supported: {supported}.")
    if resolved_paths.mode == RuntimeMode.PACKAGED and not path.exists():
        raise ValueError(
            f"Packaged ASR model {model_id!r} is not installed at {path}. "
            f"Run `transcribe models install --model {model_id}`."
        )
    return path
