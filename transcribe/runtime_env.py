from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from transcribe.runtime_defaults import (
    ALTERNATE_SESSION_NOTES_MODEL,
    DEFAULT_LIVE_TRANSCRIPTION_MODEL,
    DEFAULT_SESSION_NOTES_MODEL,
)

PACKAGED_RUNTIME_ENV = "TRANSCRIBE_PACKAGED"
APP_ROOT_ENV = "TRANSCRIBE_APP_ROOT"
DATA_ROOT_ENV = "TRANSCRIBE_DATA_DIR"

PACKAGED_ACCURACY_TRANSCRIPTION_MODEL = "nvidia/canary-qwen-2.5b"


class RuntimeMode(StrEnum):
    """Execution-mode categories used to resolve bundled assets."""

    DEVELOPMENT = "development"
    PACKAGED = "packaged"


@dataclass(frozen=True, slots=True)
class BundledModelSpec:
    """One packaged model artifact resolved relative to the install root."""

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


def bundled_notes_model_specs() -> tuple[BundledModelSpec, ...]:
    """Return the packaged note-model mapping."""
    return (
        BundledModelSpec(
            DEFAULT_SESSION_NOTES_MODEL,
            Path("models/notes/qwen3.5-4b-q4_k_m.gguf"),
        ),
        BundledModelSpec(
            ALTERNATE_SESSION_NOTES_MODEL,
            Path("models/notes/qwen3.5-2b-q4_k_m.gguf"),
        ),
    )


def bundled_transcription_model_specs() -> tuple[BundledModelSpec, ...]:
    """Return the packaged ASR model mapping."""
    return (
        BundledModelSpec(
            DEFAULT_LIVE_TRANSCRIPTION_MODEL,
            Path("models/asr/nvidia/parakeet-tdt-0.6b-v3"),
        ),
        BundledModelSpec(
            PACKAGED_ACCURACY_TRANSCRIPTION_MODEL,
            Path("models/asr/nvidia/canary-qwen-2.5b"),
        ),
    )


def resolve_app_runtime_paths() -> AppRuntimePaths:
    """Resolve install, runtime, and bundled-model paths for the current process."""
    mode = detect_runtime_mode()
    install_root = default_install_root()
    runtime_root = install_root / "runtime"
    models_root = install_root / "models"
    prompt_root = install_root / "prompts"

    binary_map = {
        spec.logical_name: install_root / spec.relative_path
        for spec in _notes_binary_specs()
    }
    notes_models = {
        spec.model_id: install_root / spec.relative_path
        for spec in bundled_notes_model_specs()
    }
    transcription_models = {
        spec.model_id: install_root / spec.relative_path
        for spec in bundled_transcription_model_specs()
    }

    return AppRuntimePaths(
        mode=mode,
        install_root=install_root,
        data_root=default_data_root(mode=mode, install_root=install_root),
        runtime_root=runtime_root,
        models_root=models_root,
        prompt_root=prompt_root,
        notes_runtime_binary=binary_map["llama_server"],
        notes_prompt_path=prompt_root / "clinical_note_synthesis_llm_prompt.md",
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

    supported = {spec.model_id for spec in bundled_transcription_model_specs()}
    if normalized in supported:
        return normalized
    supported_list = ", ".join(sorted(supported))
    raise ValueError(
        f"Packaged runtime supports only bundled ASR models: {supported_list}. "
        f"Received {normalized!r}."
    )


def resolve_bundled_notes_model_path(model_id: str, *, runtime_paths: AppRuntimePaths | None = None) -> Path:
    """Return the packaged GGUF path for a notes model identifier."""
    resolved_paths = runtime_paths or resolve_app_runtime_paths()
    path = resolved_paths.notes_models.get(model_id)
    if path is None:
        supported = ", ".join(sorted(resolved_paths.notes_models))
        raise ValueError(f"Unsupported bundled notes model {model_id!r}. Supported: {supported}.")
    return path
