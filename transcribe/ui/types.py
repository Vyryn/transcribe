from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from transcribe.models import AudioSourceMode
from transcribe.runtime_defaults import (
    DEFAULT_LIVE_TRANSCRIPTION_MODEL,
    DEFAULT_SESSION_NOTES_MODEL,
)

DEFAULT_BENCH_DATASET = "edinburghcstr/ami"
DEFAULT_BENCH_CONFIG = "ihm"
DEFAULT_BENCH_SPLIT = "test"
DEFAULT_BENCH_LIMIT = 100
DEFAULT_BENCH_MODEL = "faster-whisper-medium"
DEFAULT_MAX_MODEL_RAM_GB = 8.0
DEFAULT_NOTES_RUNTIME = "auto"
DEFAULT_LOG_LEVEL = "ERROR"
BenchmarkScenario = Literal["capture_sync", "hf_diarized_transcription"]


@dataclass(slots=True)
class UiCommonOptions:
    """Options shared by UI-triggered commands."""

    config_path: Path | None = None
    log_level: str | None = None
    debug: bool = False
    allow_network: bool = False


@dataclass(slots=True)
class DeviceInfo:
    """Display-ready representation of one audio device."""

    index: int
    name: str
    label: str
    max_input_channels: int
    default_samplerate: float | None
    hostapi_name: str


@dataclass(slots=True)
class DeviceListResult:
    """Available audio capture devices for the current platform backend."""

    devices: tuple[DeviceInfo, ...]


@dataclass(slots=True)
class CaptureRequest:
    """UI request model for capture runs."""

    common: UiCommonOptions
    duration_sec: float = 30.0
    mode: AudioSourceMode = AudioSourceMode.BOTH
    output_root: Path = Path("data/live_sessions")
    session_id: str | None = None
    mic_device: str | int | None = None
    speaker_device: str | int | None = None
    use_fixture: bool = False


@dataclass(slots=True)
class CaptureResultSummary:
    """UI-facing capture result summary."""

    manifest_path: Path
    session_dir: Path
    pair_count: int
    interrupted: bool
    raw_result: Any


@dataclass(slots=True)
class SessionRequest:
    """UI request model for live transcription sessions."""

    common: UiCommonOptions
    transcription_model: str = DEFAULT_LIVE_TRANSCRIPTION_MODEL
    duration_sec: float = 0.0
    chunk_overlap_sec: float = 0.75
    stitch_overlap_text: bool = True
    mode: AudioSourceMode = AudioSourceMode.BOTH
    chunk_sec: float = 4.0
    partial_interval_sec: float = 0.0
    output_root: Path = Path("data/live_sessions")
    session_id: str | None = None
    mic_device: str | int | None = None
    speaker_device: str | int | None = None
    single_device_per_source: bool = False
    strict_sources: bool = False
    use_fixture: bool = False
    max_model_ram_gb: float = DEFAULT_MAX_MODEL_RAM_GB
    notes_enabled: bool = True
    notes_model: str = DEFAULT_SESSION_NOTES_MODEL
    notes_runtime: str = DEFAULT_NOTES_RUNTIME


@dataclass(slots=True)
class SessionResultSummary:
    """UI-facing live session result summary."""

    session_dir: Path
    transcript_txt_path: Path
    transcript_json_path: Path
    events_path: Path
    final_segment_count: int
    partial_event_count: int
    interrupted: bool
    notes_summary: NotesResultSummary | None
    raw_result: Any


@dataclass(slots=True)
class NotesRequest:
    """UI request model for transcript cleanup and note generation."""

    common: UiCommonOptions
    transcript_path: Path
    output_dir: Path | None = None
    notes_model: str = DEFAULT_SESSION_NOTES_MODEL
    notes_runtime: str = DEFAULT_NOTES_RUNTIME


@dataclass(slots=True)
class NotesResultSummary:
    """UI-facing notes result summary."""

    transcript_path: Path
    clean_transcript_path: Path
    client_notes_path: Path
    model: str
    cpu_fallback_used: bool
    raw_result: Any


@dataclass(slots=True)
class ModelStatus:
    """One packaged model asset row for the models page."""

    model_id: str
    kind: str
    install_class: str
    installed: bool


@dataclass(slots=True)
class ModelsListResult:
    """Packaged model manifest view for the UI."""

    manifest_path: Path | None
    items: tuple[ModelStatus, ...]
    error: str | None = None


@dataclass(slots=True)
class ModelsInstallRequest:
    """UI request model for packaged model installs."""

    common: UiCommonOptions
    model_ids: tuple[str, ...] = ()
    default_only: bool = False


@dataclass(slots=True)
class ModelsInstallResultSummary:
    """UI-facing packaged model install summary."""

    installed_model_ids: tuple[str, ...]
    skipped_model_ids: tuple[str, ...]
    raw_result: Any


@dataclass(slots=True)
class ComplianceResultSummary:
    """UI-facing compliance check result."""

    name: str
    exit_code: int
    passed: bool
    target_path: Path | None = None


@dataclass(slots=True)
class BenchmarkInitRequest:
    """UI request model for benchmark cache initialization."""

    common: UiCommonOptions
    transcription_model: str = DEFAULT_BENCH_MODEL
    hf_dataset: str = DEFAULT_BENCH_DATASET
    hf_config: str = DEFAULT_BENCH_CONFIG
    hf_split: str = DEFAULT_BENCH_SPLIT
    hf_limit: int = DEFAULT_BENCH_LIMIT
    max_model_ram_gb: float = DEFAULT_MAX_MODEL_RAM_GB


@dataclass(slots=True)
class BenchmarkInitResultSummary:
    """UI-facing benchmark init summary."""

    payload: dict[str, object]


@dataclass(slots=True)
class BenchmarkRunRequest:
    """UI request model for benchmark runs."""

    common: UiCommonOptions
    scenario: BenchmarkScenario = "hf_diarized_transcription"
    runs: int = 5
    duration_sec: float = 10.0
    output_dir: Path = Path("data/benchmarks")
    real_devices: bool = False
    hf_dataset: str = DEFAULT_BENCH_DATASET
    hf_config: str = DEFAULT_BENCH_CONFIG
    hf_split: str = DEFAULT_BENCH_SPLIT
    hf_limit: int = DEFAULT_BENCH_LIMIT
    transcription_model: str = DEFAULT_BENCH_MODEL
    max_model_ram_gb: float = DEFAULT_MAX_MODEL_RAM_GB


@dataclass(slots=True)
class BenchmarkRunResultSummary:
    """UI-facing benchmark run summary."""

    scenario: BenchmarkScenario
    json_path: Path
    markdown_path: Path
    raw_result: Any


@dataclass(slots=True)
class ServiceProgressEvent:
    """Structured event forwarded from runtime services into the UI."""

    name: str
    fields: dict[str, object]

