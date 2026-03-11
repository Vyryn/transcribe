from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Callable

from transcribe.config import load_app_config
from transcribe.logging import configure_logging, security_log
from transcribe.models import AudioSourceMode, CaptureConfig
from transcribe.network_guard import install_outbound_network_guard, outbound_network_guard_installed
from transcribe.runtime_env import RuntimeMode, resolve_app_runtime_paths, validate_transcription_model_for_runtime
from transcribe.ui.types import (
    DEFAULT_BENCH_CONFIG,
    DEFAULT_BENCH_DATASET,
    DEFAULT_BENCH_LIMIT,
    DEFAULT_BENCH_MODEL,
    DEFAULT_BENCH_SPLIT,
    DEFAULT_LIVE_TRANSCRIPTION_MODEL,
    DEFAULT_NOTES_RUNTIME,
    DEFAULT_SESSION_NOTES_MODEL,
    BenchmarkInitRequest,
    BenchmarkInitResultSummary,
    BenchmarkRunRequest,
    BenchmarkRunResultSummary,
    CaptureRequest,
    CaptureResultSummary,
    ComplianceResultSummary,
    DeviceInfo,
    DeviceListResult,
    ModelStatus,
    ModelsInstallRequest,
    ModelsInstallResultSummary,
    ModelsListResult,
    NotesRequest,
    NotesResultSummary,
    ServiceProgressEvent,
    SessionRequest,
    SessionResultSummary,
    UiCommonOptions,
)

LOGGER = logging.getLogger("transcribe.ui")
ProgressCallback = Callable[[str, dict[str, object]], None]


def default_data_subdir(name: str) -> Path:
    """Resolve a writable default data directory for one UI workflow."""
    return resolve_app_runtime_paths().data_root / name


def default_session_id(prefix: str) -> str:
    """Build a UTC timestamped identifier matching CLI behavior."""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def normalize_device_reference(value: str | int | None) -> str | int | None:
    """Normalize a UI device selection into CLI-compatible device reference types."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    stripped = value.strip()
    if not stripped or stripped.lower() in {"auto", "default"}:
        return None
    if stripped.isdigit():
        return int(stripped)
    if stripped.startswith("["):
        closing_index = stripped.find("]")
        if closing_index > 1:
            inner = stripped[1:closing_index].strip()
            if inner.isdigit():
                return int(inner)
    return stripped


def _runtime_log_level(common: UiCommonOptions) -> str:
    explicit = (common.log_level or "").strip()
    if explicit:
        return explicit
    return "DEBUG" if common.debug else "ERROR"


def configure_runtime(common: UiCommonOptions) -> None:
    """Apply the same runtime logging and network policy used by CLI commands."""
    app_config = load_app_config(
        config_path=common.config_path,
        overrides={"log_level": _runtime_log_level(common)},
    )
    configure_logging(app_config.log_level, redact_logs=app_config.redact_logs)
    install_outbound_network_guard()
    if common.debug:
        security_log(LOGGER, logging.INFO, "startup", offline_only=app_config.offline_only)


def ensure_network_downloads_available(task_name: str) -> None:
    """Fail fast when a prior offline command permanently installed the socket guard."""
    if outbound_network_guard_installed():
        raise RuntimeError(
            f"{task_name} requires outbound network access, but this UI session already installed the offline socket guard. "
            "Restart transcribe-ui and run the networked task before any offline-only workflow."
        )


def list_devices(*, common: UiCommonOptions) -> DeviceListResult:
    """List available audio capture devices for the current platform backend."""
    from transcribe.audio.backend_loader import open_audio_backend

    configure_runtime(common)
    backend = open_audio_backend(use_fixture=False)
    try:
        raw_devices = backend.list_devices()
    finally:
        close_backend = getattr(backend, "close", None)
        if callable(close_backend):
            try:
                close_backend()
            except Exception:  # noqa: BLE001
                pass
    devices: list[DeviceInfo] = []
    for raw_device in raw_devices:
        if not isinstance(raw_device, dict):
            continue
        index = int(raw_device.get("index", -1))
        name = str(raw_device.get("name", "unknown device"))
        hostapi_name = str(raw_device.get("hostapi_name", "")).strip()
        max_inputs = int(raw_device.get("max_input_channels", 0))
        default_sr_value = raw_device.get("default_samplerate")
        default_samplerate = None
        if isinstance(default_sr_value, (int, float)) and float(default_sr_value) > 0:
            default_samplerate = float(default_sr_value)
        label = f"[{index}] {name}"
        if max_inputs >= 0:
            label = f"{label} inputs={max_inputs}"
        if default_samplerate is not None:
            label = f"{label} default_sr={default_samplerate:g}"
        if hostapi_name:
            label = f"{label} hostapi={hostapi_name}"
        devices.append(
            DeviceInfo(
                index=index,
                name=name,
                label=label,
                max_input_channels=max_inputs,
                default_samplerate=default_samplerate,
                hostapi_name=hostapi_name,
            )
        )
    return DeviceListResult(devices=tuple(devices))


def run_capture(
    request: CaptureRequest,
    *,
    cancel_event: Event | None = None,
) -> CaptureResultSummary:
    """Run a capture session and return a UI-facing summary."""
    from transcribe.audio.runner import run_capture_session

    configure_runtime(request.common)
    if request.mode != AudioSourceMode.BOTH and not request.use_fixture:
        raise ValueError("Real capture currently requires mode 'both'. Use the fixture for mic/speakers-only tests.")

    session_id = request.session_id or default_session_id("capture")
    output_dir = request.output_root / session_id
    config = CaptureConfig(
        source_mode=request.mode,
        mic_device=normalize_device_reference(request.mic_device),
        speaker_device=normalize_device_reference(request.speaker_device),
        session_id=session_id,
        output_dir=output_dir,
    )
    result = run_capture_session(
        config,
        duration_sec=request.duration_sec,
        use_fixture=request.use_fixture,
        cancel_event=cancel_event,
    )
    pair_count = int(result.manifest.capture_stats.get("pair_count", 0))
    return CaptureResultSummary(
        manifest_path=result.manifest_path,
        session_dir=output_dir,
        pair_count=pair_count,
        interrupted=result.interrupted,
        raw_result=result,
    )


def run_session(
    request: SessionRequest,
    *,
    cancel_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
) -> SessionResultSummary:
    """Run a live transcription session and optional notes pipeline."""
    from transcribe.live.session import LiveSessionConfig, run_live_transcription_session
    from transcribe.notes import SessionNotesConfig, run_post_transcription_notes
    from transcribe.transcription_runtime import release_transcription_runtime_resources

    configure_runtime(request.common)
    transcription_model = validate_transcription_model_for_runtime(request.transcription_model)
    session_id = request.session_id or default_session_id("live")
    output_dir = request.output_root / session_id
    config = LiveSessionConfig(
        transcription_model=transcription_model,
        duration_sec=request.duration_sec,
        chunk_sec=request.chunk_sec,
        chunk_overlap_sec=request.chunk_overlap_sec,
        stitch_overlap_text=request.stitch_overlap_text,
        partial_interval_sec=request.partial_interval_sec,
        source_mode=request.mode,
        mic_device=normalize_device_reference(request.mic_device),
        speaker_device=normalize_device_reference(request.speaker_device),
        capture_all_mic_devices=not request.single_device_per_source,
        capture_all_speaker_devices=not request.single_device_per_source,
        allow_missing_sources=not request.strict_sources,
        output_dir=output_dir,
        session_id=session_id,
        max_model_ram_gb=request.max_model_ram_gb,
    )
    result = run_live_transcription_session(
        config,
        use_fixture=request.use_fixture,
        debug=request.common.debug,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
    )

    notes_summary: NotesResultSummary | None = None
    if request.notes_enabled and result.final_segment_count > 0:
        if progress_callback is not None:
            progress_callback("notes_preparing", {"transcription_model": transcription_model})
        release_transcription_runtime_resources(transcription_model)
        notes_result = run_post_transcription_notes(
            SessionNotesConfig(
                transcript_path=result.transcript_txt_path,
                output_dir=result.session_dir,
                model=request.notes_model,
                runtime=request.notes_runtime,
            ),
            progress_callback=progress_callback,
        )
        notes_summary = NotesResultSummary(
            transcript_path=notes_result.transcript_path,
            clean_transcript_path=notes_result.clean_transcript_path,
            client_notes_path=notes_result.client_notes_path,
            model=notes_result.model,
            cpu_fallback_used=notes_result.cpu_fallback_used,
            raw_result=notes_result,
        )
    elif request.notes_enabled and progress_callback is not None:
        progress_callback("notes_skipped", {"reason": "no_final_segments"})

    return SessionResultSummary(
        session_dir=result.session_dir,
        transcript_txt_path=result.transcript_txt_path,
        transcript_json_path=result.transcript_json_path,
        events_path=result.events_path,
        final_segment_count=result.final_segment_count,
        partial_event_count=result.partial_event_count,
        interrupted=result.interrupted,
        notes_summary=notes_summary,
        raw_result=result,
    )


def run_notes(
    request: NotesRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> NotesResultSummary:
    """Run transcript cleanup and client note generation."""
    from transcribe.notes import SessionNotesConfig, run_post_transcription_notes

    configure_runtime(request.common)
    output_dir = request.output_dir or request.transcript_path.parent
    result = run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=request.transcript_path,
            output_dir=output_dir,
            model=request.notes_model,
            runtime=request.notes_runtime,
        ),
        progress_callback=progress_callback,
    )
    return NotesResultSummary(
        transcript_path=result.transcript_path,
        clean_transcript_path=result.clean_transcript_path,
        client_notes_path=result.client_notes_path,
        model=result.model,
        cpu_fallback_used=result.cpu_fallback_used,
        raw_result=result,
    )


def list_models() -> ModelsListResult:
    """List packaged model assets and install status when a manifest is available."""
    from transcribe.packaged_assets import load_packaged_asset_manifest, verify_installed_asset

    runtime_paths = resolve_app_runtime_paths()
    manifest_path = runtime_paths.packaged_assets_manifest_path
    if not manifest_path.exists():
        return ModelsListResult(manifest_path=None, items=(), error=f"Packaged asset manifest not found: {manifest_path}")

    try:
        manifest = load_packaged_asset_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return ModelsListResult(manifest_path=manifest_path, items=(), error=str(exc))

    items = tuple(
        ModelStatus(
            model_id=asset.model_id,
            kind=asset.kind,
            install_class="default" if asset.default_install else "optional",
            installed=verify_installed_asset(asset, models_root=runtime_paths.models_root),
        )
        for asset in manifest.assets
    )
    return ModelsListResult(manifest_path=manifest_path, items=items)


def install_models(
    request: ModelsInstallRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> ModelsInstallResultSummary:
    """Install packaged model assets selected from the manifest."""
    from transcribe.packaged_assets import install_packaged_model_assets, load_packaged_asset_manifest

    ensure_network_downloads_available("Packaged model install")
    runtime_paths = resolve_app_runtime_paths()
    manifest_path = runtime_paths.packaged_assets_manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Packaged asset manifest not found: {manifest_path}")
    manifest = load_packaged_asset_manifest(manifest_path)

    def _report(event: str, asset, target_path: Path) -> None:
        if progress_callback is None:
            return
        progress_callback(
            f"models_{event}",
            {
                "model_id": asset.model_id,
                "kind": asset.kind,
                "target_path": str(target_path),
            },
        )

    results = install_packaged_model_assets(
        manifest,
        models_root=runtime_paths.models_root,
        installed_state_path=runtime_paths.installed_assets_state_path,
        model_ids=list(request.model_ids),
        default_only=request.default_only,
        progress_callback=_report,
    )
    installed_model_ids = tuple(result.model_id for result in results if not result.skipped)
    skipped_model_ids = tuple(result.model_id for result in results if result.skipped)
    return ModelsInstallResultSummary(
        installed_model_ids=installed_model_ids,
        skipped_model_ids=skipped_model_ids,
        raw_result=results,
    )


def run_compliance_check_no_network(*, common: UiCommonOptions) -> ComplianceResultSummary:
    """Run the outbound-network compliance check."""
    from transcribe.compliance import run_network_compliance_check

    configure_runtime(common)
    exit_code = int(run_network_compliance_check())
    return ComplianceResultSummary(name="check-no-network", exit_code=exit_code, passed=exit_code == 0)


def run_compliance_check_no_urls(*, common: UiCommonOptions, target_path: Path | None = None) -> ComplianceResultSummary:
    """Run the no-URL-literals compliance check."""
    from transcribe.compliance import enforce_no_url_literals

    configure_runtime(common)
    runtime_paths = resolve_app_runtime_paths()
    resolved_target = target_path
    if resolved_target is None:
        if runtime_paths.mode == RuntimeMode.PACKAGED:
            resolved_target = runtime_paths.install_root
        else:
            resolved_target = Path.cwd()
    exit_code = int(enforce_no_url_literals(resolved_target))
    return ComplianceResultSummary(
        name="check-no-urls",
        exit_code=exit_code,
        passed=exit_code == 0,
        target_path=resolved_target,
    )


def initialize_bench_assets(request: BenchmarkInitRequest) -> BenchmarkInitResultSummary:
    """Warm benchmark dataset/model cache without enabling the network guard."""
    from transcribe.bench.harness import initialize_benchmark_assets

    ensure_network_downloads_available("Benchmark cache initialization")
    payload = initialize_benchmark_assets(
        dataset_id=request.hf_dataset,
        dataset_config=request.hf_config,
        split=request.hf_split,
        sample_limit=request.hf_limit,
        transcription_model=request.transcription_model,
        max_model_ram_gb=request.max_model_ram_gb,
    )
    return BenchmarkInitResultSummary(payload=dict(payload))


def run_benchmark(request: BenchmarkRunRequest) -> BenchmarkRunResultSummary:
    """Run one benchmark scenario and return report file locations."""
    from transcribe.bench.harness import (
        HF_DIARIZED_SCENARIO,
        run_capture_sync_benchmark,
        run_hf_diarized_transcription_benchmark,
    )

    configure_runtime(request.common)
    if request.scenario == "capture_sync":
        base_config = CaptureConfig(
            source_mode=AudioSourceMode.BOTH,
            session_id="bench",
            output_dir=request.output_dir,
        )
        result = run_capture_sync_benchmark(
            base_config=base_config,
            runs=request.runs,
            duration_sec=request.duration_sec,
            output_dir=request.output_dir,
            use_fixture=not request.real_devices,
        )
        scenario = "capture_sync"
    else:
        result = run_hf_diarized_transcription_benchmark(
            output_dir=request.output_dir,
            dataset_id=request.hf_dataset,
            dataset_config=request.hf_config,
            split=request.hf_split,
            sample_limit=request.hf_limit,
            transcription_model=request.transcription_model,
            max_model_ram_gb=request.max_model_ram_gb,
        )
        scenario = HF_DIARIZED_SCENARIO
    return BenchmarkRunResultSummary(
        scenario=scenario,
        json_path=result.json_path,
        markdown_path=result.markdown_path,
        raw_result=result,
    )


def wrap_progress(event: str, fields: dict[str, object]) -> ServiceProgressEvent:
    """Build a typed progress wrapper for controller and UI consumers."""
    return ServiceProgressEvent(name=event, fields=dict(fields))
