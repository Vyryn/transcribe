from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from transcribe.compliance import enforce_no_url_literals, run_network_compliance_check
from transcribe.config import load_app_config
from transcribe.logging import configure_logging, security_log
from transcribe.models import AudioSourceMode, CaptureConfig
from transcribe.network_guard import install_outbound_network_guard
from transcribe.runtime_defaults import (
    ALTERNATE_SESSION_NOTES_MODEL,
    DEFAULT_LIVE_TRANSCRIPTION_MODEL,
    DEFAULT_SESSION_NOTES_MODEL,
)
from transcribe.runtime_env import resolve_app_runtime_paths

LOGGER = logging.getLogger("transcribe")


def _default_data_subdir(name: str) -> Path:
    """Resolve a default writable data subdirectory for CLI output flags."""
    runtime_paths = resolve_app_runtime_paths()
    return runtime_paths.data_root / name


def default_session_id(prefix: str = "session") -> str:
    """Build a UTC timestamped session identifier.

    Parameters
    ----------
    prefix : str, optional
        Prefix to include before timestamp.

    Returns
    -------
    str
        Session identifier.
    """
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def parse_mode(value: str) -> AudioSourceMode:
    """Parse CLI audio mode input into ``AudioSourceMode``.

    Parameters
    ----------
    value : str
        CLI mode argument.

    Returns
    -------
    AudioSourceMode
        Parsed source mode enum.
    """
    try:
        return AudioSourceMode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported mode: {value!r}") from exc


def parse_device_ref(value: str) -> str | int:
    """Parse CLI device reference as index or name.

    Parameters
    ----------
    value : str
        Device argument passed on CLI.

    Returns
    -------
    str | int
        Integer index when value is numeric (or bracketed numeric), otherwise name string.
    """
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("Device reference cannot be empty")

    if stripped.isdigit():
        return int(stripped)

    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1].strip()
        if inner.isdigit():
            return int(inner)

    return stripped


def add_common_config_flags(parser: argparse.ArgumentParser) -> None:
    """Attach common config/log flags to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to augment.
    """
    parser.add_argument("--config", type=Path, default=None, help="Optional config file path")
    parser.add_argument("--log-level", default=None, help="Override logging level")
    parser.add_argument("--debug", action="store_true", help="Show backend logs and verbose session events")


def _build_session_progress_reporter(*, debug: bool) -> Callable[[str, dict[str, object]], None]:
    """Create CLI printer for structured live-session progress events."""

    def _format_sources(raw_devices: object) -> str:
        if not isinstance(raw_devices, dict):
            return "none"
        parts: list[str] = []
        for source_name in ("mic", "speakers"):
            devices = raw_devices.get(source_name)
            if not isinstance(devices, list) or not devices:
                continue
            label = source_name if len(devices) == 1 else f"{source_name} x{len(devices)}"
            parts.append(label)
        return ", ".join(parts) if parts else "none"

    def _report(event: str, fields: dict[str, object]) -> None:
        if event == "loading_model":
            print(f"Loading model: {fields['transcription_model']}")
            return
        if event == "model_ready":
            print("Model ready.")
            return
        if event == "capture_ready":
            requested_rate_hz = int(fields.get("requested_sample_rate_hz", 0))
            capture_rate_hz = int(fields.get("capture_sample_rate_hz", 0))
            asr_rate_hz = int(fields.get("transcription_sample_rate_hz", 0))
            print(f"Capture ready: {_format_sources(fields.get('resolved_capture_devices'))}")
            if capture_rate_hz == requested_rate_hz:
                print(f"Sample rate: {capture_rate_hz} Hz capture, {asr_rate_hz} Hz ASR")
            else:
                print(
                    "Sample rate: "
                    f"{capture_rate_hz} Hz capture (requested {requested_rate_hz} Hz), "
                    f"{asr_rate_hz} Hz ASR"
                )
            if debug:
                device_channels = fields.get("device_channels")
                if isinstance(device_channels, dict) and device_channels:
                    print(f"Device channels: {device_channels}")
            return
        if event == "transcribing_started":
            if float(fields.get("duration_sec", 0.0)) > 0:
                print("Transcribing.")
            else:
                print("Transcribing. Press Ctrl+C to stop.")
            return
        if event == "partial":
            if debug:
                print(f"[partial {fields['chunk_index']}] {fields['text']}")
            return
        if event == "final":
            text = str(fields.get("text", "")).strip()
            if not text:
                return
            if debug:
                print(f"[final {fields['chunk_index']}] {text}")
            else:
                print(text)

    return _report


def _build_notes_progress_reporter() -> Callable[[str, dict[str, object]], None]:
    """Create CLI printer for structured post-session notes progress events."""

    def _report(event: str, fields: dict[str, object]) -> None:
        if event == "notes_started":
            model = str(fields.get("model", ""))
            chunk_count = int(fields.get("cleanup_chunk_count", 0))
            if chunk_count > 1:
                print(f"Post-session notes: cleaning transcript with {model} ({chunk_count} passes)...")
            else:
                print(f"Post-session notes: cleaning transcript with {model}...")
            return
        if event == "clean_transcript_chunk_started":
            chunk_count = int(fields.get("chunk_count", 0))
            chunk_index = int(fields.get("chunk_index", 0))
            if chunk_count > 1:
                print(f"Cleanup pass {chunk_index}/{chunk_count}...")
            return
        if event == "clean_transcript_ready":
            print("Clean transcript ready.")
            return
        if event == "client_notes_started":
            print("Writing client notes...")
            return
        if event == "notes_cpu_fallback":
            print("Notes runtime: retrying on CPU...")

    return _report


def build_parser(*, packaged_runtime: bool = False) -> argparse.ArgumentParser:
    """Construct the top-level CLI parser.

    Parameters
    ----------
    packaged_runtime : bool, optional
        When ``True``, build the end-user packaged parser surface.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(prog="transcribe", description="Offline transcription CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="Audio capture commands")
    capture_subparsers = capture_parser.add_subparsers(dest="capture_command", required=True)

    capture_run = capture_subparsers.add_parser("run", help="Capture synchronized mic+speaker audio")
    add_common_config_flags(capture_run)
    capture_run.add_argument("--mode", type=parse_mode, default=AudioSourceMode.BOTH)
    capture_run.add_argument("--duration-sec", type=float, default=30.0)
    capture_run.add_argument("--out", type=Path, default=_default_data_subdir("captures"))
    capture_run.add_argument("--session-id", default=None)
    capture_run.add_argument(
        "--mic-device",
        type=parse_device_ref,
        default=None,
        help="Mic device name or index from `capture devices`",
    )
    capture_run.add_argument(
        "--speaker-device",
        type=parse_device_ref,
        default=None,
        help="Speaker playback/loopback device name or index from `capture devices`",
    )
    capture_run.add_argument("--fixture", action="store_true", help="Use synthetic audio fixture")

    capture_devices = capture_subparsers.add_parser("devices", help="List available capture devices")
    add_common_config_flags(capture_devices)

    if not packaged_runtime:
        bench_parser = subparsers.add_parser("bench", help="Benchmark commands")
        bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)

        bench_run = bench_subparsers.add_parser("run", help="Run capture synchronization benchmark")
        add_common_config_flags(bench_run)
        bench_run.add_argument("--scenario", default="capture_sync")
        bench_run.add_argument("--runs", type=int, default=5)
        bench_run.add_argument("--duration-sec", type=float, default=10.0)
        bench_run.add_argument("--out", type=Path, default=_default_data_subdir("benchmarks"))
        bench_run.add_argument("--real-devices", action="store_true", help="Use live devices instead of fixture")
        bench_run.add_argument(
            "--hf-dataset",
            default="edinburghcstr/ami",
            help="Hugging Face dataset id for diarized-transcription benchmarking",
        )
        bench_run.add_argument("--hf-config", default="ihm", help="Hugging Face dataset config/subset")
        bench_run.add_argument("--hf-split", default="test", help="Hugging Face split name")
        bench_run.add_argument(
            "--hf-limit",
            type=int,
            default=100,
            help="Row limit for Hugging Face benchmark runs",
        )
        bench_run.add_argument(
            "--model",
            "--transcription-model",
            dest="transcription_model",
            default="faster-whisper-medium",
            help="Model id under test (whisper*=>faster-whisper, nvidia/*=>nemo_asr, qwen/*=>qwen-asr)",
        )
        bench_run.add_argument(
            "--max-model-ram-gb",
            type=float,
            default=8.0,
            help="Reject models with estimated runtime RAM above this threshold",
        )

    session_parser = subparsers.add_parser("session", help="Live transcription session commands")
    session_subparsers = session_parser.add_subparsers(dest="session_command", required=True)

    session_run = session_subparsers.add_parser("run", help="Run live multi-source transcription test rig")
    add_common_config_flags(session_run)
    session_run.add_argument(
        "--model",
        "--transcription-model",
        dest="transcription_model",
        default=DEFAULT_LIVE_TRANSCRIPTION_MODEL,
        help=f"Model id for streaming ASR (default: {DEFAULT_LIVE_TRANSCRIPTION_MODEL})",
    )
    session_run.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Session runtime in seconds; 0 runs until interrupted (Ctrl+C)",
    )
    session_run.add_argument(
        "--chunk-overlap-sec",
        type=float,
        default=0.75,
        help="Audio overlap carried into the next finalized chunk to reduce clipped boundaries",
    )
    session_run.add_argument(
        "--stitch-overlap-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim clearly repeated leading text between finalized chunks (default: enabled)",
    )
    session_run.add_argument(
        "--mode",
        type=parse_mode,
        default=AudioSourceMode.BOTH,
        help="Source mode: mic, speakers, or both (default: both)",
    )
    session_run.add_argument("--chunk-sec", type=float, default=4.0, help="Finalization chunk size in seconds")
    session_run.add_argument(
        "--partial-interval-sec",
        type=float,
        default=0.0,
        help="Partial transcript refresh interval in seconds (0 disables partials)",
    )
    session_run.add_argument("--out", type=Path, default=_default_data_subdir("live_sessions"))
    session_run.add_argument("--session-id", default=None)
    session_run.add_argument(
        "--mic-device",
        type=parse_device_ref,
        default=None,
        help="Mic device name or index from `capture devices`",
    )
    session_run.add_argument(
        "--speaker-device",
        type=parse_device_ref,
        default=None,
        help="Speaker playback/loopback device name or index from `capture devices`",
    )
    session_run.add_argument(
        "--single-device-per-source",
        action="store_true",
        help="Use only one device per source type (disable all-device auto-selection)",
    )
    session_run.add_argument(
        "--strict-sources",
        action="store_true",
        help="Fail when any requested source type has no usable device",
    )
    session_run.add_argument("--fixture", action="store_true", help="Use synthetic audio fixture")
    session_run.add_argument(
        "--max-model-ram-gb",
        type=float,
        default=8.0,
        help="Reject models with estimated runtime RAM above this threshold",
    )
    session_run.add_argument(
        "--notes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run transcript cleanup + client notes generation after the session (default: enabled)",
    )
    session_run.add_argument(
        "--notes-model",
        default=DEFAULT_SESSION_NOTES_MODEL,
        help=(
            "Local notes model for transcript cleanup and client notes "
            f"(default: {DEFAULT_SESSION_NOTES_MODEL}; alternative: {ALTERNATE_SESSION_NOTES_MODEL})"
        ),
    )
    session_run.add_argument(
        "--notes-runtime",
        choices=("auto", "ollama", "llama_cpp"),
        default="auto",
        help="Notes runtime backend (default: auto)",
    )

    notes_parser = subparsers.add_parser("notes", help="Transcript cleanup and session notes commands")
    notes_subparsers = notes_parser.add_subparsers(dest="notes_command", required=True)

    notes_run = notes_subparsers.add_parser("run", help="Clean a rough transcript and generate client notes")
    add_common_config_flags(notes_run)
    notes_run.add_argument("--transcript", type=Path, required=True, help="Path to the rough transcript text file")
    notes_run.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for clean transcript and client notes (default: transcript parent)",
    )
    notes_run.add_argument(
        "--model",
        "--notes-model",
        dest="notes_model",
        default=DEFAULT_SESSION_NOTES_MODEL,
        help=(
            "Local notes model for transcript cleanup and client notes "
            f"(default: {DEFAULT_SESSION_NOTES_MODEL}; alternative: {ALTERNATE_SESSION_NOTES_MODEL})"
        ),
    )
    notes_run.add_argument(
        "--notes-runtime",
        choices=("auto", "ollama", "llama_cpp"),
        default="auto",
        help="Notes runtime backend (default: auto)",
    )

    compliance_parser = subparsers.add_parser("compliance", help="Compliance checks")
    compliance_subparsers = compliance_parser.add_subparsers(dest="compliance_command", required=True)

    compliance_network = compliance_subparsers.add_parser("check-no-network", help="Verify outbound network is blocked")
    add_common_config_flags(compliance_network)

    compliance_urls = compliance_subparsers.add_parser("check-no-urls", help="Verify no URL literals in runtime source")
    add_common_config_flags(compliance_urls)

    models_parser = subparsers.add_parser("models", help="Packaged model asset commands")
    models_subparsers = models_parser.add_subparsers(dest="models_command", required=True)

    models_list = models_subparsers.add_parser("list", help="List packaged model assets")
    add_common_config_flags(models_list)

    models_install = models_subparsers.add_parser("install", help="Install packaged model assets")
    add_common_config_flags(models_install)
    models_install.add_argument(
        "--model",
        dest="model_ids",
        action="append",
        default=[],
        help="Packaged model id to install; may be repeated",
    )
    models_install.add_argument(
        "--default",
        action="store_true",
        help="Install the default packaged model set",
    )
    models_install.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-model output",
    )

    return parser


def load_and_configure_logging(args: argparse.Namespace) -> None:
    """Load app config, configure logging, and install network guard.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.
    """
    app_config = load_app_config(
        config_path=args.config,
        overrides={
            "log_level": args.log_level or ("DEBUG" if getattr(args, "debug", False) else "ERROR")
        },
    )
    configure_logging(app_config.log_level, redact_logs=app_config.redact_logs)
    install_outbound_network_guard()
    if getattr(args, "debug", False):
        security_log(LOGGER, logging.INFO, "startup", offline_only=app_config.offline_only)


def run_capture(args: argparse.Namespace) -> int:
    """Execute the ``capture run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    from transcribe.audio.runner import run_capture_session

    load_and_configure_logging(args)

    if args.mode != AudioSourceMode.BOTH and not args.fixture:
        raise ValueError("Phase 0 requires --mode both for real synchronized capture")

    session_id = args.session_id or default_session_id("capture")
    output_dir = args.out / session_id
    config = CaptureConfig(
        source_mode=args.mode,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
        session_id=session_id,
        output_dir=output_dir,
    )
    result = run_capture_session(config, duration_sec=args.duration_sec, use_fixture=args.fixture)

    print(f"Capture complete: {result.manifest_path}")
    print(f"Pairs written: {result.manifest.capture_stats['pair_count']}")
    print(
        "Callback->Write latency p50/p95 (ms): "
        f"{result.manifest.capture_stats['callback_to_write_latency_ms_p50']:.3f}/"
        f"{result.manifest.capture_stats['callback_to_write_latency_ms_p95']:.3f}"
    )
    return 0


def run_devices(args: argparse.Namespace) -> int:
    """Execute the ``capture devices`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    from transcribe.audio.backend_loader import open_audio_backend

    load_and_configure_logging(args)
    backend = open_audio_backend(use_fixture=False)
    devices = backend.list_devices()
    if not devices:
        print("No audio devices found (or the platform capture backend is unavailable).")
        return 0

    for device in devices:
        suffix = ""
        hostapi_name = str(device.get("hostapi_name", "")).strip() if isinstance(device, dict) else ""
        if hostapi_name:
            suffix = f" hostapi={hostapi_name}"
        default_sr_value = device.get("default_samplerate", 0.0) if isinstance(device, dict) else 0.0
        default_sr = "unknown"
        if isinstance(default_sr_value, (int, float)) and float(default_sr_value) > 0:
            default_sr = str(default_sr_value)
        print(
            f"[{device['index']}] {device['name']} "
            f"inputs={device['max_input_channels']} default_sr={default_sr}"
            f"{suffix}"
        )
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """Execute the ``bench run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    from transcribe.bench.harness import (
        HF_DIARIZED_SCENARIO,
        run_capture_sync_benchmark,
        run_hf_diarized_transcription_benchmark,
    )

    load_and_configure_logging(args)
    if args.scenario == "capture_sync":
        base_config = CaptureConfig(source_mode=AudioSourceMode.BOTH, session_id="bench", output_dir=args.out)
        bench = run_capture_sync_benchmark(
            base_config=base_config,
            runs=args.runs,
            duration_sec=args.duration_sec,
            output_dir=args.out,
            use_fixture=not args.real_devices,
        )
    elif args.scenario == HF_DIARIZED_SCENARIO:
        bench = run_hf_diarized_transcription_benchmark(
            output_dir=args.out,
            dataset_id=args.hf_dataset,
            dataset_config=args.hf_config,
            split=args.hf_split,
            sample_limit=args.hf_limit,
            transcription_model=args.transcription_model,
            max_model_ram_gb=args.max_model_ram_gb,
        )
    else:
        raise ValueError(
            f"Unsupported --scenario {args.scenario!r}. Use 'capture_sync' or '{HF_DIARIZED_SCENARIO}'."
        )

    print(f"Benchmark JSON: {bench.json_path}")
    print(f"Benchmark Markdown: {bench.markdown_path}")
    return 0


def run_session(args: argparse.Namespace) -> int:
    """Execute the ``session run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    from transcribe.live.session import LiveSessionConfig, run_live_transcription_session
    from transcribe.runtime_env import validate_transcription_model_for_runtime

    load_and_configure_logging(args)
    try:
        transcription_model = validate_transcription_model_for_runtime(args.transcription_model)
    except ValueError as exc:
        print(f"Session failed: {exc}")
        return 2
    session_id = args.session_id or default_session_id("live")
    output_dir = args.out / session_id
    config = LiveSessionConfig(
        transcription_model=transcription_model,
        duration_sec=args.duration_sec,
        chunk_sec=args.chunk_sec,
        chunk_overlap_sec=args.chunk_overlap_sec,
        stitch_overlap_text=getattr(args, "stitch_overlap_text", True),
        partial_interval_sec=args.partial_interval_sec,
        source_mode=args.mode,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
        capture_all_mic_devices=not args.single_device_per_source,
        capture_all_speaker_devices=not args.single_device_per_source,
        allow_missing_sources=not args.strict_sources,
        output_dir=output_dir,
        session_id=session_id,
        max_model_ram_gb=args.max_model_ram_gb,
    )
    progress_reporter = _build_session_progress_reporter(debug=getattr(args, "debug", False))
    try:
        result = run_live_transcription_session(
            config,
            use_fixture=args.fixture,
            debug=getattr(args, "debug", False),
            progress_callback=progress_reporter,
        )
    except RuntimeError as exc:
        print(f"Session failed: {exc}")
        return 2

    print(f"Session saved: {result.session_dir}")
    print(f"Transcript: {result.transcript_txt_path}")
    print(
        "Summary: "
        f"{result.final_segment_count} final segments, "
        f"{result.total_audio_sec:.3f}s audio, "
        f"{result.total_inference_sec:.3f}s inference"
    )
    if getattr(args, "debug", False):
        print(f"Events JSONL: {result.events_path}")
        print(f"Transcript JSON: {result.transcript_json_path}")
        print(f"Partial events: {result.partial_event_count}")
        print(
            "Sample rate (requested/effective Hz): "
            f"{result.sample_rate_hz_requested}/{result.sample_rate_hz}"
        )
        print(f"Source selections: {result.source_selection_counts}")

    notes_enabled = getattr(args, "notes", False)
    if not notes_enabled:
        return 0

    from transcribe.transcription_runtime import release_transcription_runtime_resources
    from transcribe.notes import SessionNotesConfig, run_post_transcription_notes

    print("Preparing notes: releasing transcription model resources...")
    release_transcription_runtime_resources(transcription_model)
    notes_progress_reporter = _build_notes_progress_reporter()
    try:
        notes_result = run_post_transcription_notes(
            SessionNotesConfig(
                transcript_path=result.transcript_txt_path,
                output_dir=result.session_dir,
                model=getattr(args, "notes_model", DEFAULT_SESSION_NOTES_MODEL),
                runtime=getattr(args, "notes_runtime", "auto"),
            ),
            progress_callback=notes_progress_reporter,
        )
    except RuntimeError as exc:
        print(f"Notes failed: {exc}")
        return 3

    print(f"Clean transcript: {notes_result.clean_transcript_path}")
    print(f"Client notes: {notes_result.client_notes_path}")
    return 0


def run_notes(args: argparse.Namespace) -> int:
    """Execute the ``notes run`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    from transcribe.notes import SessionNotesConfig, run_post_transcription_notes

    load_and_configure_logging(args)

    output_dir = args.out_dir or args.transcript.parent
    notes_progress_reporter = _build_notes_progress_reporter()
    try:
        result = run_post_transcription_notes(
            SessionNotesConfig(
                transcript_path=args.transcript,
                output_dir=output_dir,
                model=args.notes_model,
                runtime=getattr(args, "notes_runtime", "auto"),
            ),
            progress_callback=notes_progress_reporter,
        )
    except RuntimeError as exc:
        print(f"Notes failed: {exc}")
        return 2

    print(f"Clean transcript: {result.clean_transcript_path}")
    print(f"Client notes: {result.client_notes_path}")
    return 0


def run_models(args: argparse.Namespace) -> int:
    """Execute packaged model-management commands."""
    from transcribe.packaged_assets import (
        install_packaged_model_assets,
        load_packaged_asset_manifest,
        verify_installed_asset,
    )

    runtime_paths = resolve_app_runtime_paths()
    manifest_path = runtime_paths.packaged_assets_manifest_path
    if not manifest_path.exists():
        print(f"Models command failed: packaged asset manifest not found: {manifest_path}")
        return 2

    try:
        manifest = load_packaged_asset_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        print(f"Models command failed: {exc}")
        return 2

    if args.models_command == "list":
        for asset in manifest.assets:
            install_class = "default" if asset.default_install else "optional"
            status = "installed" if verify_installed_asset(asset, models_root=runtime_paths.models_root) else "not installed"
            print(f"{asset.model_id}\t{asset.kind}\t{install_class}\t{status}")
        return 0

    if args.models_command != "install":
        print(f"Models command failed: unsupported subcommand {args.models_command!r}.")
        return 2

    if not getattr(args, "default", False) and not getattr(args, "model_ids", []):
        print("Models install failed: pass --default or --model <id>.")
        return 2

    def _report(event: str, asset, target_path: Path) -> None:
        if getattr(args, "quiet", False):
            return
        if event == "installing":
            print(f"Installing {asset.model_id} -> {target_path}")
            return
        if event == "installed":
            print(f"Installed {asset.model_id}")
            return
        if event == "skipped":
            print(f"Already installed {asset.model_id}")

    try:
        results = install_packaged_model_assets(
            manifest,
            models_root=runtime_paths.models_root,
            installed_state_path=runtime_paths.installed_assets_state_path,
            model_ids=list(getattr(args, "model_ids", [])),
            default_only=bool(getattr(args, "default", False)),
            progress_callback=_report,
        )
    except Exception as exc:
        print(f"Models install failed: {exc}")
        return 2

    if not getattr(args, "quiet", False):
        print(f"Models ready: {len(results)}")
    return 0

def run_check_no_network(args: argparse.Namespace) -> int:
    """Execute the ``compliance check-no-network`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    load_and_configure_logging(args)
    return run_network_compliance_check()


def run_check_no_urls(args: argparse.Namespace) -> int:
    """Execute the ``compliance check-no-urls`` command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code.
    """
    load_and_configure_logging(args)
    return enforce_no_url_literals(Path.cwd())


def main(argv: list[str] | None = None, *, packaged_runtime: bool = False) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI arguments. Uses process argv when ``None``.
    packaged_runtime : bool, optional
        When ``True``, expose the packaged runtime command surface.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser(packaged_runtime=packaged_runtime)
    args = parser.parse_args(argv)

    if args.command == "capture" and args.capture_command == "run":
        return run_capture(args)
    if args.command == "capture" and args.capture_command == "devices":
        return run_devices(args)
    if args.command == "bench" and args.bench_command == "run":
        return run_benchmark(args)
    if args.command == "session" and args.session_command == "run":
        return run_session(args)
    if args.command == "notes" and args.notes_command == "run":
        return run_notes(args)
    if args.command == "compliance" and args.compliance_command == "check-no-network":
        return run_check_no_network(args)
    if args.command == "compliance" and args.compliance_command == "check-no-urls":
        return run_check_no_urls(args)
    if args.command == "models":
        return run_models(args)

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())



