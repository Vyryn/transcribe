from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from transcribe.compliance import enforce_no_url_literals, run_network_compliance_check
from transcribe.config import load_app_config
from transcribe.logging import configure_logging, security_log
from transcribe.models import AudioSourceMode, CaptureConfig
from transcribe.network_guard import install_outbound_network_guard

LOGGER = logging.getLogger("transcribe")


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser.

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
    capture_run.add_argument("--out", type=Path, default=Path("data/captures"))
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
        help="Speaker monitor device name or index from `capture devices`",
    )
    capture_run.add_argument("--fixture", action="store_true", help="Use synthetic audio fixture")

    capture_devices = capture_subparsers.add_parser("devices", help="List Linux capture devices")
    add_common_config_flags(capture_devices)

    bench_parser = subparsers.add_parser("bench", help="Benchmark commands")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", required=True)

    bench_run = bench_subparsers.add_parser("run", help="Run capture synchronization benchmark")
    add_common_config_flags(bench_run)
    bench_run.add_argument("--scenario", default="capture_sync")
    bench_run.add_argument("--runs", type=int, default=5)
    bench_run.add_argument("--duration-sec", type=float, default=10.0)
    bench_run.add_argument("--out", type=Path, default=Path("data/benchmarks"))
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
        default="nvidia/canary-qwen-2.5b",
        help="Model id for streaming ASR (for example nvidia/canary-qwen-2.5b)",
    )
    session_run.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Session runtime in seconds; 0 runs until interrupted (Ctrl+C)",
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
    session_run.add_argument("--out", type=Path, default=Path("data/live_sessions"))
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
        help="Speaker monitor device name or index from `capture devices`",
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

    compliance_parser = subparsers.add_parser("compliance", help="Compliance checks")
    compliance_subparsers = compliance_parser.add_subparsers(dest="compliance_command", required=True)

    compliance_network = compliance_subparsers.add_parser("check-no-network", help="Verify outbound network is blocked")
    add_common_config_flags(compliance_network)

    compliance_urls = compliance_subparsers.add_parser("check-no-urls", help="Verify no URL literals in runtime source")
    add_common_config_flags(compliance_urls)

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
        overrides={"log_level": args.log_level} if args.log_level else None,
    )
    configure_logging(app_config.log_level, redact_logs=app_config.redact_logs)
    install_outbound_network_guard()
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
    from transcribe.audio.linux_capture import LinuxAudioCaptureBackend

    load_and_configure_logging(args)
    backend = LinuxAudioCaptureBackend(use_fixture=False)
    devices = backend.list_devices()
    if not devices:
        print("No audio devices found (or sounddevice is unavailable).")
        return 0

    for device in devices:
        print(
            f"[{device['index']}] {device['name']} "
            f"inputs={device['max_input_channels']} default_sr={device['default_samplerate']}"
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

    load_and_configure_logging(args)
    session_id = args.session_id or default_session_id("live")
    output_dir = args.out / session_id
    config = LiveSessionConfig(
        transcription_model=args.transcription_model,
        duration_sec=args.duration_sec,
        chunk_sec=args.chunk_sec,
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
    try:
        result = run_live_transcription_session(config, use_fixture=args.fixture)
    except RuntimeError as exc:
        print(f"Session failed: {exc}")
        return 2

    print(f"Session complete: {result.session_dir}")
    print(f"Events JSONL: {result.events_path}")
    print(f"Transcript JSON: {result.transcript_json_path}")
    print(f"Transcript TXT: {result.transcript_txt_path}")
    print(f"Final segments: {result.final_segment_count}")
    print(f"Partial events: {result.partial_event_count}")
    print(
        "Sample rate (requested/effective Hz): "
        f"{result.sample_rate_hz_requested}/{result.sample_rate_hz}"
    )
    print(f"Source selections: {result.source_selection_counts}")
    print(f"Audio sec: {result.total_audio_sec:.3f}")
    print(f"Inference sec: {result.total_inference_sec:.3f}")
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Parameters
    ----------
    argv : list[str] | None, optional
        CLI arguments. Uses process argv when ``None``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "capture" and args.capture_command == "run":
        return run_capture(args)
    if args.command == "capture" and args.capture_command == "devices":
        return run_devices(args)
    if args.command == "bench" and args.bench_command == "run":
        return run_benchmark(args)
    if args.command == "session" and args.session_command == "run":
        return run_session(args)
    if args.command == "compliance" and args.compliance_command == "check-no-network":
        return run_check_no_network(args)
    if args.command == "compliance" and args.compliance_command == "check-no-urls":
        return run_check_no_urls(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
