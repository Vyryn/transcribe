from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from transcribe.audio.linux_capture import LinuxAudioCaptureBackend
from transcribe.audio.sync import SyncAccumulator, build_captured_pair, record_pair_stats, summarize_stats
from transcribe.io.session_manifest import write_session_manifest
from transcribe.io.wav_writer import Pcm16MonoWavWriter
from transcribe.models import CaptureConfig, SessionManifest


class CaptureRunResult:
    """Container for capture-session outputs."""

    def __init__(self, manifest: SessionManifest, manifest_path: Path) -> None:
        """Initialize result wrapper.

        Parameters
        ----------
        manifest : SessionManifest
            Session metadata.
        manifest_path : Path
            Persisted manifest path.
        """
        self.manifest = manifest
        self.manifest_path = manifest_path


def run_capture_session(
    config: CaptureConfig,
    *,
    duration_sec: float,
    use_fixture: bool = False,
) -> CaptureRunResult:
    """Run a capture session and persist artifacts.

    Parameters
    ----------
    config : CaptureConfig
        Capture configuration.
    duration_sec : float
        Capture duration in seconds.
    use_fixture : bool, optional
        If ``True``, uses synthetic frames.

    Returns
    -------
    CaptureRunResult
        Result including manifest and manifest path.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mic_path = output_dir / "mic.wav"
    speakers_path = output_dir / "speakers.wav"
    manifest_path = output_dir / "session_manifest.json"

    backend = LinuxAudioCaptureBackend(use_fixture=use_fixture)
    backend.open(config)

    frame_index = 0
    stats = SyncAccumulator()
    started_monotonic = time.monotonic()
    started_monotonic_ns = time.monotonic_ns()

    try:
        with (
            Pcm16MonoWavWriter(mic_path, sample_rate_hz=config.sample_rate_hz) as mic_writer,
            Pcm16MonoWavWriter(speakers_path, sample_rate_hz=config.sample_rate_hz) as speakers_writer,
        ):
            while time.monotonic() - started_monotonic < duration_sec:
                try:
                    raw = backend.read_frames(timeout_ms=config.frame_ms * 3)
                except TimeoutError:
                    stats.dropped_pairs += 1
                    continue

                mic_frame, speakers_frame, drift_ns = build_captured_pair(raw, frame_index)
                mic_writer.write(mic_frame.mono_pcm16)
                speakers_writer.write(speakers_frame.mono_pcm16)

                callback_reference_ns = max(
                    mic_frame.captured_at_monotonic_ns,
                    speakers_frame.captured_at_monotonic_ns,
                )
                callback_to_write_latency_ns = max(0, time.monotonic_ns() - callback_reference_ns)
                record_pair_stats(
                    stats,
                    drift_ns=drift_ns,
                    callback_to_write_latency_ns=callback_to_write_latency_ns,
                )
                frame_index += 1
    finally:
        backend.close()

    actual_duration_sec = time.monotonic() - started_monotonic

    capture_stats = summarize_stats(stats)
    capture_stats.update(
        {
            "duration_sec_requested": duration_sec,
            "duration_sec_actual": actual_duration_sec,
            "frame_index_final": frame_index,
            "callback_drops": backend.dropped_callback_frames,
            "elapsed_monotonic_ns": time.monotonic_ns() - started_monotonic_ns,
            "frames_written_mic": frame_index,
            "frames_written_speakers": frame_index,
        }
    )

    manifest = SessionManifest(
        session_id=config.session_id,
        created_at_utc=datetime.now(timezone.utc),
        source_mode=config.source_mode,
        sample_rate_hz=config.sample_rate_hz,
        frame_ms=config.frame_ms,
        channels=config.channels,
        artifacts={
            "mic_wav": str(mic_path),
            "speakers_wav": str(speakers_path),
            "session_manifest": str(manifest_path),
        },
        capture_stats=capture_stats,
    )

    write_session_manifest(manifest, manifest_path)
    return CaptureRunResult(manifest=manifest, manifest_path=manifest_path)


def with_session_id(config: CaptureConfig, *, session_id: str, output_dir: Path) -> CaptureConfig:
    """Clone capture config with an updated session identity and output location."""
    return replace(config, session_id=session_id, output_dir=output_dir)
