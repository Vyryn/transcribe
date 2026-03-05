from __future__ import annotations

import contextlib
import io
import json
import math
import re
import struct
import time
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from transcribe.audio.linux_capture import LinuxAudioCaptureBackend
from transcribe.models import AudioSourceMode, CaptureConfig
from transcribe.runtime_defaults import DEFAULT_LIVE_TRANSCRIPTION_MODEL

ChunkTranscriber = Callable[[bytes, str], tuple[str, float]]
SessionProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(slots=True)
class LiveSessionConfig:
    """Configuration for live streaming transcription sessions."""

    transcription_model: str = DEFAULT_LIVE_TRANSCRIPTION_MODEL
    sample_rate_hz: int = 16_000
    channels: int = 1
    frame_ms: int = 20
    duration_sec: float = 0.0
    chunk_sec: float = 4.0
    chunk_overlap_sec: float = 0.75
    partial_interval_sec: float = 0.0
    source_mode: AudioSourceMode = AudioSourceMode.BOTH
    mic_device: str | int | None = None
    speaker_device: str | int | None = None
    capture_all_mic_devices: bool = True
    capture_all_speaker_devices: bool = True
    allow_missing_sources: bool = True
    output_dir: Path = Path("data/live_sessions")
    session_id: str = "live-session"
    max_model_ram_gb: float = 8.0


@dataclass(slots=True)
class LiveSessionResult:
    """Persisted outputs and high-level metrics from a live session."""

    session_dir: Path
    events_path: Path
    transcript_json_path: Path
    transcript_txt_path: Path
    final_segment_count: int
    partial_event_count: int
    sample_rate_hz: int
    sample_rate_hz_requested: int
    total_audio_sec: float
    total_inference_sec: float
    source_selection_counts: dict[str, int]
    interrupted: bool


def _normalize_text(text: str) -> str:
    """Normalize whitespace for human-readable transcript text."""
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    if lowered in {
        "transcript",
        "transcript.",
        "transcription",
        "transcription.",
        "reference",
        "reference.",
        "transcribe the following",
        "transcribe the following.",
    }:
        return ""
    if lowered.startswith("transcript:"):
        return " ".join(normalized.split(":", 1)[1].split())
    if lowered.startswith("transcribe the following"):
        prompt_remainder = normalized[len("Transcribe the following") :].strip(" :.-")
        if not prompt_remainder or len(prompt_remainder.split()) <= 3:
            return ""
        return " ".join(prompt_remainder.split())
    return normalized


def _is_parakeet_model_id(transcription_model: str) -> bool:
    """Return True when model id refers to NVIDIA Parakeet family."""
    return transcription_model.strip().lower().startswith("nvidia/parakeet-")


def _build_live_transcription_runtime_error(transcription_model: str, exc: BaseException) -> RuntimeError:
    """Build an actionable runtime error for live transcription failures."""
    if _is_parakeet_model_id(transcription_model):
        return RuntimeError(
            "Parakeet decoder failed in this runtime "
            f"({type(exc).__name__}: {exc}). "
            "Retry with fallback model: "
            "`uv run transcribe session run --model Qwen/Qwen3-ASR-0.6B --duration-sec 0`."
        )
    return RuntimeError(
        f"Live transcription failed for model {transcription_model!r} "
        f"({type(exc).__name__}: {exc})."
    )


def _pcm16_to_wav_bytes(*, pcm16_bytes: bytes, sample_rate_hz: int, channels: int) -> bytes:
    """Materialize raw PCM16 mono/stereo bytes into an in-memory WAV payload."""
    with BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate_hz)
            wav.writeframes(pcm16_bytes)
        return buffer.getvalue()


def _trim_chunk_silence_pcm16(
    pcm16: bytes,
    *,
    sample_rate_hz: int,
    activation_abs: int = 600,
    edge_padding_sec: float = 0.12,
) -> bytes:
    """Trim leading/trailing low-energy samples to focus ASR on speech regions."""
    if not pcm16:
        return b""
    sample_count = len(pcm16) // 2
    if sample_count <= 0:
        return b""

    first_active_index = -1
    last_active_index = -1
    for sample_index, (sample,) in enumerate(struct.iter_unpack("<h", pcm16)):
        if abs(sample) >= activation_abs:
            if first_active_index < 0:
                first_active_index = sample_index
            last_active_index = sample_index

    if first_active_index < 0 or last_active_index < 0:
        return b""

    padding_samples = int(sample_rate_hz * edge_padding_sec)
    start_index = max(0, first_active_index - padding_samples)
    end_index = min(sample_count, last_active_index + 1 + padding_samples)
    start_byte = start_index * 2
    end_byte = end_index * 2
    if start_byte <= 0 and end_byte >= len(pcm16):
        return pcm16
    return pcm16[start_byte:end_byte]


def _resample_pcm16_mono_linear(
    pcm16: bytes,
    *,
    source_rate_hz: int,
    target_rate_hz: int,
) -> bytes:
    """Resample mono PCM16 bytes to target sample rate via linear interpolation."""
    if not pcm16 or source_rate_hz <= 0 or target_rate_hz <= 0 or source_rate_hz == target_rate_hz:
        return pcm16

    input_samples = [sample for (sample,) in struct.iter_unpack("<h", pcm16)]
    input_count = len(input_samples)
    if input_count <= 1:
        return pcm16

    # Fast-path integer downsample ratios (for example 48k -> 16k).
    if source_rate_hz % target_rate_hz == 0:
        step = source_rate_hz // target_rate_hz
        if step > 1:
            reduced = input_samples[::step]
            return struct.pack(f"<{len(reduced)}h", *reduced)

    output_count = max(1, int(round(input_count * (target_rate_hz / source_rate_hz))))
    if output_count == input_count:
        return pcm16

    output = bytearray(output_count * 2)
    position_scale = (input_count - 1) / max(1, output_count - 1)
    for output_index in range(output_count):
        source_position = output_index * position_scale
        left_index = int(source_position)
        right_index = min(left_index + 1, input_count - 1)
        interpolation = source_position - left_index
        sample_value = ((1.0 - interpolation) * input_samples[left_index]) + (interpolation * input_samples[right_index])
        sample_int = int(round(sample_value))
        if sample_int > 32_767:
            sample_int = 32_767
        elif sample_int < -32_768:
            sample_int = -32_768
        struct.pack_into("<h", output, output_index * 2, sample_int)
    return bytes(output)


def _prepare_pcm16_for_asr(
    pcm16: bytes,
    *,
    capture_sample_rate_hz: int,
    target_sample_rate_hz: int,
    channels: int,
) -> bytes:
    """Trim silence and resample a chunk before ASR inference."""
    if not pcm16:
        return b""

    prepared = _trim_chunk_silence_pcm16(
        pcm16,
        sample_rate_hz=capture_sample_rate_hz,
    )
    if not prepared:
        return b""

    if channels != 1:
        return prepared

    return _resample_pcm16_mono_linear(
        prepared,
        source_rate_hz=capture_sample_rate_hz,
        target_rate_hz=target_sample_rate_hz,
    )


def _retain_chunk_overlap(
    chunk_pcm16_by_source: dict[str, bytearray],
    *,
    overlap_bytes: int,
) -> dict[str, bytearray]:
    """Keep a short tail from each source buffer for overlap-aware chunking."""
    if overlap_bytes <= 0:
        return {}

    retained: dict[str, bytearray] = {}
    for source_name, source_pcm16 in chunk_pcm16_by_source.items():
        if not source_pcm16:
            continue
        tail = bytes(source_pcm16[-overlap_bytes:])
        if tail:
            retained[source_name] = bytearray(tail)
    return retained


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """Return normalized words with source spans for overlap stitching."""
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[A-Za-z0-9']+", text):
        spans.append((match.group(0).lower(), match.start(), match.end()))
    return spans


def _stitch_text_overlap(previous_text: str, current_text: str, *, max_overlap_words: int = 12) -> str:
    """Remove repeated leading words caused by chunk overlap."""
    if not previous_text or not current_text:
        return current_text

    previous_words = _word_spans(previous_text)
    current_words = _word_spans(current_text)
    if not previous_words or not current_words:
        return current_text

    overlap_limit = min(max_overlap_words, len(previous_words), len(current_words))
    for overlap_size in range(overlap_limit, 0, -1):
        previous_suffix = [word for word, _, _ in previous_words[-overlap_size:]]
        current_prefix = [word for word, _, _ in current_words[:overlap_size]]
        if previous_suffix != current_prefix:
            continue
        if overlap_size >= len(current_words):
            return ""
        trim_start = current_words[overlap_size][1]
        return current_text[trim_start:].lstrip()
    return current_text


def _build_default_chunk_transcriber(transcription_model: str, max_model_ram_gb: float) -> ChunkTranscriber:
    """Build a model-routed chunk transcriber backed by benchmark ASR loaders."""
    from transcribe.bench.harness import (
        _default_hf_segment_transcriber,
        _enforce_model_ram_limit,
    )

    _enforce_model_ram_limit(transcription_model, max_model_ram_gb)
    row_transcriber = _default_hf_segment_transcriber(transcription_model)

    def _transcribe(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        row = {
            "audio": {
                "bytes": wav_bytes,
                "path": "live_chunk.wav",
            }
        }
        return row_transcriber(row, model_id)

    return _transcribe


def _emit_progress(
    progress_callback: SessionProgressCallback | None,
    event: str,
    **fields: object,
) -> None:
    """Report a structured live-session progress event."""
    if progress_callback is None:
        return
    progress_callback(event, dict(fields))


@contextlib.contextmanager
def _suppress_backend_output(enabled: bool) -> Iterator[None]:
    """Temporarily silence stdout/stderr from chatty backend libraries."""
    if not enabled:
        yield
        return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        yield


def _warm_default_chunk_transcriber(transcription_model: str, max_model_ram_gb: float) -> None:
    """Eagerly load the default backend so CLI loading feedback is explicit."""
    from transcribe.bench.harness import preload_transcription_model

    preload_transcription_model(
        transcription_model,
        max_model_ram_gb=max_model_ram_gb,
    )


def _transcribe_chunk(
    chunk_pcm16: bytearray,
    *,
    capture_sample_rate_hz: int,
    transcription_sample_rate_hz: int,
    channels: int,
    transcription_model: str,
    transcriber: ChunkTranscriber,
) -> tuple[str, float]:
    """Run one ASR pass for a buffered PCM chunk."""
    prepared_pcm16 = _prepare_pcm16_for_asr(
        bytes(chunk_pcm16),
        capture_sample_rate_hz=capture_sample_rate_hz,
        target_sample_rate_hz=transcription_sample_rate_hz,
        channels=channels,
    )
    if not prepared_pcm16:
        return "", 0.0

    wav_bytes = _pcm16_to_wav_bytes(
        pcm16_bytes=prepared_pcm16,
        sample_rate_hz=transcription_sample_rate_hz,
        channels=channels,
    )
    text, inference_latency_ms = transcriber(wav_bytes, transcription_model)
    return _normalize_text(text), float(inference_latency_ms)


def _transcribe_chunk_or_raise(
    chunk_pcm16: bytearray,
    *,
    capture_sample_rate_hz: int,
    transcription_sample_rate_hz: int,
    channels: int,
    transcription_model: str,
    transcriber: ChunkTranscriber,
    suppress_backend_output: bool = False,
) -> tuple[str, float]:
    """Transcribe one chunk and surface user-actionable runtime failures."""
    try:
        with _suppress_backend_output(suppress_backend_output):
            return _transcribe_chunk(
                chunk_pcm16,
                capture_sample_rate_hz=capture_sample_rate_hz,
                transcription_sample_rate_hz=transcription_sample_rate_hz,
                channels=channels,
                transcription_model=transcription_model,
                transcriber=transcriber,
            )
    except Exception as exc:  # noqa: BLE001
        raise _build_live_transcription_runtime_error(transcription_model, exc) from exc


def _pcm16_clarity_score(pcm16: bytes, *, sample_rate_hz: int) -> float:
    """Estimate speech clarity from chunk-level PCM16 dynamics."""
    sample_count = len(pcm16) // 2
    if sample_count <= 0:
        return float("-inf")

    sum_sq = 0.0
    sum_samples = 0
    peak_abs = 0
    clipped_count = 0
    active_count = 0
    frame_rms_db: list[float] = []
    frame_size = max(1, int(sample_rate_hz * 0.02))
    rolling_sum_sq = 0.0
    rolling_count = 0

    for sample_index, (sample,) in enumerate(struct.iter_unpack("<h", pcm16), start=1):
        abs_value = abs(sample)
        sum_sq += float(sample * sample)
        sum_samples += sample
        peak_abs = max(peak_abs, abs_value)
        if abs_value >= 500:
            active_count += 1
        if abs_value >= 32_000:
            clipped_count += 1
        rolling_sum_sq += float(sample * sample)
        rolling_count += 1
        if rolling_count >= frame_size or sample_index == sample_count:
            rms = math.sqrt(rolling_sum_sq / max(1, rolling_count)) / 32_768.0
            frame_rms_db.append(20.0 * math.log10(max(rms, 1e-8)))
            rolling_sum_sq = 0.0
            rolling_count = 0

    rms = math.sqrt(sum_sq / sample_count) / 32_768.0
    peak = peak_abs / 32_768.0
    clipping_ratio = clipped_count / sample_count
    dc_offset = abs(sum_samples / sample_count) / 32_768.0
    activity_ratio = active_count / sample_count

    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    if rms_db < -55.0:
        loudness = -0.5
    elif rms_db > -6.0:
        loudness = 0.2
    else:
        loudness = (rms_db + 55.0) / 49.0

    modulation_db = 0.0
    if frame_rms_db:
        mean_db = sum(frame_rms_db) / len(frame_rms_db)
        variance_db = sum((value - mean_db) ** 2 for value in frame_rms_db) / len(frame_rms_db)
        modulation_db = math.sqrt(variance_db)
    modulation = min(modulation_db / 18.0, 1.0)

    return (
        (1.5 * loudness)
        + (1.1 * activity_ratio)
        + (0.6 * modulation)
        + (0.2 * peak)
        - (2.8 * clipping_ratio)
        - (0.5 * dc_offset)
    )


def _select_best_source_chunk(
    chunk_pcm16_by_source: dict[str, bytearray],
    *,
    sample_rate_hz: int,
    previous_source: str | None,
) -> tuple[str, bytes, dict[str, float]] | None:
    """Select chunk audio source using clarity score with anti-flap hysteresis."""
    if not chunk_pcm16_by_source:
        return None

    source_scores: dict[str, float] = {}
    for source_name, source_pcm16 in chunk_pcm16_by_source.items():
        source_scores[source_name] = _pcm16_clarity_score(bytes(source_pcm16), sample_rate_hz=sample_rate_hz)

    best_source = max(source_scores, key=source_scores.get)
    best_score = source_scores[best_source]

    selected_source = best_source
    hysteresis_margin = 0.15
    if previous_source is not None and previous_source in source_scores:
        previous_score = source_scores[previous_source]
        if previous_score >= (best_score - hysteresis_margin):
            selected_source = previous_source

    selected_pcm16 = bytes(chunk_pcm16_by_source[selected_source])
    if not selected_pcm16:
        return None
    return selected_source, selected_pcm16, source_scores


def _max_buffered_audio_sec(
    chunk_pcm16_by_source: dict[str, bytearray],
    *,
    bytes_per_second: float,
) -> float:
    """Return maximum buffered audio seconds across all source buffers."""
    if not chunk_pcm16_by_source or bytes_per_second <= 0:
        return 0.0
    return max((len(source_pcm16) / bytes_per_second) for source_pcm16 in chunk_pcm16_by_source.values())


def _should_skip_asr_for_chunk(
    selected_pcm16: bytes,
    *,
    clarity_score: float,
    backlog_ratio: float = 0.0,
    recent_empty_streak: int = 0,
) -> bool:
    """Decide whether a chunk is silence/noise-only and safe to skip for catch-up."""
    sample_count = len(selected_pcm16) // 2
    if sample_count <= 0:
        return True

    sum_sq = 0.0
    active_count = 0
    for (sample,) in struct.iter_unpack("<h", selected_pcm16):
        sum_sq += float(sample * sample)
        if abs(sample) >= 700:
            active_count += 1
    rms = math.sqrt(sum_sq / sample_count) / 32_768.0
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    active_ratio = active_count / sample_count

    if rms_db < -45.0:
        return True
    if rms_db < -38.0 and active_ratio < 0.10:
        return True
    if clarity_score < 0.2 and rms_db < -40.0:
        return True

    # Catch-up mode: when inference load is high, be stricter about low-information chunks.
    if backlog_ratio > 0.50 and (clarity_score < 1.90 or active_ratio < 0.15):
        return True

    # If we recently produced repeated empty outputs, suppress more low-information chunks.
    if recent_empty_streak >= 2 and (clarity_score < 2.00 or rms_db < -34.0):
        return True

    return False


def run_live_transcription_session(
    config: LiveSessionConfig,
    *,
    use_fixture: bool = False,
    transcriber: ChunkTranscriber | None = None,
    debug: bool = False,
    progress_callback: SessionProgressCallback | None = None,
) -> LiveSessionResult:
    """Run a live multi-source transcription session with partial/final events."""
    if config.chunk_sec <= 0:
        raise ValueError("chunk_sec must be > 0")
    if config.chunk_overlap_sec < 0:
        raise ValueError("chunk_overlap_sec must be >= 0")
    if config.partial_interval_sec < 0:
        raise ValueError("partial_interval_sec must be >= 0")
    if config.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if config.channels <= 0:
        raise ValueError("channels must be > 0")
    if config.frame_ms <= 0:
        raise ValueError("frame_ms must be > 0")

    chunk_transcriber: ChunkTranscriber
    if transcriber is None:
        _emit_progress(
            progress_callback,
            "loading_model",
            transcription_model=config.transcription_model,
        )
        with _suppress_backend_output(not debug):
            _warm_default_chunk_transcriber(
                config.transcription_model,
                config.max_model_ram_gb,
            )
        chunk_transcriber = _build_default_chunk_transcriber(
            config.transcription_model,
            config.max_model_ram_gb,
        )
        _emit_progress(
            progress_callback,
            "model_ready",
            transcription_model=config.transcription_model,
        )
    else:
        chunk_transcriber = transcriber

    session_dir = config.output_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    events_path = session_dir / "events.jsonl"
    transcript_json_path = session_dir / "transcript.json"
    transcript_txt_path = session_dir / "transcript.txt"

    capture_config = CaptureConfig(
        sample_rate_hz=config.sample_rate_hz,
        channels=config.channels,
        frame_ms=config.frame_ms,
        source_mode=config.source_mode,
        mic_device=config.mic_device,
        speaker_device=config.speaker_device,
        capture_all_mic_devices=config.capture_all_mic_devices,
        capture_all_speaker_devices=config.capture_all_speaker_devices,
        allow_missing_sources=config.allow_missing_sources,
        session_id=config.session_id,
        output_dir=session_dir,
    )

    backend = LinuxAudioCaptureBackend(use_fixture=use_fixture)
    backend.open(capture_config)
    capture_sample_rate_hz = backend.sample_rate_hz or config.sample_rate_hz
    transcription_sample_rate_hz = int(config.sample_rate_hz)
    if transcription_sample_rate_hz <= 0:
        transcription_sample_rate_hz = int(capture_sample_rate_hz)
    resolved_capture_devices = dict(getattr(backend, "active_devices", {}))

    _emit_progress(
        progress_callback,
        "capture_ready",
        source_mode=config.source_mode.value,
        requested_sample_rate_hz=config.sample_rate_hz,
        capture_sample_rate_hz=capture_sample_rate_hz,
        transcription_sample_rate_hz=transcription_sample_rate_hz,
        resolved_capture_devices={
            key: [str(device) for device in devices]
            for key, devices in resolved_capture_devices.items()
        },
    )
    _emit_progress(
        progress_callback,
        "transcribing_started",
        chunk_sec=config.chunk_sec,
        partial_interval_sec=config.partial_interval_sec,
        duration_sec=config.duration_sec,
    )

    bytes_per_second = float(capture_sample_rate_hz * config.channels * 2)
    effective_chunk_overlap_sec = min(
        config.chunk_overlap_sec,
        max(config.chunk_sec - (config.frame_ms / 1000.0), 0.0),
    )
    sample_frame_bytes = max(2, config.channels * 2)
    chunk_overlap_bytes = int(round(bytes_per_second * effective_chunk_overlap_sec))
    chunk_overlap_bytes -= chunk_overlap_bytes % sample_frame_bytes
    chunk_index = 1
    chunk_started = time.monotonic()
    started = chunk_started
    last_partial_at = chunk_started
    chunk_pcm16_by_source: dict[str, bytearray] = {}
    last_partial_text = ""
    last_selected_source: str | None = None
    final_segments: list[dict[str, object]] = []
    partial_event_count = 0
    total_audio_bytes = 0
    total_inference_ms = 0.0
    skipped_silence_chunks = 0
    dropped_empty_chunk_count = 0
    empty_output_streak = 0
    source_selection_counts: dict[str, int] = {}
    interrupted = False
    previous_final_text = ""

    def _write_event(event: dict[str, object], handle: object) -> None:
        handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        handle.flush()

    try:
        with events_path.open("w", encoding="utf-8") as events_file:
            while True:
                now = time.monotonic()
                if config.duration_sec > 0 and (now - started) >= config.duration_sec:
                    break

                try:
                    frames = backend.read_frames(timeout_ms=config.frame_ms * 3)
                except TimeoutError:
                    continue

                for source_name, frame in frames.items():
                    source_buffer = chunk_pcm16_by_source.setdefault(source_name, bytearray())
                    source_buffer.extend(frame.mono_pcm16)
                now = time.monotonic()
                buffered_audio_sec = _max_buffered_audio_sec(
                    chunk_pcm16_by_source,
                    bytes_per_second=bytes_per_second,
                )

                if config.partial_interval_sec > 0 and (now - last_partial_at) >= config.partial_interval_sec:
                    if buffered_audio_sec < 0.5:
                        last_partial_at = now
                        continue
                    selected_chunk = _select_best_source_chunk(
                        chunk_pcm16_by_source,
                        sample_rate_hz=capture_sample_rate_hz,
                        previous_source=last_selected_source,
                    )
                    if selected_chunk is None:
                        last_partial_at = now
                        continue
                    selected_source, selected_pcm16, source_scores = selected_chunk
                    if (len(selected_pcm16) / bytes_per_second) < 0.5:
                        last_partial_at = now
                        continue
                    selected_score = source_scores.get(selected_source, float("-inf"))
                    elapsed_sec = max(now - started, 1e-6)
                    backlog_ratio = (total_inference_ms / 1000.0) / elapsed_sec
                    if _should_skip_asr_for_chunk(
                        selected_pcm16,
                        clarity_score=selected_score,
                        backlog_ratio=backlog_ratio,
                        recent_empty_streak=empty_output_streak,
                    ):
                        last_partial_at = now
                        continue
                    last_selected_source = selected_source
                    text, latency_ms = _transcribe_chunk_or_raise(
                        bytearray(selected_pcm16),
                        capture_sample_rate_hz=capture_sample_rate_hz,
                        transcription_sample_rate_hz=transcription_sample_rate_hz,
                        channels=config.channels,
                        transcription_model=config.transcription_model,
                        transcriber=chunk_transcriber,
                        suppress_backend_output=not debug,
                    )
                    total_inference_ms += latency_ms
                    if text and text != last_partial_text:
                        partial_event = {
                            "event": "partial",
                            "session_id": config.session_id,
                            "chunk_index": chunk_index,
                            "audio_sec": len(selected_pcm16) / bytes_per_second,
                            "inference_latency_ms": latency_ms,
                            "selected_source": selected_source,
                            "source_scores": source_scores,
                            "text": text,
                            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        _write_event(partial_event, events_file)
                        _emit_progress(
                            progress_callback,
                            "partial",
                            chunk_index=chunk_index,
                            text=text,
                            selected_source=selected_source,
                        )
                        partial_event_count += 1
                        last_partial_text = text
                    last_partial_at = now

                if buffered_audio_sec < config.chunk_sec:
                    continue

                selected_chunk = _select_best_source_chunk(
                    chunk_pcm16_by_source,
                    sample_rate_hz=capture_sample_rate_hz,
                    previous_source=last_selected_source,
                )
                if selected_chunk is None:
                    continue
                selected_source, selected_pcm16, source_scores = selected_chunk
                selected_audio_sec = len(selected_pcm16) / bytes_per_second
                if selected_audio_sec < config.chunk_sec:
                    continue

                last_selected_source = selected_source
                selected_score = source_scores.get(selected_source, float("-inf"))
                elapsed_sec = max(now - started, 1e-6)
                backlog_ratio = (total_inference_ms / 1000.0) / elapsed_sec
                if _should_skip_asr_for_chunk(
                    selected_pcm16,
                    clarity_score=selected_score,
                    backlog_ratio=backlog_ratio,
                    recent_empty_streak=empty_output_streak,
                ):
                    skipped_silence_chunks += 1
                    source_selection_counts[selected_source] = source_selection_counts.get(selected_source, 0) + 1
                    total_audio_bytes += len(selected_pcm16)
                    empty_output_streak += 1
                    chunk_index += 1
                    chunk_started = now
                    last_partial_at = now
                    last_partial_text = ""
                    chunk_pcm16_by_source = {}
                    continue
                else:
                    text, latency_ms = _transcribe_chunk_or_raise(
                        bytearray(selected_pcm16),
                        capture_sample_rate_hz=capture_sample_rate_hz,
                        transcription_sample_rate_hz=transcription_sample_rate_hz,
                        channels=config.channels,
                        transcription_model=config.transcription_model,
                        transcriber=chunk_transcriber,
                        suppress_backend_output=not debug,
                    )
                    total_inference_ms += latency_ms

                if not text:
                    dropped_empty_chunk_count += 1
                    source_selection_counts[selected_source] = source_selection_counts.get(selected_source, 0) + 1
                    total_audio_bytes += len(selected_pcm16)
                    empty_output_streak += 1
                    chunk_index += 1
                    chunk_started = now
                    last_partial_at = now
                    last_partial_text = ""
                    chunk_pcm16_by_source = {}
                    continue

                stitched_text = _stitch_text_overlap(previous_final_text, text)
                if stitched_text:
                    text = stitched_text

                empty_output_streak = 0
                total_audio_bytes += len(selected_pcm16)
                source_selection_counts[selected_source] = source_selection_counts.get(selected_source, 0) + 1
                segment_event = {
                    "event": "final",
                    "session_id": config.session_id,
                    "chunk_index": chunk_index,
                    "audio_sec": selected_audio_sec,
                    "inference_latency_ms": latency_ms,
                    "selected_source": selected_source,
                    "source_scores": source_scores,
                    "silence_skipped": False,
                    "text": text,
                    "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _write_event(segment_event, events_file)
                _emit_progress(
                    progress_callback,
                    "final",
                    chunk_index=chunk_index,
                    text=text,
                    selected_source=selected_source,
                )
                final_segments.append(
                    {
                        "chunk_index": chunk_index,
                        "text": text,
                        "audio_sec": selected_audio_sec,
                        "inference_latency_ms": latency_ms,
                        "selected_source": selected_source,
                        "source_scores": source_scores,
                        "silence_skipped": bool(latency_ms == 0.0 and not text),
                    }
                )
                previous_final_text = text
                chunk_index += 1
                chunk_started = now
                last_partial_at = now
                last_partial_text = ""
                chunk_pcm16_by_source = _retain_chunk_overlap(
                    chunk_pcm16_by_source,
                    overlap_bytes=chunk_overlap_bytes,
                )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        backend.close()

    if any(chunk_pcm16_by_source.values()):
        selected_chunk = _select_best_source_chunk(
            chunk_pcm16_by_source,
            sample_rate_hz=capture_sample_rate_hz,
            previous_source=last_selected_source,
        )
        if selected_chunk is None:
            selected_source = ""
            selected_pcm16 = b""
            source_scores: dict[str, float] = {}
        else:
            selected_source, selected_pcm16, source_scores = selected_chunk
        if not selected_pcm16:
            selected_source = ""
            source_scores = {}
        selected_audio_sec = len(selected_pcm16) / bytes_per_second if selected_pcm16 else 0.0
        if selected_source and selected_audio_sec >= 0.5:
            selected_score = source_scores.get(selected_source, float("-inf"))
            elapsed_sec = max((time.monotonic() - started), 1e-6)
            backlog_ratio = (total_inference_ms / 1000.0) / elapsed_sec
            if _should_skip_asr_for_chunk(
                selected_pcm16,
                clarity_score=selected_score,
                backlog_ratio=backlog_ratio,
                recent_empty_streak=empty_output_streak,
            ):
                skipped_silence_chunks += 1
                source_selection_counts[selected_source] = source_selection_counts.get(selected_source, 0) + 1
                total_audio_bytes += len(selected_pcm16)
                empty_output_streak += 1
                selected_source = ""
            else:
                text, latency_ms = _transcribe_chunk_or_raise(
                    bytearray(selected_pcm16),
                    capture_sample_rate_hz=capture_sample_rate_hz,
                    transcription_sample_rate_hz=transcription_sample_rate_hz,
                    channels=config.channels,
                    transcription_model=config.transcription_model,
                    transcriber=chunk_transcriber,
                    suppress_backend_output=not debug,
                )
                total_inference_ms += latency_ms
                if text:
                    stitched_text = _stitch_text_overlap(previous_final_text, text)
                    if stitched_text:
                        text = stitched_text
                    empty_output_streak = 0
                    total_audio_bytes += len(selected_pcm16)
                    source_selection_counts[selected_source] = source_selection_counts.get(selected_source, 0) + 1
                    with events_path.open("a", encoding="utf-8") as events_file:
                        segment_event = {
                            "event": "final",
                            "session_id": config.session_id,
                            "chunk_index": chunk_index,
                            "audio_sec": selected_audio_sec,
                            "inference_latency_ms": latency_ms,
                            "selected_source": selected_source,
                            "source_scores": source_scores,
                            "silence_skipped": False,
                            "text": text,
                            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        _write_event(segment_event, events_file)
                    _emit_progress(
                        progress_callback,
                        "final",
                        chunk_index=chunk_index,
                        text=text,
                        selected_source=selected_source,
                    )
                    final_segments.append(
                        {
                            "chunk_index": chunk_index,
                            "text": text,
                            "audio_sec": selected_audio_sec,
                            "inference_latency_ms": latency_ms,
                            "selected_source": selected_source,
                            "source_scores": source_scores,
                            "silence_skipped": False,
                        }
                    )
                    previous_final_text = text
                else:
                    dropped_empty_chunk_count += 1
                    source_selection_counts[selected_source] = source_selection_counts.get(selected_source, 0) + 1
                    total_audio_bytes += len(selected_pcm16)
                    empty_output_streak += 1

    total_audio_sec = total_audio_bytes / bytes_per_second
    session_elapsed_sec = time.monotonic() - started
    transcript_payload = {
        "schema_version": "phase1-live-session-v1",
        "session_id": config.session_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "transcription_model": config.transcription_model,
        "sample_rate_hz_requested": config.sample_rate_hz,
        "sample_rate_hz": capture_sample_rate_hz,
        "transcription_sample_rate_hz": transcription_sample_rate_hz,
        "channels": config.channels,
        "frame_ms": config.frame_ms,
        "chunk_sec": config.chunk_sec,
        "chunk_overlap_sec": config.chunk_overlap_sec,
        "chunk_overlap_sec_effective": effective_chunk_overlap_sec,
        "partial_interval_sec": config.partial_interval_sec,
        "source_mode": config.source_mode.value,
        "mic_device": config.mic_device,
        "speaker_device": config.speaker_device,
        "capture_all_mic_devices": config.capture_all_mic_devices,
        "capture_all_speaker_devices": config.capture_all_speaker_devices,
        "resolved_capture_devices": {
            key: [str(device) for device in devices]
            for key, devices in resolved_capture_devices.items()
        },
        "source_selection_strategy": "chunk_clarity_v5_silence_catchup",
        "duration_sec_requested": config.duration_sec,
        "duration_sec_actual": session_elapsed_sec,
        "interrupted": interrupted,
        "metrics": {
            "final_segment_count": len(final_segments),
            "partial_event_count": partial_event_count,
            "total_audio_sec": total_audio_sec,
            "total_inference_sec": total_inference_ms / 1000.0,
            "capture_coverage_ratio": (total_audio_sec / session_elapsed_sec) if session_elapsed_sec > 0 else 0.0,
            "silence_skipped_chunk_count": skipped_silence_chunks,
            "dropped_empty_chunk_count": dropped_empty_chunk_count,
            "source_selection_counts": source_selection_counts,
        },
        "final_segments": final_segments,
    }
    transcript_json_path.write_text(json.dumps(transcript_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    transcript_txt_path.write_text(
        "\n".join(segment["text"] for segment in final_segments if isinstance(segment.get("text"), str)) + "\n",
        encoding="utf-8",
    )

    return LiveSessionResult(
        session_dir=session_dir,
        events_path=events_path,
        transcript_json_path=transcript_json_path,
        transcript_txt_path=transcript_txt_path,
        final_segment_count=len(final_segments),
        partial_event_count=partial_event_count,
        sample_rate_hz=capture_sample_rate_hz,
        sample_rate_hz_requested=config.sample_rate_hz,
        total_audio_sec=total_audio_sec,
        total_inference_sec=total_inference_ms / 1000.0,
        source_selection_counts=source_selection_counts,
        interrupted=interrupted,
    )
