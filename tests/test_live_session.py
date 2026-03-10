from __future__ import annotations

import argparse
import json
import struct
import time
import wave
from io import BytesIO
from pathlib import Path

import pytest

from transcribe.live.session import LiveSessionConfig, _stitch_text_overlap, run_live_transcription_session
from transcribe.models import AudioSourceMode


def test_run_live_transcription_session_fixture_writes_artifacts(tmp_path) -> None:
    calls = {"count": 0}

    def fake_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        assert wav_bytes.startswith(b"RIFF")
        assert model_id == "unit-test-model"
        calls["count"] += 1
        return f"chunk-{calls['count']}", 7.5

    session_dir = tmp_path / "live-session"
    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        duration_sec=0.30,
        chunk_sec=0.12,
        partial_interval_sec=0.05,
        output_dir=session_dir,
        session_id="live-test-session",
    )
    result = run_live_transcription_session(
        config,
        use_fixture=True,
        transcriber=fake_transcriber,
    )

    assert result.session_dir == session_dir
    assert result.events_path.exists()
    assert result.transcript_json_path.exists()
    assert result.transcript_txt_path.exists()
    assert result.final_segment_count >= 1
    assert calls["count"] >= result.final_segment_count

    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "phase1-live-session-v1"
    assert payload["session_id"] == "live-test-session"
    assert payload["transcription_model"] == "unit-test-model"
    assert payload["metrics"]["final_segment_count"] == result.final_segment_count

    events = [
        json.loads(line)
        for line in result.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(event["event"] == "final" for event in events)


def test_cli_parser_accepts_session_run_command() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--model",
            "nvidia/parakeet-tdt-0.6b-v3",
            "--duration-sec",
            "0",
            "--fixture",
        ]
    )
    assert args.command == "session"
    assert args.session_command == "run"
    assert args.transcription_model == "nvidia/parakeet-tdt-0.6b-v3"
    assert args.duration_sec == 0.0
    assert args.mode == AudioSourceMode.BOTH
    assert args.fixture is True


def test_cli_parser_uses_larger_default_session_model() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--fixture",
        ]
    )
    assert args.transcription_model == "nvidia/parakeet-tdt-0.6b-v3"
    assert args.chunk_overlap_sec == 0.75
    assert args.stitch_overlap_text is True
    assert args.partial_interval_sec == 0.0


def test_cli_parser_accepts_session_overlap_stitch_flag() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--stitch-overlap-text",
            "--fixture",
        ]
    )
    assert args.stitch_overlap_text is True


def test_cli_parser_accepts_session_overlap_stitch_opt_out() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--no-stitch-overlap-text",
            "--fixture",
        ]
    )
    assert args.stitch_overlap_text is False


def test_cli_parser_accepts_session_run_mic_device_index() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--mic-device",
            "2",
            "--fixture",
        ]
    )
    assert args.mic_device == 2


def test_cli_parser_accepts_session_run_mic_device_bracketed_index() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--mic-device",
            "[2]",
            "--fixture",
        ]
    )
    assert args.mic_device == 2


def test_cli_parser_accepts_session_run_mic_device_name() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--mic-device",
            "USB Audio Mic",
            "--fixture",
        ]
    )
    assert args.mic_device == "USB Audio Mic"


def test_cli_parser_accepts_session_run_speaker_device_index() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--speaker-device",
            "4",
            "--fixture",
        ]
    )
    assert args.speaker_device == 4


def test_cli_parser_accepts_capture_run_mic_device_index() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "capture",
            "run",
            "--mode",
            "both",
            "--mic-device",
            "3",
            "--fixture",
        ]
    )
    assert args.mic_device == 3


def test_cli_parser_accepts_capture_run_speaker_device_index() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "capture",
            "run",
            "--mode",
            "both",
            "--speaker-device",
            "[5]",
            "--fixture",
        ]
    )
    assert args.speaker_device == 5


def test_live_session_uses_requested_sample_rate_for_asr_payload(monkeypatch, tmp_path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.live.session as live_session_module

    observed_rates: list[int] = []

    class FakeBackend:
        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000
            self.device_sample_rates_hz = {"mic:fixture": 48_000}
            self._closed = False

        def open(self, config) -> None:
            _ = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            time.sleep(0.02)
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            voiced_frame = struct.pack("<h", 1_200) * frame_samples
            return {
                "mic": RawFrame(
                    stream="mic",
                    mono_pcm16=voiced_frame,
                    captured_at_monotonic_ns=time.monotonic_ns(),
                    sample_rate_hz=self.sample_rate_hz,
                )
            }

        def close(self) -> None:
            self._closed = True

    def fake_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        _ = model_id
        with wave.open(BytesIO(wav_bytes), "rb") as wav:
            observed_rates.append(wav.getframerate())
        return "ok", 5.0

    monkeypatch.setattr(
        live_session_module,
        "open_audio_backend",
        lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture),
    )
    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        sample_rate_hz=16_000,
        duration_sec=0.08,
        chunk_sec=0.04,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-rate-test",
        session_id="live-rate-test",
    )
    result = run_live_transcription_session(config, use_fixture=False, transcriber=fake_transcriber)
    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))

    assert result.sample_rate_hz_requested == 16_000
    assert result.sample_rate_hz == 16_000
    assert payload["transcription_sample_rate_hz"] == 16_000
    assert payload["device_sample_rates_hz"] == {"mic:fixture": 48_000}
    assert observed_rates
    assert all(rate == 16_000 for rate in observed_rates)


def test_live_session_selects_best_source_and_tracks_source_usage(monkeypatch, tmp_path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.live.session as live_session_module

    class FakeBackend:
        captured_config = None

        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000
            self._calls = 0

        def open(self, config) -> None:
            FakeBackend.captured_config = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            self._calls += 1
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            quiet_frame = struct.pack("<h", 700) * frame_samples
            loud_frame = struct.pack("<h", 7_000) * frame_samples
            return {
                "mic": RawFrame(
                    stream="mic",
                    mono_pcm16=quiet_frame,
                    captured_at_monotonic_ns=time.monotonic_ns(),
                    sample_rate_hz=self.sample_rate_hz,
                ),
                "speakers": RawFrame(
                    stream="speakers",
                    mono_pcm16=loud_frame,
                    captured_at_monotonic_ns=time.monotonic_ns(),
                    sample_rate_hz=self.sample_rate_hz,
                ),
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        live_session_module,
        "open_audio_backend",
        lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture),
    )
    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        sample_rate_hz=16_000,
        duration_sec=0.12,
        chunk_sec=0.06,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-best-source",
        session_id="live-best-source",
    )
    result = run_live_transcription_session(
        config,
        use_fixture=False,
        transcriber=lambda wav_bytes, model_id: ("ok", 1.0),
    )

    assert result.final_segment_count >= 1
    assert FakeBackend.captured_config is not None
    assert FakeBackend.captured_config.source_mode == AudioSourceMode.BOTH
    assert FakeBackend.captured_config.capture_all_mic_devices is True
    assert FakeBackend.captured_config.capture_all_speaker_devices is True
    assert FakeBackend.captured_config.allow_missing_sources is True

    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))
    selection_counts = payload["metrics"]["source_selection_counts"]
    assert selection_counts["speakers"] > 0
    assert selection_counts.get("mic", 0) == 0


def test_live_session_capture_coverage_ratio_stays_reasonable(monkeypatch, tmp_path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.live.session as live_session_module

    class FakeBackend:
        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000

        def open(self, config) -> None:
            _ = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            time.sleep(0.02)
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            frame_pcm = struct.pack("<h", 2_000) * frame_samples
            now_ns = time.monotonic_ns()
            return {
                "mic": RawFrame(
                    stream="mic",
                    mono_pcm16=frame_pcm,
                    captured_at_monotonic_ns=now_ns,
                    sample_rate_hz=self.sample_rate_hz,
                ),
                "speakers": RawFrame(
                    stream="speakers",
                    mono_pcm16=frame_pcm,
                    captured_at_monotonic_ns=now_ns,
                    sample_rate_hz=self.sample_rate_hz,
                ),
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        live_session_module,
        "open_audio_backend",
        lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture),
    )
    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        sample_rate_hz=16_000,
        duration_sec=0.5,
        chunk_sec=0.25,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-coverage",
        session_id="live-coverage",
    )
    result = run_live_transcription_session(
        config,
        use_fixture=False,
        transcriber=lambda wav_bytes, model_id: ("ok", 1.0),
    )

    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))
    coverage_ratio = payload["metrics"]["capture_coverage_ratio"]
    assert coverage_ratio >= 0.45


def test_live_session_filters_placeholder_transcript_outputs(tmp_path) -> None:
    calls = {"count": 0}

    def fake_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        _ = (wav_bytes, model_id)
        calls["count"] += 1
        if calls["count"] == 1:
            return "Transcribe the following", 5.0
        return "Transcript: actual words", 5.0

    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        duration_sec=0.12,
        chunk_sec=0.06,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-filter",
        session_id="live-filter",
    )
    result = run_live_transcription_session(config, use_fixture=True, transcriber=fake_transcriber)
    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))

    final_texts = [segment["text"] for segment in payload["final_segments"]]
    assert all(text != "" for text in final_texts)
    assert "Transcribe the following" not in final_texts
    assert "actual words" in final_texts
    assert payload["metrics"]["dropped_empty_chunk_count"] >= 1


def test_live_session_skips_asr_for_silent_chunks(monkeypatch, tmp_path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.live.session as live_session_module

    calls = {"count": 0}

    class FakeBackend:
        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000

        def open(self, config) -> None:
            _ = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            time.sleep(0.02)
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            silent_pcm = b"\x00\x00" * frame_samples
            return {
                "mic": RawFrame(
                    stream="mic",
                    mono_pcm16=silent_pcm,
                    captured_at_monotonic_ns=time.monotonic_ns(),
                    sample_rate_hz=self.sample_rate_hz,
                )
            }

        def close(self) -> None:
            return None

    def fake_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        _ = (wav_bytes, model_id)
        calls["count"] += 1
        return "unexpected", 1.0

    monkeypatch.setattr(
        live_session_module,
        "open_audio_backend",
        lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture),
    )
    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        sample_rate_hz=16_000,
        duration_sec=0.5,
        chunk_sec=0.25,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-silence-skip",
        session_id="live-silence-skip",
    )
    result = run_live_transcription_session(config, use_fixture=False, transcriber=fake_transcriber)
    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))

    assert calls["count"] == 0
    assert payload["metrics"]["silence_skipped_chunk_count"] >= 1
    assert payload["final_segments"] == []


def test_live_session_parakeet_failure_surfaces_fallback_hint(tmp_path) -> None:
    def failing_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        _ = (wav_bytes, model_id)
        raise ValueError("decoder exploded")

    config = LiveSessionConfig(
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        duration_sec=0.08,
        chunk_sec=0.04,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-failure",
        session_id="live-failure",
    )
    with pytest.raises(RuntimeError, match="Parakeet decoder failed") as excinfo:
        run_live_transcription_session(
            config,
            use_fixture=True,
            transcriber=failing_transcriber,
        )

    message = str(excinfo.value)
    assert "Qwen/Qwen3-ASR-0.6B" in message


def test_stitch_text_overlap_removes_repeated_boundary_words() -> None:
    assert (
        _stitch_text_overlap(
            "Stand ho, who's there? Friends to this ground.",
            "ground, and liegemen to the Dane.",
        )
        == "and liegemen to the Dane."
    )


def test_stitch_text_overlap_preserves_new_text_without_overlap() -> None:
    assert _stitch_text_overlap("Who's there?", "Long live the king.") == "Long live the king."


@pytest.mark.parametrize(
    ("previous_text", "current_text", "expected_text"),
    [
        (
            "USB C port, be it with slower charging speeds, and we get the desk view upgrade.",
            "the desk view up like a pixel center stage camera. I would have really liked to see",
            "up like a pixel center stage camera. I would have really liked to see",
        ),
        (
            "But hey, it's 5k resolution and not everybody wants that.",
            "Not everybody wants. I'm just gonna say I'm jazzed enough about the new XDR that I'm willing",
            "I'm just gonna say I'm jazzed enough about the new XDR that I'm willing",
        ),
    ],
)
def test_stitch_text_overlap_handles_conservative_boundary_trim(
    previous_text: str,
    current_text: str,
    expected_text: str,
) -> None:
    assert _stitch_text_overlap(previous_text, current_text) == expected_text


def test_stitch_text_overlap_preserves_hyphenated_cutoff_recovery() -> None:
    assert (
        _stitch_text_overlap(
            "least not as much, coming in around $1,700 less than the Pro Display XDR. Also, it's finally a mini-",
            "It's finally a minity, and on top of the physical changes, we're also getting support for 16 reference modes.",
        )
        == "It's finally a minity, and on top of the physical changes, we're also getting support for 16 reference modes."
    )


def test_stitch_text_overlap_preserves_short_fuzzy_corrections() -> None:
    assert (
        _stitch_text_overlap(
            "It's finally a minity, and on top of the physical changes, we're also getting support for 16 reference modes. So color correction is sending.",
            "So color correctionists and common rejoice, along with access to the color again, and the same quality technical center stage plus...",
        )
        == "So color correctionists and common rejoice, along with access to the color again, and the same quality technical center stage plus..."
    )


def test_stitch_text_overlap_keeps_three_word_corrections() -> None:
    assert (
        _stitch_text_overlap(
            "bit the same, sixteen hundred dollars we were paying before, but now it's similarly upgraded Thunderbolt and US P C port.",
            "USB C port, be it with slower charging speeds, and we get the desk view upgrade.",
        )
        == "USB C port, be it with slower charging speeds, and we get the desk view upgrade."
    )


def test_stitch_text_overlap_removes_long_two_word_suffix_overlap() -> None:
    assert (
        _stitch_text_overlap(
            "The pricing change is pretty disruptive.",
            "pretty disruptive, but maybe overdue.",
        )
        == "but maybe overdue."
    )


def test_stitch_text_overlap_preserves_two_word_exact_overlap() -> None:
    assert (
        _stitch_text_overlap(
            "Not everybody wants. I'm just gonna say I'm jazzed enough about the new XDR that I'm willing",
            "XDR that I will give them on this one since they didn't raise the price. On that subject,",
        )
        == "XDR that I will give them on this one since they didn't raise the price. On that subject,"
    )


def test_stitch_text_overlap_preserves_partial_word_followups() -> None:
    assert (
        _stitch_text_overlap(
            "I mean who would have just slapped an M4 chip onto it and called it?",
            "onto it and c they've certainly done it before. But instead, it's",
        )
        == "onto it and c they've certainly done it before. But instead, it's"
    )


def test_run_live_transcription_session_can_disable_stitching(monkeypatch, tmp_path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.live.session as live_session_module

    class FakeBackend:
        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000
            self.active_devices = {"mic": ["fixture"]}

        def open(self, config) -> None:
            _ = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            voiced_frame = struct.pack("<h", 1_200) * frame_samples
            return {
                "mic": RawFrame(
                    stream="mic",
                    mono_pcm16=voiced_frame,
                    captured_at_monotonic_ns=time.monotonic_ns(),
                    sample_rate_hz=self.sample_rate_hz,
                )
            }

        def close(self) -> None:
            return None

    call_count = {"count": 0}

    def fake_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        _ = (wav_bytes, model_id)
        call_count["count"] += 1
        if call_count["count"] == 1:
            return "Not everybody wants that.", 1.0
        return "Not everybody wants. I'm just gonna say more.", 1.0

    def fail_if_called(previous_text: str, current_text: str, *, max_overlap_words: int = 12) -> str:
        _ = (previous_text, current_text, max_overlap_words)
        raise AssertionError("stitcher should not run by default")

    monkeypatch.setattr(
        live_session_module,
        "open_audio_backend",
        lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture),
    )
    monkeypatch.setattr(live_session_module, "_stitch_text_overlap", fail_if_called)

    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        duration_sec=0.18,
        chunk_sec=0.08,
        stitch_overlap_text=False,
        output_dir=tmp_path / "live-session",
        session_id="live-no-stitch",
    )
    result = run_live_transcription_session(config, use_fixture=False, transcriber=fake_transcriber)

    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))
    assert payload["stitch_overlap_text"] is False
    assert payload["final_segments"]


def test_run_live_transcription_session_reports_progress_events(tmp_path) -> None:
    progress_events: list[tuple[str, dict[str, object]]] = []

    def fake_transcriber(wav_bytes: bytes, model_id: str) -> tuple[str, float]:
        _ = (wav_bytes, model_id)
        return "progress line", 4.0

    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        duration_sec=0.12,
        chunk_sec=0.06,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-progress",
        session_id="live-progress",
    )
    run_live_transcription_session(
        config,
        use_fixture=True,
        transcriber=fake_transcriber,
        progress_callback=lambda event, fields: progress_events.append((event, fields)),
    )

    event_names = [event for event, _ in progress_events]
    assert "capture_ready" in event_names
    assert "transcribing_started" in event_names
    assert "final" in event_names


def test_run_session_prints_crisp_feedback_by_default(monkeypatch, tmp_path, capsys) -> None:
    import transcribe.cli as cli_module
    import transcribe.live.session as live_session_module

    monkeypatch.setattr(cli_module, "load_and_configure_logging", lambda args: None)

    def fake_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None):
        _ = (use_fixture, debug)
        assert progress_callback is not None
        progress_callback("loading_model", {"transcription_model": config.transcription_model})
        progress_callback("model_ready", {"transcription_model": config.transcription_model})
        progress_callback(
            "capture_ready",
            {
                "requested_sample_rate_hz": 16_000,
                "capture_sample_rate_hz": 16_000,
                "transcription_sample_rate_hz": 16_000,
                "resolved_capture_devices": {"mic": ["2"], "speakers": ["5"]},
                "device_sample_rates_hz": {"mic:2": 16_000, "speakers:5": 48_000},
            },
        )
        progress_callback("transcribing_started", {"duration_sec": 0.0})
        progress_callback("final", {"chunk_index": 1, "text": "clean transcript line"})
        return live_session_module.LiveSessionResult(
            session_dir=Path(tmp_path) / "live-test",
            events_path=Path(tmp_path) / "live-test" / "events.jsonl",
            transcript_json_path=Path(tmp_path) / "live-test" / "transcript.json",
            transcript_txt_path=Path(tmp_path) / "live-test" / "transcript.txt",
            final_segment_count=1,
            partial_event_count=0,
            sample_rate_hz=16_000,
            sample_rate_hz_requested=16_000,
            total_audio_sec=4.0,
            total_inference_sec=0.2,
            source_selection_counts={"mic": 1},
            interrupted=False,
        )

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", fake_runner)

    args = argparse.Namespace(
        config=None,
        log_level=None,
        debug=False,
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        duration_sec=0.0,
        chunk_overlap_sec=0.75,
        chunk_sec=4.0,
        partial_interval_sec=0.0,
        mode=AudioSourceMode.BOTH,
        mic_device=None,
        speaker_device=None,
        single_device_per_source=False,
        strict_sources=False,
        out=Path(tmp_path),
        session_id="live-test",
        max_model_ram_gb=8.0,
        fixture=False,
    )
    rc = cli_module.run_session(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert "Loading model: nvidia/parakeet-tdt-0.6b-v3" in captured.out
    assert "Model ready." in captured.out
    assert "Capture ready: mic, speakers" in captured.out
    assert "Sample rate: 48000 Hz capture (requested 16000 Hz), 16000 Hz ASR" in captured.out
    assert "Transcribing. Press Ctrl+C to stop." in captured.out
    assert "clean transcript line" in captured.out
    assert "Session saved:" in captured.out
    assert "Events JSONL:" not in captured.out
    assert "Source selections:" not in captured.out


def test_run_session_prints_debug_feedback_when_enabled(monkeypatch, tmp_path, capsys) -> None:
    import transcribe.cli as cli_module
    import transcribe.live.session as live_session_module

    monkeypatch.setattr(cli_module, "load_and_configure_logging", lambda args: None)

    def fake_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None):
        _ = use_fixture
        assert debug is True
        assert progress_callback is not None
        progress_callback("loading_model", {"transcription_model": config.transcription_model})
        progress_callback("partial", {"chunk_index": 1, "text": "partial line"})
        progress_callback("final", {"chunk_index": 1, "text": "final line"})
        return live_session_module.LiveSessionResult(
            session_dir=Path(tmp_path) / "live-test",
            events_path=Path(tmp_path) / "live-test" / "events.jsonl",
            transcript_json_path=Path(tmp_path) / "live-test" / "transcript.json",
            transcript_txt_path=Path(tmp_path) / "live-test" / "transcript.txt",
            final_segment_count=1,
            partial_event_count=1,
            sample_rate_hz=16_000,
            sample_rate_hz_requested=16_000,
            total_audio_sec=4.0,
            total_inference_sec=0.2,
            source_selection_counts={"speakers": 1},
            interrupted=False,
        )

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", fake_runner)

    args = argparse.Namespace(
        config=None,
        log_level=None,
        debug=True,
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        duration_sec=0.0,
        chunk_overlap_sec=0.75,
        chunk_sec=4.0,
        partial_interval_sec=0.0,
        mode=AudioSourceMode.BOTH,
        mic_device=None,
        speaker_device=None,
        single_device_per_source=False,
        strict_sources=False,
        out=Path(tmp_path),
        session_id="live-test",
        max_model_ram_gb=8.0,
        fixture=False,
    )
    rc = cli_module.run_session(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert "[partial 1] partial line" in captured.out
    assert "[final 1] final line" in captured.out
    assert "Events JSONL:" in captured.out
    assert "Source selections:" in captured.out


def test_run_session_returns_code_2_on_runtime_error(monkeypatch, tmp_path, capsys) -> None:
    import transcribe.cli as cli_module
    import transcribe.live.session as live_session_module

    monkeypatch.setattr(cli_module, "load_and_configure_logging", lambda args: None)

    def failing_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None):
        _ = (config, use_fixture, debug, progress_callback)
        raise RuntimeError("Parakeet decoder failed in this runtime")

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", failing_runner)

    args = argparse.Namespace(
        config=None,
        log_level=None,
        debug=False,
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        duration_sec=0.0,
        chunk_overlap_sec=0.75,
        chunk_sec=4.0,
        partial_interval_sec=1.0,
        mode=AudioSourceMode.BOTH,
        mic_device=None,
        speaker_device=None,
        single_device_per_source=False,
        strict_sources=False,
        out=Path(tmp_path),
        session_id="live-test",
        max_model_ram_gb=8.0,
        fixture=False,
    )
    rc = cli_module.run_session(args)
    captured = capsys.readouterr()

    assert rc == 2
    assert "Session failed:" in captured.out
