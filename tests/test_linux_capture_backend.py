from __future__ import annotations

import queue
import struct
import time

import pytest

from transcribe.audio.interfaces import RawFrame
from transcribe.audio.linux_capture import LinuxAudioCaptureBackend
from transcribe.models import AudioSourceMode, CaptureConfig


class _FakeSoundDevice:
    def __init__(self, *, default_rates: dict[int, int], supported_rates: dict[int, set[int]]) -> None:
        self._default_rates = default_rates
        self._supported_rates = supported_rates

    def query_devices(self, device: int | None = None):
        if device is None:
            devices = []
            for index in sorted(self._default_rates):
                devices.append(
                    {
                        "name": f"mic-{index}",
                        "max_input_channels": 1,
                        "default_samplerate": float(self._default_rates[index]),
                    }
                )
            return devices
        return {
            "name": f"mic-{device}",
            "max_input_channels": 1,
            "default_samplerate": float(self._default_rates[int(device)]),
        }

    def check_input_settings(self, *, device: int, channels: int, dtype: str, samplerate: float) -> None:
        _ = (channels, dtype)
        if int(round(samplerate)) not in self._supported_rates[int(device)]:
            raise ValueError("unsupported sample rate")


def test_negotiate_sample_rate_prefers_requested_rate() -> None:
    backend = LinuxAudioCaptureBackend(use_fixture=True)
    sd = _FakeSoundDevice(default_rates={0: 48_000}, supported_rates={0: {16_000, 48_000}})

    resolved = backend.negotiate_sample_rate(
        sd,
        devices=(0,),
        channels=1,
        requested_sample_rate_hz=16_000,
    )

    assert resolved == 16_000


def test_negotiate_sample_rate_falls_back_to_device_default() -> None:
    backend = LinuxAudioCaptureBackend(use_fixture=True)
    sd = _FakeSoundDevice(default_rates={0: 48_000}, supported_rates={0: {48_000}})

    resolved = backend.negotiate_sample_rate(
        sd,
        devices=(0,),
        channels=1,
        requested_sample_rate_hz=16_000,
    )

    assert resolved == 48_000


def test_negotiate_sample_rate_raises_when_no_candidate_supported() -> None:
    backend = LinuxAudioCaptureBackend(use_fixture=True)
    sd = _FakeSoundDevice(default_rates={0: 12_345}, supported_rates={0: set()})

    with pytest.raises(RuntimeError, match="No supported sample rate found"):
        backend.negotiate_sample_rate(
            sd,
            devices=(0,),
            channels=1,
            requested_sample_rate_hz=16_000,
        )


def test_resolve_devices_returns_all_candidates_when_include_all() -> None:
    class _Catalog:
        @staticmethod
        def query_devices():
            return [
                {"name": "USB Mic 1", "max_input_channels": 1, "default_samplerate": 48_000.0},
                {"name": "USB Mic 2", "max_input_channels": 1, "default_samplerate": 48_000.0},
                {"name": "Monitor of Built-in Output", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Loopback Source", "max_input_channels": 2, "default_samplerate": 48_000.0},
            ]

    backend = LinuxAudioCaptureBackend(use_fixture=True)
    mic_devices = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=False,
        include_all=True,
        allow_missing=False,
    )
    speaker_devices = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=True,
        include_all=True,
        allow_missing=False,
    )

    assert mic_devices == [0, 1]
    assert speaker_devices == [2, 3]


def test_resolve_devices_realistic_pipewire_catalog_prefers_expected_roles() -> None:
    class _Catalog:
        @staticmethod
        def query_devices():
            return [
                {"name": "Realtek USB MIC: Audio (hw:1,0)", "max_input_channels": 1, "default_samplerate": 44_100.0},
                {"name": "HDA Intel PCH: ALC897 Analog (hw:2,0)", "max_input_channels": 2, "default_samplerate": 44_100.0},
                {"name": "HDA NVidia: HDMI 2 (hw:3,8)", "max_input_channels": 0, "default_samplerate": 44_100.0},
                {"name": "pipewire", "max_input_channels": 64, "default_samplerate": 44_100.0},
                {"name": "pulse", "max_input_channels": 32, "default_samplerate": 44_100.0},
                {"name": "default", "max_input_channels": 32, "default_samplerate": 44_100.0},
                {"name": "USB PnP Audio Device Mono", "max_input_channels": 1, "default_samplerate": 48_000.0},
                {"name": "NoiseTorch Microphone for USB PnP Audio Device Source", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Bluetooth internal playback stream for WH-1000XM5", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "spotify", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Bluetooth internal capture stream for WH-1000XM5", "max_input_channels": 1, "default_samplerate": 48_000.0},
                {"name": "GNOME Settings", "max_input_channels": 3, "default_samplerate": 48_000.0},
            ]

    backend = LinuxAudioCaptureBackend(use_fixture=True)
    mic_devices = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=False,
        include_all=True,
        allow_missing=False,
    )
    speaker_devices = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=True,
        include_all=True,
        allow_missing=False,
    )

    # Prefer filtered/real mic-like sources.
    assert 7 in mic_devices  # NoiseTorch microphone
    assert 0 in mic_devices  # Explicit USB mic
    assert 10 in mic_devices  # Bluetooth capture stream
    assert 8 not in mic_devices  # Playback stream should not be treated as mic
    assert 9 not in mic_devices  # App playback stream should not be treated as mic
    assert 11 not in mic_devices  # Non-capture app stream should not be treated as mic

    # Prefer playback/app streams for speakers.
    assert 8 in speaker_devices  # Bluetooth playback stream
    assert 9 in speaker_devices  # App stream
    assert 10 not in speaker_devices  # Capture stream should not be treated as speaker output
    assert 7 not in speaker_devices  # NoiseTorch mic should not be treated as speaker output


def test_resolve_devices_single_device_prefers_filtered_mic_and_playback_stream() -> None:
    class _Catalog:
        @staticmethod
        def query_devices():
            return [
                {"name": "USB PnP Audio Device Mono", "max_input_channels": 1, "default_samplerate": 48_000.0},
                {"name": "NoiseTorch Microphone for USB PnP Audio Device Source", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Bluetooth internal playback stream for WH-1000XM5", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "spotify", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Bluetooth internal capture stream for WH-1000XM5", "max_input_channels": 1, "default_samplerate": 48_000.0},
            ]

    backend = LinuxAudioCaptureBackend(use_fixture=True)
    mic_device = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=False,
        include_all=False,
        allow_missing=False,
    )
    speaker_device = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=True,
        include_all=False,
        allow_missing=False,
    )

    assert mic_device == [1]
    assert speaker_device == [2]


def test_read_frames_selects_clearest_frame_per_group() -> None:
    backend = LinuxAudioCaptureBackend(use_fixture=False)
    backend.config = CaptureConfig(source_mode=AudioSourceMode.BOTH)
    backend._active_stream_names = ("mic",)
    backend._group_stream_keys = {"mic": ("mic:0", "mic:1")}
    backend._queues = {
        "mic:0": queue.Queue(maxsize=10),
        "mic:1": queue.Queue(maxsize=10),
    }

    quiet = struct.pack("<h", 500) * 320
    loud = struct.pack("<h", 5_000) * 320
    backend._queues["mic:0"].put(
        RawFrame(stream="mic", mono_pcm16=quiet, captured_at_monotonic_ns=time.monotonic_ns())
    )
    backend._queues["mic:1"].put(
        RawFrame(stream="mic", mono_pcm16=loud, captured_at_monotonic_ns=time.monotonic_ns())
    )

    frames = backend.read_frames(timeout_ms=20)

    assert frames["mic"].mono_pcm16 == loud


def test_read_frames_falls_back_when_preferred_stream_stalls() -> None:
    backend = LinuxAudioCaptureBackend(use_fixture=False)
    backend.config = CaptureConfig(source_mode=AudioSourceMode.BOTH)
    backend._active_stream_names = ("mic",)
    backend._group_stream_keys = {"mic": ("mic:0", "mic:1")}
    backend._queues = {
        "mic:0": queue.Queue(maxsize=10),
        "mic:1": queue.Queue(maxsize=10),
    }
    backend._preferred_stream_key = {"mic": "mic:0"}

    loud = struct.pack("<h", 5_000) * 320
    backend._queues["mic:1"].put(
        RawFrame(stream="mic", mono_pcm16=loud, captured_at_monotonic_ns=time.monotonic_ns())
    )

    frames = backend.read_frames(timeout_ms=20)

    assert frames["mic"].mono_pcm16 == loud
    assert backend._preferred_stream_key["mic"] == "mic:1"


def test_read_frames_recovers_after_repeated_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LinuxAudioCaptureBackend(use_fixture=False)
    backend.config = CaptureConfig(source_mode=AudioSourceMode.MIC)
    backend._active_stream_names = ("mic",)
    backend._group_stream_keys = {"mic": ("mic:0",)}
    backend._queues = {"mic:0": queue.Queue(maxsize=10)}
    backend._timeout_recovery_threshold = 1

    class BrokenStream:
        active = False

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    backend._streams = {"mic:0": BrokenStream()}

    recovered_frame = RawFrame(
        stream="mic",
        mono_pcm16=struct.pack("<h", 4_000) * 320,
        captured_at_monotonic_ns=time.monotonic_ns(),
    )
    reopened = {"count": 0}

    def fake_open(config: CaptureConfig) -> None:
        _ = config
        reopened["count"] += 1
        backend.config = config
        backend._active_stream_names = ("mic",)
        backend._group_stream_keys = {"mic": ("mic:1",)}
        backend._queues = {"mic:1": queue.Queue(maxsize=10)}
        backend._queues["mic:1"].put(recovered_frame)
        backend._streams = {"mic:1": BrokenStream()}

    monkeypatch.setattr(backend, "open", fake_open)

    with pytest.raises(TimeoutError):
        backend.read_frames(timeout_ms=5)

    frames = backend.read_frames(timeout_ms=5)

    assert reopened["count"] == 1
    assert frames["mic"] == recovered_frame
