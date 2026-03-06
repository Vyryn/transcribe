from __future__ import annotations

import pytest

import transcribe.audio.backend_loader as backend_loader
from transcribe.audio.windows_capture import WindowsAudioCaptureBackend
from transcribe.models import AudioSourceMode, CaptureConfig


def test_open_audio_backend_uses_windows_backend_when_platform_is_win32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_loader.sys, "platform", "win32", raising=False)

    backend = backend_loader.open_audio_backend(use_fixture=True)

    assert isinstance(backend, WindowsAudioCaptureBackend)


def test_windows_backend_prefers_loopback_like_devices_for_speakers() -> None:
    class _Catalog:
        @staticmethod
        def query_devices():
            return [
                {"name": "Microphone Array (Realtek)", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Stereo Mix (Realtek)", "max_input_channels": 2, "default_samplerate": 48_000.0},
                {"name": "Speakers (USB Audio)", "max_input_channels": 2, "default_samplerate": 48_000.0},
            ]

    backend = WindowsAudioCaptureBackend(use_fixture=True)
    speaker_devices = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=True,
        include_all=True,
        allow_missing=False,
    )
    mic_devices = backend.resolve_devices(
        _Catalog(),
        configured=None,
        require_monitor=False,
        include_all=True,
        allow_missing=False,
    )

    assert speaker_devices == [1, 2]
    assert mic_devices == [0]


def test_open_audio_backend_rejects_unknown_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_loader.sys, "platform", "darwin", raising=False)

    with pytest.raises(RuntimeError, match="Unsupported capture platform"):
        backend_loader.open_audio_backend(use_fixture=False)


def test_windows_backend_fixture_frames_work_without_platform_audio() -> None:
    backend = WindowsAudioCaptureBackend(use_fixture=True)
    backend.open(CaptureConfig(source_mode=AudioSourceMode.BOTH))

    try:
        frames = backend.read_frames(timeout_ms=20)
    finally:
        backend.close()

    assert set(frames) == {"mic", "speakers"}
