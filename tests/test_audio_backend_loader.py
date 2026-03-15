from __future__ import annotations

import contextlib
import importlib
import queue
import time

import numpy as np
import pytest

import transcribe.audio.backend_loader as backend_loader
import transcribe.audio.windows_capture as windows_capture_module
from transcribe.audio.interfaces import RawFrame
from transcribe.audio.windows_capture import WindowsAudioCaptureBackend
from transcribe.models import AudioSourceMode, CaptureConfig


class _FakeMicrophone:
    def __init__(self, *, name: str, device_id: str, channels: int, isloopback: bool) -> None:
        self.name = name
        self.id = device_id
        self.channels = channels
        self.isloopback = isloopback


class _FakeSpeaker:
    def __init__(self, *, device_id: str) -> None:
        self.id = device_id


class _FakeSoundCard:
    def __init__(
        self,
        *,
        microphones: list[_FakeMicrophone],
        default_microphone_id: str,
        default_speaker_id: str,
    ) -> None:
        self._microphones = microphones
        self._default_microphone_id = default_microphone_id
        self._default_speaker_id = default_speaker_id

    def all_microphones(self, include_loopback: bool = False) -> list[_FakeMicrophone]:
        if include_loopback:
            return list(self._microphones)
        return [device for device in self._microphones if not device.isloopback]

    def default_microphone(self) -> _FakeMicrophone:
        return next(device for device in self._microphones if device.id == self._default_microphone_id)

    def default_speaker(self) -> _FakeSpeaker:
        return _FakeSpeaker(device_id=self._default_speaker_id)

    def get_microphone(self, device_ref: str, include_loopback: bool = False) -> _FakeMicrophone:
        del include_loopback
        return next(device for device in self._microphones if device.id == device_ref)


@pytest.fixture
def fake_soundcard() -> _FakeSoundCard:
    return _FakeSoundCard(
        microphones=[
            _FakeMicrophone(name="USB Mic", device_id="mic-usb", channels=1, isloopback=False),
            _FakeMicrophone(name="Microphone Array", device_id="mic-default", channels=2, isloopback=False),
            _FakeMicrophone(name="Speakers (Realtek)", device_id="spk-default", channels=2, isloopback=True),
        ],
        default_microphone_id="mic-default",
        default_speaker_id="spk-default",
    )


def test_open_audio_backend_uses_windows_backend_when_platform_is_win32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend_loader.sys, "platform", "win32", raising=False)

    backend = backend_loader.open_audio_backend(use_fixture=True)

    assert isinstance(backend, WindowsAudioCaptureBackend)



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



def test_windows_backend_lists_soundcard_devices_in_wasapi_order(
    monkeypatch: pytest.MonkeyPatch,
    fake_soundcard: _FakeSoundCard,
) -> None:
    monkeypatch.setattr(windows_capture_module, "load_soundcard", lambda: fake_soundcard)

    backend = WindowsAudioCaptureBackend(use_fixture=True)
    devices = backend.list_devices()

    assert devices == [
        {
            "index": 0,
            "name": "Microphone Array",
            "max_input_channels": 2,
            "default_samplerate": 0.0,
            "hostapi_name": "windows wasapi",
            "device_id": "mic-default",
            "is_loopback": False,
        },
        {
            "index": 1,
            "name": "USB Mic",
            "max_input_channels": 1,
            "default_samplerate": 0.0,
            "hostapi_name": "windows wasapi",
            "device_id": "mic-usb",
            "is_loopback": False,
        },
        {
            "index": 2,
            "name": "Speakers (Realtek) [Loopback]",
            "max_input_channels": 2,
            "default_samplerate": 0.0,
            "hostapi_name": "windows wasapi",
            "device_id": "spk-default",
            "is_loopback": True,
        },
    ]


def test_windows_backend_list_devices_does_not_wrap_soundcard_import_with_com_init(
    monkeypatch: pytest.MonkeyPatch,
    fake_soundcard: _FakeSoundCard,
) -> None:
    observed: list[str] = []

    @contextlib.contextmanager
    def fake_initialize_com_thread():
        observed.append("enter")
        try:
            yield
        finally:
            observed.append("exit")

    monkeypatch.setattr(windows_capture_module, "_initialize_com_thread", fake_initialize_com_thread)
    monkeypatch.setattr(windows_capture_module, "load_soundcard", lambda: fake_soundcard)

    backend = WindowsAudioCaptureBackend(use_fixture=True)
    backend.list_devices()

    assert observed == []



def test_windows_backend_resolve_devices_prefers_default_mic_and_loopback(
    monkeypatch: pytest.MonkeyPatch,
    fake_soundcard: _FakeSoundCard,
) -> None:
    monkeypatch.setattr(windows_capture_module, "load_soundcard", lambda: fake_soundcard)

    backend = WindowsAudioCaptureBackend(use_fixture=True)
    mic_devices = backend.resolve_devices(
        fake_soundcard,
        configured=None,
        require_monitor=False,
        include_all=True,
        allow_missing=False,
    )
    speaker_devices = backend.resolve_devices(
        fake_soundcard,
        configured=None,
        require_monitor=True,
        include_all=True,
        allow_missing=False,
    )

    assert [device.index for device in mic_devices] == [0, 1]
    assert [device.index for device in speaker_devices] == [2]



def test_windows_backend_resolve_devices_supports_default_keyword(
    monkeypatch: pytest.MonkeyPatch,
    fake_soundcard: _FakeSoundCard,
) -> None:
    monkeypatch.setattr(windows_capture_module, "load_soundcard", lambda: fake_soundcard)

    backend = WindowsAudioCaptureBackend(use_fixture=True)
    mic_device = backend.resolve_devices(
        fake_soundcard,
        configured="default",
        require_monitor=False,
        include_all=False,
        allow_missing=False,
    )
    speaker_device = backend.resolve_devices(
        fake_soundcard,
        configured="default",
        require_monitor=True,
        include_all=False,
        allow_missing=False,
    )

    assert [device.index for device in mic_device] == [0]
    assert [device.index for device in speaker_device] == [2]



def test_windows_backend_float_audio_downmixes_and_resamples_for_speakers() -> None:
    backend = WindowsAudioCaptureBackend(use_fixture=False)
    backend.config = CaptureConfig(sample_rate_hz=16_000, frame_ms=20, channels=1)
    backend._sample_rate_hz = 16_000
    backend.frame_samples = 320
    backend._queues = {"speakers:2": queue.Queue(maxsize=4)}

    callback = backend._build_frame_callback(
        stream_key="speakers:2",
        stream_group="speakers",
        device_sample_rate_hz=48_000,
    )
    audio = np.tile(np.array([[0.25, -0.25]], dtype=np.float32), (960, 1))
    callback(audio, 960, None, None)
    frame = backend._queues["speakers:2"].get_nowait()

    assert frame.stream == "speakers"
    assert frame.sample_rate_hz == 16_000
    assert len(frame.mono_pcm16) == 320 * 2



def test_windows_backend_float_audio_prefers_loudest_mic_channel() -> None:
    backend = WindowsAudioCaptureBackend(use_fixture=False)

    pcm16 = backend._float32_audio_to_mono_pcm16(
        np.tile(np.array([[0.05, 0.8]], dtype=np.float32), (320, 1)),
        stream_group="mic",
    )
    frame = RawFrame(
        stream="mic",
        mono_pcm16=pcm16,
        captured_at_monotonic_ns=time.monotonic_ns(),
        sample_rate_hz=16_000,
    )

    samples = [sample for (sample,) in __import__("struct").iter_unpack("<h", frame.mono_pcm16)]
    assert samples
    assert max(abs(sample) for sample in samples) > 20_000



def test_windows_backend_open_supports_mixed_device_sample_rates(
    monkeypatch: pytest.MonkeyPatch,
    fake_soundcard: _FakeSoundCard,
) -> None:
    class _FakeStream:
        def __init__(
            self,
            *,
            stream_key: str,
            device_id: str,
            include_loopback: bool,
            sample_rate_hz: int,
            frame_samples: int,
            device_channels: int,
            callback,
        ) -> None:
            self.stream_key = stream_key
            self.device_id = device_id
            self.include_loopback = include_loopback
            self.sample_rate_hz = sample_rate_hz
            self.frame_samples = frame_samples
            self.device_channels = device_channels
            self.callback = callback
            self.active = True
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            supported = {
                "mic-default": 16_000,
                "spk-default": 48_000,
            }
            if supported.get(self.device_id) != self.sample_rate_hz:
                raise RuntimeError("unsupported rate")

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(windows_capture_module, "load_soundcard", lambda: fake_soundcard)
    monkeypatch.setattr(windows_capture_module, "_SoundCardRecorderStream", _FakeStream)

    backend = WindowsAudioCaptureBackend(use_fixture=False)
    backend.open(
        CaptureConfig(
            sample_rate_hz=16_000,
            frame_ms=20,
            channels=1,
            source_mode=AudioSourceMode.BOTH,
        )
    )

    assert backend.sample_rate_hz == 16_000
    assert backend.active_devices == {"mic": (0,), "speakers": (2,)}
    assert backend.device_sample_rates_hz == {"mic:0": 16_000, "speakers:2": 48_000}
    assert backend.device_channels == {"mic:0": 2, "speakers:2": 2}


def test_windows_backend_open_smoke_imports_soundcard_before_worker_thread_com_init(
    monkeypatch: pytest.MonkeyPatch,
    fake_soundcard: _FakeSoundCard,
) -> None:
    observed: list[str] = []

    class _FakeStream:
        def __init__(
            self,
            *,
            stream_key: str,
            device_id: str,
            include_loopback: bool,
            sample_rate_hz: int,
            frame_samples: int,
            device_channels: int,
            callback,
        ) -> None:
            _ = (
                stream_key,
                device_id,
                include_loopback,
                sample_rate_hz,
                frame_samples,
                device_channels,
                callback,
            )

        def start(self) -> None:
            observed.append("stream-start")

        def stop(self) -> None:
            return None

        def close(self) -> None:
            return None

    class _FakeMediaFoundation:
        numpy = np
        SoundcardRuntimeWarning = RuntimeWarning

    def fake_import_module(name: str):
        if name == "soundcard":
            observed.append("soundcard-import")
            return fake_soundcard
        if name == "soundcard.mediafoundation":
            return _FakeMediaFoundation()
        return importlib.import_module(name)

    @contextlib.contextmanager
    def fake_initialize_com_thread():
        observed.append("com-enter")
        try:
            yield
        finally:
            observed.append("com-exit")

    monkeypatch.setattr(windows_capture_module, "_SOUNDCARD", None)
    monkeypatch.setattr(windows_capture_module, "_SOUNDCARD_ATTEMPTED", False)
    monkeypatch.setattr(windows_capture_module, "_SOUNDCARD_IMPORT_ERROR", None)
    monkeypatch.setattr(windows_capture_module, "_SOUNDCARD_NUMPY_PATCHED", False)
    monkeypatch.setattr(windows_capture_module.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(windows_capture_module, "_SoundCardRecorderStream", _FakeStream)
    monkeypatch.setattr(windows_capture_module, "_initialize_com_thread", fake_initialize_com_thread)

    backend = WindowsAudioCaptureBackend(use_fixture=False)
    backend.open(
        CaptureConfig(
            sample_rate_hz=16_000,
            frame_ms=20,
            channels=1,
            source_mode=AudioSourceMode.BOTH,
        )
    )

    assert "soundcard-import" in observed
    assert "stream-start" in observed
    assert "com-enter" not in observed


def test_soundcard_recorder_stream_initializes_com_on_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []

    @contextlib.contextmanager
    def fake_initialize_com_thread():
        observed.append("enter")
        try:
            yield
        finally:
            observed.append("exit")

    class _FakeRecorder:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def record(self, *, numframes: int):
            del numframes
            return np.zeros((320, 1), dtype=np.float32)

    class _FakeDevice:
        def recorder(self, **kwargs):
            observed.append(f"recorder:{kwargs['samplerate']}")
            return _FakeRecorder()

    monkeypatch.setattr(windows_capture_module, "_initialize_com_thread", fake_initialize_com_thread)
    monkeypatch.setattr(
        windows_capture_module,
        "load_soundcard",
        lambda: type(
            "_FakeSoundcardRuntime",
            (),
            {"get_microphone": staticmethod(lambda device_id, include_loopback=False: _FakeDevice())},
        )(),
    )

    stream = windows_capture_module._SoundCardRecorderStream(
        stream_key="mic:0",
        device_id="mic-default",
        include_loopback=False,
        sample_rate_hz=16_000,
        frame_samples=320,
        device_channels=1,
        callback=lambda block, frames, time_info, status: stream._stop_event.set(),
    )

    stream.start()
    stream.stop()

    assert observed[0] == "enter"
    assert "recorder:16000" in observed
    assert observed[-1] == "exit"



def test_windows_backend_translates_microphone_access_denied() -> None:
    error = RuntimeError("Error 0x80070005")

    translated = WindowsAudioCaptureBackend._translate_open_error(error, stream_group="mic")

    assert isinstance(translated, RuntimeError)
    assert "desktop apps access your microphone" in str(translated)


def test_patch_soundcard_numpy_fromstring_uses_frombuffer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeNumpyModule:
        def __init__(self) -> None:
            self._transcribe_binary_fromstring_patch = False

        @staticmethod
        def fromstring(*args, **kwargs):
            raise AssertionError("binary fromstring should be replaced")

    class _FakeMediaFoundation:
        numpy = _FakeNumpyModule()

    def fake_import_module(name: str):
        if name == "soundcard.mediafoundation":
            return _FakeMediaFoundation()
        raise ImportError(name)

    monkeypatch.setattr(windows_capture_module, "_SOUNDCARD_NUMPY_PATCHED", False)
    monkeypatch.setattr(windows_capture_module.importlib, "import_module", fake_import_module)

    windows_capture_module._patch_soundcard_numpy_fromstring()

    values = _FakeMediaFoundation.numpy.fromstring(b"\x00\x00\x80?", dtype=np.float32)
    assert values.shape == (1,)
    assert values[0] == pytest.approx(1.0)
