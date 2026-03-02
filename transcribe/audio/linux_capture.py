from __future__ import annotations

import importlib
import math
import queue
import struct
import time
from typing import Any

from transcribe.audio.interfaces import RawFrame
from transcribe.models import AudioSourceMode, CaptureConfig

_SOUNDDEVICE: Any | None = None
_SOUNDDEVICE_ATTEMPTED = False
_SOUNDDEVICE_IMPORT_ERROR: Exception | None = None


def load_sounddevice() -> Any | None:
    """Load the optional ``sounddevice`` dependency lazily.

    Returns
    -------
    Any | None
        Imported ``sounddevice`` module when available, otherwise ``None``.
    """
    global _SOUNDDEVICE, _SOUNDDEVICE_ATTEMPTED, _SOUNDDEVICE_IMPORT_ERROR
    if _SOUNDDEVICE_ATTEMPTED:
        return _SOUNDDEVICE

    _SOUNDDEVICE_ATTEMPTED = True
    try:
        _SOUNDDEVICE = importlib.import_module("sounddevice")
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        _SOUNDDEVICE = None
        _SOUNDDEVICE_IMPORT_ERROR = exc
    return _SOUNDDEVICE


class LinuxAudioCaptureBackend:
    """Linux capture backend using PortAudio via ``sounddevice``."""

    def __init__(self, *, use_fixture: bool = False) -> None:
        """Initialize backend state.

        Parameters
        ----------
        use_fixture : bool, optional
            If ``True``, emits synthetic frames instead of live device input.
        """
        self.use_fixture = use_fixture
        self.config: CaptureConfig | None = None
        self.frame_samples = 0
        self._queues: dict[str, queue.Queue[RawFrame]] = {
            "mic": queue.Queue(maxsize=300),
            "speakers": queue.Queue(maxsize=300),
        }
        self._streams: dict[str, Any] = {}
        self._fixture_frame_index = 0
        self._dropped_callback_frames = 0

    @property
    def dropped_callback_frames(self) -> int:
        """Return the number of callback frames dropped due to queue overflow.

        Returns
        -------
        int
            Dropped callback frame count.
        """
        return self._dropped_callback_frames

    def list_devices(self) -> list[dict[str, object]]:
        """List available Linux audio input devices.

        Returns
        -------
        list[dict[str, object]]
            Device dictionaries with index, name, channels, and sample rate.
        """
        sd = load_sounddevice()
        if sd is None:
            return []
        devices = sd.query_devices()
        output: list[dict[str, object]] = []
        for index, device in enumerate(devices):
            output.append(
                {
                    "index": index,
                    "name": str(device.get("name", "")),
                    "max_input_channels": int(device.get("max_input_channels", 0)),
                    "default_samplerate": float(device.get("default_samplerate", 0.0)),
                }
            )
        return output

    def open(self, config: CaptureConfig) -> None:
        """Open capture streams for configured sources.

        Parameters
        ----------
        config : CaptureConfig
            Capture configuration.
        """
        self.config = config
        self.frame_samples = int((config.sample_rate_hz * config.frame_ms) / 1000)
        if self.frame_samples <= 0:
            raise ValueError("frame_ms and sample_rate_hz produced 0 samples per frame")

        if self.use_fixture:
            return

        sd = load_sounddevice()
        if sd is None:
            suffix = ""
            if _SOUNDDEVICE_IMPORT_ERROR is not None:
                suffix = f" ({type(_SOUNDDEVICE_IMPORT_ERROR).__name__}: {_SOUNDDEVICE_IMPORT_ERROR})"
            raise RuntimeError(f"sounddevice is required for real Linux capture and could not be imported{suffix}")
        if config.source_mode != AudioSourceMode.BOTH:
            raise ValueError("Phase 0 Linux backend currently requires source_mode='both' for synchronized capture.")

        mic_device = self.resolve_device(sd, config.mic_device, require_monitor=False)
        speaker_device = self.resolve_device(sd, config.speaker_device, require_monitor=True)

        self._streams["mic"] = self.build_stream(sd, "mic", mic_device)
        self._streams["speakers"] = self.build_stream(sd, "speakers", speaker_device)
        for stream in self._streams.values():
            stream.start()

    def resolve_device(
        self,
        sd: Any,
        configured: str | int | None,
        *,
        require_monitor: bool,
    ) -> str | int:
        """Resolve a device identifier for microphone or loopback capture."""
        if configured is not None:
            return configured

        monitor_markers = ("monitor", "loopback")
        candidates = []
        for index, device in enumerate(sd.query_devices()):
            name = str(device.get("name", ""))
            input_channels = int(device.get("max_input_channels", 0))
            if input_channels < 1:
                continue
            is_monitor = any(marker in name.lower() for marker in monitor_markers)
            if require_monitor and is_monitor:
                return index
            if not require_monitor and not is_monitor:
                candidates.append(index)

        if require_monitor:
            raise RuntimeError("No speaker monitor/loopback input device found. Provide --speaker-device explicitly.")
        if not candidates:
            raise RuntimeError("No microphone input device found. Provide --mic-device explicitly.")
        return candidates[0]

    def build_stream(self, sd: Any, stream_name: str, device: str | int) -> Any:
        """Build a ``sounddevice.RawInputStream`` for one source stream."""

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            raw_frame = RawFrame(
                stream=stream_name,  # type: ignore[arg-type]
                mono_pcm16=bytes(indata),
                captured_at_monotonic_ns=time.monotonic_ns(),
            )
            try:
                self._queues[stream_name].put_nowait(raw_frame)
            except queue.Full:
                self._dropped_callback_frames += 1

        return sd.RawInputStream(
            samplerate=self.config.sample_rate_hz,
            blocksize=self.frame_samples,
            channels=self.config.channels,
            dtype="int16",
            device=device,
            callback=callback,
        )

    def fixture_frame(self, *, stream: str, frequency_hz: float, drift_ns: int = 0) -> RawFrame:
        """Generate a deterministic synthetic frame for fixture mode."""
        assert self.config is not None
        frame_offset = self._fixture_frame_index * self.frame_samples
        amplitude = 11_000
        samples = [
            int(
                amplitude
                * math.sin(2.0 * math.pi * frequency_hz * ((frame_offset + sample_index) / self.config.sample_rate_hz))
            )
            for sample_index in range(self.frame_samples)
        ]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        return RawFrame(
            stream=stream,  # type: ignore[arg-type]
            mono_pcm16=pcm,
            captured_at_monotonic_ns=time.monotonic_ns() + drift_ns,
        )

    def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
        """Read a synchronized mic/speaker frame pair.

        Parameters
        ----------
        timeout_ms : int, optional
            Maximum wait per queue in milliseconds.

        Returns
        -------
        dict[str, RawFrame]
            Frame pair keyed by ``mic`` and ``speakers``.
        """
        if self.config is None:
            raise RuntimeError("Capture backend is not open")

        if self.use_fixture:
            time.sleep(self.config.frame_ms / 1000)
            self._fixture_frame_index += 1
            mic = self.fixture_frame(stream="mic", frequency_hz=240.0, drift_ns=0)
            speakers = self.fixture_frame(stream="speakers", frequency_hz=480.0, drift_ns=2_000_000)
            return {"mic": mic, "speakers": speakers}

        timeout_s = timeout_ms / 1000
        try:
            mic = self._queues["mic"].get(timeout=timeout_s)
            speakers = self._queues["speakers"].get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for synchronized mic+speaker frames") from exc
        return {"mic": mic, "speakers": speakers}

    def close(self) -> None:
        """Stop and close live capture streams."""
        for stream in self._streams.values():
            stream.stop()
            stream.close()
        self._streams.clear()
