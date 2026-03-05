from __future__ import annotations

import importlib
import logging
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
LOGGER = logging.getLogger("transcribe.audio.linux_capture")


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
        self._queue_max_frames = 1_200
        self._queues: dict[str, queue.Queue[RawFrame]] = {}
        self._streams: dict[str, Any] = {}
        self._active_stream_names: tuple[str, ...] = ()
        self._group_stream_keys: dict[str, tuple[str, ...]] = {}
        self._resolved_devices: dict[str, tuple[str | int, ...]] = {}
        self._stream_quality_ema: dict[str, float] = {}
        self._preferred_stream_key: dict[str, str] = {}
        self._sample_rate_hz = 0
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

    @property
    def sample_rate_hz(self) -> int:
        """Return effective capture sample rate selected by the backend."""
        return self._sample_rate_hz

    @property
    def active_devices(self) -> dict[str, tuple[str | int, ...]]:
        """Return resolved capture devices by stream group."""
        return dict(self._resolved_devices)

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
        self._active_stream_names = self._resolve_active_stream_names(config.source_mode)
        self._fixture_frame_index = 0
        self._dropped_callback_frames = 0
        self._sample_rate_hz = int(config.sample_rate_hz)
        self._queues.clear()
        self._group_stream_keys = {name: () for name in self._active_stream_names}
        self._resolved_devices = {name: () for name in self._active_stream_names}
        self._stream_quality_ema = {}
        self._preferred_stream_key = {}
        self._streams.clear()

        if self.use_fixture:
            self.frame_samples = int((self._sample_rate_hz * config.frame_ms) / 1000)
            if self.frame_samples <= 0:
                raise ValueError("frame_ms and sample_rate_hz produced 0 samples per frame")
            return

        sd = load_sounddevice()
        if sd is None:
            suffix = ""
            if _SOUNDDEVICE_IMPORT_ERROR is not None:
                suffix = f" ({type(_SOUNDDEVICE_IMPORT_ERROR).__name__}: {_SOUNDDEVICE_IMPORT_ERROR})"
            raise RuntimeError(f"sounddevice is required for real Linux capture and could not be imported{suffix}")
        resolved_devices: dict[str, list[str | int]] = {}
        if "mic" in self._active_stream_names:
            resolved_devices["mic"] = self.resolve_devices(
                sd,
                configured=config.mic_device,
                require_monitor=False,
                include_all=config.capture_all_mic_devices,
                allow_missing=config.allow_missing_sources,
            )
        if "speakers" in self._active_stream_names:
            resolved_devices["speakers"] = self.resolve_devices(
                sd,
                configured=config.speaker_device,
                require_monitor=True,
                include_all=config.capture_all_speaker_devices,
                allow_missing=config.allow_missing_sources,
            )

        self._active_stream_names = tuple(name for name in self._active_stream_names if resolved_devices.get(name))
        if not self._active_stream_names:
            raise RuntimeError("No capture devices resolved for active source mode.")

        all_devices: tuple[str | int, ...] = tuple(
            device for stream_name in self._active_stream_names for device in resolved_devices[stream_name]
        )
        self._resolved_devices = {name: tuple(resolved_devices[name]) for name in self._active_stream_names}

        self._sample_rate_hz = self.negotiate_sample_rate(
            sd,
            devices=all_devices,
            channels=config.channels,
            requested_sample_rate_hz=config.sample_rate_hz,
        )
        if self._sample_rate_hz != int(config.sample_rate_hz):
            LOGGER.warning(
                "Requested sample rate %s Hz is unsupported for selected device(s); using %s Hz instead.",
                int(config.sample_rate_hz),
                self._sample_rate_hz,
            )
        self.frame_samples = int((self._sample_rate_hz * config.frame_ms) / 1000)
        if self.frame_samples <= 0:
            raise ValueError("frame_ms and sample_rate_hz produced 0 samples per frame")

        for stream_name in self._active_stream_names:
            stream_keys: list[str] = []
            for device in resolved_devices[stream_name]:
                stream_key = f"{stream_name}:{device}"
                self._queues[stream_key] = queue.Queue(maxsize=self._queue_max_frames)
                self._streams[stream_key] = self.build_stream(
                    sd,
                    stream_key=stream_key,
                    stream_group=stream_name,
                    device=device,
                )
                stream_keys.append(stream_key)
            self._group_stream_keys[stream_name] = tuple(stream_keys)

        for stream in self._streams.values():
            stream.start()

    def negotiate_sample_rate(
        self,
        sd: Any,
        *,
        devices: tuple[str | int, ...],
        channels: int,
        requested_sample_rate_hz: int,
    ) -> int:
        """Choose a supported input sample rate for all active capture devices."""
        candidate_rates: list[int] = [int(requested_sample_rate_hz)]
        for device in devices:
            device_info = sd.query_devices(device=device)
            default_rate = int(round(float(device_info.get("default_samplerate", 0.0))))
            if default_rate > 0:
                candidate_rates.append(default_rate)
        candidate_rates.extend([48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 8_000])

        deduped_candidates: list[int] = []
        seen: set[int] = set()
        for rate in candidate_rates:
            if rate <= 0 or rate in seen:
                continue
            seen.add(rate)
            deduped_candidates.append(rate)

        for rate in deduped_candidates:
            if self._sample_rate_supported_for_devices(sd, devices=devices, channels=channels, sample_rate_hz=rate):
                return rate

        raise RuntimeError(
            "No supported sample rate found for selected input device(s). "
            "Try specifying a different --mic-device (or --speaker-device for both-mode capture)."
        )

    def _sample_rate_supported_for_devices(
        self,
        sd: Any,
        *,
        devices: tuple[str | int, ...],
        channels: int,
        sample_rate_hz: int,
    ) -> bool:
        """Check whether an input sample rate is valid across all selected devices."""
        for device in devices:
            try:
                sd.check_input_settings(
                    device=device,
                    channels=channels,
                    dtype="int16",
                    samplerate=float(sample_rate_hz),
                )
            except Exception:  # pragma: no cover - relies on backend-specific validation
                return False
        return True

    def _resolve_active_stream_names(self, source_mode: AudioSourceMode) -> tuple[str, ...]:
        """Resolve active stream list for the selected capture mode."""
        if source_mode == AudioSourceMode.MIC:
            return ("mic",)
        if source_mode == AudioSourceMode.SPEAKERS:
            return ("speakers",)
        return ("mic", "speakers")

    def find_candidate_devices(
        self,
        sd: Any,
        *,
        require_monitor: bool,
    ) -> list[int]:
        """Return candidate device indices for microphone or loopback capture."""
        monitor_markers = ("monitor", "loopback")
        candidates: list[int] = []
        for index, device in enumerate(sd.query_devices()):
            name = str(device.get("name", ""))
            input_channels = int(device.get("max_input_channels", 0))
            if input_channels < 1:
                continue
            is_monitor = any(marker in name.lower() for marker in monitor_markers)
            if require_monitor and is_monitor:
                candidates.append(index)
            if not require_monitor and not is_monitor:
                candidates.append(index)
        return candidates

    def resolve_devices(
        self,
        sd: Any,
        configured: str | int | None,
        *,
        require_monitor: bool,
        include_all: bool,
        allow_missing: bool,
    ) -> list[str | int]:
        """Resolve one or many device identifiers for microphone or loopback capture."""
        if configured is not None:
            return [configured]

        candidates = self.find_candidate_devices(sd, require_monitor=require_monitor)
        if candidates:
            if include_all:
                return candidates
            return [candidates[0]]

        if allow_missing:
            if require_monitor:
                LOGGER.warning("No speaker monitor/loopback device found; continuing without speaker capture.")
            else:
                LOGGER.warning("No microphone input device found; continuing without mic capture.")
            return []

        if require_monitor:
            raise RuntimeError("No speaker monitor/loopback input device found. Provide --speaker-device explicitly.")
        raise RuntimeError("No microphone input device found. Provide --mic-device explicitly.")

    def build_stream(self, sd: Any, *, stream_key: str, stream_group: str, device: str | int) -> Any:
        """Build a ``sounddevice.RawInputStream`` for one source stream."""

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            raw_frame = RawFrame(
                stream=stream_group,  # type: ignore[arg-type]
                mono_pcm16=bytes(indata),
                captured_at_monotonic_ns=time.monotonic_ns(),
            )
            try:
                self._queues[stream_key].put_nowait(raw_frame)
            except queue.Full:
                self._dropped_callback_frames += 1

        return sd.RawInputStream(
            samplerate=self._sample_rate_hz,
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
                amplitude * math.sin(2.0 * math.pi * frequency_hz * ((frame_offset + sample_index) / self._sample_rate_hz))
            )
            for sample_index in range(self.frame_samples)
        ]
        pcm = struct.pack(f"<{len(samples)}h", *samples)
        return RawFrame(
            stream=stream,  # type: ignore[arg-type]
            mono_pcm16=pcm,
            captured_at_monotonic_ns=time.monotonic_ns() + drift_ns,
        )

    @staticmethod
    def pcm16_clarity_score(pcm16: bytes) -> float:
        """Estimate frame clarity from PCM16 amplitude and clipping metrics."""
        sample_count = len(pcm16) // 2
        if sample_count <= 0:
            return float("-inf")

        sum_sq = 0.0
        sum_samples = 0
        peak_abs = 0
        clipped_count = 0
        for (sample,) in struct.iter_unpack("<h", pcm16):
            abs_value = abs(sample)
            sum_sq += float(sample * sample)
            sum_samples += sample
            peak_abs = max(peak_abs, abs_value)
            if abs_value >= 32_000:
                clipped_count += 1

        rms = math.sqrt(sum_sq / sample_count) / 32_768.0
        peak = peak_abs / 32_768.0
        clipping_ratio = clipped_count / sample_count
        dc_offset = abs(sum_samples / sample_count) / 32_768.0
        return (1.6 * rms) + (0.4 * peak) - (2.0 * clipping_ratio) - (0.3 * dc_offset)

    def _update_stream_quality(self, stream_key: str, frame: RawFrame) -> None:
        """Update rolling per-device quality score for one observed frame."""
        score = self.pcm16_clarity_score(frame.mono_pcm16)
        alpha = 0.25
        previous = self._stream_quality_ema.get(stream_key, score)
        self._stream_quality_ema[stream_key] = ((1.0 - alpha) * previous) + (alpha * score)

    def _read_group_frame(self, stream_name: str, timeout_s: float) -> RawFrame:
        """Read one continuous frame for a source group with low-overhead device scouting."""
        stream_keys = self._group_stream_keys.get(stream_name, ())
        if not stream_keys:
            raise TimeoutError("No streams configured for source group")

        preferred = self._preferred_stream_key.get(stream_name)
        if preferred not in stream_keys:
            preferred = stream_keys[0]

        preferred_queue = self._queues.get(preferred)
        if preferred_queue is None:
            raise TimeoutError("Preferred source stream queue is unavailable")

        try:
            preferred_frame = preferred_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(f"Timed out waiting for capture frame ({stream_name})") from exc

        self._update_stream_quality(preferred, preferred_frame)
        candidate_frames: dict[str, RawFrame] = {preferred: preferred_frame}

        # Scout non-preferred devices without blocking, then keep only the freshest frame.
        for stream_key in stream_keys:
            if stream_key == preferred:
                continue
            queue_ref = self._queues.get(stream_key)
            if queue_ref is None:
                continue
            latest = None
            while True:
                try:
                    latest = queue_ref.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                candidate_frames[stream_key] = latest
                self._update_stream_quality(stream_key, latest)

        best_stream_key = max(stream_keys, key=lambda key: self._stream_quality_ema.get(key, float("-inf")))
        preferred_score = self._stream_quality_ema.get(preferred, float("-inf"))
        best_score = self._stream_quality_ema.get(best_stream_key, float("-inf"))
        hysteresis_margin = 0.10

        selected_stream_key = preferred
        if best_stream_key != preferred and best_score > (preferred_score + hysteresis_margin):
            selected_stream_key = best_stream_key
            self._preferred_stream_key[stream_name] = best_stream_key
        else:
            self._preferred_stream_key[stream_name] = preferred

        return candidate_frames.get(selected_stream_key, preferred_frame)

    def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
        """Read one frame per active source group.

        Parameters
        ----------
        timeout_ms : int, optional
            Maximum wait per queue in milliseconds.

        Returns
        -------
        dict[str, RawFrame]
            Frames keyed by active groups (``mic`` and/or ``speakers``).
        """
        if self.config is None:
            raise RuntimeError("Capture backend is not open")

        if self.use_fixture:
            time.sleep(self.config.frame_ms / 1000)
            self._fixture_frame_index += 1
            frames: dict[str, RawFrame] = {}
            if "mic" in self._active_stream_names:
                frames["mic"] = self.fixture_frame(stream="mic", frequency_hz=240.0, drift_ns=0)
            if "speakers" in self._active_stream_names:
                frames["speakers"] = self.fixture_frame(stream="speakers", frequency_hz=480.0, drift_ns=2_000_000)
            return frames

        timeout_s = timeout_ms / 1000
        output: dict[str, RawFrame] = {}
        for stream_name in self._active_stream_names:
            output[stream_name] = self._read_group_frame(stream_name, timeout_s)
        return output

    def close(self) -> None:
        """Stop and close live capture streams."""
        for stream in self._streams.values():
            stream.stop()
            stream.close()
        self._streams.clear()
        self._queues.clear()
        self._group_stream_keys.clear()
        self._resolved_devices.clear()
        self._stream_quality_ema.clear()
        self._preferred_stream_key.clear()
