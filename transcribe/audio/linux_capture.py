from __future__ import annotations

from collections.abc import Mapping
import importlib
import logging
import math
import queue
import struct
import time
from typing import Any

from transcribe.audio.interfaces import RawFrame
from transcribe.audio.resample import resample_pcm16_mono_linear
from transcribe.models import AudioSourceMode, CaptureConfig

_SOUNDDEVICE: Any | None = None
_SOUNDDEVICE_ATTEMPTED = False
_SOUNDDEVICE_IMPORT_ERROR: Exception | None = None
LOGGER = logging.getLogger("transcribe.audio.linux_capture")

_GENERIC_BACKEND_DEVICE_NAMES = {
    "pipewire",
    "pulse",
    "default",
    "sysdefault",
    "front",
    "surround40",
    "surround41",
    "surround50",
    "surround51",
    "surround71",
}

_SPEAKER_STRONG_MARKERS = (
    "monitor",
    "loopback",
    "playback stream",
    "internal playback stream",
    "stereo mix",
    "what u hear",
)
_SPEAKER_WEAK_MARKERS = (
    "playback",
    "output",
    "sink",
    "speaker",
    "speakers",
    "hdmi",
    "iec958",
    "digital stereo",
)
_SPEAKER_APP_MARKERS = (
    "spotify",
    "chrome",
    "chromium",
    "firefox",
    "browser",
    "youtube",
    "vlc",
    "mpv",
    "discord",
    "zoom",
    "teams",
    "obs",
)

_MIC_STRONG_MARKERS = (
    "noisetorch",
    "rnnoise",
    "krisp",
    "microphone",
    " usb mic",
    "usb mic ",
)
_MIC_WEAK_MARKERS = (
    "mic",
    "capture stream",
    "capture",
    "source",
    "input",
    "headset",
)
_MIC_NEGATIVE_APP_MARKERS = (
    "spotify",
    "chrome",
    "chromium",
    "firefox",
    "browser",
    "youtube",
    "vlc",
    "mpv",
    "discord",
    "zoom",
    "teams",
    "obs",
    "settings",
)


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
        self._device_sample_rates_hz: dict[str, int] = {}
        self._device_frame_samples: dict[str, int] = {}
        self._device_channels: dict[str, int] = {}
        self._stream_quality_ema: dict[str, float] = {}
        self._preferred_stream_key: dict[str, str] = {}
        self._sample_rate_hz = 0
        self._fixture_frame_index = 0
        self._dropped_callback_frames = 0
        self._consecutive_timeouts = 0
        self._timeout_recovery_threshold = 3
        self._last_recovery_attempt_monotonic = 0.0
        self._recovery_cooldown_sec = 1.0
        self._callback_frame_count = 0
        self._last_callback_monotonic_by_stream: dict[str, float] = {}
        self._last_successful_read_monotonic = 0.0
        self._recovery_attempt_count = 0
        self._recovery_success_count = 0
        self._recovery_failure_count = 0
        self._last_recovery_error = ""

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

    @property
    def device_sample_rates_hz(self) -> dict[str, int]:
        """Return negotiated hardware sample rates keyed by stream key."""
        return dict(self._device_sample_rates_hz)

    @property
    def device_channels(self) -> dict[str, int]:
        """Return opened device channel counts keyed by stream key."""
        return dict(self._device_channels)

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
        self._device_sample_rates_hz = {}
        self._device_frame_samples = {}
        self._device_channels = {}
        self._stream_quality_ema = {}
        self._preferred_stream_key = {}
        self._streams.clear()
        self._consecutive_timeouts = 0
        self._callback_frame_count = 0
        self._last_callback_monotonic_by_stream = {}
        self._last_successful_read_monotonic = 0.0
        self._recovery_attempt_count = 0
        self._recovery_success_count = 0
        self._recovery_failure_count = 0
        self._last_recovery_error = ""

        if self._sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")

        self.frame_samples = int((self._sample_rate_hz * config.frame_ms) / 1000)
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

        opened_devices: dict[str, list[str | int]] = {}
        surviving_stream_names: list[str] = []
        for stream_name in self._active_stream_names:
            stream_keys: list[str] = []
            opened_stream_devices: list[str | int] = []
            last_open_error: Exception | None = None
            for device in resolved_devices[stream_name]:
                stream_key = f"{stream_name}:{device}"
                self._queues[stream_key] = queue.Queue(maxsize=self._queue_max_frames)
                try:
                    stream, device_sample_rate_hz, frame_samples, device_channels = self._open_stream_with_fallback(
                        sd,
                        stream_key=stream_key,
                        stream_group=stream_name,
                        device=device,
                        channels=config.channels,
                        requested_sample_rate_hz=config.sample_rate_hz,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._queues.pop(stream_key, None)
                    last_open_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                    LOGGER.warning(
                        "Skipping %s capture device %r because it could not be opened: %s",
                        stream_name,
                        device,
                        exc,
                    )
                    continue
                self._device_sample_rates_hz[stream_key] = device_sample_rate_hz
                self._device_frame_samples[stream_key] = frame_samples
                self._device_channels[stream_key] = device_channels
                self._streams[stream_key] = stream
                stream_keys.append(stream_key)
                opened_stream_devices.append(device)
            if stream_keys:
                self._group_stream_keys[stream_name] = tuple(stream_keys)
                opened_devices[stream_name] = opened_stream_devices
                surviving_stream_names.append(stream_name)
                continue

            configured_device = config.mic_device if stream_name == "mic" else config.speaker_device
            if configured_device is not None or not config.allow_missing_sources:
                if last_open_error is not None:
                    raise RuntimeError(
                        f"Failed to open {stream_name} capture device(s): {last_open_error}"
                    ) from last_open_error
                if stream_name == "speakers":
                    raise RuntimeError("No speaker playback/loopback input device found. Provide --speaker-device explicitly.")
                raise RuntimeError("No microphone input device found. Provide --mic-device explicitly.")

            LOGGER.warning("No %s capture streams could be opened; continuing without that source.", stream_name)

        self._active_stream_names = tuple(surviving_stream_names)
        if not self._active_stream_names:
            raise RuntimeError("No capture devices could be opened for active source mode.")

        self._resolved_devices = {name: tuple(opened_devices[name]) for name in self._active_stream_names}


    def negotiate_sample_rate(
        self,
        sd: Any,
        *,
        device: str | int,
        channels: int,
        requested_sample_rate_hz: int,
    ) -> int:
        """Choose a supported input sample rate for one device."""
        candidate_rates: list[int] = [int(requested_sample_rate_hz)]
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
            if self._sample_rate_supported(sd, device=device, channels=channels, sample_rate_hz=rate):
                return rate

        raise RuntimeError(
            "No supported sample rate found for selected input device(s). "
            "Try specifying a different --mic-device (or --speaker-device for both-mode capture)."
        )

    def _candidate_sample_rates(
        self,
        sd: Any,
        *,
        device: str | int,
        requested_sample_rate_hz: int,
    ) -> list[int]:
        """Return candidate sample rates for one device in preference order."""
        candidate_rates: list[int] = [int(requested_sample_rate_hz)]
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
        return deduped_candidates

    def _open_stream_with_fallback(
        self,
        sd: Any,
        *,
        stream_key: str,
        stream_group: str,
        device: str | int,
        channels: int,
        requested_sample_rate_hz: int,
    ) -> tuple[Any, int, int, int]:
        """Open one stream, retrying alternative channel/rate combinations when needed."""
        last_error: Exception | None = None
        for candidate_channels in self._candidate_channel_counts(
            sd,
            device=device,
            requested_channels=channels,
            stream_group=stream_group,
        ):
            for candidate_rate_hz in self._candidate_sample_rates(
                sd,
                device=device,
                requested_sample_rate_hz=requested_sample_rate_hz,
            ):
                if not self._sample_rate_supported(
                    sd,
                    device=device,
                    channels=candidate_channels,
                    sample_rate_hz=candidate_rate_hz,
                    stream_group=stream_group,
                ):
                    continue
                frame_samples = int((candidate_rate_hz * self.config.frame_ms) / 1000) if self.config is not None else 0
                if frame_samples <= 0:
                    continue
                stream = None
                try:
                    stream = self.build_stream(
                        sd,
                        stream_key=stream_key,
                        stream_group=stream_group,
                        device=device,
                        sample_rate_hz=candidate_rate_hz,
                        frame_samples=frame_samples,
                        device_channels=candidate_channels,
                    )
                    stream.start()
                except Exception as exc:  # noqa: BLE001
                    if stream is not None:
                        try:
                            stream.stop()
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            stream.close()
                        except Exception:  # noqa: BLE001
                            pass
                    last_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                    continue
                return stream, candidate_rate_hz, frame_samples, candidate_channels

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "No supported sample rate found for selected input device(s). "
            "Try specifying a different --mic-device (or --speaker-device for both-mode capture)."
        )

    def _sample_rate_supported(
        self,
        sd: Any,
        *,
        device: str | int,
        channels: int,
        sample_rate_hz: int,
        stream_group: str | None = None,
    ) -> bool:
        """Check whether an input sample rate is valid for one device."""
        try:
            sd.check_input_settings(
                device=device,
                channels=channels,
                dtype="int16",
                samplerate=float(sample_rate_hz),
                extra_settings=self._stream_extra_settings(sd, stream_group=stream_group, device=device),
            )
        except Exception:  # pragma: no cover - relies on backend-specific validation
            return False
        return True

    def _stream_extra_settings(
        self,
        sd: Any,
        *,
        stream_group: str | None,
        device: str | int,
    ) -> Any | None:
        """Return backend-specific extra stream settings, if any."""
        _ = (sd, stream_group, device)
        return None

    def _candidate_channel_counts(
        self,
        sd: Any,
        *,
        device: str | int,
        requested_channels: int,
        stream_group: str | None,
    ) -> list[int]:
        """Return candidate input channel counts for one device."""
        _ = (sd, device, stream_group)
        return [max(1, int(requested_channels))]

    @staticmethod
    def _pcm16_frames_by_channel(pcm16: bytes, *, channels: int) -> list[tuple[int, ...]]:
        """Decode interleaved PCM16 bytes into per-frame channel tuples."""
        if channels <= 1:
            return [(sample,) for (sample,) in struct.iter_unpack("<h", pcm16)]
        return list(struct.iter_unpack(f"<{channels}h", pcm16))

    @classmethod
    def _convert_interleaved_pcm16_to_mono(
        cls,
        pcm16: bytes,
        *,
        channels: int,
        stream_group: str,
    ) -> bytes:
        """Convert interleaved PCM16 into mono for downstream processing."""
        if channels <= 1 or not pcm16:
            return pcm16

        frames = cls._pcm16_frames_by_channel(pcm16, channels=channels)
        if not frames:
            return b""

        if stream_group == "mic":
            energy_by_channel = [0.0] * channels
            for frame in frames:
                for channel_index, sample in enumerate(frame):
                    energy_by_channel[channel_index] += float(sample * sample)
            selected_channel = max(range(channels), key=lambda index: energy_by_channel[index])
            mono_samples = [frame[selected_channel] for frame in frames]
        else:
            mono_samples = [int(round(sum(frame) / channels)) for frame in frames]

        return struct.pack(f"<{len(mono_samples)}h", *mono_samples)

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
        """Return ranked candidate device indices for mic or speaker/loopback capture."""
        scored_candidates: list[tuple[float, int]] = []
        generic_fallbacks: list[int] = []
        for index, raw_device in enumerate(sd.query_devices()):
            if not isinstance(raw_device, Mapping):
                continue
            name = self._normalize_device_name(raw_device)
            input_channels = int(raw_device.get("max_input_channels", 0))
            if input_channels < 1:
                continue

            if self._is_generic_backend_device(name):
                generic_fallbacks.append(index)

            score = self._score_device_for_role(name=name, require_monitor=require_monitor)
            minimum_score = 0.01 if require_monitor else 0.0
            if score >= minimum_score:
                # Slightly prefer devices with more input channels as tiebreaker.
                scored_candidates.append((score + min(float(input_channels), 4.0) * 0.01, index))

        if scored_candidates:
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            return [index for _, index in scored_candidates]

        # Fallback: if no confident speaker candidate exists, try generic backend nodes.
        if require_monitor and generic_fallbacks:
            return generic_fallbacks

        # Fallback: for mic mode, use all non-speaker-like devices.
        if not require_monitor:
            fallback_mic: list[int] = []
            for index, raw_device in enumerate(sd.query_devices()):
                if not isinstance(raw_device, Mapping):
                    continue
                input_channels = int(raw_device.get("max_input_channels", 0))
                if input_channels < 1:
                    continue
                name = self._normalize_device_name(raw_device)
                speaker_score = self._score_device_for_role(name=name, require_monitor=True)
                if speaker_score <= 0:
                    fallback_mic.append(index)
            return fallback_mic

        return []

    @staticmethod
    def _normalize_device_name(device: Mapping[str, object]) -> str:
        """Normalize a queried device name for role classification."""
        return str(device.get("name", "")).strip().lower()

    @staticmethod
    def _is_generic_backend_device(device_name_lower: str) -> bool:
        """Return True when name matches a generic routing backend."""
        return device_name_lower in _GENERIC_BACKEND_DEVICE_NAMES

    @staticmethod
    def _score_device_for_role(*, name: str, require_monitor: bool) -> float:
        """Score a device name for microphone or speaker/loopback suitability."""
        score = 0.0
        is_generic = LinuxAudioCaptureBackend._is_generic_backend_device(name)

        has_speaker_strong = any(marker in name for marker in _SPEAKER_STRONG_MARKERS)
        has_speaker_weak = any(marker in name for marker in _SPEAKER_WEAK_MARKERS)
        has_speaker_app = any(marker in name for marker in _SPEAKER_APP_MARKERS)
        has_mic_strong = any(marker in name for marker in _MIC_STRONG_MARKERS)
        has_mic_weak = any(marker in name for marker in _MIC_WEAK_MARKERS)
        has_capture_stream = "capture stream" in name
        has_playback_stream = "playback stream" in name

        if require_monitor:
            if has_speaker_strong:
                score += 9.0
            if has_speaker_weak:
                score += 4.0
            if has_speaker_app:
                score += 5.0
            if has_mic_strong or (" mic" in f" {name} "):
                score -= 8.0
            if has_capture_stream and not has_playback_stream:
                score -= 7.0
            if is_generic:
                score += 1.0
            return score

        if has_mic_strong:
            score += 9.0
        if has_mic_weak:
            score += 4.0
        if "bluetooth internal capture stream" in name:
            score += 2.0
        if any(marker in name for marker in _MIC_NEGATIVE_APP_MARKERS):
            score -= 6.0
        if has_speaker_strong or has_playback_stream:
            score -= 8.0
        elif has_speaker_weak or has_speaker_app:
            score -= 5.0
        if is_generic:
            score -= 2.0
        return score

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
                LOGGER.warning(
                    "No speaker playback/loopback device found; continuing without speaker capture."
                )
            else:
                LOGGER.warning("No microphone input device found; continuing without mic capture.")
            return []

        if require_monitor:
            raise RuntimeError(
                "No speaker playback/loopback input device found. Provide --speaker-device explicitly."
            )
        raise RuntimeError("No microphone input device found. Provide --mic-device explicitly.")

    def build_stream(
        self,
        sd: Any,
        *,
        stream_key: str,
        stream_group: str,
        device: str | int,
        sample_rate_hz: int,
        frame_samples: int,
        device_channels: int,
    ) -> Any:
        """Build a ``sounddevice.RawInputStream`` for one source stream."""

        def callback(indata: bytes, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            pcm16 = bytes(indata)
            frame_sample_rate_hz = sample_rate_hz
            if device_channels > 1:
                pcm16 = self._convert_interleaved_pcm16_to_mono(
                    pcm16,
                    channels=device_channels,
                    stream_group=stream_group,
                )
            if sample_rate_hz != self._sample_rate_hz and self.config is not None and self.config.channels == 1:
                pcm16 = resample_pcm16_mono_linear(
                    pcm16,
                    source_rate_hz=sample_rate_hz,
                    target_rate_hz=self._sample_rate_hz,
                )
                frame_sample_rate_hz = self._sample_rate_hz
            raw_frame = RawFrame(
                stream=stream_group,  # type: ignore[arg-type]
                mono_pcm16=pcm16,
                captured_at_monotonic_ns=time.monotonic_ns(),
                sample_rate_hz=frame_sample_rate_hz,
            )
            self._callback_frame_count += 1
            self._last_callback_monotonic_by_stream[stream_key] = time.monotonic()
            try:
                self._queues[stream_key].put_nowait(raw_frame)
            except queue.Full:
                self._dropped_callback_frames += 1

        return sd.RawInputStream(
            samplerate=sample_rate_hz,
            blocksize=frame_samples,
            channels=device_channels,
            dtype="int16",
            device=device,
            callback=callback,
            extra_settings=self._stream_extra_settings(sd, stream_group=stream_group, device=device),
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
            sample_rate_hz=self._sample_rate_hz,
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

    @staticmethod
    def _drain_latest_frame(queue_ref: queue.Queue[RawFrame], *, timeout_s: float = 0.0) -> RawFrame | None:
        """Read the freshest available frame from one queue."""
        latest: RawFrame | None = None
        try:
            if timeout_s > 0:
                latest = queue_ref.get(timeout=timeout_s)
            else:
                latest = queue_ref.get_nowait()
        except queue.Empty:
            return None

        while True:
            try:
                latest = queue_ref.get_nowait()
            except queue.Empty:
                return latest

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

        candidate_frames: dict[str, RawFrame] = {}
        preferred_frame = self._drain_latest_frame(preferred_queue, timeout_s=timeout_s)
        if preferred_frame is not None:
            self._update_stream_quality(preferred, preferred_frame)
            candidate_frames[preferred] = preferred_frame

        # Scout non-preferred devices without blocking, then keep only the freshest frame.
        for stream_key in stream_keys:
            if stream_key == preferred:
                continue
            queue_ref = self._queues.get(stream_key)
            if queue_ref is None:
                continue
            latest = self._drain_latest_frame(queue_ref)
            if latest is not None:
                candidate_frames[stream_key] = latest
                self._update_stream_quality(stream_key, latest)

        if not candidate_frames:
            raise TimeoutError(f"Timed out waiting for capture frame ({stream_name})")

        candidate_stream_keys = tuple(key for key in stream_keys if key in candidate_frames)
        best_stream_key = max(candidate_stream_keys, key=lambda key: self._stream_quality_ema.get(key, float("-inf")))
        preferred_score = self._stream_quality_ema.get(preferred, float("-inf")) if preferred in candidate_frames else float("-inf")
        best_score = self._stream_quality_ema.get(best_stream_key, float("-inf"))
        hysteresis_margin = 0.10

        selected_stream_key = preferred if preferred in candidate_frames else best_stream_key
        if best_stream_key != selected_stream_key and best_score > (preferred_score + hysteresis_margin):
            selected_stream_key = best_stream_key
            self._preferred_stream_key[stream_name] = best_stream_key
        else:
            self._preferred_stream_key[stream_name] = selected_stream_key

        return candidate_frames[selected_stream_key]

    @staticmethod
    def _stream_looks_active(stream: Any) -> bool:
        """Best-effort health check for a PortAudio stream wrapper."""
        active = getattr(stream, "active", None)
        if active is False:
            return False
        if getattr(stream, "stopped", False):
            return False
        if getattr(stream, "closed", False):
            return False
        return True

    def _recover_live_streams(self) -> None:
        """Attempt to reopen live capture streams after a stall or device change."""
        if self.use_fixture or self.config is None:
            return

        now = time.monotonic()
        if (now - self._last_recovery_attempt_monotonic) < self._recovery_cooldown_sec:
            return
        self._last_recovery_attempt_monotonic = now
        self._recovery_attempt_count += 1

        try:
            self.close()
            self.open(self.config)
        except Exception as exc:  # noqa: BLE001
            self._recovery_failure_count += 1
            self._last_recovery_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning("Capture stalled; waiting for audio device recovery: %s", exc)
            return

        self._consecutive_timeouts = 0
        self._recovery_success_count += 1
        self._last_recovery_error = ""
        LOGGER.warning("Capture stream recovered after device interruption.")

    def diagnostics_snapshot(self) -> dict[str, object]:
        """Return a best-effort snapshot of capture health and recovery state."""
        now = time.monotonic()
        stream_states: dict[str, dict[str, object]] = {}
        for stream_key, stream in self._streams.items():
            last_callback = self._last_callback_monotonic_by_stream.get(stream_key)
            stream_states[stream_key] = {
                "active": bool(getattr(stream, "active", False)),
                "stopped": bool(getattr(stream, "stopped", False)),
                "closed": bool(getattr(stream, "closed", False)),
                "last_callback_age_sec": round(now - last_callback, 3) if last_callback is not None else None,
                "runtime_error": self._stream_runtime_error(stream),
            }

        queue_depths = {stream_key: queue_ref.qsize() for stream_key, queue_ref in self._queues.items()}
        last_success_age_sec = None
        if self._last_successful_read_monotonic > 0:
            last_success_age_sec = round(now - self._last_successful_read_monotonic, 3)

        return {
            "active_stream_names": list(self._active_stream_names),
            "consecutive_timeouts": self._consecutive_timeouts,
            "dropped_callback_frames": self._dropped_callback_frames,
            "callback_frame_count": self._callback_frame_count,
            "last_successful_read_age_sec": last_success_age_sec,
            "recovery_attempt_count": self._recovery_attempt_count,
            "recovery_success_count": self._recovery_success_count,
            "recovery_failure_count": self._recovery_failure_count,
            "last_recovery_error": self._last_recovery_error,
            "queue_depths": queue_depths,
            "stream_states": stream_states,
        }

    @staticmethod
    def _stream_runtime_error(stream: Any) -> str:
        """Return a stringified runtime error for one stream when available."""
        runtime_error = getattr(stream, "_runtime_error", None)
        if runtime_error is None:
            return ""
        return f"{type(runtime_error).__name__}: {runtime_error}"

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
        try:
            for stream_name in self._active_stream_names:
                output[stream_name] = self._read_group_frame(stream_name, timeout_s)
        except TimeoutError:
            self._consecutive_timeouts += 1
            if (
                self._consecutive_timeouts >= self._timeout_recovery_threshold
                or not self._streams
                or any(not self._stream_looks_active(stream) for stream in self._streams.values())
            ):
                self._recover_live_streams()
            raise

        self._consecutive_timeouts = 0
        self._last_successful_read_monotonic = time.monotonic()
        return output

    def close(self) -> None:
        """Stop and close live capture streams."""
        for stream in self._streams.values():
            try:
                stream.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._streams.clear()
        self._queues.clear()
        self._group_stream_keys.clear()
        self._resolved_devices.clear()
        self._device_sample_rates_hz.clear()
        self._device_frame_samples.clear()
        self._device_channels.clear()
        self._stream_quality_ema.clear()
        self._preferred_stream_key.clear()
