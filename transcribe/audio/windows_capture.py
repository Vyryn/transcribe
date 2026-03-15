from __future__ import annotations

from dataclasses import dataclass
import contextlib
import ctypes
import importlib
import logging
import queue
import threading
import time
from typing import Any
import warnings

import numpy as np

from transcribe.audio.interfaces import RawFrame
from transcribe.audio.linux_capture import LinuxAudioCaptureBackend
from transcribe.audio.resample import resample_pcm16_mono_linear
from transcribe.models import CaptureConfig

LOGGER = logging.getLogger("transcribe.audio.windows_capture")
_WINDOWS_HOSTAPI_NAME = "windows wasapi"
_SOUNDCARD: Any | None = None
_SOUNDCARD_ATTEMPTED = False
_SOUNDCARD_IMPORT_ERROR: Exception | None = None
_SOUNDCARD_NUMPY_PATCHED = False
_COINIT_MULTITHREADED = 0x0
_S_OK = 0
_S_FALSE = 1
_RPC_E_CHANGED_MODE = 0x80010106


def _patch_soundcard_numpy_fromstring() -> None:
    """Patch SoundCard's NumPy binary parsing for NumPy 2 compatibility."""
    global _SOUNDCARD_NUMPY_PATCHED
    if _SOUNDCARD_NUMPY_PATCHED:
        return

    try:
        mediafoundation = importlib.import_module("soundcard.mediafoundation")
    except Exception:  # pragma: no cover - optional runtime dependency
        return

    numpy_module = getattr(mediafoundation, "numpy", None)
    if numpy_module is None:
        return
    original_fromstring = getattr(numpy_module, "fromstring", None)
    if original_fromstring is None:
        return
    if getattr(numpy_module, "_transcribe_binary_fromstring_patch", False):
        _SOUNDCARD_NUMPY_PATCHED = True
        return

    def _compat_fromstring(
        string: Any,
        dtype: Any = float,
        count: int = -1,
        *,
        sep: str = "",
        like: Any = None,
    ) -> Any:
        if sep:
            if like is None:
                return original_fromstring(string, dtype=dtype, count=count, sep=sep)
            return original_fromstring(string, dtype=dtype, count=count, sep=sep, like=like)
        try:
            buffer = memoryview(string)
        except TypeError:
            if like is None:
                return original_fromstring(string, dtype=dtype, count=count, sep=sep)
            return original_fromstring(string, dtype=dtype, count=count, sep=sep, like=like)
        if like is None:
            return np.frombuffer(buffer, dtype=dtype, count=count)
        return np.frombuffer(buffer, dtype=dtype, count=count, like=like)

    setattr(numpy_module, "fromstring", _compat_fromstring)
    setattr(numpy_module, "_transcribe_binary_fromstring_patch", True)
    _SOUNDCARD_NUMPY_PATCHED = True


def _soundcard_runtime_warning_category() -> type[Warning]:
    """Resolve SoundCard's runtime warning category when available."""
    try:
        mediafoundation = importlib.import_module("soundcard.mediafoundation")
    except Exception:  # pragma: no cover - optional runtime dependency
        return RuntimeWarning
    category = getattr(mediafoundation, "SoundcardRuntimeWarning", RuntimeWarning)
    if isinstance(category, type) and issubclass(category, Warning):
        return category
    return RuntimeWarning


@contextlib.contextmanager
def _initialize_com_thread() -> Any:
    """Initialize COM for the current thread before touching WASAPI APIs."""
    if not _running_on_windows():
        yield
        return

    ole32 = getattr(ctypes, "OleDLL", None)
    if ole32 is None:
        yield
        return

    com_library = ole32("ole32")
    result = int(com_library.CoInitializeEx(None, _COINIT_MULTITHREADED)) & 0xFFFFFFFF
    should_uninitialize = result in {_S_OK, _S_FALSE}
    if result not in {_S_OK, _S_FALSE, _RPC_E_CHANGED_MODE}:
        raise OSError(f"CoInitializeEx failed with HRESULT 0x{result:08X}")
    try:
        yield
    finally:
        if should_uninitialize:
            com_library.CoUninitialize()


def _running_on_windows() -> bool:
    """Return True when the current process is running on Windows."""
    return importlib.import_module("sys").platform == "win32"


@dataclass(slots=True)
class _WindowsCaptureDevice:
    """Resolved Windows capture endpoint backed by SoundCard/WASAPI."""

    index: int
    name: str
    soundcard_id: str
    max_input_channels: int
    default_samplerate: float
    hostapi_name: str
    is_loopback: bool


class _SoundCardRecorderStream:
    """Threaded SoundCard recorder wrapper with a stream-like surface."""

    def __init__(
        self,
        *,
        stream_key: str,
        device_id: str,
        include_loopback: bool,
        sample_rate_hz: int,
        frame_samples: int,
        device_channels: int,
        callback: Any,
    ) -> None:
        self.stream_key = stream_key
        self.device_id = device_id
        self.include_loopback = include_loopback
        self.sample_rate_hz = int(sample_rate_hz)
        self.frame_samples = int(frame_samples)
        self.device_channels = int(device_channels)
        self.callback = callback
        self.active = False
        self.stopped = True
        self.closed = False
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._open_error: Exception | None = None
        self._runtime_error: Exception | None = None

    def _resolve_device(self) -> Any:
        soundcard = load_soundcard()
        if soundcard is None:
            suffix = ""
            if _SOUNDCARD_IMPORT_ERROR is not None:
                suffix = f" ({type(_SOUNDCARD_IMPORT_ERROR).__name__}: {_SOUNDCARD_IMPORT_ERROR})"
            raise RuntimeError(f"soundcard is required for Windows capture and could not be imported{suffix}")
        return soundcard.get_microphone(self.device_id, include_loopback=self.include_loopback)

    def _run(self) -> None:
        try:
            with _initialize_com_thread():
                device = self._resolve_device()
                warning_category = _soundcard_runtime_warning_category()
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="data discontinuity in recording",
                        category=warning_category,
                    )
                    with device.recorder(
                        samplerate=self.sample_rate_hz,
                        channels=self.device_channels,
                        blocksize=self.frame_samples,
                        exclusive_mode=False,
                    ) as recorder:
                        self.active = True
                        self.stopped = False
                        self._started_event.set()
                        while not self._stop_event.is_set():
                            block = recorder.record(numframes=self.frame_samples)
                            if self._stop_event.is_set():
                                break
                            self.callback(block, self.frame_samples, None, None)
        except Exception as exc:  # noqa: BLE001
            if not self._started_event.is_set():
                self._open_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                self._started_event.set()
            else:
                self._runtime_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                LOGGER.warning("Windows capture stream %s failed: %s", self.stream_key, exc)
        finally:
            self.active = False
            self.stopped = True

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.closed = False
        self.stopped = False
        self._stop_event.clear()
        self._started_event.clear()
        self._open_error = None
        self._runtime_error = None
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.stream_key}", daemon=True)
        self._thread.start()
        start_timeout_s = max(2.0, (self.frame_samples / max(self.sample_rate_hz, 1)) * 12.0)
        if not self._started_event.wait(timeout=start_timeout_s):
            self._stop_event.set()
            raise TimeoutError(f"Timed out starting SoundCard capture stream {self.stream_key}")
        if self._open_error is not None:
            self.stop()
            raise self._open_error

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self.active = False
        self.stopped = True

    def close(self) -> None:
        self.stop()
        self.closed = True


def load_soundcard() -> Any | None:
    """Load the optional ``soundcard`` dependency lazily."""
    global _SOUNDCARD, _SOUNDCARD_ATTEMPTED, _SOUNDCARD_IMPORT_ERROR
    if _SOUNDCARD_ATTEMPTED:
        return _SOUNDCARD

    _SOUNDCARD_ATTEMPTED = True
    try:
        _SOUNDCARD = importlib.import_module("soundcard")
        _patch_soundcard_numpy_fromstring()
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        _SOUNDCARD = None
        _SOUNDCARD_IMPORT_ERROR = exc
    return _SOUNDCARD


class WindowsAudioCaptureBackend(LinuxAudioCaptureBackend):
    """Windows capture backend using native WASAPI devices via ``soundcard``."""

    def list_devices(self) -> list[dict[str, object]]:
        """List capture-capable Windows WASAPI devices."""
        soundcard = load_soundcard()
        if soundcard is None:
            return []
        return [
            {
                "index": device.index,
                "name": device.name,
                "max_input_channels": device.max_input_channels,
                "default_samplerate": device.default_samplerate,
                "hostapi_name": device.hostapi_name,
                "device_id": device.soundcard_id,
                "is_loopback": device.is_loopback,
            }
            for device in self._device_catalog(soundcard)
        ]

    def open(self, config: CaptureConfig) -> None:
        """Open Windows WASAPI capture streams for configured sources."""
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

        if self._sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")

        self.frame_samples = int((self._sample_rate_hz * config.frame_ms) / 1000)
        if self.frame_samples <= 0:
            raise ValueError("frame_ms and sample_rate_hz produced 0 samples per frame")

        if self.use_fixture:
            return

        soundcard = load_soundcard()
        if soundcard is None:
            suffix = ""
            if _SOUNDCARD_IMPORT_ERROR is not None:
                suffix = f" ({type(_SOUNDCARD_IMPORT_ERROR).__name__}: {_SOUNDCARD_IMPORT_ERROR})"
            raise RuntimeError(f"soundcard is required for real Windows capture and could not be imported{suffix}")

        resolved_devices: dict[str, list[_WindowsCaptureDevice]] = {}
        if "mic" in self._active_stream_names:
            resolved_devices["mic"] = self.resolve_devices(
                soundcard,
                configured=config.mic_device,
                require_monitor=False,
                include_all=config.capture_all_mic_devices,
                allow_missing=config.allow_missing_sources,
            )
        if "speakers" in self._active_stream_names:
            resolved_devices["speakers"] = self.resolve_devices(
                soundcard,
                configured=config.speaker_device,
                require_monitor=True,
                include_all=config.capture_all_speaker_devices,
                allow_missing=config.allow_missing_sources,
            )

        self._active_stream_names = tuple(name for name in self._active_stream_names if resolved_devices.get(name))
        if not self._active_stream_names:
            raise RuntimeError("No capture devices resolved for active source mode.")

        opened_devices: dict[str, list[int]] = {}
        surviving_stream_names: list[str] = []
        for stream_name in self._active_stream_names:
            stream_keys: list[str] = []
            opened_stream_devices: list[int] = []
            last_open_error: Exception | None = None
            for device in resolved_devices[stream_name]:
                stream_key = f"{stream_name}:{device.index}"
                self._queues[stream_key] = queue.Queue(maxsize=self._queue_max_frames)
                try:
                    stream, device_sample_rate_hz, frame_samples, device_channels = self._open_stream_with_fallback(
                        stream_key=stream_key,
                        stream_group=stream_name,
                        device=device,
                        requested_sample_rate_hz=config.sample_rate_hz,
                        requested_channels=config.channels,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._queues.pop(stream_key, None)
                    last_open_error = exc if isinstance(exc, Exception) else Exception(str(exc))
                    LOGGER.warning(
                        "Skipping %s capture device %r because it could not be opened: %s",
                        stream_name,
                        device.index,
                        exc,
                    )
                    continue
                self._device_sample_rates_hz[stream_key] = device_sample_rate_hz
                self._device_frame_samples[stream_key] = frame_samples
                self._device_channels[stream_key] = device_channels
                self._streams[stream_key] = stream
                stream_keys.append(stream_key)
                opened_stream_devices.append(device.index)
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

    def resolve_devices(
        self,
        soundcard: Any,
        configured: str | int | None,
        *,
        require_monitor: bool,
        include_all: bool,
        allow_missing: bool,
    ) -> list[_WindowsCaptureDevice]:
        """Resolve Windows SoundCard devices for microphone or loopback capture."""
        catalog = self._device_catalog(soundcard)
        candidates = [device for device in catalog if device.is_loopback is require_monitor]

        if configured is not None:
            return [self._resolve_configured_device(soundcard, catalog=catalog, configured=configured, require_monitor=require_monitor)]

        if candidates:
            if include_all:
                return candidates
            return [candidates[0]]

        if allow_missing:
            if require_monitor:
                LOGGER.warning("No speaker playback/loopback device found; continuing without speaker capture.")
            else:
                LOGGER.warning("No microphone input device found; continuing without mic capture.")
            return []

        if require_monitor:
            raise RuntimeError(
                "No speaker playback/loopback input device found. Provide --speaker-device explicitly."
            )
        raise RuntimeError("No microphone input device found. Provide --mic-device explicitly.")

    def _open_stream_with_fallback(
        self,
        *,
        stream_key: str,
        stream_group: str,
        device: _WindowsCaptureDevice,
        requested_sample_rate_hz: int,
        requested_channels: int,
    ) -> tuple[_SoundCardRecorderStream, int, int, int]:
        """Open one SoundCard stream, retrying fallback rates when necessary."""
        last_error: Exception | None = None
        for candidate_channels in self._candidate_channel_counts(device=device, requested_channels=requested_channels):
            for candidate_rate_hz in self._candidate_sample_rates(requested_sample_rate_hz=requested_sample_rate_hz):
                frame_samples = int((candidate_rate_hz * self.config.frame_ms) / 1000) if self.config is not None else 0
                if frame_samples <= 0:
                    continue
                stream = _SoundCardRecorderStream(
                    stream_key=stream_key,
                    device_id=device.soundcard_id,
                    include_loopback=device.is_loopback,
                    sample_rate_hz=candidate_rate_hz,
                    frame_samples=frame_samples,
                    device_channels=candidate_channels,
                    callback=self._build_frame_callback(
                        stream_key=stream_key,
                        stream_group=stream_group,
                        device_sample_rate_hz=candidate_rate_hz,
                    ),
                )
                try:
                    stream.start()
                except Exception as exc:  # noqa: BLE001
                    stream.close()
                    last_error = self._translate_open_error(exc, stream_group=stream_group)
                    continue
                return stream, candidate_rate_hz, frame_samples, candidate_channels

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            "No supported sample rate found for selected input device(s). "
            "Try specifying a different --mic-device (or --speaker-device for both-mode capture)."
        )

    @staticmethod
    def _candidate_sample_rates(*, requested_sample_rate_hz: int) -> list[int]:
        """Return candidate Windows shared-mode capture rates in preference order."""
        candidates = [int(requested_sample_rate_hz), 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 8_000]
        deduped: list[int] = []
        seen: set[int] = set()
        for rate in candidates:
            if rate <= 0 or rate in seen:
                continue
            seen.add(rate)
            deduped.append(rate)
        return deduped

    @staticmethod
    def _candidate_channel_counts(*, device: _WindowsCaptureDevice, requested_channels: int) -> list[int]:
        """Return candidate channel counts, preferring the device's native layout."""
        max_channels = max(1, int(device.max_input_channels))
        clamped_requested = max(1, min(int(requested_channels), max_channels))
        candidates = [max_channels, clamped_requested]
        if max_channels >= 2:
            candidates.append(2)
        candidates.append(1)

        deduped: list[int] = []
        seen: set[int] = set()
        for channel_count in candidates:
            if channel_count <= 0 or channel_count > max_channels or channel_count in seen:
                continue
            seen.add(channel_count)
            deduped.append(channel_count)
        return deduped

    def _build_frame_callback(
        self,
        *,
        stream_key: str,
        stream_group: str,
        device_sample_rate_hz: int,
    ) -> Any:
        """Build the recorder callback that normalizes audio into backend frames."""

        def callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            del frames, time_info, status
            pcm16 = self._float32_audio_to_mono_pcm16(indata, stream_group=stream_group)
            frame_sample_rate_hz = device_sample_rate_hz
            if device_sample_rate_hz != self._sample_rate_hz and self.config is not None and self.config.channels == 1:
                pcm16 = resample_pcm16_mono_linear(
                    pcm16,
                    source_rate_hz=device_sample_rate_hz,
                    target_rate_hz=self._sample_rate_hz,
                )
                frame_sample_rate_hz = self._sample_rate_hz
            raw_frame = RawFrame(
                stream=stream_group,  # type: ignore[arg-type]
                mono_pcm16=pcm16,
                captured_at_monotonic_ns=time.monotonic_ns(),
                sample_rate_hz=frame_sample_rate_hz,
            )
            try:
                self._queues[stream_key].put_nowait(raw_frame)
            except queue.Full:
                self._dropped_callback_frames += 1

        return callback

    @staticmethod
    def _float32_audio_to_mono_pcm16(audio_block: Any, *, stream_group: str) -> bytes:
        """Convert a SoundCard float32 block into mono PCM16 bytes."""
        array = np.asarray(audio_block, dtype=np.float32)
        if array.size <= 0:
            return b""
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
        if array.ndim == 1:
            mono = array
        else:
            frame_matrix = np.atleast_2d(array)
            if frame_matrix.shape[1] <= 1:
                mono = frame_matrix[:, 0]
            elif stream_group == "mic":
                energy_by_channel = np.sum(np.square(frame_matrix), axis=0)
                selected_channel = int(np.argmax(energy_by_channel))
                mono = frame_matrix[:, selected_channel]
            else:
                mono = np.mean(frame_matrix, axis=1)
        pcm16 = np.clip(np.rint(mono * 32767.0), -32_768.0, 32_767.0).astype(np.int16, copy=False)
        return pcm16.tobytes()

    @staticmethod
    def _translate_open_error(exc: Exception, *, stream_group: str) -> Exception:
        """Map low-level SoundCard errors to user-actionable Windows messages."""
        message = str(exc)
        if "0x80070005" in message:
            if stream_group == "mic":
                return RuntimeError(
                    "Failed to open the Windows microphone via WASAPI (Error 0x80070005). "
                    "Enable Settings > Privacy & security > Microphone > Let desktop apps access your microphone, then retry."
                )
            return RuntimeError(
                "Failed to open the Windows speaker loopback via WASAPI (Error 0x80070005). "
                "Check Windows audio privacy settings and ensure the playback device is enabled, then retry."
            )
        return exc if isinstance(exc, Exception) else Exception(message)

    def _device_catalog(self, soundcard: Any) -> list[_WindowsCaptureDevice]:
        """Build a deterministic Windows WASAPI capture catalog."""
        microphones = list(soundcard.all_microphones(include_loopback=True))
        default_mic_id = self._default_device_id(soundcard.default_microphone)
        default_speaker_id = self._default_device_id(soundcard.default_speaker)

        def sort_key(device: Any) -> tuple[int, str, str]:
            device_id = str(getattr(device, "id", ""))
            name = str(getattr(device, "name", "")).strip().lower()
            if bool(getattr(device, "isloopback", False)):
                return (0 if device_id == default_speaker_id else 1, name, device_id)
            return (0 if device_id == default_mic_id else 1, name, device_id)

        non_loopback = sorted((device for device in microphones if not bool(getattr(device, "isloopback", False))), key=sort_key)
        loopback = sorted((device for device in microphones if bool(getattr(device, "isloopback", False))), key=sort_key)
        ordered_devices = [*non_loopback, *loopback]

        catalog: list[_WindowsCaptureDevice] = []
        for index, device in enumerate(ordered_devices):
            is_loopback = bool(getattr(device, "isloopback", False))
            base_name = str(getattr(device, "name", "")).strip()
            display_name = base_name if not is_loopback else f"{base_name} [Loopback]"
            catalog.append(
                _WindowsCaptureDevice(
                    index=index,
                    name=display_name,
                    soundcard_id=str(getattr(device, "id", "")),
                    max_input_channels=max(1, int(getattr(device, "channels", 1) or 1)),
                    default_samplerate=0.0,
                    hostapi_name=_WINDOWS_HOSTAPI_NAME,
                    is_loopback=is_loopback,
                )
            )
        return catalog

    @staticmethod
    def _default_device_id(default_getter: Any) -> str:
        """Resolve a default SoundCard device id when available."""
        try:
            device = default_getter()
        except Exception:  # noqa: BLE001
            return ""
        return str(getattr(device, "id", ""))

    def _resolve_configured_device(
        self,
        soundcard: Any,
        *,
        catalog: list[_WindowsCaptureDevice],
        configured: str | int,
        require_monitor: bool,
    ) -> _WindowsCaptureDevice:
        """Resolve an explicit Windows device reference by index, id, or name."""
        if isinstance(configured, int):
            entry = next((device for device in catalog if device.index == configured), None)
            if entry is None:
                raise RuntimeError(f"Unknown capture device index: {configured}")
            if entry.is_loopback is not require_monitor:
                if require_monitor:
                    raise RuntimeError("Configured speaker device is not a loopback capture endpoint.")
                raise RuntimeError("Configured microphone device is a loopback endpoint. Use --speaker-device for that device.")
            return entry

        configured_text = str(configured).strip()
        if not configured_text:
            raise RuntimeError("Device reference cannot be empty.")
        lowered = configured_text.lower()

        if lowered == "default":
            default_id = self._default_device_id(soundcard.default_speaker if require_monitor else soundcard.default_microphone)
            entry = next(
                (
                    device
                    for device in catalog
                    if device.soundcard_id == default_id and device.is_loopback is require_monitor
                ),
                None,
            )
            if entry is not None:
                return entry

        exact_matches = [
            device
            for device in catalog
            if device.is_loopback is require_monitor
            and lowered in {device.name.lower(), device.soundcard_id.lower()}
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            raise RuntimeError(f"Multiple capture devices matched {configured!r}; use the numeric index from `capture devices`.")

        partial_matches = [
            device
            for device in catalog
            if device.is_loopback is require_monitor
            and (lowered in device.name.lower() or lowered in device.soundcard_id.lower())
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]
        if len(partial_matches) > 1:
            raise RuntimeError(f"Multiple capture devices matched {configured!r}; use the numeric index from `capture devices`.")

        role_name = "speaker loopback" if require_monitor else "microphone"
        raise RuntimeError(f"No Windows {role_name} device matched {configured!r}.")
