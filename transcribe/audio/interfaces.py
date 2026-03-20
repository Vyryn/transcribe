from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from transcribe.models import CaptureConfig

StreamName = Literal["mic", "speakers"]


@dataclass(slots=True)
class RawFrame:
    """Raw mono PCM frame emitted by a capture backend."""

    stream: StreamName
    mono_pcm16: bytes
    captured_at_monotonic_ns: int
    sample_rate_hz: int


class AudioBackend(Protocol):
    """Protocol for audio capture backend implementations."""

    sample_rate_hz: int
    dropped_callback_frames: int

    def list_devices(self) -> list[dict[str, object]]: ...

    def open(self, config: CaptureConfig) -> None: ...

    def read_frames(self, timeout_ms: int = 500) -> dict[StreamName, RawFrame]: ...

    def close(self) -> None: ...

