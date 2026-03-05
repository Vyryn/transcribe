from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal


class AudioSourceMode(StrEnum):
    """Supported audio capture source modes."""

    MIC = "mic"
    SPEAKERS = "speakers"
    BOTH = "both"


class DataClassification(StrEnum):
    """Data sensitivity classes used in persisted metadata."""

    PHI = "phi"
    PERSONAL = "personal"
    OPERATIONAL = "operational"


@dataclass(slots=True)
class AppConfig:
    """Application-level configuration values."""

    offline_only: bool = True
    network_block_mode: str = "deny_outbound"
    log_level: str = "INFO"
    redact_logs: bool = True
    data_dir: Path = Path("data")


@dataclass(slots=True)
class CaptureConfig:
    """Audio capture runtime configuration."""

    sample_rate_hz: int = 16_000
    channels: int = 1
    frame_ms: int = 20
    source_mode: AudioSourceMode = AudioSourceMode.BOTH
    mic_device: str | int | None = None
    speaker_device: str | int | None = None
    capture_all_mic_devices: bool = False
    capture_all_speaker_devices: bool = False
    allow_missing_sources: bool = False
    session_id: str = "session"
    output_dir: Path = Path("data")


@dataclass(slots=True)
class CapturedFrame:
    """Single captured PCM frame with timing metadata."""

    stream: Literal["mic", "speakers"]
    frame_index: int
    mono_pcm16: bytes
    captured_at_monotonic_ns: int
    captured_at_utc: datetime


@dataclass(slots=True)
class SessionManifest:
    """Session metadata persisted alongside captured artifacts."""

    schema_version: str = "phase0-session-v1"
    session_id: str = "session"
    created_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_mode: AudioSourceMode = AudioSourceMode.BOTH
    sample_rate_hz: int = 16_000
    frame_ms: int = 20
    channels: int = 1
    contains_phi: bool = True
    data_classification: DataClassification = DataClassification.PHI
    artifacts: dict[str, str] = field(default_factory=dict)
    capture_stats: dict[str, int | float | str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialize manifest to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, object]
            Serialized manifest payload.
        """
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at_utc": self.created_at_utc.astimezone(timezone.utc).isoformat(),
            "source_mode": self.source_mode.value,
            "sample_rate_hz": self.sample_rate_hz,
            "frame_ms": self.frame_ms,
            "channels": self.channels,
            "contains_phi": self.contains_phi,
            "data_classification": self.data_classification.value,
            "artifacts": dict(self.artifacts),
            "capture_stats": dict(self.capture_stats),
            "notes": list(self.notes),
        }
