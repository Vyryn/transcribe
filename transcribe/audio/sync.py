from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from transcribe.audio.interfaces import RawFrame
from transcribe.models import CapturedFrame
from transcribe.utils.stats import percentile


@dataclass(slots=True)
class SyncAccumulator:
    """Accumulator for synchronization and latency metrics."""

    pair_count: int = 0
    dropped_pairs: int = 0
    total_drift_ns: int = 0
    max_drift_ns: int = 0
    total_callback_to_write_ns: int = 0
    callback_to_write_latency_samples_ms: list[float] = None

    def __post_init__(self) -> None:
        if self.callback_to_write_latency_samples_ms is None:
            self.callback_to_write_latency_samples_ms = []


def build_captured_pair(
    raw_frames: dict[str, RawFrame],
    frame_index: int,
) -> tuple[CapturedFrame, CapturedFrame, int]:
    """Convert raw stream frames into typed captured frames.

    Parameters
    ----------
    raw_frames : dict[str, RawFrame]
        Raw frames keyed by stream name.
    frame_index : int
        Monotonic frame index to assign to outputs.

    Returns
    -------
    tuple[CapturedFrame, CapturedFrame, int]
        Mic frame, speaker frame, and absolute drift in nanoseconds.
    """
    mic = raw_frames["mic"]
    speakers = raw_frames["speakers"]
    captured_at = datetime.now(timezone.utc)

    mic_frame = CapturedFrame(
        stream="mic",
        frame_index=frame_index,
        mono_pcm16=mic.mono_pcm16,
        captured_at_monotonic_ns=mic.captured_at_monotonic_ns,
        captured_at_utc=captured_at,
    )
    speakers_frame = CapturedFrame(
        stream="speakers",
        frame_index=frame_index,
        mono_pcm16=speakers.mono_pcm16,
        captured_at_monotonic_ns=speakers.captured_at_monotonic_ns,
        captured_at_utc=captured_at,
    )
    drift_ns = abs(mic.captured_at_monotonic_ns - speakers.captured_at_monotonic_ns)
    return mic_frame, speakers_frame, drift_ns


def record_pair_stats(
    accumulator: SyncAccumulator,
    *,
    drift_ns: int,
    callback_to_write_latency_ns: int,
) -> None:
    """Record drift and callback-to-write latency for one frame pair."""
    accumulator.pair_count += 1
    accumulator.total_drift_ns += drift_ns
    accumulator.max_drift_ns = max(accumulator.max_drift_ns, drift_ns)
    accumulator.total_callback_to_write_ns += callback_to_write_latency_ns
    accumulator.callback_to_write_latency_samples_ms.append(callback_to_write_latency_ns / 1_000_000)


def summarize_stats(accumulator: SyncAccumulator) -> dict[str, int | float]:
    """Summarize accumulated synchronization metrics.

    Parameters
    ----------
    accumulator : SyncAccumulator
        Statistics accumulator.

    Returns
    -------
    dict[str, int | float]
        Aggregate metrics for reporting.
    """
    pair_count = accumulator.pair_count
    avg_drift = accumulator.total_drift_ns / pair_count if pair_count else 0
    avg_latency_ms = (accumulator.total_callback_to_write_ns / pair_count) / 1_000_000 if pair_count else 0
    return {
        "pair_count": pair_count,
        "dropped_pairs": accumulator.dropped_pairs,
        "drift_ns_avg": avg_drift,
        "drift_ns_max": accumulator.max_drift_ns,
        "callback_to_write_latency_ms_avg": avg_latency_ms,
        "callback_to_write_latency_ms_p50": percentile(
            accumulator.callback_to_write_latency_samples_ms,
            0.5,
        ),
        "callback_to_write_latency_ms_p95": percentile(
            accumulator.callback_to_write_latency_samples_ms,
            0.95,
        ),
    }
