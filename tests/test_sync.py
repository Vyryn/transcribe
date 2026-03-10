from __future__ import annotations

from transcribe.audio.interfaces import RawFrame
from transcribe.audio.sync import SyncAccumulator, build_captured_pair, record_pair_stats, summarize_stats


def test_build_captured_pair_and_stats() -> None:
    raw = {
        "mic": RawFrame(stream="mic", mono_pcm16=b"\\x00\\x00", captured_at_monotonic_ns=10, sample_rate_hz=16_000),
        "speakers": RawFrame(stream="speakers", mono_pcm16=b"\\x00\\x00", captured_at_monotonic_ns=30, sample_rate_hz=16_000),
    }
    mic_frame, speakers_frame, drift_ns = build_captured_pair(raw, frame_index=2)

    assert mic_frame.frame_index == 2
    assert speakers_frame.frame_index == 2
    assert drift_ns == 20

    acc = SyncAccumulator()
    record_pair_stats(acc, drift_ns=20, callback_to_write_latency_ns=2_000_000)
    summary = summarize_stats(acc)
    assert summary["pair_count"] == 1
    assert summary["drift_ns_max"] == 20
    assert summary["callback_to_write_latency_ms_p50"] == 2.0

