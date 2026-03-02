from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcribe.bench.harness import (
    DEFAULT_HF_SAMPLE_LIMIT,
    DEFAULT_TRANSCRIPTION_MODEL,
    HF_DIARIZED_SCENARIO,
    _extract_audio_input,
    run_capture_sync_benchmark,
    run_hf_diarized_transcription_benchmark,
)
from transcribe.models import AudioSourceMode, CaptureConfig


def test_run_capture_sync_benchmark_fixture(tmp_path: Path) -> None:
    base_config = CaptureConfig(source_mode=AudioSourceMode.BOTH, session_id="bench", output_dir=tmp_path)
    result = run_capture_sync_benchmark(
        base_config=base_config,
        runs=2,
        duration_sec=0.2,
        output_dir=tmp_path / "bench",
        use_fixture=True,
    )

    assert result.json_path.exists()
    assert result.markdown_path.exists()

    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report["summary"]["run_count"] == 2
    assert len(report["runs"]) == 2


def test_run_hf_diarized_transcription_benchmark(tmp_path: Path) -> None:
    rows = [
        {
            "meeting_id": "EN2001a",
            "speaker_id": "A",
            "begin_time": 0.0,
            "end_time": 1.5,
            "text": "hello there",
        },
        {
            "meeting_id": "EN2001a",
            "speaker_id": "B",
            "begin_time": 1.5,
            "end_time": 3.5,
            "text": "general kenobi",
        },
    ]

    def fake_loader(
        dataset_id: str,
        dataset_config: str | None,
        split: str,
        sample_limit: int | None,
    ) -> list[dict[str, object]]:
        assert dataset_id == "edinburghcstr/ami"
        assert dataset_config == "ihm"
        assert split == "test"
        assert sample_limit == 2
        return rows

    def fake_transcriber(row: dict[str, object], model_id: str) -> tuple[str, float]:
        assert model_id == "faster-whisper-medium"
        if row["speaker_id"] == "A":
            return "hello there", 120.0
        return "general kenobi there", 80.0

    result = run_hf_diarized_transcription_benchmark(
        output_dir=tmp_path / "hf-bench",
        dataset_id="edinburghcstr/ami",
        dataset_config="ihm",
        split="test",
        sample_limit=2,
        transcription_model="faster-whisper-medium",
        rows_loader=fake_loader,
        transcriber=fake_transcriber,
    )

    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report["scenario"] == HF_DIARIZED_SCENARIO
    assert report["summary"]["run_count"] == 2
    assert report["summary"]["dataset_id"] == "edinburghcstr/ami"
    assert report["summary"]["dataset_config"] == "ihm"
    assert report["summary"]["dataset_split"] == "test"
    assert report["summary"]["transcription_model"] == "faster-whisper-medium"
    assert report["summary"]["max_model_ram_gb"] == pytest.approx(8.0)
    assert report["summary"]["total_segment_duration_sec"] == 3.5
    assert report["summary"]["total_inference_time_sec"] == pytest.approx(0.2)
    assert report["summary"]["avg_inference_latency_ms"] == pytest.approx(100.0)
    assert report["summary"]["inference_latency_ms_p50"] == pytest.approx(100.0)
    assert report["summary"]["inference_latency_ms_p95"] == pytest.approx(118.0)
    assert report["summary"]["inference_speed_x_realtime"] == pytest.approx(17.5)
    assert report["summary"]["avg_word_error_rate"] == pytest.approx(0.25)
    assert report["summary"]["median_word_error_rate"] == pytest.approx(0.25)
    assert report["summary"]["total_reference_words"] == 4
    assert report["summary"]["total_predicted_words"] == 5
    assert report["summary"]["unique_meeting_count"] == 1
    assert report["summary"]["unique_speaker_stream_count"] == 2
    assert all(run["transcription_model"] == "faster-whisper-medium" for run in report["runs"])
    assert report["runs"][0]["inference_latency_ms"] == pytest.approx(120.0)
    assert report["runs"][1]["word_error_rate"] == pytest.approx(0.5)


def test_run_hf_diarized_transcription_benchmark_defaults(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_loader(
        dataset_id: str,
        dataset_config: str | None,
        split: str,
        sample_limit: int | None,
    ) -> list[dict[str, object]]:
        observed["dataset_id"] = dataset_id
        observed["dataset_config"] = dataset_config
        observed["split"] = split
        observed["sample_limit"] = sample_limit
        return [{"meeting_id": "EN2001a", "speaker_id": "A", "begin_time": 0.0, "end_time": 1.0, "text": "hello"}]

    def fake_transcriber(row: dict[str, object], model_id: str) -> tuple[str, float]:
        _ = (row, model_id)
        return "hello", 12.5

    result = run_hf_diarized_transcription_benchmark(
        output_dir=tmp_path / "hf-bench",
        rows_loader=fake_loader,
        transcriber=fake_transcriber,
    )

    assert observed["dataset_id"] == "edinburghcstr/ami"
    assert observed["dataset_config"] == "ihm"
    assert observed["split"] == "test"
    assert observed["sample_limit"] == DEFAULT_HF_SAMPLE_LIMIT

    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report["summary"]["transcription_model"] == DEFAULT_TRANSCRIPTION_MODEL


def test_run_hf_diarized_transcription_benchmark_surfaces_offline_error(tmp_path: Path) -> None:
    def failing_loader(
        dataset_id: str,
        dataset_config: str | None,
        split: str,
        sample_limit: int | None,
    ) -> list[dict[str, object]]:
        _ = (dataset_id, dataset_config, split, sample_limit)
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    with pytest.raises(RuntimeError, match="offline policy"):
        run_hf_diarized_transcription_benchmark(
            output_dir=tmp_path / "hf-bench",
            rows_loader=failing_loader,
        )


def test_run_hf_diarized_transcription_benchmark_rejects_large_model() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        run_hf_diarized_transcription_benchmark(
            output_dir=Path("unused"),
            transcription_model="whisper-large-v3",
            rows_loader=lambda *_: [],
        )


def test_extract_audio_input_prefers_bytes_over_path() -> None:
    input_audio = _extract_audio_input({"audio": {"bytes": b"abc", "path": "missing.wav"}})
    assert hasattr(input_audio, "read")
    assert input_audio.read() == b"abc"
