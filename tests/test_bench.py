from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import transcribe.bench.harness as bench_harness
from transcribe.bench.harness import (
    DEFAULT_HF_SAMPLE_LIMIT,
    DEFAULT_TRANSCRIPTION_MODEL,
    HF_DIARIZED_SCENARIO,
    _apply_parakeet_runtime_compatibility,
    _default_hf_segment_transcriber,
    _extract_audio_input,
    _extract_audio_path,
    _extract_qwen_audio_input,
    _prepare_nemo_model_for_inference,
    _is_nemo_speechlm_snapshot,
    _normalize_transcription_model_id,
    run_capture_sync_benchmark,
    run_hf_diarized_transcription_benchmark,
    transcribe_row_with_faster_whisper,
    transcribe_row_with_nemo_asr,
    transcribe_row_with_qwen_asr,
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


def test_normalize_transcription_model_id_supports_backend_aliases() -> None:
    assert _normalize_transcription_model_id("whisper-small") == "small"
    assert _normalize_transcription_model_id("faster-whisper-medium") == "medium"
    assert _normalize_transcription_model_id("parakeet-tdt-0.6b-v3") == "nvidia/parakeet-tdt-0.6b-v3"
    assert _normalize_transcription_model_id("nvidia/canary-1b") == "nvidia/canary-1b"
    assert _normalize_transcription_model_id("qwen/qwen3-asr-1.7b") == "Qwen/Qwen3-ASR-1.7B"
    assert _normalize_transcription_model_id("qwen/custom-asr") == "qwen/custom-asr"


def test_default_hf_segment_transcriber_routes_by_model_slug_prefix() -> None:
    assert _default_hf_segment_transcriber("whisper-small") is transcribe_row_with_faster_whisper
    assert _default_hf_segment_transcriber("nvidia/parakeet-tdt-0.6b-v3") is transcribe_row_with_nemo_asr
    assert _default_hf_segment_transcriber("nvidia/canary-1b") is transcribe_row_with_nemo_asr
    assert _default_hf_segment_transcriber("Qwen/Qwen3-ASR-1.7B") is transcribe_row_with_qwen_asr
    assert _default_hf_segment_transcriber("qwen/custom-asr") is transcribe_row_with_qwen_asr


def test_cache_transcription_model_uses_nvidia_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_snapshot(repo_id: str, *, local_files_only: bool) -> str:
        observed["repo_id"] = repo_id
        observed["local_files_only"] = local_files_only
        return "/tmp/fake-nvidia-model"

    def fake_get_nemo_model(model_id: str, *, local_files_only: bool) -> object:
        observed["model_id"] = model_id
        observed["loader_local_files_only"] = local_files_only
        return object()

    monkeypatch.setattr(bench_harness, "_get_hf_repo_snapshot", fake_snapshot)
    monkeypatch.setattr(bench_harness, "_get_nemo_asr_model", fake_get_nemo_model)
    result = bench_harness.cache_transcription_model(
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        max_model_ram_gb=16.0,
    )

    assert observed["repo_id"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert observed["local_files_only"] is False
    assert observed["model_id"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert observed["loader_local_files_only"] is False
    assert result["model_cache_source"] == "nemo_toolkit_asr"
    assert result["model_cache_dir"] == "/tmp/fake-nvidia-model"


def test_cache_transcription_model_uses_qwen_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_snapshot(repo_id: str, *, local_files_only: bool) -> str:
        observed["repo_id"] = repo_id
        observed["local_files_only"] = local_files_only
        return "/tmp/fake-qwen-model"

    def fake_get_qwen_model(model_id: str, *, local_files_only: bool) -> object:
        observed["model_id"] = model_id
        observed["loader_local_files_only"] = local_files_only
        return object()

    monkeypatch.setattr(bench_harness, "_get_hf_repo_snapshot", fake_snapshot)
    monkeypatch.setattr(bench_harness, "_get_qwen_asr_model", fake_get_qwen_model)
    result = bench_harness.cache_transcription_model(
        transcription_model="qwen/qwen3-asr-0.6b",
        max_model_ram_gb=16.0,
    )

    assert observed["repo_id"] == "qwen/qwen3-asr-0.6b"
    assert observed["local_files_only"] is False
    assert observed["model_id"] == "qwen/qwen3-asr-0.6b"
    assert observed["loader_local_files_only"] is False
    assert result["model_cache_source"] == "qwen_asr"
    assert result["model_cache_dir"] == "/tmp/fake-qwen-model"


def test_enforce_hf_offline_mode_sets_all_flags() -> None:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

    bench_harness._enforce_hf_offline_mode()

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_allow_hf_network_access_clears_all_offline_flags() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    bench_harness._allow_hf_network_access()

    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["HF_DATASETS_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_extract_audio_input_prefers_bytes_over_path() -> None:
    input_audio = _extract_audio_input({"audio": {"bytes": b"abc", "path": "missing.wav"}})
    assert hasattr(input_audio, "read")
    assert input_audio.read() == b"abc"


def test_extract_qwen_audio_input_prefers_bytes_over_path() -> None:
    input_audio = _extract_qwen_audio_input({"audio": {"bytes": b"abc", "path": "missing.wav"}})
    assert isinstance(input_audio, str)
    materialized = Path(input_audio)
    assert materialized.exists()
    assert materialized.read_bytes() == b"abc"


def test_extract_audio_path_prefers_materialized_bytes_when_path_missing() -> None:
    input_audio = _extract_audio_path({"audio": {"bytes": b"abc", "path": "missing.wav"}})
    materialized = Path(input_audio)
    assert materialized.exists()
    assert materialized.read_bytes() == b"abc"


def test_is_nemo_speechlm_snapshot_detects_expected_config(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(
        json.dumps({"pretrained_llm": "Qwen/Qwen3-1.7B", "perception": {"encoder": {}}}),
        encoding="utf-8",
    )
    assert _is_nemo_speechlm_snapshot(str(snapshot_dir)) is True


def test_apply_parakeet_runtime_compatibility_disables_cuda_graph_decoder() -> None:
    class FakeGreedyCfg:
        def __init__(self) -> None:
            self.use_cuda_graph_decoder = True

    class FakeDecodingCfg:
        def __init__(self) -> None:
            self.greedy = FakeGreedyCfg()

    class FakeCfg:
        def __init__(self) -> None:
            self.decoding = FakeDecodingCfg()

    class FakeModel:
        def __init__(self) -> None:
            self.cfg = FakeCfg()
            self.change_calls: list[tuple[object, bool]] = []

        def change_decoding_strategy(self, decoding_cfg: object, verbose: bool = True) -> None:
            self.change_calls.append((decoding_cfg, verbose))

    model = FakeModel()
    _apply_parakeet_runtime_compatibility(model, "nvidia/parakeet-tdt-0.6b-v3")

    assert model.cfg.decoding.greedy.use_cuda_graph_decoder is False
    assert len(model.change_calls) == 1
    assert model.change_calls[0][0] is model.cfg.decoding
    assert model.change_calls[0][1] is False


def test_apply_parakeet_runtime_compatibility_skips_non_parakeet_models() -> None:
    class FakeGreedyCfg:
        def __init__(self) -> None:
            self.use_cuda_graph_decoder = True

    class FakeDecodingCfg:
        def __init__(self) -> None:
            self.greedy = FakeGreedyCfg()

    class FakeCfg:
        def __init__(self) -> None:
            self.decoding = FakeDecodingCfg()

    class FakeModel:
        def __init__(self) -> None:
            self.cfg = FakeCfg()
            self.change_calls: list[tuple[object, bool]] = []

        def change_decoding_strategy(self, decoding_cfg: object, verbose: bool = True) -> None:
            self.change_calls.append((decoding_cfg, verbose))

    model = FakeModel()
    _apply_parakeet_runtime_compatibility(model, "nvidia/canary-qwen-2.5b")

    assert model.cfg.decoding.greedy.use_cuda_graph_decoder is True
    assert model.change_calls == []


def test_apply_parakeet_runtime_compatibility_supports_structured_omegaconf() -> None:
    from omegaconf import OmegaConf

    class FakeModel:
        def __init__(self) -> None:
            self.cfg = OmegaConf.create({"decoding": {"greedy": {"max_symbols": 10}}})
            OmegaConf.set_struct(self.cfg, True)
            self.change_calls: list[tuple[object, bool]] = []

        def change_decoding_strategy(self, decoding_cfg: object, verbose: bool = True) -> None:
            self.change_calls.append((decoding_cfg, verbose))

    model = FakeModel()
    _apply_parakeet_runtime_compatibility(model, "nvidia/parakeet-tdt-0.6b-v3")

    assert model.cfg.decoding.greedy.use_cuda_graph_decoder is False
    assert len(model.change_calls) == 1


def test_prepare_nemo_model_for_inference_sets_eval_and_disables_grad() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.eval_called = False
            self.grad_flag = True

        def eval(self) -> "FakeModel":
            self.eval_called = True
            return self

        def requires_grad_(self, flag: bool) -> "FakeModel":
            self.grad_flag = flag
            return self

    model = FakeModel()
    _prepare_nemo_model_for_inference(model, "nvidia/canary-qwen-2.5b")

    assert model.eval_called is True
    assert model.grad_flag is False
