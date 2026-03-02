from __future__ import annotations

import pytest

import transcribe.bench.init_cli as init_bench_cli
from transcribe.bench.harness import (
    DEFAULT_HF_DIARIZED_CONFIG,
    DEFAULT_HF_DIARIZED_DATASET,
    DEFAULT_HF_DIARIZED_SPLIT,
    DEFAULT_HF_SAMPLE_LIMIT,
    DEFAULT_MAX_MODEL_RAM_GB,
    DEFAULT_TRANSCRIPTION_MODEL,
)


def _fake_init_result() -> dict[str, object]:
    return {
        "dataset_id": DEFAULT_HF_DIARIZED_DATASET,
        "dataset_config": DEFAULT_HF_DIARIZED_CONFIG,
        "dataset_split": DEFAULT_HF_DIARIZED_SPLIT,
        "dataset_rows_cached": DEFAULT_HF_SAMPLE_LIMIT,
        "dataset_audio_rows_cached": DEFAULT_HF_SAMPLE_LIMIT,
        "transcription_model": DEFAULT_TRANSCRIPTION_MODEL,
        "normalized_model_id": "medium",
        "model_cache_source": "faster_whisper",
        "model_cache_dir": "/tmp/fake-model",
    }


def test_init_bench_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_initialize_benchmark_assets(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return _fake_init_result()

    monkeypatch.setattr(init_bench_cli, "initialize_benchmark_assets", fake_initialize_benchmark_assets)
    rc = init_bench_cli.main([])

    assert rc == 0
    assert captured["dataset_id"] == DEFAULT_HF_DIARIZED_DATASET
    assert captured["dataset_config"] == DEFAULT_HF_DIARIZED_CONFIG
    assert captured["split"] == DEFAULT_HF_DIARIZED_SPLIT
    assert captured["sample_limit"] == DEFAULT_HF_SAMPLE_LIMIT
    assert captured["transcription_model"] == DEFAULT_TRANSCRIPTION_MODEL
    assert captured["max_model_ram_gb"] == DEFAULT_MAX_MODEL_RAM_GB


def test_init_bench_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_initialize_benchmark_assets(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        result = _fake_init_result()
        result["dataset_id"] = str(kwargs["dataset_id"])
        result["dataset_config"] = str(kwargs["dataset_config"])
        result["dataset_split"] = str(kwargs["split"])
        result["transcription_model"] = str(kwargs["transcription_model"])
        return result

    monkeypatch.setattr(init_bench_cli, "initialize_benchmark_assets", fake_initialize_benchmark_assets)
    rc = init_bench_cli.main(
        [
            "--hf-dataset",
            "custom/ds",
            "--hf-config",
            "mycfg",
            "--hf-split",
            "eval",
            "--hf-limit",
            "12",
            "--model",
            "whisper-small",
            "--max-model-ram-gb",
            "7.5",
        ]
    )

    assert rc == 0
    assert captured["dataset_id"] == "custom/ds"
    assert captured["dataset_config"] == "mycfg"
    assert captured["split"] == "eval"
    assert captured["sample_limit"] == 12
    assert captured["transcription_model"] == "whisper-small"
    assert captured["max_model_ram_gb"] == 7.5


def test_init_bench_cli_error_returns_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_initialize_benchmark_assets(**kwargs: object) -> dict[str, object]:
        _ = kwargs
        raise RuntimeError("network failed")

    monkeypatch.setattr(init_bench_cli, "initialize_benchmark_assets", fake_initialize_benchmark_assets)
    rc = init_bench_cli.main([])
    assert rc == 2

