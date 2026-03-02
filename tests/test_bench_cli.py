from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import transcribe.bench.cli as bench_cli
from transcribe.bench.harness import (
    DEFAULT_HF_DIARIZED_CONFIG,
    DEFAULT_HF_DIARIZED_DATASET,
    DEFAULT_HF_DIARIZED_SPLIT,
    DEFAULT_HF_SAMPLE_LIMIT,
    DEFAULT_TRANSCRIPTION_MODEL,
    HF_DIARIZED_SCENARIO,
)


def test_bench_cli_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, argparse.Namespace] = {}
    observed: dict[str, str] = {}

    expected_out = Path("data/benchmarks/hf_diarized/faster-whisper-medium_20260302_010203")

    def fake_default_out(model_id: str) -> Path:
        observed["model_id"] = model_id
        return expected_out

    def fake_run_benchmark(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(bench_cli, "_default_output_dir", fake_default_out)
    monkeypatch.setattr(bench_cli, "run_benchmark", fake_run_benchmark)

    rc = bench_cli.main([])
    assert rc == 0

    args = captured["args"]
    assert args.scenario == HF_DIARIZED_SCENARIO
    assert args.hf_dataset == DEFAULT_HF_DIARIZED_DATASET
    assert args.hf_config == DEFAULT_HF_DIARIZED_CONFIG
    assert args.hf_split == DEFAULT_HF_DIARIZED_SPLIT
    assert args.hf_limit == DEFAULT_HF_SAMPLE_LIMIT
    assert args.transcription_model == DEFAULT_TRANSCRIPTION_MODEL
    assert observed["model_id"] == DEFAULT_TRANSCRIPTION_MODEL
    assert args.out == expected_out


def test_bench_cli_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, argparse.Namespace] = {}

    def fake_run_benchmark(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    def fail_default_out(model_id: str) -> Path:
        raise AssertionError(f"_default_output_dir should not be called when --out is set: {model_id}")

    monkeypatch.setattr(bench_cli, "_default_output_dir", fail_default_out)
    monkeypatch.setattr(bench_cli, "run_benchmark", fake_run_benchmark)

    rc = bench_cli.main(
        [
            "--hf-dataset",
            "custom/dataset",
            "--hf-config",
            "subset",
            "--hf-split",
            "eval",
            "--hf-limit",
            "7",
            "--model",
            "whisper-tiny",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0

    args = captured["args"]
    assert args.hf_dataset == "custom/dataset"
    assert args.hf_config == "subset"
    assert args.hf_split == "eval"
    assert args.hf_limit == 7
    assert args.transcription_model == "whisper-tiny"
    assert args.out == tmp_path


def test_bench_cli_runtime_error_returns_code_2(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_benchmark(args: argparse.Namespace) -> int:
        _ = args
        raise RuntimeError("offline")

    monkeypatch.setattr(bench_cli, "run_benchmark", fake_run_benchmark)
    rc = bench_cli.main([])
    assert rc == 2
