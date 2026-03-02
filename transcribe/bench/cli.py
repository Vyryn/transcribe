from __future__ import annotations

import argparse
from pathlib import Path

from transcribe.bench.harness import (
    DEFAULT_HF_DIARIZED_CONFIG,
    DEFAULT_HF_DIARIZED_DATASET,
    DEFAULT_HF_DIARIZED_SPLIT,
    DEFAULT_MAX_MODEL_RAM_GB,
    DEFAULT_HF_SAMPLE_LIMIT,
    DEFAULT_TRANSCRIPTION_MODEL,
    HF_DIARIZED_SCENARIO,
)
from transcribe.cli import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    """Build parser for the default Hugging Face diarized benchmark runner."""
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Run Hugging Face diarized transcription benchmark",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config file path")
    parser.add_argument("--log-level", default=None, help="Override logging level")
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/hf_diarized"), help="Output directory")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DIARIZED_DATASET, help="Hugging Face dataset id")
    parser.add_argument("--hf-config", default=DEFAULT_HF_DIARIZED_CONFIG, help="Hugging Face dataset config/subset")
    parser.add_argument("--hf-split", default=DEFAULT_HF_DIARIZED_SPLIT, help="Hugging Face split name")
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=DEFAULT_HF_SAMPLE_LIMIT,
        help="Row limit for Hugging Face benchmark runs",
    )
    parser.add_argument(
        "--model",
        "--transcription-model",
        dest="transcription_model",
        default=DEFAULT_TRANSCRIPTION_MODEL,
        help="Transcription runtime/model identifier under test",
    )
    parser.add_argument(
        "--max-model-ram-gb",
        type=float,
        default=DEFAULT_MAX_MODEL_RAM_GB,
        help="Reject models with estimated runtime RAM above this threshold",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``uv run bench`` convenience command."""
    args = build_parser().parse_args(argv)
    benchmark_args = argparse.Namespace(
        config=args.config,
        log_level=args.log_level,
        scenario=HF_DIARIZED_SCENARIO,
        runs=1,
        duration_sec=0.0,
        out=args.out,
        real_devices=False,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_limit=args.hf_limit,
        transcription_model=args.transcription_model,
        max_model_ram_gb=args.max_model_ram_gb,
    )
    try:
        return run_benchmark(benchmark_args)
    except Exception as exc:  # noqa: BLE001
        print(f"Benchmark failed: {exc}")
        return 2
