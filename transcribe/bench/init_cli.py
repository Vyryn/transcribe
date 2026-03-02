from __future__ import annotations

import argparse

from transcribe.bench.harness import (
    DEFAULT_HF_DIARIZED_CONFIG,
    DEFAULT_HF_DIARIZED_DATASET,
    DEFAULT_HF_DIARIZED_SPLIT,
    DEFAULT_HF_SAMPLE_LIMIT,
    DEFAULT_MAX_MODEL_RAM_GB,
    DEFAULT_TRANSCRIPTION_MODEL,
    initialize_benchmark_assets,
)


def build_parser() -> argparse.ArgumentParser:
    """Build parser for benchmark cache initialization."""
    parser = argparse.ArgumentParser(
        prog="init-bench",
        description="Download model and dataset cache for benchmark runs",
    )
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DIARIZED_DATASET, help="Hugging Face dataset id")
    parser.add_argument("--hf-config", default=DEFAULT_HF_DIARIZED_CONFIG, help="Hugging Face dataset config/subset")
    parser.add_argument("--hf-split", default=DEFAULT_HF_DIARIZED_SPLIT, help="Hugging Face split name")
    parser.add_argument(
        "--hf-limit",
        type=int,
        default=DEFAULT_HF_SAMPLE_LIMIT,
        help="Number of dataset rows to cache",
    )
    parser.add_argument(
        "--model",
        "--transcription-model",
        dest="transcription_model",
        default=DEFAULT_TRANSCRIPTION_MODEL,
        help="Transcription runtime/model identifier to cache",
    )
    parser.add_argument(
        "--max-model-ram-gb",
        type=float,
        default=DEFAULT_MAX_MODEL_RAM_GB,
        help="Reject models with estimated runtime RAM above this threshold",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for ``uv run init-bench``.

    This command intentionally does not install the runtime outbound network guard.
    It is for one-time cache priming only.
    """
    args = build_parser().parse_args(argv)
    try:
        result = initialize_benchmark_assets(
            dataset_id=args.hf_dataset,
            dataset_config=args.hf_config,
            split=args.hf_split,
            sample_limit=args.hf_limit,
            transcription_model=args.transcription_model,
            max_model_ram_gb=args.max_model_ram_gb,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"init-bench failed: {exc}")
        return 2

    print("Benchmark cache initialization complete.")
    print(
        "Dataset cache: "
        f"{result['dataset_id']} ({result['dataset_config']}) split={result['dataset_split']} "
        f"rows={result['dataset_rows_cached']} audio_rows={result['dataset_audio_rows_cached']}"
    )
    print(
        "Model cache: "
        f"{result['transcription_model']} normalized={result['normalized_model_id']} "
        f"source={result['model_cache_source']}"
    )
    if result.get("model_cache_dir"):
        print(f"Model cache directory: {result['model_cache_dir']}")
    return 0

