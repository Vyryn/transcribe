from __future__ import annotations

import io
import os
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transcribe.bench.report import build_benchmark_report, write_benchmark_report
from transcribe.models import CaptureConfig
from transcribe.utils.stats import percentile

DEFAULT_HF_DIARIZED_DATASET = "edinburghcstr/ami"
DEFAULT_HF_DIARIZED_CONFIG = "ihm"
DEFAULT_HF_DIARIZED_SPLIT = "test"
DEFAULT_HF_SAMPLE_LIMIT = 100
DEFAULT_TRANSCRIPTION_MODEL = "faster-whisper-medium"
DEFAULT_MAX_MODEL_RAM_GB = 8.0
HF_DIARIZED_SCENARIO = "hf_diarized_transcription"


class BenchmarkResult:
    """Container for benchmark report artifacts."""

    def __init__(self, report: dict[str, object], json_path: Path, markdown_path: Path) -> None:
        """Initialize benchmark result metadata."""
        self.report = report
        self.json_path = json_path
        self.markdown_path = markdown_path


HfDatasetRowsLoader = Callable[[str, str | None, str, int | None], Iterable[Mapping[str, object]]]
HfSegmentTranscriber = Callable[[Mapping[str, object], str], tuple[str, float]]

_FASTER_WHISPER_MODEL_CACHE: dict[str, Any] = {}
_MODEL_RAM_GB_ESTIMATES = {
    "tiny": 0.7,
    "base": 1.0,
    "small": 2.0,
    "medium": 5.5,
    "large-v1": 10.0,
    "large-v2": 10.0,
    "large-v3": 10.5,
    "large-v3-turbo": 9.0,
    "turbo": 9.0,
}


class HfOfflineDatasetUnavailableError(RuntimeError):
    """Raised when HF dataset access requires network under offline policy."""


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Iterate an exception and its cause/context chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None:
        ident = id(current)
        if ident in seen:
            break
        seen.add(ident)
        yield current
        current = current.__cause__ or current.__context__


def _is_hf_offline_network_error(exc: BaseException) -> bool:
    """Check whether an exception chain indicates offline-network HF access failure."""
    needles = (
        "outbound network blocked by offline policy",
        "cannot send a request, as the client has been closed",
        "couldn't reach",
        "couldn't find",
        "local cache",
        "offline mode",
        "offlinemodeisenabled",
        "local_files_only",
    )
    for item in _iter_exception_chain(exc):
        message = str(item).lower()
        if any(needle in message for needle in needles):
            return True
    return False


def _offline_hf_error(dataset_id: str) -> HfOfflineDatasetUnavailableError:
    """Create a user-friendly offline cache error for HF benchmark datasets."""
    return HfOfflineDatasetUnavailableError(
        "Hugging Face dataset access failed under offline policy. "
        f"Dataset {dataset_id!r} must be pre-populated in local cache before running this benchmark."
    )


def _allow_hf_network_access() -> None:
    """Ensure Hugging Face clients can use network for cache initialization flows."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"


def _normalize_transcription_model_id(transcription_model: str) -> str:
    """Normalize user-facing model identifiers to faster-whisper ids."""
    normalized = transcription_model.strip()
    for prefix in ("faster-whisper-", "whisper-"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def _estimate_model_ram_gb(transcription_model: str) -> float | None:
    """Return estimated runtime RAM/VRAM requirement for known model ids."""
    return _MODEL_RAM_GB_ESTIMATES.get(_normalize_transcription_model_id(transcription_model))


def _enforce_model_ram_limit(transcription_model: str, max_model_ram_gb: float) -> None:
    """Reject models estimated to exceed runtime memory policy."""
    estimated = _estimate_model_ram_gb(transcription_model)
    if estimated is None:
        return
    if estimated > max_model_ram_gb:
        raise ValueError(
            f"Model {transcription_model!r} is estimated at {estimated:.1f} GB and exceeds "
            f"the configured {max_model_ram_gb:.1f} GB limit. "
            "Choose a smaller model like whisper-small or whisper-medium."
        )


def _extract_audio_input(row: Mapping[str, object]) -> object:
    """Extract audio payload from dataset rows for local transcription."""
    audio = row.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, Mapping):
        payload = audio.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            return io.BytesIO(bytes(payload))
        array = audio.get("array")
        if array is not None:
            return array
        path = audio.get("path")
        if isinstance(path, str) and path:
            return path
    raise ValueError("Dataset row does not include a usable audio payload")


def _is_model_cache_miss_error(exc: BaseException) -> bool:
    """Check whether an exception indicates missing local model artifacts."""
    needles = (
        "local_files_only",
        "offline mode",
        "offlinemodeisenabled",
        "not found in local cache",
        "couldn't find",
        "cannot find",
        "no such file",
        "does not exist",
    )
    for item in _iter_exception_chain(exc):
        message = str(item).lower()
        if any(needle in message for needle in needles):
            return True
    return False


def _offline_model_error(model_id: str) -> RuntimeError:
    """Create a user-friendly offline cache error for transcription models."""
    return RuntimeError(
        f"Transcription model {model_id!r} is not available in local cache. "
        "Pre-populate it before running this benchmark."
    )


def _get_faster_whisper_model(model_id: str, *, local_files_only: bool = True) -> Any:
    """Return cached faster-whisper model instance for local benchmarking."""
    normalized = _normalize_transcription_model_id(model_id)
    cache_key = f"{normalized}|local_files_only={local_files_only}"
    if cache_key in _FASTER_WHISPER_MODEL_CACHE:
        return _FASTER_WHISPER_MODEL_CACHE[cache_key]
    if local_files_only and normalized in _FASTER_WHISPER_MODEL_CACHE:
        return _FASTER_WHISPER_MODEL_CACHE[normalized]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Transcription benchmarking requires `faster-whisper`. "
            "Install it before running this scenario."
        ) from exc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        model = WhisperModel(
            normalized,
            device="cpu",
            compute_type="int8",
            local_files_only=local_files_only,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_model_cache_miss_error(exc) or _is_hf_offline_network_error(exc):
            raise _offline_model_error(model_id) from exc
        raise

    _FASTER_WHISPER_MODEL_CACHE[cache_key] = model
    if local_files_only:
        _FASTER_WHISPER_MODEL_CACHE[normalized] = model
    return model


def transcribe_row_with_faster_whisper(row: Mapping[str, object], transcription_model: str) -> tuple[str, float]:
    """Transcribe one diarized row and return text plus inference latency."""
    model = _get_faster_whisper_model(transcription_model, local_files_only=True)
    audio_input = _extract_audio_input(row)

    started_at = time.perf_counter()
    segments, _ = model.transcribe(
        audio_input,
        language="en",
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
    )
    predicted_text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return predicted_text, elapsed_ms


def _word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Compute word error rate between reference and hypothesis text."""
    reference = [token for token in reference_text.lower().split() if token]
    hypothesis = [token for token in hypothesis_text.lower().split() if token]
    if not reference:
        return 0.0 if not hypothesis else 1.0

    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp: list[list[int]] = [[0] * cols for _ in range(rows)]
    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if reference[row - 1] == hypothesis[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + substitution_cost,
            )

    return float(dp[-1][-1]) / float(len(reference))


def _to_float(value: object, default: float = 0.0) -> float:
    """Convert a value into float with safe default fallback."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return float(stripped)
        except ValueError:
            return default
    return default


def _to_text(value: object, default: str = "") -> str:
    """Convert values into normalized display text."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return default
    return str(value).strip()


def _extract_text(row: Mapping[str, object]) -> str:
    """Extract transcript text from a diarized-transcription row."""
    for key in ("text", "transcript", "sentence"):
        if key in row:
            text = _to_text(row[key])
            if text:
                return text
    return ""


def _extract_meeting_id(row: Mapping[str, object], index: int) -> str:
    """Extract meeting/session identifier from a row."""
    for key in ("meeting_id", "session_id", "recording_id"):
        if key in row:
            value = _to_text(row[key])
            if value:
                return value
    return f"meeting-{index:05d}"


def _extract_speaker_id(row: Mapping[str, object], index: int) -> str:
    """Extract speaker label from a row."""
    for key in ("speaker_id", "speaker"):
        if key in row:
            value = _to_text(row[key])
            if value:
                return value
    if "speakers" in row and isinstance(row["speakers"], list):
        speakers = [entry for entry in row["speakers"] if _to_text(entry)]
        if speakers:
            return _to_text(speakers[0])
    return f"speaker-{index:05d}"


def _extract_segment_duration_sec(row: Mapping[str, object]) -> float:
    """Extract segment duration in seconds from diarized row timestamps."""
    for begin_key, end_key in (
        ("begin_time", "end_time"),
        ("start_time", "end_time"),
        ("timestamps_start", "timestamps_end"),
    ):
        begin = _to_float(row.get(begin_key))
        end = _to_float(row.get(end_key))
        if end > begin:
            return end - begin
    return 0.0


def load_hf_diarized_rows(
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    sample_limit: int | None,
) -> list[dict[str, object]]:
    """Load diarized transcription rows from Hugging Face datasets."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    try:
        from datasets import Audio, DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face dataset benchmarking requires the `datasets` package. "
            "Install it before running this scenario."
        ) from exc

    split_candidates = [split]
    if split == "test":
        split_candidates.append("eval")
    elif split == "eval":
        split_candidates.append("test")

    dataset: Any | None = None
    for split_name in split_candidates:
        split_expr = f"{split_name}[:{sample_limit}]" if sample_limit is not None else split_name
        try:
            dataset = load_dataset(
                path=dataset_id,
                name=dataset_config,
                split=split_expr,
                streaming=False,
                trust_remote_code=False,
                download_config=DownloadConfig(local_files_only=True),
            )
            break
        except ValueError:
            continue
        except Exception as exc:  # noqa: BLE001
            if _is_hf_offline_network_error(exc):
                raise _offline_hf_error(dataset_id) from exc
            raise

    if dataset is None:
        split_list = ", ".join(split_candidates)
        raise ValueError(
            f"Unable to load split for dataset {dataset_id!r}. Tried: {split_list}. "
            "Verify dataset id/config/split."
        )

    if hasattr(dataset, "features") and "audio" in dataset.features:
        dataset = dataset.cast_column("audio", Audio(decode=False))

    rows: list[dict[str, object]] = []
    for idx, row in enumerate(dataset, start=1):
        if isinstance(row, Mapping):
            rows.append(dict(row))
        else:
            rows.append({})
        if sample_limit is not None and idx >= sample_limit:
            break
    return rows


def cache_hf_diarized_dataset(
    *,
    dataset_id: str,
    dataset_config: str | None,
    split: str,
    sample_limit: int | None,
) -> dict[str, object]:
    """Download and cache a Hugging Face diarized dataset subset."""
    _allow_hf_network_access()

    try:
        from datasets import Audio, DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Dataset cache initialization requires the `datasets` package. "
            "Install it before running init-bench."
        ) from exc

    split_expr = f"{split}[:{sample_limit}]" if sample_limit is not None else split
    dataset = load_dataset(
        path=dataset_id,
        name=dataset_config,
        split=split_expr,
        streaming=False,
        trust_remote_code=False,
        download_config=DownloadConfig(local_files_only=False),
    )

    has_audio = hasattr(dataset, "features") and "audio" in dataset.features
    cached_audio_rows = 0
    if has_audio:
        dataset = dataset.cast_column("audio", Audio(decode=False))
        for row in dataset:
            audio = row.get("audio") if isinstance(row, Mapping) else None
            if isinstance(audio, Mapping):
                path = audio.get("path")
                payload = audio.get("bytes")
                if (isinstance(path, str) and path) or isinstance(payload, (bytes, bytearray)):
                    cached_audio_rows += 1
            elif isinstance(audio, str) and audio:
                cached_audio_rows += 1

    return {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config or "",
        "dataset_split": split,
        "dataset_rows_cached": len(dataset),
        "dataset_has_audio": has_audio,
        "dataset_audio_rows_cached": cached_audio_rows,
    }


def cache_transcription_model(
    *,
    transcription_model: str,
    max_model_ram_gb: float,
) -> dict[str, object]:
    """Download and cache a transcription model for local benchmarking."""
    _enforce_model_ram_limit(transcription_model, max_model_ram_gb)
    normalized_model_id = _normalize_transcription_model_id(transcription_model)
    _allow_hf_network_access()

    try:
        model = _get_faster_whisper_model(transcription_model, local_files_only=False)
        cache_source = "faster_whisper"
        cache_dir = ""
        if hasattr(model, "model_path"):
            path_value = getattr(model, "model_path")
            if isinstance(path_value, str):
                cache_dir = path_value
        if not cache_dir and hasattr(model, "model_dir"):
            path_value = getattr(model, "model_dir")
            if isinstance(path_value, str):
                cache_dir = path_value
    except RuntimeError as exc:
        if "faster-whisper" not in str(exc).lower():
            raise
        try:
            from huggingface_hub import snapshot_download
        except ImportError as import_exc:
            raise RuntimeError(
                "Model cache initialization requires either `faster-whisper` or `huggingface_hub`."
            ) from import_exc

        repo_id = f"Systran/faster-whisper-{normalized_model_id}"
        cache_dir = snapshot_download(repo_id=repo_id, local_files_only=False)
        cache_source = "huggingface_hub"

    return {
        "transcription_model": transcription_model,
        "normalized_model_id": normalized_model_id,
        "max_model_ram_gb": max_model_ram_gb,
        "model_cache_dir": cache_dir,
        "model_cache_source": cache_source,
    }


def initialize_benchmark_assets(
    *,
    dataset_id: str = DEFAULT_HF_DIARIZED_DATASET,
    dataset_config: str | None = DEFAULT_HF_DIARIZED_CONFIG,
    split: str = DEFAULT_HF_DIARIZED_SPLIT,
    sample_limit: int | None = DEFAULT_HF_SAMPLE_LIMIT,
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
    max_model_ram_gb: float = DEFAULT_MAX_MODEL_RAM_GB,
) -> dict[str, object]:
    """Initialize benchmark prerequisites by downloading model and dataset cache."""
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be >= 1 when provided")
    if max_model_ram_gb <= 0:
        raise ValueError("max_model_ram_gb must be > 0")
    if not transcription_model.strip():
        raise ValueError("transcription_model must be non-empty")
    _enforce_model_ram_limit(transcription_model, max_model_ram_gb)

    dataset_cache = cache_hf_diarized_dataset(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        split=split,
        sample_limit=sample_limit,
    )
    model_cache = cache_transcription_model(
        transcription_model=transcription_model,
        max_model_ram_gb=max_model_ram_gb,
    )
    return {
        **dataset_cache,
        **model_cache,
    }


def run_capture_sync_benchmark(
    *,
    base_config: CaptureConfig,
    runs: int,
    duration_sec: float,
    output_dir: Path,
    use_fixture: bool,
) -> BenchmarkResult:
    """Run repeated capture sessions and aggregate benchmark metrics.

    Parameters
    ----------
    base_config : CaptureConfig
        Base capture configuration.
    runs : int
        Number of benchmark runs.
    duration_sec : float
        Duration per run in seconds.
    output_dir : Path
        Directory to store per-run artifacts and aggregate reports.
    use_fixture : bool
        If ``True``, runs synthetic capture instead of live devices.

    Returns
    -------
    BenchmarkResult
        Report payload and output file paths.
    """
    from transcribe.audio.runner import run_capture_session, with_session_id

    if runs < 1:
        raise ValueError("runs must be >= 1")

    scenario = "capture_sync"
    run_results: list[dict[str, object]] = []

    benchmark_started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for run_index in range(1, runs + 1):
        run_id = f"{benchmark_started}-run{run_index:03d}"
        run_dir = output_dir / run_id
        run_config = with_session_id(base_config, session_id=run_id, output_dir=run_dir)
        result = run_capture_session(run_config, duration_sec=duration_sec, use_fixture=use_fixture)

        run_result = {
            "run_id": run_id,
            **result.manifest.capture_stats,
        }
        run_results.append(run_result)

    report = build_benchmark_report(scenario=scenario, run_results=run_results)
    json_path, markdown_path = write_benchmark_report(report, output_dir=output_dir)
    return BenchmarkResult(report=report, json_path=json_path, markdown_path=markdown_path)


def run_hf_diarized_transcription_benchmark(
    *,
    output_dir: Path,
    dataset_id: str = DEFAULT_HF_DIARIZED_DATASET,
    dataset_config: str | None = DEFAULT_HF_DIARIZED_CONFIG,
    split: str = DEFAULT_HF_DIARIZED_SPLIT,
    sample_limit: int | None = DEFAULT_HF_SAMPLE_LIMIT,
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL,
    max_model_ram_gb: float = DEFAULT_MAX_MODEL_RAM_GB,
    rows_loader: HfDatasetRowsLoader | None = None,
    transcriber: HfSegmentTranscriber | None = None,
) -> BenchmarkResult:
    """Run diarized-transcription benchmark with real model inference."""
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be >= 1 when provided")
    if not transcription_model.strip():
        raise ValueError("transcription_model must be non-empty")
    if max_model_ram_gb <= 0:
        raise ValueError("max_model_ram_gb must be > 0")
    _enforce_model_ram_limit(transcription_model, max_model_ram_gb)

    loader = rows_loader or load_hf_diarized_rows
    try:
        rows = list(loader(dataset_id, dataset_config, split, sample_limit))
    except HfOfflineDatasetUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_hf_offline_network_error(exc):
            raise _offline_hf_error(dataset_id) from exc
        raise
    if not rows:
        raise ValueError("No dataset rows were loaded for benchmark run")

    benchmark_started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_results: list[dict[str, object]] = []
    segment_transcriber = transcriber or transcribe_row_with_faster_whisper

    total_reference_words = 0
    total_reference_chars = 0
    total_predicted_words = 0
    total_duration_sec = 0.0
    total_inference_ms = 0.0
    meeting_ids: set[str] = set()
    speaker_streams: set[tuple[str, str]] = set()
    latency_values_ms: list[float] = []
    wer_values: list[float] = []
    rtf_values: list[float] = []

    for index, row in enumerate(rows, start=1):
        reference_text = _extract_text(row)
        meeting_id = _extract_meeting_id(row, index)
        speaker_id = _extract_speaker_id(row, index)
        duration_sec = _extract_segment_duration_sec(row)
        reference_word_count = len([token for token in reference_text.split() if token])
        reference_char_count = len(reference_text)

        predicted_text, inference_latency_ms = segment_transcriber(row, transcription_model)
        predicted_word_count = len([token for token in predicted_text.split() if token])
        word_error_rate = _word_error_rate(reference_text, predicted_text)
        real_time_factor = (inference_latency_ms / 1000.0) / duration_sec if duration_sec > 0 else 0.0

        total_reference_words += reference_word_count
        total_reference_chars += reference_char_count
        total_predicted_words += predicted_word_count
        total_duration_sec += duration_sec
        total_inference_ms += inference_latency_ms
        meeting_ids.add(meeting_id)
        speaker_streams.add((meeting_id, speaker_id))
        latency_values_ms.append(inference_latency_ms)
        wer_values.append(word_error_rate)
        rtf_values.append(real_time_factor)

        run_results.append(
            {
                "run_id": f"{benchmark_started}-seg{index:05d}",
                "meeting_id": meeting_id,
                "speaker_id": speaker_id,
                "segment_duration_sec": duration_sec,
                "reference_word_count": reference_word_count,
                "reference_char_count": reference_char_count,
                "predicted_word_count": predicted_word_count,
                "inference_latency_ms": inference_latency_ms,
                "real_time_factor": real_time_factor,
                "word_error_rate": word_error_rate,
                "transcription_model": transcription_model,
            }
        )

    row_count = len(run_results)
    total_inference_sec = total_inference_ms / 1000.0
    summary = {
        "run_count": row_count,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config or "",
        "dataset_split": split,
        "transcription_model": transcription_model,
        "max_model_ram_gb": max_model_ram_gb,
        "total_segment_duration_sec": total_duration_sec,
        "avg_segment_duration_sec": total_duration_sec / row_count,
        "total_inference_time_sec": total_inference_sec,
        "avg_inference_latency_ms": total_inference_ms / row_count,
        "inference_latency_ms_p50": percentile(latency_values_ms, 0.5),
        "inference_latency_ms_p95": percentile(latency_values_ms, 0.95),
        "inference_speed_x_realtime": (total_duration_sec / total_inference_sec) if total_inference_sec > 0 else 0.0,
        "avg_real_time_factor": sum(rtf_values) / row_count,
        "median_real_time_factor": percentile(rtf_values, 0.5),
        "avg_word_error_rate": sum(wer_values) / row_count,
        "median_word_error_rate": percentile(wer_values, 0.5),
        "total_reference_words": total_reference_words,
        "total_predicted_words": total_predicted_words,
        "avg_reference_words_per_segment": total_reference_words / row_count,
        "avg_reference_chars_per_segment": total_reference_chars / row_count,
        "unique_meeting_count": len(meeting_ids),
        "unique_speaker_stream_count": len(speaker_streams),
    }
    report = build_benchmark_report(
        scenario=HF_DIARIZED_SCENARIO,
        run_results=run_results,
        summary=summary,
    )
    json_path, markdown_path = write_benchmark_report(report, output_dir=output_dir)
    return BenchmarkResult(report=report, json_path=json_path, markdown_path=markdown_path)
