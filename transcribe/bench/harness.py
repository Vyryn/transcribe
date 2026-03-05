from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from transcribe.bench.report import build_benchmark_report, write_benchmark_report
from transcribe.models import CaptureConfig
from transcribe.utils.stats import percentile

LOGGER = logging.getLogger("transcribe.bench.harness")

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
TranscriptionBackend = Literal["faster_whisper", "nemo_asr", "qwen_asr"]

_FASTER_WHISPER_MODEL_CACHE: dict[str, Any] = {}
_NEMO_ASR_MODEL_CACHE: dict[str, Any] = {}
_QWEN_ASR_MODEL_CACHE: dict[str, Any] = {}
_HF_REPO_SNAPSHOT_CACHE: dict[str, str] = {}
_QWEN_AUDIO_BYTES_PATH_CACHE: dict[str, str] = {}
_NEMO_AUDIO_BYTES_PATH_CACHE: dict[str, str] = {}
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
    os.environ["TRANSFORMERS_OFFLINE"] = "0"


def _enforce_hf_offline_mode() -> None:
    """Force offline mode for Hugging Face and Transformers clients."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _canonical_transcription_model_id(transcription_model: str) -> str:
    """Map known transcription model aliases to canonical ids."""
    normalized = transcription_model.strip()
    if not normalized:
        return normalized
    aliases = {
        "parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        "qwen/qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    }
    return aliases.get(normalized.lower(), normalized)


def _is_parakeet_model_id(transcription_model: str) -> bool:
    """Return True when model id refers to NVIDIA Parakeet family."""
    return _canonical_transcription_model_id(transcription_model).lower().startswith("nvidia/parakeet-")


def _apply_parakeet_runtime_compatibility(model: Any, resolved_model_id: str) -> None:
    """Disable NeMo CUDA-graph decoder for Parakeet models in this runtime.

    Some NeMo runtime combinations can fail in TDT decoder paths when CUDA-graph
    decoding is enabled. For Parakeet models, force the safer non-CUDA-graph path.
    """
    if not _is_parakeet_model_id(resolved_model_id):
        return

    cfg = getattr(model, "cfg", None)
    if cfg is None:
        return
    decoding_cfg = getattr(cfg, "decoding", None)
    if decoding_cfg is None:
        return
    greedy_cfg = getattr(decoding_cfg, "greedy", None)
    if greedy_cfg is None:
        return

    current_value = None
    if hasattr(greedy_cfg, "get"):
        try:
            current_value = greedy_cfg.get("use_cuda_graph_decoder", None)
        except Exception:  # noqa: BLE001
            current_value = None
    if current_value is None:
        current_value = getattr(greedy_cfg, "use_cuda_graph_decoder", None)
    if current_value is False:
        return

    updated = False
    try:
        setattr(greedy_cfg, "use_cuda_graph_decoder", False)
        updated = True
    except Exception:  # noqa: BLE001
        updated = False
    if not updated:
        try:
            greedy_cfg["use_cuda_graph_decoder"] = False
            updated = True
        except Exception:  # noqa: BLE001
            updated = False
    if not updated:
        try:
            from omegaconf import open_dict

            with open_dict(greedy_cfg):
                greedy_cfg["use_cuda_graph_decoder"] = False
            updated = True
        except Exception:  # noqa: BLE001
            updated = False
    if not updated:
        LOGGER.warning(
            "Unable to disable CUDA graph decoder for Parakeet model %s; "
            "continuing with default decoding configuration.",
            resolved_model_id,
        )
        return

    if hasattr(model, "change_decoding_strategy"):
        try:
            model.change_decoding_strategy(decoding_cfg, verbose=False)
        except TypeError:
            model.change_decoding_strategy(decoding_cfg)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Disabled CUDA graph decoder for %s, but failed to refresh decoding strategy: %s",
                resolved_model_id,
                exc,
            )
            return

    LOGGER.info(
        "Disabled NeMo CUDA graph decoder for Parakeet model %s for runtime stability.",
        resolved_model_id,
    )


def _prepare_nemo_model_for_inference(model: Any, resolved_model_id: str) -> None:
    """Ensure loaded NeMo model is configured for inference-time execution."""
    if hasattr(model, "eval"):
        try:
            model.eval()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to set NeMo model %s to eval mode: %s", resolved_model_id, exc)
    if hasattr(model, "requires_grad_"):
        try:
            model.requires_grad_(False)
        except Exception:  # noqa: BLE001
            # Best effort only; not all wrappers expose requires_grad_ safely.
            pass


def _resolve_transcription_backend(transcription_model: str) -> tuple[TranscriptionBackend, str]:
    """Resolve backend type and normalized model id."""
    canonical = _canonical_transcription_model_id(transcription_model)
    canonical_lower = canonical.lower()
    if canonical_lower.startswith("nvidia/"):
        return "nemo_asr", canonical
    if canonical_lower.startswith("qwen/"):
        return "qwen_asr", canonical

    normalized = canonical
    for prefix in ("faster-whisper-", "whisper-"):
        if canonical_lower.startswith(prefix):
            normalized = canonical[len(prefix) :]
            break
    return "faster_whisper", normalized


def _normalize_transcription_model_id(transcription_model: str) -> str:
    """Normalize user-facing model identifiers for cache/report metadata."""
    _, normalized = _resolve_transcription_backend(transcription_model)
    return normalized


def _estimate_model_ram_gb(transcription_model: str) -> float | None:
    """Return estimated runtime RAM/VRAM requirement for known model ids."""
    normalized = _normalize_transcription_model_id(transcription_model).lower()
    return _MODEL_RAM_GB_ESTIMATES.get(normalized)


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


def _get_hf_repo_snapshot(repo_id: str, *, local_files_only: bool) -> str:
    """Resolve local snapshot path for a Hugging Face repo id."""
    cache_key = f"{repo_id}|local_files_only={local_files_only}"
    if cache_key in _HF_REPO_SNAPSHOT_CACHE:
        return _HF_REPO_SNAPSHOT_CACHE[cache_key]
    if local_files_only and repo_id in _HF_REPO_SNAPSHOT_CACHE:
        return _HF_REPO_SNAPSHOT_CACHE[repo_id]

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Transcription benchmarking model caching requires `huggingface_hub`."
        ) from exc

    if local_files_only:
        _enforce_hf_offline_mode()
    try:
        snapshot_dir = snapshot_download(repo_id=repo_id, local_files_only=local_files_only)
    except Exception as exc:  # noqa: BLE001
        if _is_model_cache_miss_error(exc) or _is_hf_offline_network_error(exc):
            raise _offline_model_error(repo_id) from exc
        raise

    _HF_REPO_SNAPSHOT_CACHE[cache_key] = snapshot_dir
    if local_files_only:
        _HF_REPO_SNAPSHOT_CACHE[repo_id] = snapshot_dir
    return snapshot_dir


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
            "Transcription benchmarking requires `faster-whisper`. Install it before running this scenario."
        ) from exc

    if local_files_only:
        _enforce_hf_offline_mode()
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


def _load_nemo_model_from_snapshot(nemo_asr: Any, resolved_model_id: str, snapshot_dir: str) -> Any:
    """Load a NeMo model from a locally cached HF snapshot directory."""
    snapshot_path = Path(snapshot_dir)
    preferred_nemo_filename = f"{Path(resolved_model_id).name}.nemo"
    nemo_files = sorted(snapshot_path.glob("*.nemo"))

    restore_errors: list[BaseException] = []
    if nemo_files:
        ordered_files = sorted(nemo_files, key=lambda item: (item.name != preferred_nemo_filename, item.name))
        for nemo_file in ordered_files:
            try:
                return nemo_asr.models.ASRModel.restore_from(restore_path=str(nemo_file))
            except Exception as exc:  # noqa: BLE001
                restore_errors.append(exc)
                continue
    else:
        try:
            from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import NeMo SaveRestoreConnector required for local snapshot loading."
            ) from exc

        for class_name in ("ASRModel", "EncDecMultiTaskModel"):
            model_class = getattr(nemo_asr.models, class_name, None)
            if model_class is None:
                continue
            connector = SaveRestoreConnector()
            connector.model_extracted_dir = snapshot_dir
            try:
                return model_class.restore_from(restore_path=snapshot_dir, save_restore_connector=connector)
            except Exception as exc:  # noqa: BLE001
                restore_errors.append(exc)
                continue

    if restore_errors:
        raise RuntimeError(
            f"Failed to restore NeMo model {resolved_model_id!r} from local snapshot {snapshot_dir}: "
            f"{restore_errors[-1]}"
        ) from restore_errors[-1]
    raise RuntimeError(
        f"NeMo model snapshot {snapshot_dir!r} for {resolved_model_id!r} did not contain loadable artifacts."
    )


def _is_nemo_speechlm_snapshot(snapshot_dir: str) -> bool:
    """Check whether a snapshot directory looks like a NeMo SpeechLM checkpoint."""
    config_path = Path(snapshot_dir) / "config.json"
    if not config_path.exists():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and "pretrained_llm" in data and "perception" in data


def _load_nemo_speechlm_model_from_snapshot(
    resolved_model_id: str,
    snapshot_dir: str,
    *,
    local_files_only: bool,
) -> Any:
    """Load a NeMo SpeechLM (for example Canary-Qwen) model from snapshot or repo id."""
    if not _is_nemo_speechlm_snapshot(snapshot_dir):
        raise RuntimeError(
            f"Snapshot {snapshot_dir!r} for {resolved_model_id!r} does not look like a NeMo SpeechLM checkpoint."
        )

    try:
        from nemo.collections.speechlm2.models import SALM
    except ImportError as exc:
        raise RuntimeError(
            "Loading NVIDIA SpeechLM models requires `nemo_toolkit` with speechlm2 support."
        ) from exc

    load_errors: list[BaseException] = []
    load_sources = [snapshot_dir]
    if not local_files_only:
        load_sources.append(resolved_model_id)

    for load_source in load_sources:
        try:
            return SALM.from_pretrained(load_source, local_files_only=local_files_only)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(exc)
            continue

    raise RuntimeError(
        f"Failed to restore NeMo SpeechLM model {resolved_model_id!r} from snapshot {snapshot_dir!r}: "
        f"{load_errors[-1]}"
    ) from load_errors[-1]


def _get_nemo_asr_model(model_id: str, *, local_files_only: bool = True) -> Any:
    """Return cached NeMo ASR model instance for local benchmarking."""
    _, resolved_model_id = _resolve_transcription_backend(model_id)
    cache_key = f"{resolved_model_id}|local_files_only={local_files_only}"
    if cache_key in _NEMO_ASR_MODEL_CACHE:
        model = _NEMO_ASR_MODEL_CACHE[cache_key]
        _prepare_nemo_model_for_inference(model, resolved_model_id)
        return model
    if local_files_only and resolved_model_id in _NEMO_ASR_MODEL_CACHE:
        model = _NEMO_ASR_MODEL_CACHE[resolved_model_id]
        _prepare_nemo_model_for_inference(model, resolved_model_id)
        return model

    if local_files_only:
        _enforce_hf_offline_mode()

    snapshot_dir = _get_hf_repo_snapshot(resolved_model_id, local_files_only=local_files_only)
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError(
            "Transcription benchmarking for NVIDIA ASR models requires `nemo_toolkit[asr]`."
        ) from exc

    load_errors: list[BaseException] = []
    model = None
    try:
        model = _load_nemo_model_from_snapshot(nemo_asr, resolved_model_id, snapshot_dir)
    except Exception as exc:  # noqa: BLE001
        load_errors.append(exc)
    if model is None:
        try:
            model = _load_nemo_speechlm_model_from_snapshot(
                resolved_model_id,
                snapshot_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:  # noqa: BLE001
            load_errors.append(exc)

    if model is None and not local_files_only:
        try:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=resolved_model_id)
        except Exception as exc:  # noqa: BLE001
            load_errors.append(exc)
            model = None

    if model is None:
        for exc in load_errors:
            if _is_model_cache_miss_error(exc) or _is_hf_offline_network_error(exc):
                raise _offline_model_error(resolved_model_id) from exc
        raise RuntimeError(
            f"Failed to initialize NeMo ASR model for {resolved_model_id!r} from snapshot "
            f"{snapshot_dir!r}: {load_errors[-1]}"
        ) from load_errors[-1]

    _apply_parakeet_runtime_compatibility(model, resolved_model_id)
    _prepare_nemo_model_for_inference(model, resolved_model_id)

    _NEMO_ASR_MODEL_CACHE[cache_key] = model
    if local_files_only:
        _NEMO_ASR_MODEL_CACHE[resolved_model_id] = model
    return model


def _get_qwen_asr_model(model_id: str, *, local_files_only: bool = True) -> Any:
    """Return cached Qwen ASR model instance for local benchmarking."""
    _, resolved_model_id = _resolve_transcription_backend(model_id)
    cache_key = f"{resolved_model_id}|local_files_only={local_files_only}"
    if cache_key in _QWEN_ASR_MODEL_CACHE:
        return _QWEN_ASR_MODEL_CACHE[cache_key]
    if local_files_only and resolved_model_id in _QWEN_ASR_MODEL_CACHE:
        return _QWEN_ASR_MODEL_CACHE[resolved_model_id]

    if local_files_only:
        _enforce_hf_offline_mode()

    snapshot_dir = _get_hf_repo_snapshot(resolved_model_id, local_files_only=local_files_only)
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Transcription benchmarking for Qwen ASR models requires `qwen-asr`."
        ) from exc

    load_errors: list[BaseException] = []
    for load_source in (snapshot_dir, resolved_model_id):
        try:
            model = Qwen3ASRModel.from_pretrained(load_source)
            _QWEN_ASR_MODEL_CACHE[cache_key] = model
            if local_files_only:
                _QWEN_ASR_MODEL_CACHE[resolved_model_id] = model
            return model
        except Exception as exc:  # noqa: BLE001
            load_errors.append(exc)
            continue

    for exc in load_errors:
        if _is_model_cache_miss_error(exc) or _is_hf_offline_network_error(exc):
            raise _offline_model_error(resolved_model_id) from exc
    raise RuntimeError(
        f"Failed to initialize qwen-asr model for {resolved_model_id!r}: {load_errors[-1]}"
    ) from load_errors[-1]


def _extract_audio_path(row: Mapping[str, object]) -> str:
    """Extract local audio path from a dataset row."""
    audio = row.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, Mapping):
        path = audio.get("path")
        if isinstance(path, str) and path:
            path_obj = Path(path)
            if path_obj.exists():
                return path
        payload = audio.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            payload_bytes = bytes(payload)
            cache_key = hashlib.sha1(payload_bytes).hexdigest()
            cached_path = _NEMO_AUDIO_BYTES_PATH_CACHE.get(cache_key)
            if cached_path and Path(cached_path).exists():
                return cached_path

            suffix = ".wav"
            if isinstance(path, str) and path:
                path_suffix = Path(path).suffix
                if path_suffix:
                    suffix = path_suffix
            temp_dir = Path(tempfile.gettempdir()) / "transcribe-nemo-audio"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{cache_key}{suffix}"
            if not temp_path.exists():
                temp_path.write_bytes(payload_bytes)
            materialized_path = str(temp_path)
            _NEMO_AUDIO_BYTES_PATH_CACHE[cache_key] = materialized_path
            return materialized_path
        if isinstance(path, str) and path:
            return path
    raise ValueError("Dataset row does not include an audio file path")


def _extract_qwen_audio_input(row: Mapping[str, object]) -> object:
    """Extract qwen-asr compatible audio input payload from a dataset row."""
    audio = row.get("audio")
    if isinstance(audio, str) and audio:
        return audio
    if isinstance(audio, Mapping):
        path = audio.get("path")
        if isinstance(path, str) and path and Path(path).exists():
            return path
        payload = audio.get("bytes")
        if isinstance(payload, (bytes, bytearray)):
            payload_bytes = bytes(payload)
            cache_key = hashlib.sha1(payload_bytes).hexdigest()
            cached_path = _QWEN_AUDIO_BYTES_PATH_CACHE.get(cache_key)
            if cached_path and Path(cached_path).exists():
                return cached_path

            suffix = ".wav"
            if isinstance(path, str) and path:
                path_suffix = Path(path).suffix
                if path_suffix:
                    suffix = path_suffix
            temp_dir = Path(tempfile.gettempdir()) / "transcribe-qwen-audio"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f"{cache_key}{suffix}"
            if not temp_path.exists():
                temp_path.write_bytes(payload_bytes)
            materialized_path = str(temp_path)
            _QWEN_AUDIO_BYTES_PATH_CACHE[cache_key] = materialized_path
            return materialized_path
        if isinstance(path, str) and path:
            return path
        array = audio.get("array")
        if array is not None:
            sampling_rate = int(_to_float(audio.get("sampling_rate"), default=16_000))
            sampling_rate = sampling_rate if sampling_rate > 0 else 16_000
            return (array, sampling_rate)
    return _extract_audio_input(row)


def _transcription_item_to_text(item: object) -> str:
    """Extract transcript text from one backend response item."""
    text_attr = getattr(item, "text", None)
    if isinstance(text_attr, str):
        return text_attr.strip()
    if isinstance(item, Mapping):
        text = item.get("text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(item, str):
        return item.strip()
    return ""


def _transcription_output_to_text(output: object) -> str:
    """Normalize backend-specific transcription outputs into plain text."""
    if isinstance(output, (list, tuple)):
        return " ".join(piece for piece in (_transcription_item_to_text(item) for item in output) if piece).strip()
    return _transcription_item_to_text(output)


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


def transcribe_row_with_nemo_asr(row: Mapping[str, object], transcription_model: str) -> tuple[str, float]:
    """Transcribe one diarized row with NeMo ASR and return text plus latency."""
    model = _get_nemo_asr_model(transcription_model, local_files_only=True)
    audio_path = _extract_audio_path(row)

    started_at = time.perf_counter()
    if hasattr(model, "transcribe"):
        output = model.transcribe([audio_path])
        predicted_text = _transcription_output_to_text(output)
    elif hasattr(model, "generate"):
        audio_locator_tag = getattr(model, "audio_locator_tag", "<|audioplaceholder|>")
        prompt = f"Transcribe the following: {audio_locator_tag}"
        output_ids = model.generate(
            prompts=[[{"role": "user", "content": prompt, "audio": [audio_path]}]],
            max_new_tokens=256,
            do_sample=False,
        )
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "ids_to_text"):
            predicted_text = str(tokenizer.ids_to_text(output_ids[0].cpu())).strip()
        else:
            raise RuntimeError("Loaded NeMo model does not expose tokenizer.ids_to_text for transcription decoding.")
    else:
        raise RuntimeError(
            f"Loaded NeMo model for {transcription_model!r} has no supported inference API (expected transcribe or generate)."
        )
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return predicted_text, elapsed_ms


def transcribe_row_with_qwen_asr(row: Mapping[str, object], transcription_model: str) -> tuple[str, float]:
    """Transcribe one diarized row with qwen-asr and return text plus latency."""
    model = _get_qwen_asr_model(transcription_model, local_files_only=True)
    audio_input = _extract_qwen_audio_input(row)

    started_at = time.perf_counter()
    output = model.transcribe(audio=audio_input)
    predicted_text = _transcription_output_to_text(output)
    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    return predicted_text, elapsed_ms


def _default_hf_segment_transcriber(transcription_model: str) -> HfSegmentTranscriber:
    """Resolve the default segment transcriber for a model identifier."""
    backend, _ = _resolve_transcription_backend(transcription_model)
    if backend == "nemo_asr":
        return transcribe_row_with_nemo_asr
    if backend == "qwen_asr":
        return transcribe_row_with_qwen_asr
    return transcribe_row_with_faster_whisper


def preload_transcription_model(
    transcription_model: str,
    *,
    max_model_ram_gb: float = DEFAULT_MAX_MODEL_RAM_GB,
) -> None:
    """Load the requested transcription backend into cache without running inference."""
    _enforce_model_ram_limit(transcription_model, max_model_ram_gb)
    backend, resolved_model_id = _resolve_transcription_backend(transcription_model)

    if backend == "nemo_asr":
        _get_nemo_asr_model(resolved_model_id, local_files_only=True)
        return
    if backend == "qwen_asr":
        _get_qwen_asr_model(resolved_model_id, local_files_only=True)
        return
    _get_faster_whisper_model(resolved_model_id, local_files_only=True)


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
    _enforce_hf_offline_mode()

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
            f"Unable to load split for dataset {dataset_id!r}. Tried: {split_list}. Verify dataset id/config/split."
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
            "Dataset cache initialization requires the `datasets` package. Install it before running init-bench."
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
    backend, normalized_model_id = _resolve_transcription_backend(transcription_model)
    _allow_hf_network_access()

    cache_dir = ""
    if backend == "nemo_asr":
        cache_dir = _get_hf_repo_snapshot(normalized_model_id, local_files_only=False)
        try:
            _get_nemo_asr_model(transcription_model, local_files_only=False)
            cache_source = "nemo_toolkit_asr"
        except RuntimeError as exc:
            if "nemo_toolkit[asr]" not in str(exc).lower():
                raise
            cache_source = "huggingface_hub"
    elif backend == "qwen_asr":
        cache_dir = _get_hf_repo_snapshot(normalized_model_id, local_files_only=False)
        try:
            _get_qwen_asr_model(transcription_model, local_files_only=False)
            cache_source = "qwen_asr"
        except RuntimeError as exc:
            if "qwen-asr" not in str(exc).lower():
                raise
            cache_source = "huggingface_hub"
    else:
        try:
            model = _get_faster_whisper_model(transcription_model, local_files_only=False)
            cache_source = "faster_whisper"
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
            repo_id = f"Systran/faster-whisper-{normalized_model_id}"
            cache_dir = _get_hf_repo_snapshot(repo_id, local_files_only=False)
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
    segment_transcriber = transcriber or _default_hf_segment_transcriber(transcription_model)

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
