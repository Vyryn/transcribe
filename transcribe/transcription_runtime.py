from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from transcribe.runtime_env import RuntimeMode, resolve_app_runtime_paths, resolve_bundled_transcription_model_path

LOGGER = logging.getLogger("transcribe.transcription_runtime")

DEFAULT_MAX_MODEL_RAM_GB = 8.0
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


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Check whether an exception chain indicates CUDA memory exhaustion."""
    needles = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cudnn_status_alloc_failed",
        "hip out of memory",
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
    runtime_paths = resolve_app_runtime_paths()
    if runtime_paths.mode == RuntimeMode.PACKAGED and model_id in runtime_paths.transcription_models:
        return RuntimeError(
            f"Packaged ASR model {model_id!r} is not installed. "
            f"Run `transcribe models install --model {model_id}` and retry."
        )
    return RuntimeError(
        f"Transcription model {model_id!r} is not available in local cache. "
        "Pre-populate it before retrying."
    )


def _resolve_packaged_snapshot_path(repo_id: str) -> str | None:
    runtime_paths = resolve_app_runtime_paths()
    if runtime_paths.mode != RuntimeMode.PACKAGED:
        return None
    if repo_id not in runtime_paths.transcription_models:
        return None
    try:
        return str(resolve_bundled_transcription_model_path(repo_id, runtime_paths=runtime_paths))
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def _get_hf_repo_snapshot(repo_id: str, *, local_files_only: bool) -> str:
    """Resolve local snapshot path for a Hugging Face repo id."""
    cache_key = f"{repo_id}|local_files_only={local_files_only}"
    if cache_key in _HF_REPO_SNAPSHOT_CACHE:
        return _HF_REPO_SNAPSHOT_CACHE[cache_key]
    if local_files_only and repo_id in _HF_REPO_SNAPSHOT_CACHE:
        return _HF_REPO_SNAPSHOT_CACHE[repo_id]

    packaged_snapshot = _resolve_packaged_snapshot_path(repo_id)
    if packaged_snapshot is not None:
        _HF_REPO_SNAPSHOT_CACHE[cache_key] = packaged_snapshot
        _HF_REPO_SNAPSHOT_CACHE[repo_id] = packaged_snapshot
        return packaged_snapshot

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Transcription model caching requires `huggingface_hub`."
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
            "Transcription runtime requires `faster-whisper`. Install it before retrying."
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

    def _restore_with_optional_cpu_fallback(model_loader: Callable[..., Any], **kwargs: Any) -> Any:
        try:
            return model_loader(**kwargs)
        except Exception as exc:  # noqa: BLE001
            restore_errors.append(exc)
            if not _is_cuda_oom_error(exc):
                raise

            try:
                import torch
            except ImportError:
                raise

            LOGGER.warning(
                "CUDA memory exhausted while restoring %s; retrying on CPU.",
                resolved_model_id,
            )
            try:
                return model_loader(**kwargs, map_location=torch.device("cpu"))
            except Exception as cpu_exc:  # noqa: BLE001
                restore_errors.append(cpu_exc)
                raise

    snapshot_path = Path(snapshot_dir)
    preferred_nemo_filename = f"{Path(resolved_model_id).name}.nemo"
    nemo_files = sorted(snapshot_path.glob("*.nemo"))

    restore_errors: list[BaseException] = []
    if nemo_files:
        ordered_files = sorted(nemo_files, key=lambda item: (item.name != preferred_nemo_filename, item.name))
        for nemo_file in ordered_files:
            try:
                return _restore_with_optional_cpu_fallback(
                    nemo_asr.models.ASRModel.restore_from,
                    restore_path=str(nemo_file),
                )
            except Exception:  # noqa: BLE001
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
                return _restore_with_optional_cpu_fallback(
                    model_class.restore_from,
                    restore_path=snapshot_dir,
                    save_restore_connector=connector,
                )
            except Exception:  # noqa: BLE001
                continue

    if restore_errors:
        raise RuntimeError(
            f"Failed to restore NeMo model {resolved_model_id!r} from local snapshot {snapshot_dir}: "
            f"{restore_errors[-1]}"
        ) from restore_errors[-1]
    raise RuntimeError(
        f"NeMo model snapshot {snapshot_dir!r} for {resolved_model_id!r} did not contain loadable artifacts."
    )


def _snapshot_contains_nemo_archive(snapshot_dir: str) -> bool:
    """Return True when a snapshot directory contains one or more `.nemo` archives."""
    return any(Path(snapshot_dir).glob("*.nemo"))


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
            "Transcription runtime for NVIDIA ASR models requires `nemo_toolkit[asr]`."
        ) from exc

    load_errors: list[BaseException] = []
    model = None
    snapshot_contains_nemo_archive = _snapshot_contains_nemo_archive(snapshot_dir)
    nemo_restore_attempts = 2 if snapshot_contains_nemo_archive else 1
    for attempt_index in range(nemo_restore_attempts):
        try:
            model = _load_nemo_model_from_snapshot(nemo_asr, resolved_model_id, snapshot_dir)
            break
        except Exception as exc:  # noqa: BLE001
            load_errors.append(exc)
            if (attempt_index + 1) < nemo_restore_attempts:
                continue

    if model is None and not snapshot_contains_nemo_archive:
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
        preferred_error = load_errors[-1]
        if snapshot_contains_nemo_archive:
            preferred_error = load_errors[0]
        raise RuntimeError(
            f"Failed to initialize NeMo ASR model for {resolved_model_id!r} from snapshot "
            f"{snapshot_dir!r}: {preferred_error}"
        ) from preferred_error

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
            "Transcription runtime for Qwen ASR models requires `qwen-asr`."
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


def release_transcription_runtime_resources(transcription_model: str | None = None) -> int:
    """Release cached ASR model instances so later stages can reclaim accelerator memory."""
    caches = _selected_transcription_model_caches(transcription_model)
    models_to_release = _collect_unique_cached_models(caches)

    for cache in caches:
        cache.clear()
    for model in models_to_release:
        _offload_model_to_cpu(model)

    gc.collect()
    _clear_accelerator_caches()
    return len(models_to_release)


def _selected_transcription_model_caches(transcription_model: str | None) -> list[dict[str, Any]]:
    """Select cache dictionaries relevant to a given transcription model."""
    if transcription_model is None:
        return [
            _FASTER_WHISPER_MODEL_CACHE,
            _NEMO_ASR_MODEL_CACHE,
            _QWEN_ASR_MODEL_CACHE,
        ]

    backend, _ = _resolve_transcription_backend(transcription_model)
    if backend == "nemo_asr":
        return [_NEMO_ASR_MODEL_CACHE]
    if backend == "qwen_asr":
        return [_QWEN_ASR_MODEL_CACHE]
    return [_FASTER_WHISPER_MODEL_CACHE]


def _collect_unique_cached_models(caches: Iterable[dict[str, Any]]) -> list[Any]:
    """Collect unique cached model objects across one or more cache mappings."""
    seen_model_ids: set[int] = set()
    unique_models: list[Any] = []
    for cache in caches:
        for model in cache.values():
            model_id = id(model)
            if model_id in seen_model_ids:
                continue
            seen_model_ids.add(model_id)
            unique_models.append(model)
    return unique_models


def _offload_model_to_cpu(model: Any) -> None:
    """Best-effort move a cached model off accelerator memory before dropping references."""
    for candidate in (model, getattr(model, "model", None)):
        if candidate is None:
            continue
        to_method = getattr(candidate, "to", None)
        if callable(to_method):
            try:
                to_method("cpu")
                return
            except TypeError:
                try:
                    import torch

                    to_method(torch.device("cpu"))
                    return
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass

        cpu_method = getattr(candidate, "cpu", None)
        if callable(cpu_method):
            try:
                cpu_method()
                return
            except Exception:  # noqa: BLE001
                pass


def _clear_accelerator_caches() -> None:
    """Best-effort flush of Torch accelerator allocators after model release."""
    try:
        import torch
    except ImportError:
        return

    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is not None:
        try:
            if cuda_module.is_available():
                cuda_module.empty_cache()
                ipc_collect = getattr(cuda_module, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
        except Exception:  # noqa: BLE001
            pass

    mps_module = getattr(torch, "mps", None)
    empty_cache = getattr(mps_module, "empty_cache", None)
    if callable(empty_cache):
        try:
            empty_cache()
        except Exception:  # noqa: BLE001
            pass


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

