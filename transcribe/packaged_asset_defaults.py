from __future__ import annotations

from transcribe.packaged_assets import (
    PACKAGED_ASSET_SCHEMA_VERSION,
    PackagedAssetFile,
    PackagedAssetsManifest,
    PackagedModelAsset,
)
from transcribe.runtime_defaults import (
    ALTERNATE_SESSION_NOTES_MODEL,
    DEFAULT_LIVE_TRANSCRIPTION_MODEL,
    DEFAULT_SESSION_NOTES_MODEL,
    PACKAGED_ACCURACY_TRANSCRIPTION_MODEL,
    PACKAGED_GRANITE_TRANSCRIPTION_MODEL,
)

DEFAULT_NOTES_MODEL_4B_REPO = "unsloth/Qwen3.5-4B-GGUF"
DEFAULT_NOTES_MODEL_4B_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
DEFAULT_NOTES_MODEL_4B_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
DEFAULT_NOTES_MODEL_2B_REPO = "unsloth/Qwen3.5-2B-GGUF"
DEFAULT_NOTES_MODEL_2B_REVISION = "f6d5376be1edb4d416d56da11e5397a961aca8ae"
DEFAULT_NOTES_MODEL_2B_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_PARAKEET_MODEL_REPO = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_PARAKEET_MODEL_REVISION = "6d590f77001d318fb17a0b5bf7ee329a91b52598"
DEFAULT_PARAKEET_REQUIRED_FILES = ("parakeet-tdt-0.6b-v3.nemo",)
DEFAULT_CANARY_MODEL_REPO = "nvidia/canary-qwen-2.5b"
DEFAULT_CANARY_MODEL_REVISION = "6cfc37ec7edc35a0545c403f551ecdfa28133d72"
DEFAULT_CANARY_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "LICENSES",
    "model.safetensors",
    "tokenizer.model",
)
DEFAULT_GRANITE_MODEL_REPO = "ibm-granite/granite-4.0-1b-speech"
DEFAULT_GRANITE_MODEL_REVISION = "4eaf14d77837c989d00f59c26262b6b9d10a9091"
DEFAULT_GRANITE_REQUIRED_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
UNKNOWN_PACKAGED_ASSET_SHA256 = "0" * 64
UNKNOWN_PACKAGED_ASSET_SIZE_BYTES = 0


def build_default_packaged_assets_manifest() -> PackagedAssetsManifest:
    """Return the shared packaged-model manifest used by runtime and build tools."""

    def zero_file(path: str) -> PackagedAssetFile:
        return PackagedAssetFile(
            path=path,
            sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
            size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
        )

    return PackagedAssetsManifest(
        schema_version=PACKAGED_ASSET_SCHEMA_VERSION,
        assets=(
            PackagedModelAsset(
                model_id=DEFAULT_SESSION_NOTES_MODEL,
                kind="notes",
                relative_path="notes/qwen3.5-4b-q4_k_m.gguf",
                source_type="huggingface_file",
                repo_id=DEFAULT_NOTES_MODEL_4B_REPO,
                revision=DEFAULT_NOTES_MODEL_4B_REVISION,
                filename=DEFAULT_NOTES_MODEL_4B_FILE,
                required_files=(),
                sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
                size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
                default_install=True,
            ),
            PackagedModelAsset(
                model_id=ALTERNATE_SESSION_NOTES_MODEL,
                kind="notes",
                relative_path="notes/qwen3.5-2b-q4_k_m.gguf",
                source_type="huggingface_file",
                repo_id=DEFAULT_NOTES_MODEL_2B_REPO,
                revision=DEFAULT_NOTES_MODEL_2B_REVISION,
                filename=DEFAULT_NOTES_MODEL_2B_FILE,
                required_files=(),
                sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
                size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
                default_install=False,
            ),
            PackagedModelAsset(
                model_id=DEFAULT_LIVE_TRANSCRIPTION_MODEL,
                kind="transcription",
                relative_path="asr/nvidia/parakeet-tdt-0.6b-v3",
                source_type="huggingface_snapshot",
                repo_id=DEFAULT_PARAKEET_MODEL_REPO,
                revision=DEFAULT_PARAKEET_MODEL_REVISION,
                filename=None,
                required_files=tuple(zero_file(path) for path in DEFAULT_PARAKEET_REQUIRED_FILES),
                sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
                size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
                default_install=True,
            ),
            PackagedModelAsset(
                model_id=PACKAGED_ACCURACY_TRANSCRIPTION_MODEL,
                kind="transcription",
                relative_path="asr/nvidia/canary-qwen-2.5b",
                source_type="huggingface_snapshot",
                repo_id=DEFAULT_CANARY_MODEL_REPO,
                revision=DEFAULT_CANARY_MODEL_REVISION,
                filename=None,
                required_files=tuple(zero_file(path) for path in DEFAULT_CANARY_REQUIRED_FILES),
                sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
                size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
                default_install=False,
            ),
            PackagedModelAsset(
                model_id=PACKAGED_GRANITE_TRANSCRIPTION_MODEL,
                kind="transcription",
                relative_path="asr/ibm-granite/granite-4.0-1b-speech",
                source_type="huggingface_snapshot",
                repo_id=DEFAULT_GRANITE_MODEL_REPO,
                revision=DEFAULT_GRANITE_MODEL_REVISION,
                filename=None,
                required_files=tuple(zero_file(path) for path in DEFAULT_GRANITE_REQUIRED_FILES),
                sha256=UNKNOWN_PACKAGED_ASSET_SHA256,
                size_bytes=UNKNOWN_PACKAGED_ASSET_SIZE_BYTES,
                default_install=False,
            ),
        ),
    )
