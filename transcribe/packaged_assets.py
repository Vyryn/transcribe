from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PACKAGED_ASSET_MANIFEST_FILENAME = "packaged-assets.json"
INSTALLED_ASSET_STATE_FILENAME = "installed-assets.json"
PACKAGED_ASSET_SCHEMA_VERSION = "transcribe-packaged-assets-v1"
_HF_CACHE_DIRNAME = "hf-cache"


@dataclass(frozen=True, slots=True)
class PackagedAssetFile:
    """One downloadable file that belongs to a packaged model asset."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PackagedModelAsset:
    """Manifest entry describing one installable model asset."""

    model_id: str
    kind: str
    relative_path: str
    source_type: str
    repo_id: str
    revision: str
    filename: str | None
    required_files: tuple[PackagedAssetFile, ...]
    sha256: str
    size_bytes: int
    default_install: bool


@dataclass(frozen=True, slots=True)
class PackagedAssetsManifest:
    """Manifest of all packaged model assets."""

    schema_version: str
    assets: tuple[PackagedModelAsset, ...]


@dataclass(frozen=True, slots=True)
class PackagedAssetInstallResult:
    """Result metadata for one asset install attempt."""

    model_id: str
    target_path: Path
    skipped: bool


def _normalize_relative_path(value: str | Path) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute():
        raise ValueError(f"relative path must not be absolute: {value!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise ValueError("relative path must not be empty")
    if any(part == ".." for part in parts):
        raise ValueError(f"relative path must not traverse upward: {value!r}")
    return str(PurePosixPath(*parts))


def _sha256_for_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_packaged_asset_file(source_path: Path, *, relative_path: str | Path) -> PackagedAssetFile:
    """Create a manifest file entry from an existing local file."""
    resolved = source_path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(resolved)
    return PackagedAssetFile(
        path=_normalize_relative_path(relative_path),
        sha256=_sha256_for_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _aggregate_file_records(files: tuple[PackagedAssetFile, ...]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_size = 0
    for file_entry in sorted(files, key=lambda item: item.path):
        digest.update(file_entry.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_entry.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(file_entry.size_bytes).encode("ascii"))
        digest.update(b"\n")
        total_size += file_entry.size_bytes
    return digest.hexdigest(), total_size


def build_single_file_asset(
    *,
    model_id: str,
    kind: str,
    relative_path: str | Path,
    repo_id: str,
    revision: str,
    filename: str,
    source_path: Path,
    default_install: bool,
) -> PackagedModelAsset:
    """Build a manifest entry for one file-backed asset."""
    normalized_relative_path = _normalize_relative_path(relative_path)
    normalized_filename = _normalize_relative_path(filename)
    file_entry = build_packaged_asset_file(source_path, relative_path=normalized_filename)
    return PackagedModelAsset(
        model_id=model_id,
        kind=kind,
        relative_path=normalized_relative_path,
        source_type="huggingface_file",
        repo_id=repo_id,
        revision=revision,
        filename=normalized_filename,
        required_files=(),
        sha256=file_entry.sha256,
        size_bytes=file_entry.size_bytes,
        default_install=default_install,
    )


def build_directory_asset(
    *,
    model_id: str,
    kind: str,
    relative_path: str | Path,
    repo_id: str,
    revision: str,
    source_root: Path,
    required_files: tuple[str | Path, ...],
    default_install: bool,
) -> PackagedModelAsset:
    """Build a manifest entry for a multi-file directory asset."""
    resolved_root = source_root.resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise FileNotFoundError(resolved_root)
    file_entries = tuple(
        build_packaged_asset_file(
            resolved_root / Path(str(file_relative_path).replace("/", os.sep)),
            relative_path=file_relative_path,
        )
        for file_relative_path in required_files
    )
    aggregate_sha256, total_size = _aggregate_file_records(file_entries)
    return PackagedModelAsset(
        model_id=model_id,
        kind=kind,
        relative_path=_normalize_relative_path(relative_path),
        source_type="huggingface_snapshot",
        repo_id=repo_id,
        revision=revision,
        filename=None,
        required_files=file_entries,
        sha256=aggregate_sha256,
        size_bytes=total_size,
        default_install=default_install,
    )


def packaged_asset_manifest_to_dict(manifest: PackagedAssetsManifest) -> dict[str, object]:
    """Convert a manifest into a JSON-safe dictionary."""
    return {
        "schema_version": manifest.schema_version,
        "assets": [
            {
                "model_id": asset.model_id,
                "kind": asset.kind,
                "relative_path": asset.relative_path,
                "source_type": asset.source_type,
                "repo_id": asset.repo_id,
                "revision": asset.revision,
                "filename": asset.filename,
                "required_files": [
                    {
                        "path": file_entry.path,
                        "sha256": file_entry.sha256,
                        "size_bytes": file_entry.size_bytes,
                    }
                    for file_entry in asset.required_files
                ],
                "sha256": asset.sha256,
                "size_bytes": asset.size_bytes,
                "default_install": asset.default_install,
            }
            for asset in manifest.assets
        ],
    }


def write_packaged_asset_manifest(manifest: PackagedAssetsManifest, destination: Path) -> None:
    """Write a packaged asset manifest to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(packaged_asset_manifest_to_dict(manifest), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _parse_required_file(raw: object) -> PackagedAssetFile:
    if not isinstance(raw, dict):
        raise ValueError(f"required file entry must be a JSON object, got {type(raw).__name__}")
    path = raw.get("path")
    sha256 = raw.get("sha256")
    size_bytes = raw.get("size_bytes")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("required file entry must include non-empty string field 'path'")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError("required file entry must include 64-char string field 'sha256'")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError("required file entry must include non-negative integer field 'size_bytes'")
    return PackagedAssetFile(
        path=_normalize_relative_path(path),
        sha256=sha256,
        size_bytes=size_bytes,
    )


def _parse_asset(raw: object) -> PackagedModelAsset:
    if not isinstance(raw, dict):
        raise ValueError(f"asset entry must be a JSON object, got {type(raw).__name__}")

    model_id = raw.get("model_id")
    kind = raw.get("kind")
    relative_path = raw.get("relative_path")
    source_type = raw.get("source_type")
    repo_id = raw.get("repo_id")
    revision = raw.get("revision")
    filename = raw.get("filename")
    required_files_raw = raw.get("required_files", [])
    sha256 = raw.get("sha256")
    size_bytes = raw.get("size_bytes")
    default_install = raw.get("default_install")

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("asset entry must include non-empty string field 'model_id'")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"asset {model_id!r} must include non-empty string field 'kind'")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"asset {model_id!r} must include non-empty string field 'relative_path'")
    if not isinstance(source_type, str) or source_type not in {"huggingface_file", "huggingface_snapshot"}:
        raise ValueError(f"asset {model_id!r} has unsupported source_type {source_type!r}")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError(f"asset {model_id!r} must include non-empty string field 'repo_id'")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError(f"asset {model_id!r} must include non-empty string field 'revision'")
    if filename is not None and not isinstance(filename, str):
        raise ValueError(f"asset {model_id!r} field 'filename' must be a string when present")
    if not isinstance(required_files_raw, list):
        raise ValueError(f"asset {model_id!r} field 'required_files' must be a list")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"asset {model_id!r} must include 64-char string field 'sha256'")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"asset {model_id!r} must include non-negative integer field 'size_bytes'")
    if not isinstance(default_install, bool):
        raise ValueError(f"asset {model_id!r} must include boolean field 'default_install'")

    required_files = tuple(_parse_required_file(item) for item in required_files_raw)
    normalized_relative_path = _normalize_relative_path(relative_path)
    normalized_filename = _normalize_relative_path(filename) if isinstance(filename, str) else None
    if source_type == "huggingface_file" and normalized_filename is None:
        raise ValueError(f"asset {model_id!r} must include 'filename' for source_type 'huggingface_file'")
    if source_type == "huggingface_snapshot" and not required_files:
        raise ValueError(f"asset {model_id!r} must include 'required_files' for source_type 'huggingface_snapshot'")

    return PackagedModelAsset(
        model_id=model_id.strip(),
        kind=kind.strip(),
        relative_path=normalized_relative_path,
        source_type=source_type,
        repo_id=repo_id.strip(),
        revision=revision.strip(),
        filename=normalized_filename,
        required_files=required_files,
        sha256=sha256,
        size_bytes=size_bytes,
        default_install=default_install,
    )


def load_packaged_asset_manifest(path: Path) -> PackagedAssetsManifest:
    """Load and validate one packaged asset manifest from disk."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("packaged asset manifest must be a JSON object")
    schema_version = raw.get("schema_version")
    assets_raw = raw.get("assets")
    if schema_version != PACKAGED_ASSET_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported packaged asset manifest schema {schema_version!r}; "
            f"expected {PACKAGED_ASSET_SCHEMA_VERSION!r}"
        )
    if not isinstance(assets_raw, list):
        raise ValueError("packaged asset manifest must include a list field 'assets'")
    assets = tuple(_parse_asset(item) for item in assets_raw)
    seen_model_ids: set[str] = set()
    for asset in assets:
        if asset.model_id in seen_model_ids:
            raise ValueError(f"duplicate packaged asset model id: {asset.model_id!r}")
        seen_model_ids.add(asset.model_id)
    return PackagedAssetsManifest(schema_version=schema_version, assets=assets)


def select_packaged_model_assets(
    manifest: PackagedAssetsManifest,
    *,
    model_ids: list[str] | None = None,
    default_only: bool = False,
) -> tuple[PackagedModelAsset, ...]:
    """Select installable assets from a manifest."""
    requested = {model_id.strip() for model_id in (model_ids or []) if model_id.strip()}
    selected: list[PackagedModelAsset] = []
    for asset in manifest.assets:
        if default_only and asset.default_install:
            selected.append(asset)
            continue
        if asset.model_id in requested:
            selected.append(asset)

    if requested:
        selected_ids = {asset.model_id for asset in selected}
        if not requested.issubset(selected_ids):
            available = ", ".join(sorted(asset.model_id for asset in manifest.assets))
            missing = ", ".join(sorted(requested.difference(selected_ids)))
            raise ValueError(f"unknown model ids: {missing}. Available: {available}")

    if not selected:
        raise ValueError("select_packaged_model_assets requires model_ids or default_only=True")

    deduplicated: list[PackagedModelAsset] = []
    seen_model_ids: set[str] = set()
    for asset in selected:
        if asset.model_id in seen_model_ids:
            continue
        seen_model_ids.add(asset.model_id)
        deduplicated.append(asset)
    return tuple(deduplicated)


def resolve_asset_target_path(asset: PackagedModelAsset, *, models_root: Path) -> Path:
    """Resolve the final filesystem path for an installed asset."""
    return (models_root / Path(asset.relative_path.replace("/", os.sep))).resolve()


def _asset_component_files(asset: PackagedModelAsset) -> tuple[PackagedAssetFile, ...]:
    if asset.source_type == "huggingface_snapshot":
        return asset.required_files
    assert asset.filename is not None
    return (PackagedAssetFile(path=asset.filename, sha256=asset.sha256, size_bytes=asset.size_bytes),)


def verify_installed_asset(asset: PackagedModelAsset, *, models_root: Path) -> bool:
    """Return True when an installed asset matches manifest size and hash metadata."""
    target_path = resolve_asset_target_path(asset, models_root=models_root)
    if asset.source_type == "huggingface_file":
        if not target_path.exists() or not target_path.is_file():
            return False
        return target_path.stat().st_size == asset.size_bytes and _sha256_for_file(target_path) == asset.sha256

    if not target_path.exists() or not target_path.is_dir():
        return False

    observed_files: list[PackagedAssetFile] = []
    for file_entry in _asset_component_files(asset):
        installed_file = target_path / Path(file_entry.path.replace("/", os.sep))
        if not installed_file.exists() or not installed_file.is_file():
            return False
        size_bytes = installed_file.stat().st_size
        sha256 = _sha256_for_file(installed_file)
        if size_bytes != file_entry.size_bytes or sha256 != file_entry.sha256:
            return False
        observed_files.append(PackagedAssetFile(path=file_entry.path, sha256=sha256, size_bytes=size_bytes))
    aggregate_sha256, total_size = _aggregate_file_records(tuple(observed_files))
    return aggregate_sha256 == asset.sha256 and total_size == asset.size_bytes


def load_installed_asset_state(path: Path) -> dict[str, dict[str, object]]:
    """Load installed asset state from disk."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    assets = raw.get("assets")
    if not isinstance(assets, dict):
        return {}
    state: dict[str, dict[str, object]] = {}
    for model_id, metadata in assets.items():
        if isinstance(model_id, str) and isinstance(metadata, dict):
            state[model_id] = dict(metadata)
    return state


def write_installed_asset_state(path: Path, state: dict[str, dict[str, object]]) -> None:
    """Persist installed asset state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": PACKAGED_ASSET_SCHEMA_VERSION, "assets": state}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _temporary_environment(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, original in previous.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _hf_download_file(*, repo_id: str, revision: str, filename: str, cache_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Downloading packaged model assets requires `huggingface_hub`.") from exc

    resolved_cache_dir = cache_dir.resolve()
    hub_cache = resolved_cache_dir / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    with _temporary_environment(
        {
            "HF_HOME": str(resolved_cache_dir),
            "HF_HUB_CACHE": str(hub_cache),
            "HUGGINGFACE_HUB_CACHE": str(hub_cache),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_DATASETS_OFFLINE": "0",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        }
    ):
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=False,
            cache_dir=str(hub_cache),
        )
    path = Path(downloaded).resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Downloaded Hugging Face asset is missing: {path}")
    return path


def _write_verified_file(
    source_path: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> None:
    source_resolved = source_path.resolve()
    observed_size = source_resolved.stat().st_size
    if observed_size != expected_size_bytes:
        raise RuntimeError(
            f"Downloaded file size mismatch for {source_resolved}: "
            f"expected {expected_size_bytes}, got {observed_size}"
        )
    observed_sha256 = _sha256_for_file(source_resolved)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Downloaded file hash mismatch for {source_resolved}: "
            f"expected {expected_sha256}, got {observed_sha256}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_destination = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}")
    shutil.copy2(source_resolved, temp_destination)
    os.replace(temp_destination, destination)


def _installed_asset_metadata(asset: PackagedModelAsset, *, target_path: Path) -> dict[str, object]:
    installed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "relative_path": asset.relative_path,
        "sha256": asset.sha256,
        "size_bytes": asset.size_bytes,
        "source_type": asset.source_type,
        "revision": asset.revision,
        "installed_at": installed_at,
        "target_path": str(target_path),
    }


def install_packaged_model_assets(
    manifest: PackagedAssetsManifest,
    *,
    models_root: Path,
    installed_state_path: Path,
    model_ids: list[str] | None = None,
    default_only: bool = False,
    hf_cache_dir: Path | None = None,
    progress_callback: Callable[[str, PackagedModelAsset, Path], None] | None = None,
) -> tuple[PackagedAssetInstallResult, ...]:
    """Install one or more packaged assets into the writable models root."""
    selected_assets = select_packaged_model_assets(manifest, model_ids=model_ids, default_only=default_only)
    resolved_models_root = models_root.resolve()
    resolved_models_root.mkdir(parents=True, exist_ok=True)
    state = load_installed_asset_state(installed_state_path)
    cache_dir = (hf_cache_dir or (installed_state_path.parent / _HF_CACHE_DIRNAME)).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[PackagedAssetInstallResult] = []
    for asset in selected_assets:
        target_path = resolve_asset_target_path(asset, models_root=resolved_models_root)
        if verify_installed_asset(asset, models_root=resolved_models_root):
            state[asset.model_id] = _installed_asset_metadata(asset, target_path=target_path)
            if progress_callback is not None:
                progress_callback("skipped", asset, target_path)
            results.append(PackagedAssetInstallResult(model_id=asset.model_id, target_path=target_path, skipped=True))
            continue

        if progress_callback is not None:
            progress_callback("installing", asset, target_path)

        if asset.source_type == "huggingface_file":
            if target_path.exists() and target_path.is_dir():
                shutil.rmtree(target_path)
            assert asset.filename is not None
            downloaded = _hf_download_file(
                repo_id=asset.repo_id,
                revision=asset.revision,
                filename=asset.filename,
                cache_dir=cache_dir,
            )
            _write_verified_file(
                downloaded,
                target_path,
                expected_sha256=asset.sha256,
                expected_size_bytes=asset.size_bytes,
            )
        else:
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                else:
                    target_path.unlink()
            temp_root = target_path.parent / f".{target_path.name}.tmp-{os.getpid()}-{int(time.time() * 1000)}"
            if temp_root.exists():
                shutil.rmtree(temp_root)
            for file_entry in asset.required_files:
                downloaded = _hf_download_file(
                    repo_id=asset.repo_id,
                    revision=asset.revision,
                    filename=file_entry.path,
                    cache_dir=cache_dir,
                )
                _write_verified_file(
                    downloaded,
                    temp_root / Path(file_entry.path.replace("/", os.sep)),
                    expected_sha256=file_entry.sha256,
                    expected_size_bytes=file_entry.size_bytes,
                )
            temp_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_root, target_path)

        if not verify_installed_asset(asset, models_root=resolved_models_root):
            raise RuntimeError(
                f"Installed packaged asset {asset.model_id!r} failed verification at {target_path}."
            )
        state[asset.model_id] = _installed_asset_metadata(asset, target_path=target_path)
        if progress_callback is not None:
            progress_callback("installed", asset, target_path)
        results.append(PackagedAssetInstallResult(model_id=asset.model_id, target_path=target_path, skipped=False))

    write_installed_asset_state(installed_state_path, state)
    return tuple(results)


