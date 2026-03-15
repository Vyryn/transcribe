from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_build_module() -> ModuleType:
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "build_windows_standalone.py"
    spec = importlib.util.spec_from_file_location("build_windows_standalone", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sanitize_version_accepts_semver() -> None:
    module = _load_build_module()

    assert module.sanitize_version("1.2.3") == "1.2.3"
    assert module.sanitize_version("1.2.3-rc.1") == "1.2.3-rc.1"
    assert module.sanitize_version("1.2.3+build.5") == "1.2.3+build.5"


def test_sanitize_version_rejects_invalid_semver() -> None:
    module = _load_build_module()

    with pytest.raises(ValueError, match="semver"):
        module.sanitize_version("1.2")

    with pytest.raises(ValueError, match="semver"):
        module.sanitize_version("version-1.2.3")


def test_build_packaged_assets_manifest_defaults() -> None:
    module = _load_build_module()
    manifest = module.build_packaged_assets_manifest()
    defaults = {asset.model_id for asset in manifest.assets if asset.default_install}
    canary_asset = next(asset for asset in manifest.assets if asset.model_id == "nvidia/canary-qwen-2.5b")

    assert manifest.schema_version == "transcribe-packaged-assets-v1"
    assert defaults == {
        "nvidia/parakeet-tdt-0.6b-v3",
        "qwen3.5:4b-q4_K_M",
    }
    assert tuple(file_entry.path for file_entry in canary_asset.required_files) == (
        "config.json",
        "generation_config.json",
        "LICENSES",
        "model.safetensors",
        "tokenizer.model",
    )


def test_safe_filename_component_normalizes_build_metadata() -> None:
    module = _load_build_module()

    assert module.safe_filename_component("1.2.3+build.5") == "1.2.3-build.5"
