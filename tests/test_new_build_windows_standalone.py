from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

from transcribe.packaged_assets import load_packaged_asset_manifest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "new_build_windows_standalone.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("new_build_windows_standalone", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_script = _load_module()


def _make_inputs(tmp_path: Path, *, version: str = "1.2.3") -> object:
    llama_runtime_dir = tmp_path / "llama-runtime"
    llama_runtime_dir.mkdir()
    (llama_runtime_dir / "llama-server.exe").write_bytes(b"exe")
    (llama_runtime_dir / "ggml-base.dll").write_bytes(b"dll")

    return build_script.ResolvedBuildInputs(
        version=version,
        inno_setup_exe="iscc.exe",
        nuitka_command=("python", "-m", "nuitka"),
        llama_runtime_dir=llama_runtime_dir,
    )


def test_build_parser_defaults_to_package_version() -> None:
    args = build_script.build_parser().parse_args([])

    assert args.version == build_script.PACKAGE_VERSION


def test_build_packaged_assets_manifest_marks_default_and_optional_assets() -> None:
    manifest = build_script._build_packaged_assets_manifest()

    assert manifest.schema_version == build_script.PACKAGED_ASSET_SCHEMA_VERSION
    assert [asset.model_id for asset in manifest.assets] == [
        "qwen3.5:4b-q4_K_M",
        "qwen3.5:2b-q4_K_M",
        "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/canary-qwen-2.5b",
    ]
    assert [asset.default_install for asset in manifest.assets] == [True, False, True, False]
    assert all(asset.sha256 == build_script.UNKNOWN_PACKAGED_ASSET_SHA256 for asset in manifest.assets)
    assert all(asset.size_bytes == build_script.UNKNOWN_PACKAGED_ASSET_SIZE_BYTES for asset in manifest.assets)


def test_stage_runtime_assets_writes_manifest_prompt_and_llama_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_inputs = _make_inputs(tmp_path)
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("prompt body\n", encoding="utf-8")
    monkeypatch.setattr(build_script, "PROMPT_SOURCE_PATH", prompt_path)
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    build_script._stage_runtime_assets(stage_dir, build_inputs)

    manifest = load_packaged_asset_manifest(stage_dir / build_script.PACKAGED_ASSET_MANIFEST_FILENAME)
    assert (stage_dir / "prompts" / "clinical_note_synthesis_llm_prompt.md").read_text(encoding="utf-8") == "prompt body\n"
    assert (stage_dir / "runtime" / "llm" / "llama-server.exe").read_bytes() == b"exe"
    assert (stage_dir / "runtime" / "llm" / "ggml-base.dll").read_bytes() == b"dll"
    assert [asset.model_id for asset in manifest.assets] == [
        "qwen3.5:4b-q4_K_M",
        "qwen3.5:2b-q4_K_M",
        "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/canary-qwen-2.5b",
    ]


def test_build_nuitka_command_targets_packaged_entrypoint_and_attach_console(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_script, "_supports_nuitka_option", lambda command, option_fragment: option_fragment == "--windows-console-mode=")
    monkeypatch.setattr(build_script, "_nuitka_distribution_metadata_names", lambda build_dir: ("libcst", "transformers"))
    monkeypatch.setattr(build_script, "_module_available", lambda module_name: module_name == "_yaml")

    command = build_script._build_nuitka_command(
        nuitka_command=("python", "-m", "nuitka"),
        build_dir=tmp_path / "nuitka",
        version="1.2.3",
    )

    assert "--standalone" in command
    assert "--windows-console-mode=attach" in command
    assert "--enable-plugin=tk-inter" in command
    assert "--include-module=transcribe.packaged_ui" in command
    assert "--include-package=soundcard" in command
    assert "--include-module=_yaml" in command
    assert "--include-distribution-metadata=libcst" in command
    assert "--include-distribution-metadata=transformers" in command
    assert str(build_script.REPO_ROOT / "packaged_main.py") == command[-1]


def test_require_inno_setup_uses_existing_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(build_script, "_resolve_executable", lambda candidate: "C:/Tools/ISCC.exe")

    resolved = build_script._require_inno_setup()

    assert resolved == "C:/Tools/ISCC.exe"


def test_require_inno_setup_bootstraps_local_install_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer_path = tmp_path / "downloads" / "innosetup.exe"
    installer_path.parent.mkdir(parents=True, exist_ok=True)
    installer_path.write_bytes(b"installer")
    observed: dict[str, list[str]] = {}

    monkeypatch.setattr(build_script, "TOOLS_DIR", tmp_path)
    monkeypatch.setattr(build_script, "_resolve_executable", lambda candidate: None)
    monkeypatch.setattr(build_script, "_resolve_known_executable", lambda paths: None)
    monkeypatch.setattr(
        build_script,
        "_download_inno_setup_installer",
        lambda *, tools_dir, release: installer_path,
    )

    def fake_run(command, **kwargs):
        del kwargs
        observed["command"] = list(command)
        iscc_path = tmp_path / "inno-setup" / "ISCC.exe"
        iscc_path.parent.mkdir(parents=True, exist_ok=True)
        iscc_path.write_bytes(b"iscc")

    monkeypatch.setattr(build_script, "_run", fake_run)

    resolved = build_script._require_inno_setup()

    assert Path(resolved) == (tmp_path / "inno-setup" / "ISCC.exe").resolve()
    assert observed["command"][0] == str(installer_path)
    assert "/VERYSILENT" in observed["command"]


def test_require_nuitka_uses_lowercase_module_name(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_module_names: list[str] = []

    def fake_module_available(module_name: str) -> bool:
        seen_module_names.append(module_name)
        return module_name == "nuitka"

    monkeypatch.setattr(build_script, "_module_available", fake_module_available)

    command = build_script._require_nuitka()

    assert command == (build_script.sys.executable, "-m", "nuitka")
    assert seen_module_names
    assert set(seen_module_names) == {"nuitka"}


def test_require_nuitka_bootstraps_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = {"ready": False}

    def fake_module_available(module_name: str) -> bool:
        return module_name == "nuitka" and installed["ready"]

    def fake_bootstrap_nuitka() -> None:
        installed["ready"] = True

    monkeypatch.setattr(build_script, "_module_available", fake_module_available)
    monkeypatch.setattr(build_script, "_bootstrap_nuitka", fake_bootstrap_nuitka)

    command = build_script._require_nuitka()

    assert command == (build_script.sys.executable, "-m", "nuitka")


def test_require_repo_runtime_dependencies_bootstraps_missing_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"ready": False}

    def fake_module_available(module_name: str) -> bool:
        if module_name in build_script.REQUIRED_RUNTIME_MODULES:
            return installed["ready"]
        return True

    def fake_bootstrap_runtime_dependencies() -> None:
        installed["ready"] = True

    monkeypatch.setattr(build_script, "_module_available", fake_module_available)
    monkeypatch.setattr(
        build_script,
        "_bootstrap_repo_runtime_dependencies",
        fake_bootstrap_runtime_dependencies,
    )

    build_script._require_repo_runtime_dependencies()


def test_seed_distribution_metadata_names_skips_blocked_seed_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_script.importlib.metadata,
        "packages_distributions",
        lambda: {
            "nemo": ["nemo-toolkit"],
            "torch": ["torch"],
            "transformers": ["transformers"],
            "libcst": ["libcst"],
        },
    )
    monkeypatch.setattr(
        build_script,
        "_distribution_top_level_packages",
        lambda distribution_name: {
            "nemo-toolkit": ("nemo", "examples"),
            "torch": ("torch",),
            "transformers": ("transformers",),
            "libcst": ("libcst",),
        }.get(build_script._normalize_distribution_name(distribution_name), (distribution_name,)),
    )

    names = {
        build_script._normalize_distribution_name(distribution_name)
        for distribution_name in build_script._seed_distribution_metadata_names()
    }

    assert "nemo-toolkit" not in names
    assert {"torch", "transformers", "libcst"}.issubset(names)


def test_seed_distribution_metadata_names_skips_nofollowed_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_script.importlib.metadata,
        "packages_distributions",
        lambda: {
            "datasets": ["datasets"],
            "requests": ["requests"],
            "transcribe": ["transcribe"],
        },
    )
    monkeypatch.setattr(
        build_script,
        "_distribution_top_level_packages",
        lambda distribution_name: {
            "datasets": ("datasets",),
            "requests": ("requests",),
            "transcribe": ("transcribe",),
        }.get(build_script._normalize_distribution_name(distribution_name), (distribution_name,)),
    )

    names = {
        build_script._normalize_distribution_name(distribution_name)
        for distribution_name in build_script._seed_distribution_metadata_names()
    }

    assert "datasets" not in names
    assert {"requests", "transcribe"}.issubset(names)


def test_seed_distribution_metadata_names_includes_transformers_runtime_metadata_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_script.importlib.metadata,
        "packages_distributions",
        lambda: {
            "datasets": ["datasets"],
            "requests": ["requests"],
            "packaging": ["packaging"],
            "filelock": ["filelock"],
            "yaml": ["PyYAML"],
            "tqdm": ["tqdm"],
            "regex": ["regex"],
            "accelerate": ["accelerate"],
            "numpy": ["numpy"],
            "transformers": ["transformers"],
            "tokenizers": ["tokenizers"],
            "huggingface_hub": ["huggingface_hub"],
            "safetensors": ["safetensors"],
            "torch": ["torch"],
            "soundcard": ["SoundCard"],
            "omegaconf": ["omegaconf"],
            "libcst": ["libcst"],
            "transcribe": ["transcribe"],
        },
    )
    monkeypatch.setattr(
        build_script,
        "_distribution_top_level_packages",
        lambda distribution_name: {
            "datasets": ("datasets",),
            "requests": ("requests",),
            "packaging": ("packaging",),
            "filelock": ("filelock",),
            "pyyaml": ("yaml",),
            "tqdm": ("tqdm",),
            "regex": ("regex",),
            "accelerate": ("accelerate",),
            "numpy": ("numpy",),
            "transformers": ("transformers",),
            "tokenizers": ("tokenizers",),
            "huggingface-hub": ("huggingface_hub",),
            "safetensors": ("safetensors",),
            "torch": ("torch",),
            "soundcard": ("soundcard",),
            "omegaconf": ("omegaconf",),
            "libcst": ("libcst",),
            "transcribe": ("transcribe",),
        }.get(build_script._normalize_distribution_name(distribution_name), (distribution_name,)),
    )

    names = {
        build_script._normalize_distribution_name(distribution_name)
        for distribution_name in build_script._seed_distribution_metadata_names()
    }

    assert {
        "accelerate",
        "filelock",
        "huggingface-hub",
        "numpy",
        "packaging",
        "pyyaml",
        "regex",
        "requests",
        "tqdm",
    }.issubset(names)


def test_download_llama_cpp_runtime_scans_recent_releases_when_latest_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_script, "TOOLS_DIR", tmp_path / "tools")
    monkeypatch.setattr(
        build_script,
        "_fetch_json",
        lambda url: {
            "assets": [
                {
                    "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                    "browser_download_url": "https://example.test/cuda.zip",
                }
            ]
        },
    )
    monkeypatch.setattr(
        build_script,
        "_fetch_json_list",
        lambda url: [
            {
                "tag_name": "b9999",
                "assets": [
                    {
                        "name": "cudart-llama-bin-win-cuda-12.4-x64.zip",
                        "browser_download_url": "https://example.test/cuda.zip",
                    }
                ],
            },
            {
                "tag_name": "b9998",
                "assets": [
                    {
                        "name": "llama-b9998-bin-win-cpu-x64.zip",
                        "browser_download_url": "https://example.test/cpu.zip",
                    }
                ],
            },
        ],
    )

    def fake_download_file(url: str, destination: Path) -> Path:
        del url
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("bundle/llama-server.exe", "exe")
            archive.writestr("bundle/ggml-base.dll", "dll")
        return destination.resolve()

    monkeypatch.setattr(build_script, "_download_file", fake_download_file)

    runtime_dir = build_script._download_llama_cpp_runtime()

    assert runtime_dir == (tmp_path / "tools" / "llama.cpp" / "llama-b9998-bin-win-cpu-x64" / "bundle").resolve()
    assert (runtime_dir / "llama-server.exe").read_text(encoding="utf-8") == "exe"


def test_resolve_build_inputs_uses_prerequisites_and_llama_runtime_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_runtime_dir = tmp_path / "llama-runtime"
    llama_runtime_dir.mkdir()
    calls: list[str] = []

    monkeypatch.setattr(build_script, "_ensure_supported_python", lambda: calls.append("python"))
    monkeypatch.setattr(build_script, "_require_repo_runtime_dependencies", lambda: calls.append("runtime-deps"))
    monkeypatch.setattr(build_script, "_require_nuitka", lambda: ("python", "-m", "nuitka"))
    monkeypatch.setattr(build_script, "_require_inno_setup", lambda: "C:/Tools/ISCC.exe")
    monkeypatch.setattr(build_script, "_download_llama_cpp_runtime", lambda: llama_runtime_dir)

    build_inputs = build_script._resolve_build_inputs("2.0.0")

    assert calls == ["python", "runtime-deps"]
    assert build_inputs == build_script.ResolvedBuildInputs(
        version="2.0.0",
        inno_setup_exe="C:/Tools/ISCC.exe",
        nuitka_command=("python", "-m", "nuitka"),
        llama_runtime_dir=llama_runtime_dir,
    )


def test_build_nuitka_command_keeps_existing_report_available_for_metadata_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = (tmp_path / "nuitka" / build_script.NUITKA_REPORT_FILENAME)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("<report />", encoding="utf-8")
    seen: dict[str, bool] = {}

    def fake_metadata_names(build_dir: Path) -> tuple[str, ...]:
        seen["report_exists"] = build_script._default_nuitka_report_path(build_dir).exists()
        return ()

    monkeypatch.setattr(build_script, "_supports_nuitka_option", lambda command, option_fragment: False)
    monkeypatch.setattr(build_script, "_nuitka_distribution_metadata_names", fake_metadata_names)

    build_script._build_nuitka_command(
        nuitka_command=("python", "-m", "nuitka"),
        build_dir=tmp_path / "nuitka",
        version="1.2.3",
    )

    assert seen == {"report_exists": True}
    assert report_path.exists()


def test_publish_release_installer_clears_existing_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir()
    (releases_dir / "old.exe").write_bytes(b"old")
    stale_dir = releases_dir / "stale"
    stale_dir.mkdir()
    (stale_dir / "artifact.txt").write_text("stale", encoding="utf-8")
    installer_path = tmp_path / "installer.exe"
    installer_path.write_bytes(b"installer")
    monkeypatch.setattr(build_script, "RELEASES_DIR", releases_dir)

    published = build_script._publish_release_installer(installer_path, "1.2.3")

    assert published == (releases_dir / "Transcribe-1.2.3.exe").resolve()
    assert published.read_bytes() == b"installer"
    assert sorted(path.name for path in releases_dir.iterdir()) == ["Transcribe-1.2.3.exe"]


def test_main_smoke_builds_installer_and_replaces_release_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "wbuild"
    releases_dir = tmp_path / "releases"
    releases_dir.mkdir()
    (releases_dir / "old.exe").write_bytes(b"old")
    build_inputs = _make_inputs(tmp_path, version="9.9.9")
    installer_path = tmp_path / "installer" / "transcribe-setup.exe"
    installer_path.parent.mkdir(parents=True, exist_ok=True)
    installer_path.write_bytes(b"installer")

    monkeypatch.setattr(build_script.os, "name", "nt")
    monkeypatch.setattr(build_script, "BUILD_ROOT", build_root)
    monkeypatch.setattr(build_script, "STAGE_DIR", build_root / "stage")
    monkeypatch.setattr(build_script, "NUITKA_BUILD_DIR", build_root / "nuitka")
    monkeypatch.setattr(build_script, "INSTALLER_DIR", build_root / "installer")
    monkeypatch.setattr(build_script, "CACHE_DIR", build_root / "_cache")
    monkeypatch.setattr(build_script, "RELEASES_DIR", releases_dir)
    monkeypatch.setattr(build_script, "_resolve_build_inputs", lambda version: build_inputs)
    monkeypatch.setattr(build_script, "_build_nuitka_bundle", lambda stage_dir, build_dir, build_inputs: stage_dir.mkdir(parents=True, exist_ok=True))

    def fake_stage_runtime_assets(stage_dir: Path, build_inputs: object) -> None:
        del build_inputs
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "transcribe.exe").write_bytes(b"exe")
        soundcard_dir = stage_dir / "soundcard"
        soundcard_dir.mkdir(parents=True, exist_ok=True)
        (soundcard_dir / "mediafoundation.py.h").write_text("header", encoding="utf-8")
        (stage_dir / "packaged-assets.json").write_text("{}", encoding="utf-8")
        prompt_dir = stage_dir / "prompts"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "clinical_note_synthesis_llm_prompt.md").write_text("prompt", encoding="utf-8")
        runtime_dir = stage_dir / "runtime" / "llm"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "llama-server.exe").write_bytes(b"exe")

    monkeypatch.setattr(build_script, "_stage_runtime_assets", fake_stage_runtime_assets)
    monkeypatch.setattr(build_script, "_build_inno_installer", lambda stage_dir, build_inputs: installer_path)

    rc = build_script.main(["--version", "9.9.9"])

    assert rc == 0
    assert (releases_dir / "Transcribe-9.9.9.exe").read_bytes() == b"installer"
    assert sorted(path.name for path in releases_dir.iterdir()) == ["Transcribe-9.9.9.exe"]
