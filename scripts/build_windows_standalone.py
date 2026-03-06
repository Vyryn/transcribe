from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from transcribe.runtime_defaults import (
    ALTERNATE_SESSION_NOTES_MODEL,
    DEFAULT_SESSION_NOTES_MODEL,
)
from transcribe.runtime_env import bundled_notes_model_specs, bundled_transcription_model_specs

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGE_DIR = REPO_ROOT / "dist" / "windows" / "transcribe"
DEFAULT_BUILD_DIR = REPO_ROOT / "dist" / "windows" / "build"
DEFAULT_INSTALLER_DIR = REPO_ROOT / "dist" / "windows" / "installer"
PROMPT_SOURCE_PATH = REPO_ROOT / "clinical note synthesis llm prompt.md"
INNO_SCRIPT_PATH = REPO_ROOT / "packaging" / "windows" / "transcribe.iss"


def _copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _build_pyinstaller_bundle(*, pyinstaller_exe: str, stage_dir: Path, build_dir: Path) -> Path:
    stage_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        pyinstaller_exe,
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "transcribe",
        "--distpath",
        str(stage_dir.parent),
        "--workpath",
        str(build_dir / "work"),
        "--specpath",
        str(build_dir / "spec"),
        "--paths",
        str(REPO_ROOT),
        "--hidden-import",
        "transcribe.audio.windows_capture",
        "--hidden-import",
        "transcribe.audio.backend_loader",
        "--hidden-import",
        "transcribe.notes",
        str(REPO_ROOT / "main.py"),
    ]
    subprocess.run(command, check=True)
    bundle_dir = stage_dir.parent / "transcribe"
    if bundle_dir != stage_dir:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        shutil.move(str(bundle_dir), str(stage_dir))
    return stage_dir


def _stage_runtime_assets(
    *,
    stage_dir: Path,
    llama_server: Path,
    notes_model_4b: Path,
    notes_model_2b: Path,
    parakeet_model_dir: Path,
    canary_model_dir: Path,
) -> None:
    _copy_file(PROMPT_SOURCE_PATH, stage_dir / "prompts" / "clinical_note_synthesis_llm_prompt.md")
    _copy_file(llama_server, stage_dir / "runtime" / "llm" / "llama-server.exe")

    notes_specs = {spec.model_id: spec.relative_path for spec in bundled_notes_model_specs()}
    _copy_file(notes_model_4b, stage_dir / notes_specs[DEFAULT_SESSION_NOTES_MODEL])
    _copy_file(notes_model_2b, stage_dir / notes_specs[ALTERNATE_SESSION_NOTES_MODEL])

    transcription_specs = {spec.model_id: spec.relative_path for spec in bundled_transcription_model_specs()}
    _copy_tree(parakeet_model_dir, stage_dir / transcription_specs["nvidia/parakeet-tdt-0.6b-v3"])
    _copy_tree(canary_model_dir, stage_dir / transcription_specs["nvidia/canary-qwen-2.5b"])


def _build_inno_installer(*, inno_setup_exe: str, stage_dir: Path, installer_dir: Path) -> Path:
    installer_dir.mkdir(parents=True, exist_ok=True)
    command = [
        inno_setup_exe,
        f"/DSourceDir={stage_dir}",
        f"/DOutputDir={installer_dir}",
        str(INNO_SCRIPT_PATH),
    ]
    subprocess.run(command, check=True)
    installer_candidates = sorted(installer_dir.glob("*.exe"))
    if not installer_candidates:
        raise FileNotFoundError(f"No installer executable was created in {installer_dir}")
    return installer_candidates[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Windows standalone transcribe package")
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--installer-dir", type=Path, default=DEFAULT_INSTALLER_DIR)
    parser.add_argument("--pyinstaller-exe", default="pyinstaller")
    parser.add_argument("--inno-setup-exe", default="iscc")
    parser.add_argument("--skip-pyinstaller", action="store_true")
    parser.add_argument("--skip-installer", action="store_true")
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--notes-model-4b", type=Path, required=True)
    parser.add_argument("--notes-model-2b", type=Path, required=True)
    parser.add_argument("--parakeet-model-dir", type=Path, required=True)
    parser.add_argument("--canary-model-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    stage_dir = args.stage_dir.resolve()
    if not args.skip_pyinstaller:
        _build_pyinstaller_bundle(
            pyinstaller_exe=args.pyinstaller_exe,
            stage_dir=stage_dir,
            build_dir=args.build_dir.resolve(),
        )
    elif not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory does not exist for --skip-pyinstaller: {stage_dir}")

    _stage_runtime_assets(
        stage_dir=stage_dir,
        llama_server=args.llama_server.resolve(),
        notes_model_4b=args.notes_model_4b.resolve(),
        notes_model_2b=args.notes_model_2b.resolve(),
        parakeet_model_dir=args.parakeet_model_dir.resolve(),
        canary_model_dir=args.canary_model_dir.resolve(),
    )

    installer_path: Path | None = None
    if not args.skip_installer:
        installer_path = _build_inno_installer(
            inno_setup_exe=args.inno_setup_exe,
            stage_dir=stage_dir,
            installer_dir=args.installer_dir.resolve(),
        )

    print(f"Windows stage: {stage_dir}")
    if installer_path is not None:
        print(f"Installer: {installer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
