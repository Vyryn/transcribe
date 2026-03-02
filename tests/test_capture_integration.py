from __future__ import annotations

import json
from pathlib import Path

from transcribe.audio.runner import run_capture_session
from transcribe.models import AudioSourceMode, CaptureConfig


def test_run_capture_session_fixture_writes_artifacts(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    config = CaptureConfig(
        source_mode=AudioSourceMode.BOTH,
        session_id="fixture-session",
        output_dir=session_dir,
    )
    result = run_capture_session(config, duration_sec=0.25, use_fixture=True)

    manifest_path = result.manifest_path
    mic_path = session_dir / "mic.wav"
    speakers_path = session_dir / "speakers.wav"

    assert manifest_path.exists()
    assert mic_path.exists()
    assert speakers_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_classification"] == "phi"
    assert manifest["contains_phi"] is True
    assert manifest["capture_stats"]["pair_count"] > 0
