from __future__ import annotations

import json
from pathlib import Path

from transcribe.bench.harness import run_capture_sync_benchmark
from transcribe.models import AudioSourceMode, CaptureConfig


def test_run_capture_sync_benchmark_fixture(tmp_path: Path) -> None:
    base_config = CaptureConfig(source_mode=AudioSourceMode.BOTH, session_id="bench", output_dir=tmp_path)
    result = run_capture_sync_benchmark(
        base_config=base_config,
        runs=2,
        duration_sec=0.2,
        output_dir=tmp_path / "bench",
        use_fixture=True,
    )

    assert result.json_path.exists()
    assert result.markdown_path.exists()

    report = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert report["summary"]["run_count"] == 2
    assert len(report["runs"]) == 2
