from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from transcribe.audio.runner import run_capture_session, with_session_id
from transcribe.bench.report import build_benchmark_report, write_benchmark_report
from transcribe.models import CaptureConfig


class BenchmarkResult:
    """Container for benchmark report artifacts."""

    def __init__(self, report: dict[str, object], json_path: Path, markdown_path: Path) -> None:
        """Initialize benchmark result metadata."""
        self.report = report
        self.json_path = json_path
        self.markdown_path = markdown_path


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
