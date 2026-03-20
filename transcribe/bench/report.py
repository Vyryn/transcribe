from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transcribe.utils.stats import percentile


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    """Return one float-like value or a fallback."""
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_int(value: object, *, default: int = 0) -> int:
    """Return one int-like value or a fallback."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _build_capture_sync_summary(run_results: list[dict[str, object]]) -> dict[str, object]:
    """Build default capture-sync summary metrics."""
    latency_p50_values = [
        _coerce_float(run["callback_to_write_latency_ms_p50"])
        for run in run_results
        if "callback_to_write_latency_ms_p50" in run
    ]
    latency_p95_values = [
        _coerce_float(run["callback_to_write_latency_ms_p95"])
        for run in run_results
        if "callback_to_write_latency_ms_p95" in run
    ]
    drift_avg_values = [_coerce_float(run["drift_ns_avg"]) for run in run_results if "drift_ns_avg" in run]

    return {
        "run_count": len(run_results),
        "callback_to_write_latency_ms_p50": percentile(latency_p50_values, 0.5),
        "callback_to_write_latency_ms_p95": percentile(latency_p95_values, 0.95),
        "drift_ns_avg": percentile(drift_avg_values, 0.5),
        "max_pair_count": max((_coerce_int(run.get("pair_count", 0)) for run in run_results), default=0),
        "total_dropped_pairs": sum(_coerce_int(run.get("dropped_pairs", 0)) for run in run_results),
    }


def build_benchmark_report(
    *,
    scenario: str,
    run_results: list[dict[str, object]],
    summary: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build an aggregate benchmark report document.

    Parameters
    ----------
    scenario : str
        Benchmark scenario name.
    run_results : list[dict[str, object]]
        Per-run metric dictionaries.
    summary : dict[str, object] | None, optional
        Optional scenario-specific summary payload.

    Returns
    -------
    dict[str, object]
        Aggregated report payload.
    """
    resolved_summary = summary if summary is not None else _build_capture_sync_summary(run_results)
    return {
        "schema_version": "phase0-benchmark-v1",
        "scenario": scenario,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": run_results,
        "summary": resolved_summary,
    }


def _format_summary_value(value: Any) -> str:
    """Format summary values for stable Markdown output."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_benchmark_report(
    report: dict[str, object],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write benchmark report files in JSON and Markdown formats.

    Parameters
    ----------
    report : dict[str, object]
        Benchmark report payload.
    output_dir : Path
        Output directory for report files.

    Returns
    -------
    tuple[Path, Path]
        Paths to JSON and Markdown report files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark_report.json"
    md_path = output_dir / "benchmark_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")

    summary = report["summary"]
    if not isinstance(summary, dict):
        raise ValueError("Benchmark report summary must be a JSON object.")
    summary_map = dict(summary)
    is_capture_sync_summary = all(
        key in summary_map
        for key in (
            "callback_to_write_latency_ms_p50",
            "callback_to_write_latency_ms_p95",
            "drift_ns_avg",
            "total_dropped_pairs",
        )
    )
    md_lines = [
        "# Benchmark Report",
        "",
        f"- Scenario: `{report['scenario']}`",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
    ]

    if is_capture_sync_summary:
        md_lines.extend(
            [
                f"- Run count: `{_coerce_int(summary_map.get('run_count'))}`",
                f"- Callback->Write Latency p50 (ms): `{_coerce_float(summary_map.get('callback_to_write_latency_ms_p50')):.3f}`",
                f"- Callback->Write Latency p95 (ms): `{_coerce_float(summary_map.get('callback_to_write_latency_ms_p95')):.3f}`",
                f"- Drift avg (ns): `{_coerce_float(summary_map.get('drift_ns_avg')):.3f}`",
                f"- Total dropped pairs: `{_coerce_int(summary_map.get('total_dropped_pairs'))}`",
            ]
        )
    else:
        for key, value in summary_map.items():
            label = key.replace("_", " ").title()
            md_lines.append(f"- {label}: `{_format_summary_value(value)}`")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path
