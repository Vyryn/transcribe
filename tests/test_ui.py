from __future__ import annotations

import importlib
import json
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

from transcribe.audio.runner import run_capture_session
from transcribe.live.session import LiveSessionConfig, LiveSessionResult, run_live_transcription_session
from transcribe.models import AudioSourceMode, CaptureConfig
from transcribe.ui.controller import UiTaskController
from transcribe.ui.types import SessionRequest, UiCommonOptions


def test_run_capture_session_can_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.audio.runner as runner_module

    cancel_event = threading.Event()

    class FakeBackend:
        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000
            self.dropped_callback_frames = 0
            self.calls = 0

        def open(self, config) -> None:
            _ = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            self.calls += 1
            if self.calls >= 2:
                cancel_event.set()
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            frame = struct.pack("<h", 1_500) * frame_samples
            now_ns = time.monotonic_ns()
            return {
                "mic": RawFrame("mic", frame, now_ns, self.sample_rate_hz),
                "speakers": RawFrame("speakers", frame, now_ns, self.sample_rate_hz),
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(runner_module, "open_audio_backend", lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture))
    config = CaptureConfig(source_mode=AudioSourceMode.BOTH, session_id="capture-stop", output_dir=tmp_path / "capture-stop")
    result = run_capture_session(config, duration_sec=5.0, use_fixture=False, cancel_event=cancel_event)

    assert result.interrupted is True
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_stats"]["interrupted"] is True
    assert manifest["capture_stats"]["pair_count"] >= 1


def test_run_live_transcription_session_can_cancel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from transcribe.audio.interfaces import RawFrame
    import transcribe.live.session as live_session_module

    cancel_event = threading.Event()

    class FakeBackend:
        def __init__(self, *, use_fixture: bool = False) -> None:
            _ = use_fixture
            self.sample_rate_hz = 16_000
            self.calls = 0

        def open(self, config) -> None:
            _ = config

        def read_frames(self, timeout_ms: int = 500) -> dict[str, RawFrame]:
            _ = timeout_ms
            self.calls += 1
            if self.calls >= 4:
                cancel_event.set()
            frame_samples = int((self.sample_rate_hz * 20) / 1000)
            voiced_frame = struct.pack("<h", 1_200) * frame_samples
            return {
                "mic": RawFrame(
                    stream="mic",
                    mono_pcm16=voiced_frame,
                    captured_at_monotonic_ns=time.monotonic_ns(),
                    sample_rate_hz=self.sample_rate_hz,
                )
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        live_session_module,
        "open_audio_backend",
        lambda *, use_fixture=False: FakeBackend(use_fixture=use_fixture),
    )
    config = LiveSessionConfig(
        transcription_model="unit-test-model",
        duration_sec=5.0,
        chunk_sec=0.06,
        partial_interval_sec=0.0,
        output_dir=tmp_path / "live-stop",
        session_id="live-stop",
    )
    result = run_live_transcription_session(
        config,
        use_fixture=False,
        transcriber=lambda wav_bytes, model_id: ("stopped line", 1.0),
        cancel_event=cancel_event,
    )

    assert result.interrupted is True
    payload = json.loads(result.transcript_json_path.read_text(encoding="utf-8"))
    assert payload["interrupted"] is True
    assert result.events_path.exists()


def test_ui_run_session_skips_notes_when_no_final_segments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transcribe.live.session as live_session_module
    import transcribe.notes as notes_module
    import transcribe.transcription_runtime as transcription_runtime
    import transcribe.ui.services as services_module

    progress_events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(services_module, "configure_runtime", lambda common: None)
    monkeypatch.setattr(services_module, "validate_transcription_model_for_runtime", lambda model: model)

    def fake_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None, cancel_event=None):
        _ = (config, use_fixture, debug, cancel_event)
        if progress_callback is not None:
            progress_callback("capture_ready", {"source_mode": "both"})
        return LiveSessionResult(
            session_dir=tmp_path / "live-empty",
            events_path=tmp_path / "live-empty" / "events.jsonl",
            transcript_json_path=tmp_path / "live-empty" / "transcript.json",
            transcript_txt_path=tmp_path / "live-empty" / "transcript.txt",
            final_segment_count=0,
            partial_event_count=0,
            sample_rate_hz=16_000,
            sample_rate_hz_requested=16_000,
            total_audio_sec=0.0,
            total_inference_sec=0.0,
            source_selection_counts={},
            interrupted=True,
        )

    def fail_notes(*args, **kwargs):
        raise AssertionError("notes should not run when there are no final segments")

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", fake_runner)
    monkeypatch.setattr(notes_module, "run_post_transcription_notes", fail_notes)
    monkeypatch.setattr(transcription_runtime, "release_transcription_runtime_resources", lambda transcription_model: 0)

    request = SessionRequest(
        common=UiCommonOptions(),
        transcription_model="unit-test-model",
        output_root=tmp_path,
        session_id="live-empty",
        notes_enabled=True,
    )
    result = services_module.run_session(request, progress_callback=lambda event, fields: progress_events.append((event, fields)))

    assert result.notes_summary is None
    assert ("notes_skipped", {"reason": "no_final_segments"}) in progress_events


def test_ui_networked_tasks_require_fresh_process_after_guard_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.ui.services as services_module

    monkeypatch.setattr(services_module, "outbound_network_guard_installed", lambda: True)

    with pytest.raises(RuntimeError, match="Restart transcribe-ui"):
        services_module.ensure_network_downloads_available("Benchmark cache initialization")


def test_ui_task_controller_emits_progress_result_and_finished() -> None:
    controller = UiTaskController()
    controller.start_task(
        "demo",
        lambda cancel, progress: (progress("step", {"value": 1}), "done")[1],
        cancelable=False,
    )

    deadline = time.monotonic() + 2.0
    messages = []
    while time.monotonic() < deadline:
        messages.extend(controller.drain_messages())
        if any(message.kind == "finished" for message in messages):
            break
        time.sleep(0.01)

    kinds = [message.kind for message in messages]
    assert kinds[0] == "started"
    assert "progress" in kinds
    assert "result" in kinds
    assert kinds[-1] == "finished"


def test_ui_task_controller_supports_cancellation() -> None:
    controller = UiTaskController()

    def runner(cancel_event, progress_callback):
        _ = progress_callback
        assert cancel_event is not None
        while not cancel_event.is_set():
            time.sleep(0.01)
        return "stopped"

    controller.start_task("cancelable", runner, cancelable=True)
    assert controller.cancel_active_task() is True

    deadline = time.monotonic() + 2.0
    messages = []
    while time.monotonic() < deadline:
        messages.extend(controller.drain_messages())
        if any(message.kind == "finished" for message in messages):
            break
        time.sleep(0.01)

    assert any(message.kind == "result" and message.payload == "stopped" for message in messages)


def test_ui_app_import_is_lazy() -> None:
    for module_name in (
        "transcribe.ui.app",
        "transcribe.live.session",
        "transcribe.notes",
        "transcribe.audio.windows_capture",
        "transcribe.audio.linux_capture",
        "transcribe.bench.harness",
    ):
        sys.modules.pop(module_name, None)

    app_module = importlib.import_module("transcribe.ui.app")

    assert app_module is not None
    assert "transcribe.live.session" not in sys.modules
    assert "transcribe.notes" not in sys.modules
    assert "transcribe.audio.windows_capture" not in sys.modules
    assert "transcribe.audio.linux_capture" not in sys.modules
    assert "transcribe.bench.harness" not in sys.modules


def test_ui_page_order_hides_bench_in_packaged_mode() -> None:
    app_module = importlib.import_module("transcribe.ui.app")

    assert "bench" in app_module.page_order(packaged_runtime=False)
    assert "bench" not in app_module.page_order(packaged_runtime=True)


def test_ui_smoke_instantiates_when_tk_available() -> None:
    app_module = importlib.import_module("transcribe.ui.app")
    try:
        root = app_module.tk.Tk()
    except app_module.tk.TclError:
        pytest.skip("Tk is unavailable in this environment")
    try:
        root.withdraw()
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        session_page = app.pages["session"]
        capture_page = app.pages["capture"]
        assert "session" in app.pages
        assert "logs" in app.pages
        assert app.log_level_combo.cget("values") == app_module.LOG_LEVEL_OPTIONS
        assert session_page.advanced_visible is False
        assert capture_page.advanced_visible is False
        app.advanced_ui_var.set(True)
        app._apply_advanced_ui_state()
        assert session_page.advanced_visible is True
        assert capture_page.advanced_visible is True
    finally:
        root.destroy()
