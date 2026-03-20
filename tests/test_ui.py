from __future__ import annotations

import importlib
import json
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcribe.audio.runner import run_capture_session
from transcribe.live.session import LiveSessionConfig, LiveSessionResult, run_live_transcription_session
from transcribe.models import AudioSourceMode, CaptureConfig
from transcribe.ui.controller import UiTaskController
from transcribe.ui.types import ComplianceResultSummary, NotesRequest, SessionRequest, UiCommonOptions


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


def test_ui_run_session_passes_notes_reasoning_to_notes_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transcribe.live.session as live_session_module
    import transcribe.notes as notes_module
    import transcribe.transcription_runtime as transcription_runtime
    import transcribe.ui.services as services_module

    observed: dict[str, object] = {}

    monkeypatch.setattr(services_module, "configure_runtime", lambda common: None)
    monkeypatch.setattr(services_module, "validate_transcription_model_for_runtime", lambda model: model)

    def fake_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None, cancel_event=None):
        _ = (config, use_fixture, debug, progress_callback, cancel_event)
        return LiveSessionResult(
            session_dir=tmp_path / "live-with-notes",
            events_path=tmp_path / "live-with-notes" / "events.jsonl",
            transcript_json_path=tmp_path / "live-with-notes" / "transcript.json",
            transcript_txt_path=tmp_path / "live-with-notes" / "transcript.txt",
            final_segment_count=1,
            partial_event_count=0,
            sample_rate_hz=16_000,
            sample_rate_hz_requested=16_000,
            total_audio_sec=1.0,
            total_inference_sec=0.1,
            source_selection_counts={"mic": 1},
            interrupted=True,
        )

    def fake_notes(config, *, progress_callback=None):
        _ = progress_callback
        observed["config"] = config
        return notes_module.SessionNotesResult(
            transcript_path=config.transcript_path,
            clean_transcript_path=config.output_dir / "clean_transcript.txt",
            client_notes_path=config.output_dir / "client_notes.txt",
            model=config.model,
            cpu_fallback_used=False,
            clean_duration_sec=0.1,
            notes_duration_sec=0.2,
        )

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", fake_runner)
    monkeypatch.setattr(notes_module, "run_post_transcription_notes", fake_notes)
    monkeypatch.setattr(transcription_runtime, "release_transcription_runtime_resources", lambda transcription_model: 0)

    request = SessionRequest(
        common=UiCommonOptions(),
        transcription_model="unit-test-model",
        output_root=tmp_path,
        session_id="live-with-notes",
        notes_enabled=True,
        notes_allow_reasoning=False,
        notes_max_output_tokens=1536,
    )

    result = services_module.run_session(request)

    assert result.notes_summary is not None
    assert observed["config"].allow_reasoning is False
    assert observed["config"].notes_max_output_tokens == 1536


def test_ui_run_notes_passes_max_output_tokens_to_notes_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transcribe.notes as notes_module
    import transcribe.ui.services as services_module

    observed: dict[str, object] = {}

    monkeypatch.setattr(services_module, "configure_runtime", lambda common: None)

    def fake_notes(config, *, progress_callback=None):
        _ = progress_callback
        observed["config"] = config
        return notes_module.SessionNotesResult(
            transcript_path=config.transcript_path,
            clean_transcript_path=config.output_dir / "clean_transcript.txt",
            client_notes_path=config.output_dir / "client_notes.txt",
            model=config.model,
            cpu_fallback_used=False,
            clean_duration_sec=0.1,
            notes_duration_sec=0.2,
        )

    monkeypatch.setattr(notes_module, "run_post_transcription_notes", fake_notes)

    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("hello\n", encoding="utf-8")
    result = services_module.run_notes(
        NotesRequest(
            common=UiCommonOptions(),
            transcript_path=transcript_path,
            output_dir=tmp_path,
            max_output_tokens=2048,
        )
    )

    assert result.model == observed["config"].model
    assert observed["config"].notes_max_output_tokens == 2048


def test_ui_networked_tasks_require_network_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.ui.services as services_module

    common = UiCommonOptions(allow_network=False)

    with pytest.raises(RuntimeError, match="Allow Network Access"):
        services_module.ensure_network_downloads_available("Benchmark cache initialization", common=common)


def test_transcription_model_options_include_granite_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.runtime_env as runtime_env
    import transcribe.ui.services as services_module

    monkeypatch.setattr(
        services_module,
        "resolve_app_runtime_paths",
        lambda: SimpleNamespace(mode=runtime_env.RuntimeMode.DEVELOPMENT, transcription_models={}),
    )

    options = services_module.transcription_model_options()

    assert runtime_env.PACKAGED_GRANITE_TRANSCRIPTION_MODEL in options



def test_ui_networked_tasks_require_fresh_process_after_guard_install(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.ui.services as services_module

    monkeypatch.setattr(services_module, "outbound_network_guard_installed", lambda: True)

    with pytest.raises(RuntimeError, match="Restart transcribe-ui"):
        services_module.ensure_network_downloads_available(
            "Benchmark cache initialization",
            common=UiCommonOptions(allow_network=True),
        )


def test_configure_runtime_skips_guard_when_network_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.ui.services as services_module

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        services_module,
        "load_app_config",
        lambda config_path=None, overrides=None: SimpleNamespace(
            log_level=(overrides or {}).get("log_level", "ERROR"),
            redact_logs=False,
            offline_only=True,
        ),
    )
    monkeypatch.setattr(services_module, "configure_logging", lambda log_level, redact_logs=False: observed.setdefault("log_level", log_level))
    monkeypatch.setattr(services_module, "set_network_access_allowed", lambda allowed: observed.setdefault("allow_network", allowed))
    monkeypatch.setattr(services_module, "install_outbound_network_guard", lambda: observed.setdefault("guard_installed", True))

    services_module.configure_runtime(UiCommonOptions(allow_network=True))

    assert observed["allow_network"] is True
    assert observed["log_level"] == "ERROR"
    assert "guard_installed" not in observed


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


def test_ui_preferences_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    preferences_module = importlib.import_module("transcribe.ui.preferences")
    prefs_path = tmp_path / "ui-preferences.json"
    monkeypatch.setattr(preferences_module, "preferences_path", lambda: prefs_path)

    preferences_module.save_ui_preferences(
        preferences_module.UiPreferences(advanced_ui=False, allow_network=True)
    )
    loaded = preferences_module.load_ui_preferences()

    assert loaded.allow_network is True
    assert loaded.advanced_ui is True


def _skip_tk_unavailable(error: Exception) -> None:
    detail = str(error).strip()
    if detail:
        pytest.skip(f"Tk is unavailable in this environment: {detail}")
    pytest.skip("Tk is unavailable in this environment")


@pytest.fixture(scope="module")
def tk_host_root() -> tuple[object, object]:
    app_module = importlib.import_module("transcribe.ui.app")
    try:
        root = app_module.tk.Tk()
    except app_module.tk.TclError as exc:
        _skip_tk_unavailable(exc)
    root.withdraw()
    try:
        yield app_module, root
    finally:
        if root.winfo_exists():
            root.destroy()


def _create_ui_test_window(app_module: object, host_root: object) -> object:
    try:
        root = app_module.tk.Toplevel(host_root)
    except app_module.tk.TclError as exc:
        _skip_tk_unavailable(exc)
    root.withdraw()
    return root


def _close_ui_app(app: object | None, root: object) -> None:
    if app is not None and hasattr(app, "_on_close"):
        app._on_close()
        return
    root.destroy()


def test_ui_loads_persisted_network_preference(
    monkeypatch: pytest.MonkeyPatch, tk_host_root: tuple[object, object]
) -> None:
    app_module, host_root = tk_host_root
    observed: list[bool] = []
    monkeypatch.setattr(
        app_module,
        "load_ui_preferences",
        lambda: app_module.UiPreferences(advanced_ui=True, allow_network=True),
    )
    monkeypatch.setattr(app_module, "set_network_access_allowed", lambda allowed: observed.append(allowed))
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)

        assert app.advanced_ui_var.get() is True
        assert app.allow_network_var.get() is True
        assert observed[-1] is True
    finally:
        _close_ui_app(app, root)


def test_ui_persists_network_toggle(
    monkeypatch: pytest.MonkeyPatch, tk_host_root: tuple[object, object]
) -> None:
    app_module, host_root = tk_host_root
    saved: list[object] = []
    observed_network: list[bool] = []
    monkeypatch.setattr(app_module, "load_ui_preferences", lambda: app_module.UiPreferences())
    monkeypatch.setattr(app_module, "save_ui_preferences", lambda preferences: saved.append(preferences))
    monkeypatch.setattr(app_module, "set_network_access_allowed", lambda allowed: observed_network.append(allowed))
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        app.advanced_ui_var.set(True)
        app._handle_advanced_ui_toggle()
        app.allow_network_var.set(True)
        app._handle_allow_network_toggle()

        assert saved
        assert saved[-1].advanced_ui is True
        assert saved[-1].allow_network is True
        assert observed_network[-1] is True
    finally:
        _close_ui_app(app, root)


def test_ui_models_result_handler_failure_does_not_freeze_polling(
    monkeypatch: pytest.MonkeyPatch, tk_host_root: tuple[object, object]
) -> None:
    app_module, host_root = tk_host_root
    controller_module = importlib.import_module("transcribe.ui.controller")
    captured_errors: list[tuple[str, str]] = []
    scheduled: list[tuple[int, object]] = []
    monkeypatch.setattr(app_module, "load_ui_preferences", lambda: app_module.UiPreferences())
    monkeypatch.setattr(app_module.messagebox, "showerror", lambda title, body: captured_errors.append((title, body)))
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        monkeypatch.setattr(root, "after", lambda delay, callback: scheduled.append((delay, callback)))
        models_page = app.pages["models"]
        app._active_binding = app_module.TaskBinding(
            task_name="models-install",
            on_result=models_page.handle_install,
        )
        app._set_busy(True, "models-install", cancelable=False)
        app.controller._messages.put(
            controller_module.ControllerMessage(kind="result", task_name="models-install", payload=None)
        )
        app.controller._messages.put(
            controller_module.ControllerMessage(kind="finished", task_name="models-install")
        )

        app._poll_messages()

        assert captured_errors
        assert "models-install failed while processing its result update" in captured_errors[0][1]
        assert app.busy_var.get() == "Idle"
        assert app._active_binding is None
        assert scheduled and scheduled[-1][0] == app_module.POLL_INTERVAL_MS
    finally:
        _close_ui_app(app, root)


def test_compliance_page_shows_informative_failure(
    monkeypatch: pytest.MonkeyPatch, tk_host_root: tuple[object, object]
) -> None:
    app_module, host_root = tk_host_root
    monkeypatch.setattr(app_module, "load_ui_preferences", lambda: app_module.UiPreferences())
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        page = app.pages["compliance"]
        assert isinstance(page, app_module.CompliancePage)

        page.handle_result(
            ComplianceResultSummary(
                name="check-no-network",
                exit_code=1,
                passed=False,
                summary="Outbound network is currently allowed in this process.",
                details=(
                    "Observed outbound_blocked=False, loopback_allowed=True.",
                    "If you enabled network in the UI, disable 'Allow Network Access' and restart before rerunning this check.",
                ),
            )
        )
        rendered = page.text.get("1.0", "end-1c")

        assert "Outbound network is currently allowed in this process." in rendered
        assert "disable 'Allow Network Access' and restart" in rendered
        assert page.status_var.get().endswith("Outbound network is currently allowed in this process.")
    finally:
        _close_ui_app(app, root)


def test_ui_smoke_instantiates_when_tk_available(
    monkeypatch: pytest.MonkeyPatch, tk_host_root: tuple[object, object]
) -> None:
    app_module, host_root = tk_host_root
    monkeypatch.setattr(app_module, "load_ui_preferences", lambda: app_module.UiPreferences())
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        session_page = app.pages["session"]
        capture_page = app.pages["capture"]
        notes_page = app.pages["notes"]
        assert "session" in app.pages
        assert "logs" in app.pages
        assert app.log_level_combo.cget("values") == app_module.LOG_LEVEL_OPTIONS
        assert app.allow_network_var.get() is False
        assert not hasattr(app, "debug_var")
        assert session_page.start_button.cget("style") == "Primary.TButton"
        assert session_page.stop_button.cget("style") == "Danger.TButton"
        assert hasattr(session_page, "transcription_model_combo")
        assert hasattr(session_page, "notes_model_combo")
        assert hasattr(session_page, "notes_reasoning_var")
        assert hasattr(session_page, "notes_max_output_tokens_var")
        assert hasattr(notes_page, "model_combo")
        assert hasattr(notes_page, "max_output_tokens_var")
        assert session_page.transcription_model_combo.cget("state") == "readonly"
        assert session_page.notes_model_combo.cget("state") == "readonly"
        assert session_page.notes_reasoning_var.get() is False
        assert session_page.notes_max_output_tokens_var.get() == ""
        assert notes_page.model_combo.cget("state") == "readonly"
        assert notes_page.max_output_tokens_var.get() == ""
        assert hasattr(session_page, "advanced_scrollbar")
        assert session_page.advanced_visible is False
        assert capture_page.advanced_visible is False
        assert notes_page.advanced_visible is False
        app.advanced_ui_var.set(True)
        app.allow_network_var.set(True)
        app._apply_advanced_ui_state()
        assert session_page.advanced_visible is True
        assert capture_page.advanced_visible is True
        assert notes_page.advanced_visible is True
        assert app.common_options().allow_network is True
    finally:
        _close_ui_app(app, root)


def test_session_page_start_passes_notes_reasoning_flag(
    monkeypatch: pytest.MonkeyPatch,
    tk_host_root: tuple[object, object],
    tmp_path: Path,
) -> None:
    app_module, host_root = tk_host_root
    monkeypatch.setattr(app_module, "load_ui_preferences", lambda: app_module.UiPreferences())
    observed: dict[str, object] = {}
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        session_page = app.pages["session"]
        assert isinstance(session_page, app_module.SessionPage)
        session_page.output_root_var.set(str(tmp_path))
        session_page.session_id_var.set("ui-live")
        session_page.notes_reasoning_var.set(False)
        session_page.notes_max_output_tokens_var.set("1536")

        monkeypatch.setattr(
            app_module.services,
            "run_session",
            lambda request, cancel_event=None, progress_callback=None: observed.setdefault("request", request),
        )

        def fake_start_task(task_name, runner, **kwargs):
            _ = kwargs
            observed["task_name"] = task_name
            runner(None, None)

        monkeypatch.setattr(app, "start_task", fake_start_task)

        session_page.start()

        assert observed["task_name"] == "session"
        assert observed["request"].notes_allow_reasoning is False
        assert observed["request"].notes_max_output_tokens == 1536
    finally:
        _close_ui_app(app, root)


def test_notes_page_start_passes_max_output_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tk_host_root: tuple[object, object],
    tmp_path: Path,
) -> None:
    app_module, host_root = tk_host_root
    monkeypatch.setattr(app_module, "load_ui_preferences", lambda: app_module.UiPreferences())
    observed: dict[str, object] = {}
    app = None
    root = _create_ui_test_window(app_module, host_root)
    try:
        app = app_module.TranscribeUiApp(root, packaged_runtime=False)
        notes_page = app.pages["notes"]
        assert isinstance(notes_page, app_module.NotesPage)
        transcript_path = tmp_path / "transcript.txt"
        transcript_path.write_text("hello\n", encoding="utf-8")
        notes_page.transcript_var.set(str(transcript_path))
        notes_page.output_dir_var.set(str(tmp_path))
        notes_page.max_output_tokens_var.set("2048")

        monkeypatch.setattr(
            app_module.services,
            "run_notes",
            lambda request, progress_callback=None: observed.setdefault("request", request),
        )

        def fake_start_task(task_name, runner, **kwargs):
            _ = kwargs
            observed["task_name"] = task_name
            runner(None, None)

        monkeypatch.setattr(app, "start_task", fake_start_task)

        notes_page.start()

        assert observed["task_name"] == "notes"
        assert observed["request"].max_output_tokens == 2048
    finally:
        _close_ui_app(app, root)
