from __future__ import annotations

import os
import subprocess
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from transcribe.models import AudioSourceMode
from transcribe.runtime_env import (
    RuntimeMode,
    detect_runtime_mode,
    resolve_app_runtime_paths,
    set_network_access_allowed,
)
from transcribe.ui import services
from transcribe.ui.controller import ControllerMessage, UiTaskController
from transcribe.ui.preferences import UiPreferences, load_ui_preferences, save_ui_preferences
from transcribe.ui.services import default_data_subdir
from transcribe.ui.types import (
    BenchmarkInitRequest,
    BenchmarkInitResultSummary,
    BenchmarkRunRequest,
    BenchmarkRunResultSummary,
    CaptureRequest,
    CaptureResultSummary,
    ComplianceResultSummary,
    DeviceInfo,
    DeviceListResult,
    ModelsInstallRequest,
    ModelsInstallResultSummary,
    ModelsListResult,
    NotesRequest,
    NotesResultSummary,
    ServiceProgressEvent,
    SessionRequest,
    SessionResultSummary,
    UiCommonOptions,
)

MAX_LOG_LINES = 2_000
POLL_INTERVAL_MS = 75
LOG_LEVEL_OPTIONS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class StableCombobox(ttk.Combobox):
    """Combobox variant that normalizes Tk state values into plain strings."""

    def cget(self, key: str) -> object:
        value = super().cget(key)
        if key == "state":
            return str(value)
        return value


@dataclass(slots=True)
class TaskBinding:
    """UI handlers bound to the currently active background task."""

    task_name: str
    on_result: Callable[[object], None] | None = None
    on_progress: Callable[[ServiceProgressEvent], None] | None = None
    on_error: Callable[[BaseException], None] | None = None
    cancelable: bool = False


class BasePage(ttk.Frame):
    """Base frame for workflow-specific UI pages."""

    title = ""

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, padding=16)
        self.app = app

    def set_busy(self, busy: bool, *, cancelable: bool) -> None:
        """Update action availability while background work is running."""

    def set_devices(self, devices: tuple[DeviceInfo, ...]) -> None:
        """Update audio-device widgets when the device list changes."""

    def set_advanced_mode(self, advanced: bool) -> None:
        """Show or hide advanced controls for this page."""


class LogsPage(BasePage):
    """Global application log viewer."""

    title = "Logs"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.text = ScrolledText(self, wrap="word", height=30)
        self.text.grid(row=0, column=0, sticky="nsew")
        self.text.configure(state="disabled")

    def replace_lines(self, lines: list[str]) -> None:
        """Replace the log view contents with the retained log buffer."""
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, "\n".join(lines) + ("\n" if lines else ""))
        self.text.see(tk.END)
        self.text.configure(state="disabled")


class SessionPage(BasePage):
    """Live transcription session controls and transcript display."""

    title = "Session"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)

        self.transcription_model_var = tk.StringVar(value=services.DEFAULT_LIVE_TRANSCRIPTION_MODEL)
        self.duration_var = tk.StringVar(value="0")
        self.chunk_var = tk.StringVar(value="6.0")
        self.mode_var = tk.StringVar(value=AudioSourceMode.BOTH.value)
        self.output_root_var = tk.StringVar(value=str(default_data_subdir("live_sessions")))
        self.session_id_var = tk.StringVar(value="")
        self.mic_device_var = tk.StringVar(value="Auto")
        self.speaker_device_var = tk.StringVar(value="Auto")
        self.partial_interval_var = tk.StringVar(value="0.0")
        self.chunk_overlap_var = tk.StringVar(value="1.0")
        self.max_model_ram_var = tk.StringVar(value="8.0")
        self.stitch_overlap_var = tk.BooleanVar(value=True)
        self.single_device_var = tk.BooleanVar(value=False)
        self.strict_sources_var = tk.BooleanVar(value=False)
        self.fixture_var = tk.BooleanVar(value=False)
        self.notes_enabled_var = tk.BooleanVar(value=True)
        self.notes_model_var = tk.StringVar(value=services.DEFAULT_SESSION_NOTES_MODEL)
        self.notes_runtime_var = tk.StringVar(value=services.DEFAULT_NOTES_RUNTIME)
        self.transcription_model_choices = services.transcription_model_options()
        self.notes_model_choices = services.notes_model_options()
        self.partial_text_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self.session_dir_var = tk.StringVar(value="")
        self.transcript_path_var = tk.StringVar(value="")
        self.notes_path_var = tk.StringVar(value="")
        self.device_labels: tuple[str, ...] = ("Auto",)
        self.advanced_visible = False
        self._advanced_window_id: int | None = None
        self._notes_progress_active = False
        self._notes_stream_started = False

        controls = ttk.LabelFrame(self, text="Live Session")
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for index in range(3):
            controls.columnconfigure(index, weight=1)

        actions = ttk.Frame(controls)
        actions.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(
            actions,
            text="Run notes after transcription stops",
            variable=self.notes_enabled_var,
        ).pack(side="left")
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self.stop,
            state="disabled",
            style="Danger.TButton",
            width=12,
        )
        self.stop_button.pack(side="right", padx=(8, 0))
        self.start_button = ttk.Button(
            actions,
            text="Start Session",
            command=self.start,
            style="Primary.TButton",
            width=16,
        )
        self.start_button.pack(side="right")

        self.advanced_frame = ttk.LabelFrame(self, text="Advanced Session Options")
        self.advanced_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.advanced_frame.columnconfigure(0, weight=1)
        self.advanced_frame.rowconfigure(0, weight=1)
        self.advanced_canvas = tk.Canvas(self.advanced_frame, highlightthickness=0, borderwidth=0, height=190)
        self.advanced_canvas.grid(row=0, column=0, sticky="nsew")
        self.advanced_scrollbar = ttk.Scrollbar(
            self.advanced_frame,
            orient="vertical",
            command=self.advanced_canvas.yview,
        )
        self.advanced_scrollbar.grid(row=0, column=1, sticky="ns")
        self.advanced_canvas.configure(yscrollcommand=self.advanced_scrollbar.set)
        self.advanced_content = ttk.Frame(self.advanced_canvas)
        for index in range(4):
            self.advanced_content.columnconfigure(index, weight=1)
        self._advanced_window_id = self.advanced_canvas.create_window(
            (0, 0),
            window=self.advanced_content,
            anchor="nw",
        )
        self.advanced_content.bind("<Configure>", self._refresh_advanced_scroll_region)
        self.advanced_canvas.bind("<Configure>", self._resize_advanced_content)

        ttk.Label(self.advanced_content, text="Voice Model").grid(row=0, column=0, sticky="w")
        self.transcription_model_combo = StableCombobox(
            self.advanced_content,
            textvariable=self.transcription_model_var,
            values=self.transcription_model_choices,
            state="readonly",
        )
        self.transcription_model_combo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Label(self.advanced_content, text="Notes Model").grid(row=0, column=2, sticky="w")
        self.notes_model_combo = StableCombobox(
            self.advanced_content,
            textvariable=self.notes_model_var,
            values=self.notes_model_choices,
            state="readonly",
        )
        self.notes_model_combo.grid(row=1, column=2, columnspan=2, sticky="ew", pady=(0, 6))

        ttk.Label(self.advanced_content, text="Duration (sec)").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.duration_var).grid(
            row=3, column=0, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Label(self.advanced_content, text="Audio Source").grid(row=2, column=1, sticky="w")
        StableCombobox(
            self.advanced_content,
            textvariable=self.mode_var,
            values=[mode.value for mode in AudioSourceMode],
            state="readonly",
        ).grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Label(self.advanced_content, text="Notes Runtime").grid(row=2, column=2, sticky="w")
        StableCombobox(
            self.advanced_content,
            textvariable=self.notes_runtime_var,
            values=["auto", "ollama", "llama_cpp"],
            state="readonly",
        ).grid(row=3, column=2, sticky="ew", padx=(0, 6), pady=(0, 6))
        self.refresh_button = ttk.Button(self.advanced_content, text="Refresh Devices", command=self.refresh_devices)
        self.refresh_button.grid(row=3, column=3, sticky="ew", pady=(0, 6))

        ttk.Label(self.advanced_content, text="Mic Device").grid(row=4, column=0, sticky="w")
        self.mic_combo = StableCombobox(
            self.advanced_content,
            textvariable=self.mic_device_var,
            values=self.device_labels,
            state="readonly",
        )
        self.mic_combo.grid(row=5, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Label(self.advanced_content, text="Speaker Device").grid(row=4, column=1, sticky="w")
        self.speaker_combo = StableCombobox(
            self.advanced_content,
            textvariable=self.speaker_device_var,
            values=self.device_labels,
            state="readonly",
        )
        self.speaker_combo.grid(row=5, column=1, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Label(self.advanced_content, text="Output Root").grid(row=4, column=2, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.output_root_var).grid(
            row=5, column=2, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Button(
            self.advanced_content,
            text="Browse",
            command=lambda: self.app.choose_directory(self.output_root_var),
        ).grid(row=5, column=3, sticky="ew", pady=(0, 6))

        ttk.Label(self.advanced_content, text="Session Id").grid(row=6, column=0, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.session_id_var).grid(
            row=7, column=0, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Label(self.advanced_content, text="Chunk (sec)").grid(row=6, column=1, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.chunk_var).grid(
            row=7, column=1, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Label(self.advanced_content, text="Partial Interval").grid(row=6, column=2, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.partial_interval_var).grid(
            row=7, column=2, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Label(self.advanced_content, text="Chunk Overlap").grid(row=6, column=3, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.chunk_overlap_var).grid(
            row=7, column=3, sticky="ew", pady=(0, 6)
        )
        ttk.Label(self.advanced_content, text="Max Model RAM").grid(row=8, column=0, sticky="w")
        ttk.Entry(self.advanced_content, textvariable=self.max_model_ram_var).grid(
            row=9, column=0, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Checkbutton(self.advanced_content, text="Fixture", variable=self.fixture_var).grid(
            row=9, column=1, sticky="w", pady=(0, 6)
        )
        ttk.Checkbutton(self.advanced_content, text="Single Device/Source", variable=self.single_device_var).grid(
            row=9, column=2, sticky="w", pady=(0, 6)
        )
        ttk.Checkbutton(self.advanced_content, text="Strict Sources", variable=self.strict_sources_var).grid(
            row=9, column=3, sticky="w", pady=(0, 6)
        )
        ttk.Checkbutton(self.advanced_content, text="Stitch Overlap", variable=self.stitch_overlap_var).grid(
            row=10, column=0, sticky="w", pady=(4, 0)
        )

        devices = ttk.LabelFrame(self, text="Available Devices")
        devices.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(0, 12))
        devices.columnconfigure(0, weight=1)
        devices.rowconfigure(0, weight=1)
        self.device_list = tk.Listbox(devices, height=8)
        self.device_list.grid(row=0, column=0, sticky="nsew")

        outputs = ttk.LabelFrame(self, text="Outputs")
        outputs.grid(row=2, column=1, sticky="nsew", pady=(0, 12))
        outputs.columnconfigure(1, weight=1)
        ttk.Label(outputs, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Label(outputs, textvariable=self.status_var).grid(row=0, column=1, sticky="w")
        ttk.Label(outputs, text="Session Dir").grid(row=1, column=0, sticky="w")
        ttk.Entry(outputs, textvariable=self.session_dir_var).grid(row=1, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(outputs, text="Open", command=lambda: self.app.open_path_var(self.session_dir_var)).grid(
            row=1, column=2, sticky="ew"
        )
        ttk.Label(outputs, text="Transcript").grid(row=2, column=0, sticky="w")
        ttk.Entry(outputs, textvariable=self.transcript_path_var).grid(row=2, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(outputs, text="Open", command=lambda: self.app.open_path_var(self.transcript_path_var)).grid(
            row=2, column=2, sticky="ew"
        )
        ttk.Label(outputs, text="Notes").grid(row=3, column=0, sticky="w")
        ttk.Entry(outputs, textvariable=self.notes_path_var).grid(row=3, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(outputs, text="Open", command=lambda: self.app.open_path_var(self.notes_path_var)).grid(
            row=3, column=2, sticky="ew"
        )
        ttk.Label(outputs, text="Current Partial").grid(row=4, column=0, sticky="nw")
        ttk.Label(outputs, textvariable=self.partial_text_var, wraplength=420, justify="left").grid(
            row=4, column=1, columnspan=2, sticky="w"
        )

        self.transcript_frame = ttk.LabelFrame(self, text="Final Transcript")
        self.transcript_frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.transcript_frame.columnconfigure(0, weight=1)
        self.transcript_frame.rowconfigure(0, weight=1)
        self.transcript_text = ScrolledText(self.transcript_frame, wrap="word", height=16)
        self.transcript_text.grid(row=0, column=0, sticky="nsew")
        self.transcript_text.configure(state="disabled")
        self.set_advanced_mode(False)

    def _set_transcript_frame_title(self, title: str) -> None:
        """Update the session output panel title."""
        self.transcript_frame.configure(text=title)

    def _replace_transcript_text(self, text: str) -> None:
        """Replace the session output text widget contents."""
        self.transcript_text.configure(state="normal")
        self.transcript_text.delete("1.0", tk.END)
        if text:
            self.transcript_text.insert(tk.END, text)
        self.transcript_text.see(tk.END)
        self.transcript_text.configure(state="disabled")

    def _append_transcript_text(self, text: str) -> None:
        """Append text to the session output text widget."""
        if not text:
            return
        self.transcript_text.configure(state="normal")
        self.transcript_text.insert(tk.END, text)
        self.transcript_text.see(tk.END)
        self.transcript_text.configure(state="disabled")

    def _append_transcript_line(self, text: str = "") -> None:
        """Append one line to the session output text widget."""
        self._append_transcript_text(f"{text}\n")

    def _begin_notes_progress_view(self) -> None:
        """Switch the transcript area into a notes-progress console."""
        if self._notes_progress_active:
            return
        self._notes_progress_active = True
        self._notes_stream_started = False
        self._set_transcript_frame_title("Notes Progress")
        self._replace_transcript_text("Preparing notes model...\n")

    def _prepare_launch_paths(self) -> tuple[Path, str, Path]:
        output_root_text = self.output_root_var.get().strip()
        output_root = Path(output_root_text) if output_root_text else default_data_subdir("live_sessions")
        session_id = self.session_id_var.get().strip() or services.default_session_id("live")
        session_dir = output_root / session_id
        self.output_root_var.set(str(output_root))
        self.session_id_var.set(session_id)
        self.session_dir_var.set(str(session_dir))
        self.transcript_path_var.set(str(session_dir / "transcript.txt"))
        if self.notes_enabled_var.get():
            self.notes_path_var.set(str(session_dir / "client_notes.txt"))
        else:
            self.notes_path_var.set("")
        return output_root, session_id, session_dir

    def set_devices(self, devices: tuple[DeviceInfo, ...]) -> None:
        self.device_labels = ("Auto", *[device.label for device in devices])
        self.mic_combo.configure(values=self.device_labels)
        self.speaker_combo.configure(values=self.device_labels)
        self.device_list.delete(0, tk.END)
        for device in devices:
            self.device_list.insert(tk.END, device.label)

    def set_advanced_mode(self, advanced: bool) -> None:
        self.advanced_visible = advanced
        if advanced:
            self.advanced_frame.grid()
            self.after_idle(self._refresh_advanced_scroll_region)
        else:
            self.advanced_frame.grid_remove()

    def _refresh_advanced_scroll_region(self, _event: tk.Event | None = None) -> None:
        """Recompute the scrollable region for advanced session controls."""
        self.advanced_canvas.configure(scrollregion=self.advanced_canvas.bbox("all"))

    def _resize_advanced_content(self, event: tk.Event) -> None:
        """Keep the embedded advanced-options frame width aligned with the canvas."""
        if self._advanced_window_id is not None:
            self.advanced_canvas.itemconfigure(self._advanced_window_id, width=event.width)

    def refresh_devices(self) -> None:
        self.app.start_task(
            "session-devices",
            lambda cancel, progress: services.list_devices(common=self.app.common_options()),
            on_result=self._handle_devices_result,
        )

    def _handle_devices_result(self, result: object) -> None:
        assert isinstance(result, DeviceListResult)
        self.app.set_devices(result.devices)
        self.status_var.set(f"Loaded {len(result.devices)} devices")

    def start(self) -> None:
        self._notes_progress_active = False
        self._notes_stream_started = False
        self._set_transcript_frame_title("Final Transcript")
        self._replace_transcript_text("")
        self.partial_text_var.set("")
        output_root, session_id, _ = self._prepare_launch_paths()
        self.status_var.set("Starting session...")
        request = SessionRequest(
            common=self.app.common_options(),
            transcription_model=self.transcription_model_var.get().strip(),
            duration_sec=float(self.duration_var.get() or 0.0),
            chunk_overlap_sec=float(self.chunk_overlap_var.get() or 1.0),
            stitch_overlap_text=self.stitch_overlap_var.get(),
            mode=AudioSourceMode(self.mode_var.get()),
            chunk_sec=float(self.chunk_var.get() or 6.0),
            partial_interval_sec=float(self.partial_interval_var.get() or 0.0),
            output_root=output_root,
            session_id=session_id,
            mic_device=self.mic_device_var.get(),
            speaker_device=self.speaker_device_var.get(),
            single_device_per_source=self.single_device_var.get(),
            strict_sources=self.strict_sources_var.get(),
            use_fixture=self.fixture_var.get(),
            max_model_ram_gb=float(self.max_model_ram_var.get() or 8.0),
            notes_enabled=self.notes_enabled_var.get(),
            notes_model=self.notes_model_var.get().strip(),
            notes_runtime=self.notes_runtime_var.get().strip(),
        )
        self.app.start_task(
            "session",
            lambda cancel, progress: services.run_session(request, cancel_event=cancel, progress_callback=progress),
            cancelable=True,
            on_progress=self.handle_progress,
            on_result=self.handle_result,
        )

    def stop(self) -> None:
        if self.app.cancel_active_task():
            self.status_var.set("Stopping session...")

    def handle_progress(self, progress: ServiceProgressEvent) -> None:
        name = progress.name
        fields = progress.fields
        if name == "capture_ready":
            self.status_var.set("Capture ready")
        elif name == "loading_model":
            self.status_var.set(f"Loading model: {fields.get('transcription_model', '')}")
        elif name == "model_ready":
            self.status_var.set("Model ready")
        elif name == "transcribing_started":
            self.status_var.set("Listening and transcribing")
        elif name == "partial":
            self.partial_text_var.set(str(fields.get("text", "")).strip())
        elif name == "final":
            text = str(fields.get("text", "")).strip()
            if text:
                self._append_transcript_line(text)
                self.partial_text_var.set("")
        elif name == "capture_timeout":
            streak = int(fields.get("read_timeout_streak", 0))
            stall_duration_sec = float(fields.get("stall_duration_sec", 0.0))
            self.status_var.set("Waiting for audio")
            self._append_transcript_line(
                f"[capture] stalled after {streak} timeout{'s' if streak != 1 else ''} ({stall_duration_sec:.1f}s)"
            )
        elif name == "capture_resumed":
            stall_duration_sec = float(fields.get("stall_duration_sec", 0.0))
            self.status_var.set("Capture resumed")
            self._append_transcript_line(f"[capture] resumed after {stall_duration_sec:.1f}s")
        elif name == "notes_preparing":
            self.status_var.set("Preparing notes")
            self._begin_notes_progress_view()
        elif name == "notes_started":
            self.status_var.set("Generating notes")
            model = str(fields.get("model", "")).strip()
            if model:
                self._append_transcript_line(f"Starting notes model: {model}")
            else:
                self._append_transcript_line("Starting notes generation")
        elif name == "clean_transcript_started":
            chunk_count = fields.get("chunk_count")
            self.status_var.set(
                f"Cleaning transcript ({chunk_count} chunk{'s' if chunk_count != 1 else ''})"
                if isinstance(chunk_count, int)
                else "Cleaning transcript"
            )
            self._append_transcript_line("")
            self._append_transcript_line("Transcript cleanup")
        elif name == "clean_transcript_chunk_started":
            chunk_index = fields.get("chunk_index")
            chunk_count = fields.get("chunk_count")
            if isinstance(chunk_index, int) and isinstance(chunk_count, int):
                self.status_var.set(f"Cleaning transcript ({chunk_index}/{chunk_count})")
                self._append_transcript_line(f"[Cleanup {chunk_index}/{chunk_count}] Running")
            else:
                self.status_var.set("Cleaning transcript")
                self._append_transcript_line("[Cleanup] Running")
        elif name == "clean_transcript_chunk_fallback":
            chunk_index = fields.get("chunk_index")
            chunk_count = fields.get("chunk_count")
            if isinstance(chunk_index, int) and isinstance(chunk_count, int):
                self.status_var.set(f"Cleanup fallback used ({chunk_index}/{chunk_count})")
                self._append_transcript_line(f"[Cleanup {chunk_index}/{chunk_count}] Fallback to raw transcript chunk")
            else:
                self.status_var.set("Cleanup fallback used")
                self._append_transcript_line("[Cleanup] Fallback to raw transcript chunk")
        elif name == "clean_transcript_chunk_ready":
            chunk_index = fields.get("chunk_index")
            chunk_count = fields.get("chunk_count")
            text = str(fields.get("text", "")).strip()
            if isinstance(chunk_index, int) and isinstance(chunk_count, int):
                self._append_transcript_line(f"[Cleanup {chunk_index}/{chunk_count}] Result")
            else:
                self._append_transcript_line("[Cleanup] Result")
            if text:
                self._append_transcript_line(text)
                self._append_transcript_line("")
        elif name == "clean_transcript_ready":
            self.status_var.set("Clean transcript ready")
            self._append_transcript_line("Clean transcript complete")
            self._append_transcript_line("")
        elif name == "notes_cpu_fallback":
            self.status_var.set("Retrying notes on CPU")
            self._append_transcript_line("Retrying notes generation on CPU")
        elif name == "client_notes_started":
            self.status_var.set("Generating client notes")
            self._notes_stream_started = False
            self._append_transcript_line("Client notes generation")
            self._append_transcript_line("")
        elif name == "client_notes_ready":
            self.status_var.set("Client notes ready")
            text = str(fields.get("text", ""))
            streamed = bool(fields.get("streamed", False))
            if not streamed and text.strip():
                self._append_transcript_text(text.rstrip())
            if not self._notes_stream_started:
                self._append_transcript_line("Client notes ready")
            else:
                self._append_transcript_line("")
                self._append_transcript_line("Client notes ready")
        elif name == "client_notes_delta":
            text = str(fields.get("text", ""))
            if text:
                self._notes_stream_started = True
                self._append_transcript_text(text)
        elif name == "transcription_resources_released":
            released_models = fields.get("released_models")
            if isinstance(released_models, int):
                self._append_transcript_line(
                    f"Released transcription resources ({released_models} model cache entr{'y' if released_models == 1 else 'ies'})"
                )

    def handle_result(self, result: object) -> None:
        assert isinstance(result, SessionResultSummary)
        self.session_dir_var.set(str(result.session_dir))
        self.transcript_path_var.set(str(result.transcript_txt_path))
        if result.notes_summary is not None:
            self.notes_path_var.set(str(result.notes_summary.client_notes_path))
            self.status_var.set("Notes ready")
        elif not self.notes_enabled_var.get():
            self.notes_path_var.set("")
            self.status_var.set(
                "Session stopped" if result.interrupted else f"Saved {result.final_segment_count} final segments"
            )
        else:
            self.status_var.set(
                "Session stopped" if result.interrupted else f"Saved {result.final_segment_count} final segments"
            )

    def set_busy(self, busy: bool, *, cancelable: bool) -> None:
        self.start_button.configure(state="disabled" if busy else "normal")
        self.refresh_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy and cancelable else "disabled")


class CapturePage(BasePage):
    """Audio capture workflow page."""

    title = "Capture"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.duration_var = tk.StringVar(value="30")
        self.mode_var = tk.StringVar(value=AudioSourceMode.BOTH.value)
        self.output_root_var = tk.StringVar(value=str(default_data_subdir("live_sessions")))
        self.session_id_var = tk.StringVar(value="")
        self.mic_device_var = tk.StringVar(value="Auto")
        self.speaker_device_var = tk.StringVar(value="Auto")
        self.fixture_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Idle")
        self.session_dir_var = tk.StringVar(value="")
        self.manifest_path_var = tk.StringVar(value="")
        self.device_labels: tuple[str, ...] = ("Auto",)
        self.advanced_visible = False

        controls = ttk.LabelFrame(self, text="Capture")
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        controls.columnconfigure(0, weight=1)

        actions = ttk.Frame(controls)
        actions.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self.stop,
            state="disabled",
            style="Danger.TButton",
            width=12,
        )
        self.stop_button.pack(side="right", padx=(8, 0))
        self.start_button = ttk.Button(
            actions,
            text="Start Capture",
            command=self.start,
            style="Primary.TButton",
            width=16,
        )
        self.start_button.pack(side="right")

        self.advanced_frame = ttk.LabelFrame(self, text="Advanced Capture Options")
        self.advanced_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for index in range(4):
            self.advanced_frame.columnconfigure(index, weight=1)
        ttk.Label(self.advanced_frame, text="Duration (sec)").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.advanced_frame, textvariable=self.duration_var).grid(
            row=1, column=0, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Label(self.advanced_frame, text="Audio Source").grid(row=0, column=1, sticky="w")
        StableCombobox(
            self.advanced_frame,
            textvariable=self.mode_var,
            values=[mode.value for mode in AudioSourceMode],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Label(self.advanced_frame, text="Mic Device").grid(row=0, column=2, sticky="w")
        self.mic_combo = StableCombobox(
            self.advanced_frame,
            textvariable=self.mic_device_var,
            values=self.device_labels,
            state="readonly",
        )
        self.mic_combo.grid(row=1, column=2, sticky="ew", padx=(0, 6), pady=(0, 6))
        ttk.Label(self.advanced_frame, text="Speaker Device").grid(row=0, column=3, sticky="w")
        self.speaker_combo = StableCombobox(
            self.advanced_frame,
            textvariable=self.speaker_device_var,
            values=self.device_labels,
            state="readonly",
        )
        self.speaker_combo.grid(row=1, column=3, sticky="ew", pady=(0, 6))
        self.refresh_button = ttk.Button(self.advanced_frame, text="Refresh Devices", command=self.refresh_devices)
        self.refresh_button.grid(row=2, column=3, sticky="ew", pady=(0, 6))
        ttk.Label(self.advanced_frame, text="Output Root").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.advanced_frame, textvariable=self.output_root_var).grid(
            row=3, column=0, columnspan=3, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Button(
            self.advanced_frame, text="Browse", command=lambda: self.app.choose_directory(self.output_root_var)
        ).grid(row=3, column=3, sticky="ew", pady=(0, 6))
        ttk.Label(self.advanced_frame, text="Session Id").grid(row=4, column=0, sticky="w")
        ttk.Entry(self.advanced_frame, textvariable=self.session_id_var).grid(
            row=5, column=0, sticky="ew", padx=(0, 6), pady=(0, 6)
        )
        ttk.Checkbutton(self.advanced_frame, text="Fixture", variable=self.fixture_var).grid(
            row=5, column=1, sticky="w", pady=(0, 6)
        )

        devices = ttk.LabelFrame(self, text="Available Devices")
        devices.grid(row=2, column=0, sticky="nsew", padx=(0, 8), pady=(0, 12))
        devices.columnconfigure(0, weight=1)
        devices.rowconfigure(0, weight=1)
        self.device_list = tk.Listbox(devices, height=10)
        self.device_list.grid(row=0, column=0, sticky="nsew")

        outputs = ttk.LabelFrame(self, text="Outputs")
        outputs.grid(row=2, column=1, sticky="nsew", pady=(0, 12))
        outputs.columnconfigure(1, weight=1)
        ttk.Label(outputs, text="Status").grid(row=0, column=0, sticky="w")
        ttk.Label(outputs, textvariable=self.status_var).grid(row=0, column=1, sticky="w")
        ttk.Label(outputs, text="Session Dir").grid(row=1, column=0, sticky="w")
        ttk.Entry(outputs, textvariable=self.session_dir_var).grid(row=1, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(outputs, text="Open", command=lambda: self.app.open_path_var(self.session_dir_var)).grid(
            row=1, column=2, sticky="ew"
        )
        ttk.Label(outputs, text="Manifest").grid(row=2, column=0, sticky="w")
        ttk.Entry(outputs, textvariable=self.manifest_path_var).grid(row=2, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(outputs, text="Open", command=lambda: self.app.open_path_var(self.manifest_path_var)).grid(
            row=2, column=2, sticky="ew"
        )
        self.set_advanced_mode(False)

    def _prepare_launch_paths(self) -> tuple[Path, str, Path]:
        output_root_text = self.output_root_var.get().strip()
        output_root = Path(output_root_text) if output_root_text else default_data_subdir("live_sessions")
        session_id = self.session_id_var.get().strip() or services.default_session_id("capture")
        session_dir = output_root / session_id
        self.output_root_var.set(str(output_root))
        self.session_id_var.set(session_id)
        self.session_dir_var.set(str(session_dir))
        self.manifest_path_var.set(str(session_dir / "session_manifest.json"))
        return output_root, session_id, session_dir

    def refresh_devices(self) -> None:
        self.app.start_task(
            "capture-devices",
            lambda cancel, progress: services.list_devices(common=self.app.common_options()),
            on_result=self._handle_devices_result,
        )

    def _handle_devices_result(self, result: object) -> None:
        assert isinstance(result, DeviceListResult)
        self.app.set_devices(result.devices)
        self.status_var.set(f"Loaded {len(result.devices)} devices")

    def set_devices(self, devices: tuple[DeviceInfo, ...]) -> None:
        labels = ("Auto", *[device.label for device in devices])
        self.mic_combo.configure(values=labels)
        self.speaker_combo.configure(values=labels)
        self.device_list.delete(0, tk.END)
        for device in devices:
            self.device_list.insert(tk.END, device.label)

    def set_advanced_mode(self, advanced: bool) -> None:
        self.advanced_visible = advanced
        if advanced:
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def start(self) -> None:
        output_root, session_id, _ = self._prepare_launch_paths()
        request = CaptureRequest(
            common=self.app.common_options(),
            duration_sec=float(self.duration_var.get() or 30.0),
            mode=AudioSourceMode(self.mode_var.get()),
            output_root=output_root,
            session_id=session_id,
            mic_device=self.mic_device_var.get(),
            speaker_device=self.speaker_device_var.get(),
            use_fixture=self.fixture_var.get(),
        )
        self.status_var.set("Starting capture...")
        self.app.start_task(
            "capture",
            lambda cancel, progress: services.run_capture(request, cancel_event=cancel),
            cancelable=True,
            on_result=self.handle_result,
        )

    def stop(self) -> None:
        if self.app.cancel_active_task():
            self.status_var.set("Stopping capture...")

    def handle_result(self, result: object) -> None:
        assert isinstance(result, CaptureResultSummary)
        self.session_dir_var.set(str(result.session_dir))
        self.manifest_path_var.set(str(result.manifest_path))
        self.status_var.set(
            "Capture stopped" if result.interrupted else f"Capture complete ({result.pair_count} pairs)"
        )

    def set_busy(self, busy: bool, *, cancelable: bool) -> None:
        self.start_button.configure(state="disabled" if busy else "normal")
        self.refresh_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if busy and cancelable else "disabled")


class NotesPage(BasePage):
    """Post-session notes workflow page."""

    title = "Notes"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.transcript_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value=services.DEFAULT_SESSION_NOTES_MODEL)
        self.model_choices = services.notes_model_options()
        self.runtime_var = tk.StringVar(value=services.DEFAULT_NOTES_RUNTIME)
        self.status_var = tk.StringVar(value="Idle")
        self.clean_path_var = tk.StringVar(value="")
        self.notes_path_var = tk.StringVar(value="")

        controls = ttk.LabelFrame(self, text="Transcript Cleanup and Notes")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Transcript").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.transcript_var).grid(
            row=0, column=1, sticky="ew", padx=(6, 6), pady=(0, 6)
        )
        ttk.Button(controls, text="Browse", command=lambda: self.app.choose_file(self.transcript_var)).grid(
            row=0, column=2, sticky="ew"
        )
        ttk.Label(controls, text="Output Dir").grid(row=1, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.output_dir_var).grid(
            row=1, column=1, sticky="ew", padx=(6, 6), pady=(0, 6)
        )
        ttk.Button(controls, text="Browse", command=lambda: self.app.choose_directory(self.output_dir_var)).grid(
            row=1, column=2, sticky="ew"
        )
        ttk.Label(controls, text="Model").grid(row=2, column=0, sticky="w")
        self.model_combo = StableCombobox(
            controls, textvariable=self.model_var, values=self.model_choices, state="readonly"
        )
        self.model_combo.grid(row=2, column=1, sticky="ew", padx=(6, 6), pady=(0, 6))
        ttk.Label(controls, text="Runtime").grid(row=3, column=0, sticky="w")
        StableCombobox(
            controls, textvariable=self.runtime_var, values=["auto", "ollama", "llama_cpp"], state="readonly"
        ).grid(row=3, column=1, sticky="ew", padx=(6, 6), pady=(0, 6))
        self.start_button = ttk.Button(controls, text="Run Notes", command=self.start)
        self.start_button.grid(row=3, column=2, sticky="ew")
        ttk.Label(controls, text="Status").grid(row=4, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.status_var).grid(row=4, column=1, sticky="w")
        ttk.Label(controls, text="Clean Transcript").grid(row=5, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.clean_path_var).grid(
            row=5, column=1, sticky="ew", padx=(6, 6), pady=(0, 6)
        )
        ttk.Button(controls, text="Open", command=lambda: self.app.open_path_var(self.clean_path_var)).grid(
            row=5, column=2, sticky="ew"
        )
        ttk.Label(controls, text="Client Notes").grid(row=6, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.notes_path_var).grid(row=6, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(controls, text="Open", command=lambda: self.app.open_path_var(self.notes_path_var)).grid(
            row=6, column=2, sticky="ew"
        )

        log_frame = ttk.LabelFrame(self, text="Notes Progress")
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.progress_text = ScrolledText(log_frame, wrap="word")
        self.progress_text.grid(row=0, column=0, sticky="nsew")
        self.progress_text.configure(state="disabled")

    def start(self) -> None:
        self.progress_text.configure(state="normal")
        self.progress_text.delete("1.0", tk.END)
        self.progress_text.configure(state="disabled")
        transcript_path = Path(self.transcript_var.get().strip())
        output_dir = Path(self.output_dir_var.get().strip()) if self.output_dir_var.get().strip() else None
        request = NotesRequest(
            common=self.app.common_options(),
            transcript_path=transcript_path,
            output_dir=output_dir,
            notes_model=self.model_var.get().strip(),
            notes_runtime=self.runtime_var.get().strip(),
        )
        self.status_var.set("Running notes...")
        self.app.start_task(
            "notes",
            lambda cancel, progress: services.run_notes(request, progress_callback=progress),
            on_progress=self.handle_progress,
            on_result=self.handle_result,
        )

    def handle_progress(self, progress: ServiceProgressEvent) -> None:
        self._append_progress(f"{progress.name}: {progress.fields}")
        self.status_var.set(progress.name.replace("_", " ").title())

    def handle_result(self, result: object) -> None:
        assert isinstance(result, NotesResultSummary)
        self.clean_path_var.set(str(result.clean_transcript_path))
        self.notes_path_var.set(str(result.client_notes_path))
        self.status_var.set("Notes ready")

    def _append_progress(self, text: str) -> None:
        self.progress_text.configure(state="normal")
        self.progress_text.insert(tk.END, text + "\n")
        self.progress_text.see(tk.END)
        self.progress_text.configure(state="disabled")

    def set_busy(self, busy: bool, *, cancelable: bool) -> None:
        del cancelable
        self.start_button.configure(state="disabled" if busy else "normal")


class ModelsPage(BasePage):
    """Packaged model manifest page."""

    title = "Models"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.status_var = tk.StringVar(value="Idle")

        actions = ttk.Frame(self)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.refresh_button = ttk.Button(actions, text="Refresh", command=self.refresh_models)
        self.refresh_button.pack(side="left")
        self.install_selected_button = ttk.Button(actions, text="Install Selected", command=self.install_selected)
        self.install_selected_button.pack(side="left", padx=(6, 0))
        self.install_default_button = ttk.Button(actions, text="Install Defaults", command=self.install_defaults)
        self.install_default_button.pack(side="left", padx=(6, 0))
        ttk.Label(actions, textvariable=self.status_var).pack(side="right")

        self.tree = ttk.Treeview(self, columns=("kind", "class", "status"), show="tree headings", selectmode="extended")
        self.tree.heading("#0", text="Model Id")
        self.tree.heading("kind", text="Kind")
        self.tree.heading("class", text="Install Class")
        self.tree.heading("status", text="Status")
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.tree.column("#0", width=300)
        self.tree.column("kind", width=160)
        self.tree.column("class", width=120)
        self.tree.column("status", width=120)

    def refresh_models(self) -> None:
        self.status_var.set("Loading manifest...")
        self.app.start_task("models-list", lambda cancel, progress: services.list_models(), on_result=self.handle_list)

    def handle_list(self, result: object) -> None:
        if not isinstance(result, ModelsListResult):
            raise RuntimeError(
                f"Models list returned an unexpected result payload: {type(result).__name__}."
            )
        for item_id in self.tree.get_children():
            self.tree.delete(item_id)
        if result.error:
            self.status_var.set(result.error)
            return
        for item in result.items:
            self.tree.insert(
                "",
                tk.END,
                iid=item.model_id,
                text=item.model_id,
                values=(item.kind, item.install_class, "installed" if item.installed else "not installed"),
            )
        self.status_var.set(f"Loaded {len(result.items)} models")

    def install_selected(self) -> None:
        selected = tuple(self.tree.selection())
        if not selected:
            messagebox.showinfo("Models", "Select one or more models first.")
            return
        request = ModelsInstallRequest(common=self.app.common_options(), model_ids=selected, default_only=False)
        self.status_var.set("Installing selected models...")
        self.app.start_task(
            "models-install",
            lambda cancel, progress: services.install_models(request, progress_callback=progress),
            on_progress=self.handle_progress,
            on_result=self.handle_install,
        )

    def install_defaults(self) -> None:
        request = ModelsInstallRequest(common=self.app.common_options(), default_only=True)
        self.status_var.set("Installing default models...")
        self.app.start_task(
            "models-install-default",
            lambda cancel, progress: services.install_models(request, progress_callback=progress),
            on_progress=self.handle_progress,
            on_result=self.handle_install,
        )

    def handle_progress(self, progress: ServiceProgressEvent) -> None:
        fields = progress.fields if isinstance(progress.fields, dict) else {}
        model_id = fields.get("model_id", "")
        self.status_var.set(f"{progress.name.replace('_', ' ')} {model_id}".strip())

    def handle_install(self, result: object) -> None:
        if not isinstance(result, ModelsInstallResultSummary):
            raise RuntimeError(
                f"Model install returned an unexpected result payload: {type(result).__name__}."
            )
        for model_id in result.installed_model_ids:
            if self.tree.exists(model_id):
                self.tree.set(model_id, "status", "installed")
        self.status_var.set(
            f"Installed {len(result.installed_model_ids)} models, skipped {len(result.skipped_model_ids)}"
        )

    def set_busy(self, busy: bool, *, cancelable: bool) -> None:
        del cancelable
        state = "disabled" if busy else "normal"
        self.refresh_button.configure(state=state)
        self.install_selected_button.configure(state=state)
        self.install_default_button.configure(state=state)


class BenchPage(BasePage):
    """Benchmark init/run page for development mode."""

    title = "Bench"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Idle")
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.init_model_var = tk.StringVar(value=services.DEFAULT_BENCH_MODEL)
        self.init_dataset_var = tk.StringVar(value=services.DEFAULT_BENCH_DATASET)
        self.init_config_var = tk.StringVar(value=services.DEFAULT_BENCH_CONFIG)
        self.init_split_var = tk.StringVar(value=services.DEFAULT_BENCH_SPLIT)
        self.init_limit_var = tk.StringVar(value=str(services.DEFAULT_BENCH_LIMIT))
        self.init_ram_var = tk.StringVar(value="8.0")
        self.run_scenario_var = tk.StringVar(value="hf_diarized_transcription")
        self.run_runs_var = tk.StringVar(value="5")
        self.run_duration_var = tk.StringVar(value="10")
        self.run_output_var = tk.StringVar(value=str(default_data_subdir("benchmarks")))
        self.run_real_devices_var = tk.BooleanVar(value=False)
        self.run_dataset_var = tk.StringVar(value=services.DEFAULT_BENCH_DATASET)
        self.run_config_var = tk.StringVar(value=services.DEFAULT_BENCH_CONFIG)
        self.run_split_var = tk.StringVar(value=services.DEFAULT_BENCH_SPLIT)
        self.run_limit_var = tk.StringVar(value=str(services.DEFAULT_BENCH_LIMIT))
        self.run_model_var = tk.StringVar(value=services.DEFAULT_BENCH_MODEL)
        self.run_ram_var = tk.StringVar(value="8.0")
        self.json_path_var = tk.StringVar(value="")
        self.markdown_path_var = tk.StringVar(value="")

        init_frame = ttk.Frame(self.notebook, padding=16)
        init_frame.columnconfigure(1, weight=1)
        self.notebook.add(init_frame, text="Init Cache")
        self._bench_entry(init_frame, 0, "Model", self.init_model_var)
        self._bench_entry(init_frame, 1, "Dataset", self.init_dataset_var)
        self._bench_entry(init_frame, 2, "Config", self.init_config_var)
        self._bench_entry(init_frame, 3, "Split", self.init_split_var)
        self._bench_entry(init_frame, 4, "Row Limit", self.init_limit_var)
        self._bench_entry(init_frame, 5, "Max Model RAM", self.init_ram_var)
        self.init_button = ttk.Button(init_frame, text="Initialize Cache", command=self.start_init)
        self.init_button.grid(row=6, column=1, sticky="e", pady=(12, 0))

        run_frame = ttk.Frame(self.notebook, padding=16)
        run_frame.columnconfigure(1, weight=1)
        self.notebook.add(run_frame, text="Run Benchmark")
        ttk.Label(run_frame, text="Scenario").grid(row=0, column=0, sticky="w")
        StableCombobox(
            run_frame,
            textvariable=self.run_scenario_var,
            values=["hf_diarized_transcription", "capture_sync"],
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", pady=(0, 6))
        self._bench_entry(run_frame, 1, "Runs", self.run_runs_var)
        self._bench_entry(run_frame, 2, "Duration", self.run_duration_var)
        self._bench_entry(run_frame, 3, "Output Dir", self.run_output_var)
        ttk.Button(run_frame, text="Browse", command=lambda: self.app.choose_directory(self.run_output_var)).grid(
            row=3, column=2, sticky="ew", padx=(6, 0)
        )
        ttk.Checkbutton(run_frame, text="Use Real Devices", variable=self.run_real_devices_var).grid(
            row=4, column=1, sticky="w"
        )
        self._bench_entry(run_frame, 5, "Dataset", self.run_dataset_var)
        self._bench_entry(run_frame, 6, "Config", self.run_config_var)
        self._bench_entry(run_frame, 7, "Split", self.run_split_var)
        self._bench_entry(run_frame, 8, "Row Limit", self.run_limit_var)
        self._bench_entry(run_frame, 9, "Model", self.run_model_var)
        self._bench_entry(run_frame, 10, "Max Model RAM", self.run_ram_var)
        self.run_button = ttk.Button(run_frame, text="Run Benchmark", command=self.start_run)
        self.run_button.grid(row=11, column=1, sticky="e", pady=(12, 0))
        ttk.Label(run_frame, text="JSON Report").grid(row=12, column=0, sticky="w")
        ttk.Entry(run_frame, textvariable=self.json_path_var).grid(row=12, column=1, sticky="ew", pady=(6, 6))
        ttk.Button(run_frame, text="Open", command=lambda: self.app.open_path_var(self.json_path_var)).grid(
            row=12, column=2, sticky="ew", padx=(6, 0)
        )
        ttk.Label(run_frame, text="Markdown Report").grid(row=13, column=0, sticky="w")
        ttk.Entry(run_frame, textvariable=self.markdown_path_var).grid(row=13, column=1, sticky="ew")
        ttk.Button(run_frame, text="Open", command=lambda: self.app.open_path_var(self.markdown_path_var)).grid(
            row=13, column=2, sticky="ew", padx=(6, 0)
        )

        status_frame = ttk.Frame(self)
        status_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor="w")

    def _bench_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=(0, 6))

    def start_init(self) -> None:
        request = BenchmarkInitRequest(
            common=self.app.common_options(),
            transcription_model=self.init_model_var.get().strip(),
            hf_dataset=self.init_dataset_var.get().strip(),
            hf_config=self.init_config_var.get().strip(),
            hf_split=self.init_split_var.get().strip(),
            hf_limit=int(self.init_limit_var.get() or services.DEFAULT_BENCH_LIMIT),
            max_model_ram_gb=float(self.init_ram_var.get() or 8.0),
        )
        self.status_var.set("Initializing benchmark cache...")
        self.app.start_task(
            "bench-init",
            lambda cancel, progress: services.initialize_bench_assets(request),
            on_result=self.handle_init_result,
        )

    def handle_init_result(self, result: object) -> None:
        assert isinstance(result, BenchmarkInitResultSummary)
        self.status_var.set(
            f"Cached {result.payload.get('transcription_model', '')} and {result.payload.get('dataset_id', '')}"
        )

    def start_run(self) -> None:
        request = BenchmarkRunRequest(
            common=self.app.common_options(),
            scenario=self.run_scenario_var.get(),
            runs=int(self.run_runs_var.get() or 5),
            duration_sec=float(self.run_duration_var.get() or 10.0),
            output_dir=Path(self.run_output_var.get().strip()),
            real_devices=self.run_real_devices_var.get(),
            hf_dataset=self.run_dataset_var.get().strip(),
            hf_config=self.run_config_var.get().strip(),
            hf_split=self.run_split_var.get().strip(),
            hf_limit=int(self.run_limit_var.get() or services.DEFAULT_BENCH_LIMIT),
            transcription_model=self.run_model_var.get().strip(),
            max_model_ram_gb=float(self.run_ram_var.get() or 8.0),
        )
        self.status_var.set("Running benchmark...")
        self.app.start_task(
            "bench-run", lambda cancel, progress: services.run_benchmark(request), on_result=self.handle_run_result
        )

    def handle_run_result(self, result: object) -> None:
        assert isinstance(result, BenchmarkRunResultSummary)
        self.json_path_var.set(str(result.json_path))
        self.markdown_path_var.set(str(result.markdown_path))
        self.status_var.set(f"Benchmark complete: {result.scenario}")

    def set_busy(self, busy: bool, *, cancelable: bool) -> None:
        del cancelable
        state = "disabled" if busy else "normal"
        self.init_button.configure(state=state)
        self.run_button.configure(state=state)


class CompliancePage(BasePage):
    """Compliance checks page."""

    title = "Compliance"

    def __init__(self, master: ttk.Frame, app: "TranscribeUiApp") -> None:
        super().__init__(master, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.urls_target_var = tk.StringVar(value=str(Path.cwd()))
        self.status_var = tk.StringVar(value="Idle")

        controls = ttk.Frame(self)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Button(controls, text="Check No Network", command=self.run_network).pack(side="left")
        ttk.Button(controls, text="Check No URLs", command=self.run_urls).pack(side="left", padx=(6, 0))
        ttk.Entry(controls, textvariable=self.urls_target_var, width=60).pack(side="left", padx=(12, 6))
        ttk.Button(controls, text="Browse", command=lambda: self.app.choose_directory(self.urls_target_var)).pack(
            side="left"
        )
        ttk.Label(controls, textvariable=self.status_var).pack(side="right")

        self.text = ScrolledText(self, wrap="word")
        self.text.grid(row=1, column=0, sticky="nsew")
        self.text.configure(state="disabled")

    def run_network(self) -> None:
        self.status_var.set("Running network check...")
        self.app.start_task(
            "compliance-network",
            lambda cancel, progress: services.run_compliance_check_no_network(common=self.app.common_options()),
            on_result=self.handle_result,
        )

    def run_urls(self) -> None:
        self.status_var.set("Running URL check...")
        target = Path(self.urls_target_var.get().strip()) if self.urls_target_var.get().strip() else None
        self.app.start_task(
            "compliance-urls",
            lambda cancel, progress: services.run_compliance_check_no_urls(
                common=self.app.common_options(), target_path=target
            ),
            on_result=self.handle_result,
        )

    def handle_result(self, result: object) -> None:
        assert isinstance(result, ComplianceResultSummary)
        status = "PASS" if result.passed else f"FAIL ({result.exit_code})"
        line = f"{result.name}: {status} - {result.summary}"
        detail_lines = [line]
        if result.target_path is not None:
            detail_lines.append(f"Target: {result.target_path}")
        detail_lines.extend(result.details)
        self.text.configure(state="normal")
        self.text.insert(tk.END, "\n".join(detail_lines) + "\n\n")
        self.text.see(tk.END)
        self.text.configure(state="disabled")
        self.status_var.set(line)


class TranscribeUiApp:
    """Tkinter application shell for `transcribe`."""

    def __init__(self, root: tk.Tk | tk.Toplevel, *, packaged_runtime: bool | None = None) -> None:
        self.root = root
        self.root.title("Transcribe; the offline therapy session note taker")
        self.root.geometry("980x680")
        self.root.minsize(900, 620)
        self.controller = UiTaskController()
        self._active_binding: TaskBinding | None = None
        self._log_lines: list[str] = []
        self._devices: tuple[DeviceInfo, ...] = ()
        self._poll_after_id: str | None = None
        self._closing = False
        runtime_mode = (
            detect_runtime_mode()
            if packaged_runtime is None
            else (RuntimeMode.PACKAGED if packaged_runtime else RuntimeMode.DEVELOPMENT)
        )
        self.packaged_runtime = runtime_mode == RuntimeMode.PACKAGED
        self.preferences = load_ui_preferences()
        set_network_access_allowed(self.preferences.allow_network)

        self._apply_window_icon()
        self._apply_theme()
        self._build_shell()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_messages()

    def _apply_window_icon(self) -> None:
        """Apply the packaged application icon to the Tk root when available."""
        icon_path = self._resolve_window_icon_path()
        if icon_path is None:
            return
        try:
            self.root.iconbitmap(default=str(icon_path))
        except tk.TclError:
            return

    def _resolve_window_icon_path(self) -> Path | None:
        """Resolve the preferred icon file for the desktop UI."""
        runtime_paths = resolve_app_runtime_paths()
        candidates = [
            runtime_paths.install_root / "transcribe.ico",
            Path(__file__).resolve().parents[2] / "packaging" / "windows" / "transcribe.ico",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _apply_theme(self) -> None:
        style = ttk.Style(self.root)
        theme_names = set(style.theme_names())
        for candidate in ("vista", "aqua", "xpnative", "clam"):
            if candidate in theme_names:
                style.theme_use(candidate)
                break
        style.configure("Primary.TButton", padding=(18, 12), font=("TkDefaultFont", 10, "bold"))
        style.configure("Danger.TButton", padding=(18, 12), font=("TkDefaultFont", 10, "bold"))

    def _build_shell(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.nav = ttk.Frame(self.root, padding=12)
        self.nav.grid(row=0, column=0, sticky="ns")
        self.content = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(1, weight=1)

        header = ttk.Frame(self.content)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(1, weight=1)
        self.advanced_ui_var = tk.BooleanVar(value=self.preferences.advanced_ui)
        ttk.Checkbutton(
            header,
            text="Expand Options",
            variable=self.advanced_ui_var,
            command=self._handle_advanced_ui_toggle,
        ).grid(row=0, column=0, sticky="w")
        self.busy_var = tk.StringVar(value="Idle")
        ttk.Label(header, textvariable=self.busy_var).grid(row=0, column=1, sticky="e", padx=(12, 0))

        self.global_advanced_frame = ttk.LabelFrame(header, text="Global Options")
        self.global_advanced_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.global_advanced_frame.columnconfigure(1, weight=1)
        ttk.Label(self.global_advanced_frame, text="Runtime Config").grid(row=0, column=0, sticky="w")
        self.config_path_var = tk.StringVar(value="")
        ttk.Entry(self.global_advanced_frame, textvariable=self.config_path_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 6)
        )
        ttk.Button(
            self.global_advanced_frame, text="Browse", command=lambda: self.choose_file(self.config_path_var)
        ).grid(row=0, column=2, sticky="ew")
        ttk.Label(self.global_advanced_frame, text="Log Level").grid(row=0, column=3, sticky="w", padx=(12, 0))
        self.log_level_var = tk.StringVar(value=services.DEFAULT_LOG_LEVEL)
        self.log_level_combo = StableCombobox(
            self.global_advanced_frame,
            textvariable=self.log_level_var,
            values=LOG_LEVEL_OPTIONS,
            state="readonly",
            width=10,
        )
        self.log_level_combo.grid(row=0, column=4, sticky="ew", padx=(6, 6))
        self.allow_network_var = tk.BooleanVar(value=self.preferences.allow_network)
        ttk.Checkbutton(
            self.global_advanced_frame,
            text="Allow Network Access",
            variable=self.allow_network_var,
            command=self._handle_allow_network_toggle,
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(8, 0))

        self.page_container = ttk.Frame(self.content)
        self.page_container.grid(row=1, column=0, sticky="nsew")
        self.page_container.columnconfigure(0, weight=1)
        self.page_container.rowconfigure(0, weight=1)

        self.pages: dict[str, BasePage] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        page_specs: list[tuple[str, type[BasePage]]] = [
            ("session", SessionPage),
            ("capture", CapturePage),
            ("notes", NotesPage),
            ("models", ModelsPage),
        ]
        if not self.packaged_runtime:
            page_specs.append(("bench", BenchPage))
        page_specs.extend(
            [
                ("compliance", CompliancePage),
                ("logs", LogsPage),
            ]
        )
        for row, (page_id, page_type) in enumerate(page_specs):
            page = page_type(self.page_container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[page_id] = page
            button = ttk.Button(self.nav, text=page.title, command=lambda page_id=page_id: self.show_page(page_id))
            button.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            self.nav_buttons[page_id] = button

        ttk.Separator(self.nav, orient="horizontal").grid(row=len(page_specs), column=0, sticky="ew", pady=(10, 10))
        self.cancel_button = ttk.Button(
            self.nav, text="Stop Active Task", command=self.cancel_active_task, state="disabled"
        )
        self.cancel_button.grid(row=len(page_specs) + 1, column=0, sticky="ew")
        self._apply_advanced_ui_state()
        self.show_page("session")

    def common_options(self) -> UiCommonOptions:
        """Build common runtime options from the global settings row."""
        config_value = self.config_path_var.get().strip()
        log_level = self.log_level_var.get().strip() or None
        return UiCommonOptions(
            config_path=Path(config_value) if config_value else None,
            log_level=log_level,
            debug=(log_level or "").upper() == "DEBUG",
            allow_network=self.allow_network_var.get(),
        )

    def _current_preferences(self) -> UiPreferences:
        """Return the current persisted UI preference state."""
        return UiPreferences(
            advanced_ui=self.advanced_ui_var.get(),
            allow_network=self.allow_network_var.get(),
        )

    def _persist_preferences(self, *, show_error: bool = True) -> None:
        """Write the current UI preference state to disk."""
        try:
            save_ui_preferences(self._current_preferences())
        except OSError as exc:
            if show_error:
                messagebox.showerror("Preferences save failed", str(exc))

    def _handle_advanced_ui_toggle(self) -> None:
        """Persist the expanded-options toggle and refresh the page layout."""
        self._persist_preferences()
        self._apply_advanced_ui_state()

    def _handle_allow_network_toggle(self) -> None:
        """Persist the network-access toggle for restart-sensitive workflows."""
        set_network_access_allowed(self.allow_network_var.get())
        self._persist_preferences()

    def _on_close(self) -> None:
        """Persist UI preferences before closing the application window."""
        self._closing = True
        self._cancel_poll_callback()
        self._persist_preferences(show_error=False)
        self.root.destroy()

    def _cancel_poll_callback(self) -> None:
        """Cancel the pending message-poll callback when the app is closing."""
        if self._poll_after_id is None:
            return
        try:
            self.root.after_cancel(self._poll_after_id)
        except tk.TclError:
            pass
        finally:
            self._poll_after_id = None

    def _apply_advanced_ui_state(self) -> None:
        """Toggle advanced controls across all workflow pages."""
        advanced_enabled = self.advanced_ui_var.get() if hasattr(self, "advanced_ui_var") else False
        if advanced_enabled:
            self.global_advanced_frame.grid()
        else:
            self.global_advanced_frame.grid_remove()
        for page in self.pages.values():
            page.set_advanced_mode(bool(advanced_enabled))

    def show_page(self, page_id: str) -> None:
        """Raise one page into view."""
        self.pages[page_id].tkraise()

    def start_task(
        self,
        task_name: str,
        runner,
        *,
        cancelable: bool = False,
        on_result: Callable[[object], None] | None = None,
        on_progress: Callable[[ServiceProgressEvent], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Start a background task and bind UI handlers for it."""
        if self.controller.is_busy():
            messagebox.showinfo("Task in progress", "Wait for the current task to finish before starting another.")
            return
        self._active_binding = TaskBinding(
            task_name=task_name,
            on_result=on_result,
            on_progress=on_progress,
            on_error=on_error,
            cancelable=cancelable,
        )
        self.controller.start_task(task_name, runner, cancelable=cancelable)
        self._set_busy(True, task_name, cancelable=cancelable)

    def cancel_active_task(self) -> bool:
        """Forward a stop request to the active cancelable task."""
        canceled = self.controller.cancel_active_task()
        if canceled:
            self.append_log("Stop requested for active task.")
        return canceled

    def set_devices(self, devices: tuple[DeviceInfo, ...]) -> None:
        """Share freshly loaded devices across audio pages."""
        self._devices = devices
        for page in self.pages.values():
            page.set_devices(devices)

    def choose_directory(self, variable: tk.StringVar) -> None:
        """Open a folder picker and store the selected path."""
        initial = variable.get().strip() or str(Path.cwd())
        selected = filedialog.askdirectory(initialdir=initial)
        if selected:
            variable.set(selected)

    def choose_file(self, variable: tk.StringVar) -> None:
        """Open a file picker and store the selected path."""
        initial = variable.get().strip()
        initialdir = str(Path(initial).parent if initial else Path.cwd())
        selected = filedialog.askopenfilename(initialdir=initialdir)
        if selected:
            variable.set(selected)

    def open_path_var(self, variable: tk.StringVar) -> None:
        """Open the path currently held by a string variable."""
        value = variable.get().strip()
        if not value:
            return
        self.open_path(Path(value))

    def open_path(self, path: Path) -> None:
        """Open a file or directory with the platform default handler."""
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            messagebox.showerror("Path not found", str(resolved))
            return
        try:
            if os.name == "nt":
                os.startfile(str(resolved))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(resolved)])
            else:
                subprocess.Popen(["xdg-open", str(resolved)])
        except OSError as exc:
            messagebox.showerror("Open failed", str(exc))

    def append_log(self, message: str) -> None:
        """Append one line to the retained log buffer and refresh the log page."""
        timestamp = time.strftime("%H:%M:%S")
        self._log_lines.append(f"[{timestamp}] {message}")
        if len(self._log_lines) > MAX_LOG_LINES:
            self._log_lines = self._log_lines[-MAX_LOG_LINES:]
        logs_page = self.pages.get("logs")
        if isinstance(logs_page, LogsPage):
            logs_page.replace_lines(self._log_lines)

    def _poll_messages(self) -> None:
        """Drain controller messages on the Tk main thread."""
        try:
            for message in self.controller.drain_messages():
                try:
                    self._handle_message(message)
                except Exception as exc:  # noqa: BLE001
                    self._handle_message_dispatch_failure(message, exc)
        finally:
            self._poll_after_id = None
            if not self._closing and self.root.winfo_exists():
                self._poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_messages)

    def _handle_message_dispatch_failure(self, message: ControllerMessage, exc: BaseException) -> None:
        """Surface UI callback failures without stopping background-task polling."""
        context = message.kind.replace("_", " ")
        self.append_log(f"UI callback failed during {message.task_name} {context}: {exc}")
        messagebox.showerror(
            "Task failed",
            f"{message.task_name} failed while processing its {context} update:\n{exc}",
        )

    def _handle_message(self, message: ControllerMessage) -> None:
        binding = self._active_binding
        if message.kind == "started":
            self.append_log(f"Task started: {message.task_name}")
            self.busy_var.set(f"Running {message.task_name}")
            return
        if message.kind == "progress":
            progress = message.payload
            if isinstance(progress, ServiceProgressEvent):
                self._handle_progress(progress)
                if binding is not None and binding.on_progress is not None:
                    binding.on_progress(progress)
            return
        if message.kind == "result":
            self.append_log(f"Task completed: {message.task_name}")
            if binding is not None and binding.on_result is not None:
                binding.on_result(message.payload)
            return
        if message.kind == "error":
            exc = message.payload if isinstance(message.payload, BaseException) else RuntimeError(str(message.payload))
            self.append_log(f"Task failed: {message.task_name}: {exc}")
            if binding is not None and binding.on_error is not None:
                binding.on_error(exc)
            else:
                messagebox.showerror("Task failed", str(exc))
            return
        if message.kind == "finished":
            self._set_busy(False, message.task_name, cancelable=False)
            self.busy_var.set("Idle")
            self._active_binding = None

    def _handle_progress(self, progress: ServiceProgressEvent) -> None:
        if progress.name not in {"partial"}:
            self.append_log(self._format_progress(progress))

    def _format_progress(self, progress: ServiceProgressEvent) -> str:
        important_keys = (
            "text",
            "transcription_model",
            "buffered_audio_sec",
            "model_id",
            "target_path",
            "chunk_index",
            "reason",
            "selected_source",
            "audio_sec",
            "inference_latency_ms",
            "text_length",
            "read_timeout_streak",
            "stall_duration_sec",
            "total_inference_sec",
            "empty_output_streak",
        )
        details = ", ".join(f"{key}={progress.fields[key]}" for key in important_keys if key in progress.fields)
        if details:
            return f"{progress.name}: {details}"
        return progress.name

    def _set_busy(self, busy: bool, task_name: str, *, cancelable: bool) -> None:
        for page in self.pages.values():
            page.set_busy(busy, cancelable=cancelable)
        self.cancel_button.configure(state="normal" if busy and cancelable else "disabled")
        if busy:
            self.append_log(f"Running task: {task_name}")


def page_order(*, packaged_runtime: bool | None = None) -> tuple[str, ...]:
    """Return the visible UI page order for tests and packaged-mode checks."""
    runtime_mode = (
        detect_runtime_mode()
        if packaged_runtime is None
        else (RuntimeMode.PACKAGED if packaged_runtime else RuntimeMode.DEVELOPMENT)
    )
    pages = ["session", "capture", "notes", "models"]
    if runtime_mode != RuntimeMode.PACKAGED:
        pages.append("bench")
    pages.extend(["compliance", "logs"])
    return tuple(pages)


def main() -> int:
    """Launch the desktop UI."""
    root = tk.Tk()
    TranscribeUiApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
