from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, SimpleQueue
from threading import Event, Thread
from typing import Callable, Literal

from transcribe.ui.services import wrap_progress

TaskKind = Literal["started", "progress", "result", "error", "finished"]
TaskRunner = Callable[[Event | None, Callable[[str, dict[str, object]], None]], object]


@dataclass(slots=True)
class ControllerMessage:
    """One message emitted by the background task controller."""

    kind: TaskKind
    task_name: str
    payload: object | None = None


class UiTaskController:
    """Run one background UI task at a time and buffer thread-safe messages."""

    def __init__(self) -> None:
        self._messages: SimpleQueue[ControllerMessage] = SimpleQueue()
        self._thread: Thread | None = None
        self._active_task_name: str | None = None
        self._cancel_event: Event | None = None
        self._cancelable = False

    @property
    def active_task_name(self) -> str | None:
        """Name of the currently running task, if any."""
        return self._active_task_name

    @property
    def cancelable(self) -> bool:
        """True when the active task supports cooperative cancellation."""
        return self._cancelable and self._active_task_name is not None

    def is_busy(self) -> bool:
        """Return True when a task is currently running."""
        return self._active_task_name is not None

    def start_task(self, task_name: str, runner: TaskRunner, *, cancelable: bool = False) -> None:
        """Start a background task and stream messages back to the caller."""
        if self.is_busy():
            raise RuntimeError(f"Task already running: {self._active_task_name}")

        cancel_event = Event() if cancelable else None
        self._active_task_name = task_name
        self._cancel_event = cancel_event
        self._cancelable = cancelable

        def _emit_progress(event: str, fields: dict[str, object]) -> None:
            self._messages.put(ControllerMessage(kind="progress", task_name=task_name, payload=wrap_progress(event, fields)))

        def _worker() -> None:
            self._messages.put(ControllerMessage(kind="started", task_name=task_name))
            try:
                result = runner(cancel_event, _emit_progress)
            except Exception as exc:  # noqa: BLE001
                self._messages.put(ControllerMessage(kind="error", task_name=task_name, payload=exc))
            else:
                self._messages.put(ControllerMessage(kind="result", task_name=task_name, payload=result))
            finally:
                self._messages.put(ControllerMessage(kind="finished", task_name=task_name))

        self._thread = Thread(target=_worker, name=f"ui-task-{task_name}", daemon=True)
        self._thread.start()

    def cancel_active_task(self) -> bool:
        """Request cooperative cancellation for the running task."""
        if self._cancel_event is None or self._active_task_name is None:
            return False
        self._cancel_event.set()
        return True

    def drain_messages(self) -> list[ControllerMessage]:
        """Return all queued messages emitted by background tasks."""
        messages: list[ControllerMessage] = []
        while True:
            try:
                message = self._messages.get_nowait()
            except Empty:
                break
            messages.append(message)
            if message.kind == "finished":
                self._active_task_name = None
                self._cancel_event = None
                self._thread = None
                self._cancelable = False
        return messages
