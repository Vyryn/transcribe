from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from transcribe.runtime_defaults import DEFAULT_SESSION_NOTES_MODEL
from transcribe.runtime_env import (
    RuntimeMode,
    default_notes_runtime,
    resolve_app_runtime_paths,
    resolve_bundled_notes_model_path,
)

DEFAULT_CLEAN_TRANSCRIPT_FILENAME = "clean_transcript.txt"
DEFAULT_CLIENT_NOTES_FILENAME = "client_notes.txt"
DEFAULT_OLLAMA_HOST = "127.0.0.1:11434"
DEFAULT_LLAMA_SERVER_HOST = "127.0.0.1"
_MAX_CLEAN_TRANSCRIPT_CHUNK_WORDS = 700
_TEMP_SERVER_START_TIMEOUT_SEC = 20.0
_MODEL_LOADING_RETRY_TIMEOUT_SEC = 120.0
_MODEL_LOADING_RETRY_INTERVAL_SEC = 0.5
_EMPTY_MODEL_OUTPUT_RETRY_ATTEMPTS = 2
_PROMPT_TIMEOUT_SEC = 1_800.0
_CLEAN_TRANSCRIPT_PROMPT = """You are repairing a rough ASR transcript into a readable transcript.

Return only the cleaned transcript text.

Requirements:
- Preserve the original meaning, sequence, and level of detail.
- Preserve speaker labels exactly when they are already present in the input.
- Keep each completed speaker turn or paragraph on its own paragraph line instead of collapsing everything into one block.
- Merge adjacent fragments when they clearly belong to the same sentence or speaker turn.
- Fix obvious punctuation, capitalization, spacing, and transcript chunk-boundary artifacts.
- Correct obvious ASR wording mistakes only when the intended phrasing is strongly supported by nearby context.
- Remove only text that is clearly ASR garbage, such as isolated non-words, impossible fragments, or duplicated overlap text.
- When a short span is not recoverable, prefer the least-committal readable wording supported by the input instead of inventing specifics.
- Do not summarize.
- Do not add facts, interpretations, stage directions, or speaker labels that are not directly supported.
- Do not include commentary, headings, markdown fences, or explanations.

Rough transcript:
"""
_NOTES_TRANSCRIPT_PREFIX = """Use the cleaned psychotherapy session transcript below as the only source of truth.

Output only the finished client note.

Cleaned transcript:
"""
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
_GPU_ERROR_NEEDLES = (
    "cuda",
    "gpu",
    "metal",
    "rocm",
    "hip",
    "no compatible gpus",
    "out of memory",
    "failed to load gpu",
)
_SERVER_UNAVAILABLE_NEEDLES = (
    "connection refused",
    "connect: operation not permitted",
    "dial tcp",
    "cannot assign requested address",
    "client has been closed",
    "actively refused",
    "connection reset by peer",
    "timed out",
)


class NotesRuntimeError(RuntimeError):
    """Raised when local session-note generation cannot complete."""


class NotesGpuRuntimeError(NotesRuntimeError):
    """Raised when the default notes runtime should retry on CPU."""


SessionNotesProgressCallback = Callable[[str, dict[str, object]], None]


def _subprocess_text_mode_kwargs() -> dict[str, str | bool]:
    """Return text-mode subprocess options that tolerate undecodable output."""
    return {
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }


@dataclass(slots=True)
class SessionNotesConfig:
    """Configuration for transcript cleanup and post-session notes generation."""

    transcript_path: Path
    output_dir: Path
    model: str = DEFAULT_SESSION_NOTES_MODEL
    clean_transcript_filename: str = DEFAULT_CLEAN_TRANSCRIPT_FILENAME
    client_notes_filename: str = DEFAULT_CLIENT_NOTES_FILENAME
    prompt_path: Path | None = None
    runtime: str = "auto"


@dataclass(slots=True)
class SessionNotesResult:
    """Persisted artifacts from post-transcription notes generation."""

    transcript_path: Path
    clean_transcript_path: Path
    client_notes_path: Path
    model: str
    cpu_fallback_used: bool
    clean_duration_sec: float
    notes_duration_sec: float


class PromptRuntime(Protocol):
    """Protocol for runtime backends used in notes-generation tests and production."""

    def ensure_model_available(self, model: str) -> None:
        """Raise when the requested model is unavailable."""

    def run_prompt(
        self,
        *,
        model: str,
        prompt: str,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Run one local prompt and return raw model output."""


@dataclass(slots=True)
class OllamaRuntimeSession:
    """Thin wrapper around a local Ollama runtime host."""

    env: dict[str, str]
    host: str
    server_process: subprocess.Popen[str] | None = None

    def ensure_model_available(self, model: str) -> None:
        result = _run_ollama_command(
            ["show", model],
            env=self.env,
            timeout_sec=30.0,
        )
        if result.returncode == 0:
            return

        detail = _command_error_detail(result)
        if _is_model_missing_error(detail):
            raise NotesRuntimeError(
                f"Session notes model {model!r} is not installed locally in Ollama. "
                "Import or pull it before running notes."
            )
        if _is_gpu_runtime_error(detail):
            raise NotesGpuRuntimeError(detail)
        raise NotesRuntimeError(f"Unable to use Ollama model {model!r}: {detail}")

    def run_prompt(
        self,
        *,
        model: str,
        prompt: str,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        result = _run_ollama_command(
            [
                "run",
                model,
                "--hidethinking",
                "--think=false",
                "--nowordwrap",
                "--keepalive=10m",
            ],
            env=self.env,
            input_text=prompt,
            timeout_sec=_PROMPT_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            detail = _command_error_detail(result)
            if _is_model_missing_error(detail):
                raise NotesRuntimeError(
                    f"Session notes model {model!r} is not installed locally in Ollama. "
                    "Import or pull it before running notes."
                )
            if _is_gpu_runtime_error(detail):
                raise NotesGpuRuntimeError(detail)
            if _is_server_unavailable_error(detail):
                raise NotesRuntimeError(
                    "Unable to reach a local Ollama runtime for session notes generation. "
                    "Start `ollama serve` or ensure the local daemon can accept requests."
                )
            raise NotesRuntimeError(f"Ollama notes generation failed: {detail}")
        if on_text_delta is not None and result.stdout:
            on_text_delta(result.stdout)
        return result.stdout


@dataclass(slots=True)
class LlamaCppRuntimeSession:
    """Private llama.cpp server runtime used by bundled builds."""

    binary_path: Path
    model_path: Path
    host: str
    server_process: subprocess.Popen[str] | None = None

    def ensure_model_available(self, model: str) -> None:
        if not self.binary_path.exists():
            raise NotesRuntimeError(
                f"Bundled notes runtime is missing llama-server binary: {self.binary_path}"
            )
        if not self.model_path.exists():
            raise NotesRuntimeError(
                f"Bundled notes model for {model!r} is missing: {self.model_path}"
            )

    def run_prompt(
        self,
        *,
        model: str,
        prompt: str,
        on_text_delta: Callable[[str], None] | None = None,
    ) -> str:
        del model
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": on_text_delta is not None,
        }
        if on_text_delta is not None:
            return _llama_server_stream_chat_completion_request(
                host=self.host,
                payload=payload,
                on_text_delta=on_text_delta,
            )
        response = _llama_server_chat_completion_request(
            host=self.host,
            payload=payload,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise NotesRuntimeError("Bundled notes runtime returned no completion choices.")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise NotesRuntimeError("Bundled notes runtime returned an invalid completion payload.")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise NotesRuntimeError("Bundled notes runtime returned no message content.")
        content = message.get("content")
        if not isinstance(content, str):
            raise NotesRuntimeError("Bundled notes runtime returned non-text content.")
        return content


def _llama_server_chat_completion_request(
    *,
    host: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Call llama.cpp chat completions, retrying while the model is still loading."""
    deadline = time.monotonic() + min(_MODEL_LOADING_RETRY_TIMEOUT_SEC, _PROMPT_TIMEOUT_SEC)
    last_loading_error: NotesRuntimeError | None = None
    while True:
        try:
            return _llama_server_request(
                host=host,
                path="/v1/chat/completions",
                payload=payload,
                timeout_sec=_PROMPT_TIMEOUT_SEC,
            )
        except NotesRuntimeError as exc:
            if not _is_model_loading_error(str(exc)):
                raise
            last_loading_error = exc
            if time.monotonic() >= deadline:
                raise NotesRuntimeError(
                    "Bundled llama.cpp runtime did not finish loading the model before timeout."
                ) from exc
            time.sleep(_MODEL_LOADING_RETRY_INTERVAL_SEC)


def _llama_server_stream_chat_completion_request(
    *,
    host: str,
    payload: dict[str, object],
    on_text_delta: Callable[[str], None],
) -> str:
    """Stream llama.cpp chat completions while retrying transient model-loading responses."""
    deadline = time.monotonic() + min(_MODEL_LOADING_RETRY_TIMEOUT_SEC, _PROMPT_TIMEOUT_SEC)
    while True:
        try:
            return _llama_server_stream_request_once(
                host=host,
                path="/v1/chat/completions",
                payload=payload,
                timeout_sec=_PROMPT_TIMEOUT_SEC,
                on_text_delta=on_text_delta,
            )
        except NotesRuntimeError as exc:
            if not _is_model_loading_error(str(exc)):
                raise
            if time.monotonic() >= deadline:
                raise NotesRuntimeError(
                    "Bundled llama.cpp runtime did not finish loading the model before timeout."
                ) from exc
            time.sleep(_MODEL_LOADING_RETRY_INTERVAL_SEC)


def default_notes_prompt_path() -> Path:
    """Return the repository-managed clinical note prompt path."""
    runtime_paths = resolve_app_runtime_paths()
    if runtime_paths.mode == RuntimeMode.PACKAGED:
        return runtime_paths.notes_prompt_path
    return Path(__file__).resolve().parent.parent / "clinical_note_synthesis_llm_prompt.md"


def load_session_note_prompt(prompt_path: Path | None = None) -> str:
    """Load the clinician-note prompt template from disk."""
    resolved_path = prompt_path or default_notes_prompt_path()
    prompt = resolved_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise NotesRuntimeError(f"Session notes prompt is empty: {resolved_path}")
    return prompt


def build_clean_transcript_prompt(rough_transcript: str) -> str:
    """Build the cleanup prompt for rough ASR transcript text."""
    return f"{_CLEAN_TRANSCRIPT_PROMPT}\n{rough_transcript.strip()}\n"


def build_client_notes_prompt(prompt_template: str, clean_transcript: str) -> str:
    """Build the grounded session-notes prompt from template plus transcript."""
    return f"{prompt_template.strip()}\n\n{_NOTES_TRANSCRIPT_PREFIX}\n{clean_transcript.strip()}\n"


def _normalize_model_output(text: str) -> str:
    """Strip auxiliary output so persisted artifacts contain only model content."""
    normalized = _ANSI_ESCAPE_RE.sub("", text).strip()
    normalized = _THINK_BLOCK_RE.sub("", normalized).strip()
    if normalized.startswith("```") and normalized.endswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3:
            normalized = "\n".join(lines[1:-1]).strip()
    return normalized


def _write_text_artifact(path: Path, text: str) -> None:
    """Persist a UTF-8 text artifact with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _run_normalized_prompt_with_retries(
    *,
    runtime: PromptRuntime,
    model: str,
    prompt: str,
    attempts: int = _EMPTY_MODEL_OUTPUT_RETRY_ATTEMPTS,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    """Run one prompt, retrying when the model returns only empty/auxiliary output."""
    last_output = ""
    for _attempt_index in range(max(1, attempts)):
        raw_output = _run_runtime_prompt(
            runtime=runtime,
            model=model,
            prompt=prompt,
            on_text_delta=on_text_delta,
        )
        normalized_output = _normalize_model_output(raw_output)
        if normalized_output:
            return normalized_output
        last_output = raw_output
    return _normalize_model_output(last_output)


def _run_runtime_prompt(
    *,
    runtime: PromptRuntime,
    model: str,
    prompt: str,
    on_text_delta: Callable[[str], None] | None = None,
) -> str:
    """Call one runtime prompt method while tolerating older non-streaming runtimes."""
    if on_text_delta is None:
        return runtime.run_prompt(model=model, prompt=prompt)
    try:
        return runtime.run_prompt(
            model=model,
            prompt=prompt,
            on_text_delta=on_text_delta,
        )
    except TypeError as exc:
        if "on_text_delta" not in str(exc):
            raise
        result = runtime.run_prompt(model=model, prompt=prompt)
        if result:
            on_text_delta(result)
        return result


def _subprocess_creationflags_no_window() -> int:
    """Return Windows subprocess flags that avoid spawning a console window."""
    create_no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return int(create_no_window) if os.name == "nt" else 0


def run_post_transcription_notes(
    config: SessionNotesConfig,
    *,
    runtime_factory: Callable[..., contextlib.AbstractContextManager[PromptRuntime]] | None = None,
    progress_callback: SessionNotesProgressCallback | None = None,
) -> SessionNotesResult:
    """Clean a transcript locally, then generate client notes locally."""
    transcript_units = load_transcript_units(config.transcript_path)
    if not transcript_units:
        raise NotesRuntimeError(f"Transcript is empty: {config.transcript_path}")
    cleanup_chunks = build_cleanup_chunks(transcript_units, max_words=_MAX_CLEAN_TRANSCRIPT_CHUNK_WORDS)
    _emit_progress(
        progress_callback,
        "notes_started",
        model=config.model,
        cleanup_chunk_count=len(cleanup_chunks),
    )

    prompt_template = load_session_note_prompt(config.prompt_path)
    clean_transcript_path = config.output_dir / config.clean_transcript_filename
    client_notes_path = config.output_dir / config.client_notes_filename
    factory = runtime_factory or _default_runtime_factory(config)

    def _generate_once(*, cpu_only: bool) -> tuple[str, str, float, float]:
        with factory(cpu_only=cpu_only) as runtime:
            runtime.ensure_model_available(config.model)

            _emit_progress(
                progress_callback,
                "clean_transcript_started",
                chunk_count=len(cleanup_chunks),
                cpu_only=cpu_only,
            )
            clean_started = time.monotonic()
            cleaned_chunks: list[str] = []
            for chunk_index, cleanup_chunk in enumerate(cleanup_chunks, start=1):
                _emit_progress(
                    progress_callback,
                    "clean_transcript_chunk_started",
                    chunk_index=chunk_index,
                    chunk_count=len(cleanup_chunks),
                    cpu_only=cpu_only,
                )
                cleaned_chunk = _run_normalized_prompt_with_retries(
                    runtime=runtime,
                    model=config.model,
                    prompt=build_clean_transcript_prompt(cleanup_chunk),
                )
                if not cleaned_chunk:
                    cleaned_chunk = cleanup_chunk.strip()
                    _emit_progress(
                        progress_callback,
                        "clean_transcript_chunk_fallback",
                        chunk_index=chunk_index,
                        chunk_count=len(cleanup_chunks),
                        cpu_only=cpu_only,
                    )
                _emit_progress(
                    progress_callback,
                    "clean_transcript_chunk_ready",
                    chunk_index=chunk_index,
                    chunk_count=len(cleanup_chunks),
                    cpu_only=cpu_only,
                    text=cleaned_chunk,
                )
                cleaned_chunks.append(cleaned_chunk)
            clean_transcript = "\n\n".join(chunk.strip() for chunk in cleaned_chunks if chunk.strip()).strip()
            clean_duration_sec = time.monotonic() - clean_started
            if not clean_transcript:
                raise NotesRuntimeError("Clean transcript generation returned no content.")
            _emit_progress(
                progress_callback,
                "clean_transcript_ready",
                chunk_count=len(cleanup_chunks),
                cpu_only=cpu_only,
            )

            _emit_progress(
                progress_callback,
                "client_notes_started",
                cpu_only=cpu_only,
            )
            notes_started = time.monotonic()
            streamed_note_text = False

            def _emit_notes_delta(text: str) -> None:
                nonlocal streamed_note_text
                if not text:
                    return
                streamed_note_text = True
                _emit_progress(
                    progress_callback,
                    "client_notes_delta",
                    cpu_only=cpu_only,
                    text=text,
                )

            client_notes = _run_normalized_prompt_with_retries(
                runtime=runtime,
                model=config.model,
                prompt=build_client_notes_prompt(prompt_template, clean_transcript),
                on_text_delta=_emit_notes_delta,
            )
            notes_duration_sec = time.monotonic() - notes_started
            if not client_notes:
                raise NotesRuntimeError("Client notes generation returned no content.")
            _emit_progress(
                progress_callback,
                "client_notes_ready",
                cpu_only=cpu_only,
                text=client_notes,
                streamed=streamed_note_text,
            )

            return clean_transcript, client_notes, clean_duration_sec, notes_duration_sec

    cpu_fallback_used = False
    try:
        clean_transcript, client_notes, clean_duration_sec, notes_duration_sec = _generate_once(cpu_only=False)
    except NotesGpuRuntimeError:
        _emit_progress(progress_callback, "notes_cpu_fallback")
        clean_transcript, client_notes, clean_duration_sec, notes_duration_sec = _generate_once(cpu_only=True)
        cpu_fallback_used = True

    _write_text_artifact(clean_transcript_path, clean_transcript)
    _write_text_artifact(client_notes_path, client_notes)
    return SessionNotesResult(
        transcript_path=config.transcript_path,
        clean_transcript_path=clean_transcript_path,
        client_notes_path=client_notes_path,
        model=config.model,
        cpu_fallback_used=cpu_fallback_used,
        clean_duration_sec=clean_duration_sec,
        notes_duration_sec=notes_duration_sec,
    )


def _emit_progress(
    progress_callback: SessionNotesProgressCallback | None,
    event: str,
    **fields: object,
) -> None:
    """Report a structured notes-generation progress event."""
    if progress_callback is None:
        return
    progress_callback(event, dict(fields))


def _default_runtime_factory(
    config: SessionNotesConfig,
) -> Callable[..., contextlib.AbstractContextManager[PromptRuntime]]:
    """Resolve the default notes runtime factory for one request."""
    runtime_name = (config.runtime or "auto").strip().lower()
    if runtime_name in {"", "auto"}:
        preferred_runtime = default_notes_runtime()
        alternate_runtime = "llama_cpp" if preferred_runtime == "ollama" else "ollama"
        return _open_auto_notes_runtime(
            model=config.model,
            preferred_runtime=preferred_runtime,
            alternate_runtime=alternate_runtime,
        )
    return _runtime_factory_for_name(runtime_name, model=config.model)


def _runtime_factory_for_name(
    runtime_name: str,
    *,
    model: str,
) -> Callable[..., contextlib.AbstractContextManager[PromptRuntime]]:
    """Resolve one explicit runtime name into a context-manager factory."""
    if runtime_name == "ollama":
        return open_ollama_runtime
    if runtime_name == "llama_cpp":
        return lambda *, cpu_only=False: open_llama_cpp_runtime(model=model, cpu_only=cpu_only)
    raise NotesRuntimeError(f"Unsupported notes runtime {runtime_name!r}. Use 'auto', 'ollama', or 'llama_cpp'.")


def _open_auto_notes_runtime(
    *,
    model: str,
    preferred_runtime: str,
    alternate_runtime: str,
) -> Callable[..., contextlib.AbstractContextManager[PromptRuntime]]:
    """Build an auto runtime factory that can fall back across local runtimes."""

    @contextlib.contextmanager
    def _factory(*, cpu_only: bool = False) -> Iterator[PromptRuntime]:
        last_error: NotesRuntimeError | None = None
        attempted_messages: list[str] = []
        for runtime_name in (preferred_runtime, alternate_runtime):
            runtime_factory = _runtime_factory_for_name(runtime_name, model=model)
            try:
                with runtime_factory(cpu_only=cpu_only) as runtime:
                    yield runtime
                    return
            except NotesRuntimeError as exc:
                if not _is_runtime_bootstrap_unavailable_error(str(exc)):
                    raise
                last_error = exc
                attempted_messages.append(f"{runtime_name}: {exc}")
                continue

        detail = "; ".join(attempted_messages) if attempted_messages else "no runtime candidates attempted"
        if last_error is not None:
            raise NotesRuntimeError(
                f"Unable to initialize any local notes runtime. Tried {detail}"
            ) from last_error
        raise NotesRuntimeError("Unable to initialize any local notes runtime.")

    return _factory


def load_transcript_units(transcript_path: Path) -> list[str]:
    """Load transcript text as paragraph/turn units, using structured session JSON when available."""
    structured_units = _load_structured_session_units(transcript_path)
    if structured_units:
        return structured_units

    transcript_text = transcript_path.read_text(encoding="utf-8")
    return [line.strip() for line in transcript_text.splitlines() if line.strip()]


def build_cleanup_chunks(transcript_units: list[str], *, max_words: int) -> list[str]:
    """Group transcript units into word-budgeted cleanup chunks."""
    if max_words <= 0:
        raise ValueError("max_words must be > 0")

    normalized_units: list[str] = []
    for unit in transcript_units:
        normalized_units.extend(_split_oversized_unit(unit.strip(), max_words=max_words))
    normalized_units = [unit for unit in normalized_units if unit]
    if not normalized_units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    current_words = 0
    for unit in normalized_units:
        unit_words = _word_count(unit)
        if current_units and (current_words + unit_words) > max_words:
            chunks.append("\n\n".join(current_units))
            current_units = []
            current_words = 0
        current_units.append(unit)
        current_words += unit_words

    if current_units:
        chunks.append("\n\n".join(current_units))
    return chunks


def _load_structured_session_units(transcript_path: Path) -> list[str]:
    """Extract turn-preserving units from a live-session transcript JSON when present."""
    json_path = _candidate_transcript_json_path(transcript_path)
    if json_path is None or not json_path.exists():
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    raw_segments = payload.get("final_segments")
    if not isinstance(raw_segments, list):
        return []

    segments: list[tuple[str | None, str]] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        raw_text = raw_segment.get("text")
        if not isinstance(raw_text, str):
            continue
        text = raw_text.strip()
        if not text:
            continue
        raw_source = raw_segment.get("selected_source")
        source = raw_source if isinstance(raw_source, str) and raw_source.strip() else None
        segments.append((source, text))
    if not segments:
        return []

    ordered_sources: list[str] = []
    for source, _ in segments:
        if source is None or source in ordered_sources:
            continue
        ordered_sources.append(source)
    source_labels = {
        source: f"Speaker {chr(ord('A') + index)}"
        for index, source in enumerate(ordered_sources)
    }

    if len(source_labels) <= 1:
        return [text for _, text in segments]

    turns: list[str] = []
    current_label: str | None = None
    current_parts: list[str] = []
    for source, text in segments:
        label = source_labels.get(source, "Speaker")
        if label != current_label and current_parts:
            turns.append(f"{current_label}: {' '.join(current_parts)}")
            current_parts = []
        current_label = label
        current_parts.append(text)

    if current_label is not None and current_parts:
        turns.append(f"{current_label}: {' '.join(current_parts)}")
    return turns


def _candidate_transcript_json_path(transcript_path: Path) -> Path | None:
    """Find a structured transcript JSON file associated with the provided transcript path."""
    if transcript_path.suffix.lower() == ".json":
        return transcript_path

    direct_match = transcript_path.with_suffix(".json")
    if direct_match.exists():
        return direct_match

    sibling_transcript_json = transcript_path.parent / "transcript.json"
    if transcript_path.name.endswith(".txt") and sibling_transcript_json.exists():
        return sibling_transcript_json
    return None


def _split_oversized_unit(unit: str, *, max_words: int) -> list[str]:
    """Split a very large transcript unit without losing simple speaker labels."""
    if not unit:
        return []
    if _word_count(unit) <= max_words:
        return [unit]

    label, body = _split_unit_label(unit)
    words = body.split()
    chunks: list[str] = []
    for start_index in range(0, len(words), max_words):
        body_chunk = " ".join(words[start_index : start_index + max_words]).strip()
        if not body_chunk:
            continue
        if label:
            chunks.append(f"{label} {body_chunk}")
        else:
            chunks.append(body_chunk)
    return chunks


def _split_unit_label(unit: str) -> tuple[str, str]:
    """Separate a synthetic speaker label prefix from transcript text."""
    match = re.match(r"^(Speaker [A-Z]:)\s+(.*)$", unit)
    if match is None:
        return "", unit
    return match.group(1), match.group(2)


def _word_count(text: str) -> int:
    """Count words for chunk-size budgeting."""
    return len(text.split())


@contextlib.contextmanager
def open_llama_cpp_runtime(
    *,
    model: str,
    cpu_only: bool = False,
) -> Iterator[PromptRuntime]:
    """Yield a private bundled llama.cpp server runtime."""
    runtime_paths = resolve_app_runtime_paths()
    try:
        model_path = resolve_bundled_notes_model_path(model, runtime_paths=runtime_paths)
    except ValueError as exc:
        raise NotesRuntimeError(str(exc)) from exc
    executable = runtime_paths.notes_runtime_binary
    if not executable.exists():
        path_lookup = shutil.which("llama-server")
        if path_lookup is None:
            raise NotesRuntimeError(
                "Session notes generation requires bundled llama-server or a local `llama-server` on PATH."
            )
        executable = Path(path_lookup)
    with _temporary_llama_cpp_runtime(
        executable=executable,
        model_path=model_path,
        cpu_only=cpu_only,
    ) as runtime:
        yield runtime


@contextlib.contextmanager
def _temporary_llama_cpp_runtime(
    *,
    executable: Path,
    model_path: Path,
    cpu_only: bool,
) -> Iterator[LlamaCppRuntimeSession]:
    """Run a private llama.cpp server process on a free loopback port."""
    host = _loopback_host_for_free_port()
    host_name, port = _split_host_port(host)
    env = os.environ.copy()
    if cpu_only:
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "HIP_VISIBLE_DEVICES": "-1",
                "ROCR_VISIBLE_DEVICES": "-1",
                "GPU_DEVICE_ORDINAL": "-1",
            }
        )
    command = [
        str(executable),
        "-m",
        str(model_path),
        "--host",
        host_name or DEFAULT_LLAMA_SERVER_HOST,
        "--port",
        str(port),
        "--n-gpu-layers",
        "0",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **_subprocess_text_mode_kwargs(),
        creationflags=_subprocess_creationflags_no_window(),
    )
    try:
        _wait_for_llama_cpp_runtime_ready(process=process, host=host)
        yield LlamaCppRuntimeSession(
            binary_path=executable,
            model_path=model_path,
            host=host,
            server_process=process,
        )
    finally:
        _terminate_process(process)


@contextlib.contextmanager
def open_ollama_runtime(*, cpu_only: bool = False) -> Iterator[PromptRuntime]:
    """Yield a usable local Ollama runtime, starting a private server when needed."""
    default_env = _base_ollama_env(host=DEFAULT_OLLAMA_HOST)
    if cpu_only:
        with _temporary_ollama_runtime(cpu_only=True) as runtime:
            yield runtime
        return

    availability = _run_ollama_command(["list"], env=default_env, timeout_sec=20.0)
    if availability.returncode == 0:
        yield OllamaRuntimeSession(env=default_env, host=DEFAULT_OLLAMA_HOST)
        return

    detail = _command_error_detail(availability)
    if _is_gpu_runtime_error(detail):
        raise NotesGpuRuntimeError(detail)
    if _is_server_unavailable_error(detail):
        with _temporary_ollama_runtime(cpu_only=False) as runtime:
            yield runtime
        return
    raise NotesRuntimeError(f"Unable to initialize local Ollama runtime: {detail}")


@contextlib.contextmanager
def _temporary_ollama_runtime(*, cpu_only: bool) -> Iterator[OllamaRuntimeSession]:
    """Run a private `ollama serve` instance on a free loopback port."""
    executable = _ollama_executable()
    host = _loopback_host_for_free_port()
    env = _base_ollama_env(host=host)
    if cpu_only:
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "HIP_VISIBLE_DEVICES": "-1",
                "ROCR_VISIBLE_DEVICES": "-1",
                "GPU_DEVICE_ORDINAL": "-1",
            }
        )
    process = subprocess.Popen(
        [executable, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        **_subprocess_text_mode_kwargs(),
        creationflags=_subprocess_creationflags_no_window(),
    )
    try:
        _wait_for_runtime_ready(process=process, env=env)
        yield OllamaRuntimeSession(env=env, host=host, server_process=process)
    finally:
        _terminate_process(process)


def _base_ollama_env(*, host: str) -> dict[str, str]:
    """Build the environment used for local Ollama CLI calls."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = host
    env["OLLAMA_NOHISTORY"] = "1"
    return env


def _ollama_executable() -> str:
    """Resolve the Ollama executable or raise a user-facing runtime error."""
    executable = shutil.which("ollama")
    if executable is None:
        raise NotesRuntimeError(
            "Session notes generation requires the local `ollama` command to be installed and on PATH."
        )
    return executable


def _split_host_port(host: str) -> tuple[str, int]:
    """Split a loopback host string into host and integer port."""
    host_name, separator, raw_port = host.rpartition(":")
    if not separator or not host_name:
        raise NotesRuntimeError(f"Invalid runtime host value: {host!r}")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise NotesRuntimeError(f"Invalid runtime port value in host {host!r}") from exc
    return host_name, port


def _loopback_host_for_free_port() -> str:
    """Allocate a loopback port for a private Ollama runtime."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"127.0.0.1:{port}"


def _wait_for_llama_cpp_runtime_ready(*, process: subprocess.Popen[str], host: str) -> None:
    """Poll until a private llama.cpp server is ready to accept requests."""
    deadline = time.monotonic() + _TEMP_SERVER_START_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read().strip()
            if _is_gpu_runtime_error(stderr):
                raise NotesGpuRuntimeError(stderr)
            detail = stderr or "temporary llama.cpp server exited before becoming ready"
            raise NotesRuntimeError(f"Unable to start bundled llama.cpp runtime: {detail}")

        try:
            _llama_server_request(
                host=host,
                path="/v1/models",
                payload=None,
                timeout_sec=5.0,
                method="GET",
            )
            return
        except NotesRuntimeError as exc:
            detail = str(exc)
            if not _is_server_unavailable_error(detail) and not _is_model_loading_error(detail):
                raise
        time.sleep(0.25)

    raise NotesRuntimeError("Timed out while waiting for a private bundled llama.cpp runtime to start.")


def _wait_for_runtime_ready(*, process: subprocess.Popen[str], env: dict[str, str]) -> None:
    """Poll until a private Ollama server is ready to accept requests."""
    deadline = time.monotonic() + _TEMP_SERVER_START_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = ""
            if process.stderr is not None:
                stderr = process.stderr.read().strip()
            if _is_gpu_runtime_error(stderr):
                raise NotesGpuRuntimeError(stderr)
            detail = stderr or "temporary Ollama server exited before becoming ready"
            raise NotesRuntimeError(f"Unable to start local Ollama runtime: {detail}")

        probe = _run_ollama_command(["list"], env=env, timeout_sec=5.0)
        if probe.returncode == 0:
            return
        time.sleep(0.25)

    raise NotesRuntimeError("Timed out while waiting for a private local Ollama runtime to start.")


def _llama_server_request(
    *,
    host: str,
    path: str,
    payload: dict[str, object] | None,
    timeout_sec: float,
    method: str = "POST",
) -> dict[str, object]:
    """Call one private llama.cpp server endpoint and decode JSON."""
    data: bytes | None = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    runtime_url = urllib_parse.urlunsplit(("http", host, path, "", ""))
    request = urllib_request.Request(
        runtime_url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8").strip()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if _is_gpu_runtime_error(detail):
            raise NotesGpuRuntimeError(detail) from exc
        raise NotesRuntimeError(detail or f"llama.cpp server returned HTTP {exc.code}") from exc
    except urllib_error.URLError as exc:
        detail = str(exc.reason or exc).strip()
        if _is_server_unavailable_error(detail):
            raise NotesRuntimeError(detail) from exc
        raise NotesRuntimeError(f"Unable to contact bundled llama.cpp runtime: {detail}") from exc
    except TimeoutError as exc:
        raise NotesRuntimeError("Bundled llama.cpp request timed out.") from exc

    if not body:
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise NotesRuntimeError("Bundled llama.cpp runtime returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise NotesRuntimeError("Bundled llama.cpp runtime returned an unexpected JSON payload.")
    return parsed


def _llama_server_stream_request_once(
    *,
    host: str,
    path: str,
    payload: dict[str, object],
    timeout_sec: float,
    on_text_delta: Callable[[str], None],
) -> str:
    """Call one llama.cpp streaming endpoint and emit text deltas as they arrive."""
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    runtime_url = urllib_parse.urlunsplit(("http", host, path, "", ""))
    request = urllib_request.Request(
        runtime_url,
        data=data,
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    content_parts: list[str] = []
    try:
        with urllib_request.urlopen(request, timeout=timeout_sec) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_text = line[5:].strip()
                if not payload_text or payload_text == "[DONE]":
                    continue
                try:
                    parsed = json.loads(payload_text)
                except json.JSONDecodeError as exc:
                    raise NotesRuntimeError("Bundled llama.cpp runtime returned invalid stream JSON.") from exc
                if not isinstance(parsed, dict):
                    raise NotesRuntimeError("Bundled llama.cpp runtime returned an unexpected stream payload.")
                delta_text = _extract_llama_stream_delta(parsed)
                if not delta_text:
                    continue
                content_parts.append(delta_text)
                on_text_delta(delta_text)
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if _is_gpu_runtime_error(detail):
            raise NotesGpuRuntimeError(detail) from exc
        raise NotesRuntimeError(detail or f"llama.cpp server returned HTTP {exc.code}") from exc
    except urllib_error.URLError as exc:
        detail = str(exc.reason or exc).strip()
        if _is_server_unavailable_error(detail):
            raise NotesRuntimeError(detail) from exc
        raise NotesRuntimeError(f"Unable to contact bundled llama.cpp runtime: {detail}") from exc
    except TimeoutError as exc:
        raise NotesRuntimeError("Bundled llama.cpp request timed out.") from exc
    return "".join(content_parts)


def _extract_llama_stream_delta(payload: dict[str, object]) -> str:
    """Extract one streamed text delta from a llama.cpp SSE payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _run_ollama_command(
    argv: list[str],
    *,
    env: dict[str, str],
    input_text: str | None = None,
    timeout_sec: float,
) -> subprocess.CompletedProcess[str]:
    """Run one Ollama CLI command and capture UTF-8 output."""
    try:
        return subprocess.run(
            [_ollama_executable(), *argv],
            env=env,
            input=input_text,
            capture_output=True,
            **_subprocess_text_mode_kwargs(),
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NotesRuntimeError(f"Ollama command timed out: {' '.join(argv)}") from exc
    except OSError as exc:
        raise NotesRuntimeError(f"Failed to execute Ollama command {' '.join(argv)!r}: {exc}") from exc


def _command_error_detail(result: subprocess.CompletedProcess[str]) -> str:
    """Extract the most useful error detail from a completed process."""
    detail = (result.stderr or "").strip()
    if detail:
        return detail
    return (result.stdout or "").strip() or f"exit code {result.returncode}"


def _is_model_missing_error(message: str) -> bool:
    """Detect missing-model messages from local Ollama commands."""
    lowered = message.lower()
    return "not found" in lowered or "pull model" in lowered or ("manifest" in lowered and "not" in lowered)


def _is_gpu_runtime_error(message: str) -> bool:
    """Detect GPU-runtime failures that warrant a CPU retry."""
    lowered = message.lower()
    return any(needle in lowered for needle in _GPU_ERROR_NEEDLES)


def _is_model_loading_error(message: str) -> bool:
    """Detect llama.cpp responses that mean the server is alive but still loading a model."""
    lowered = message.lower()
    return (
        "loading model" in lowered
        or '"type":"unavailable_error"' in lowered
        or "'type': 'unavailable_error'" in lowered
    )


def _is_server_unavailable_error(message: str) -> bool:
    """Detect inability to connect to a local Ollama runtime."""
    lowered = message.lower()
    return any(needle in lowered for needle in _SERVER_UNAVAILABLE_NEEDLES)


def _is_runtime_bootstrap_unavailable_error(message: str) -> bool:
    """Detect runtime startup failures where trying another local backend is reasonable."""
    lowered = message.lower()
    needles = (
        "requires bundled llama-server",
        "missing llama-server binary",
        "bundled notes model",
        "not installed locally in ollama",
        "requires the local `ollama` command",
        "unable to reach a local ollama runtime",
        "unable to initialize local ollama runtime",
        "unable to start local ollama runtime",
        "timed out while waiting for a private local ollama runtime to start",
        "timed out while waiting for a private bundled llama.cpp runtime to start",
        "unable to start bundled llama.cpp runtime",
    )
    return any(needle in lowered for needle in needles)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    """Best-effort shutdown for a temporary Ollama server process."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)
