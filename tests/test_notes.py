from __future__ import annotations

import argparse
import contextlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from transcribe.notes import (
    LlamaCppRuntimeSession,
    NotesGpuRuntimeError,
    PromptRequestOptions,
    SessionNotesConfig,
    SessionNotesResult,
    _default_runtime_factory,
    build_notes_execution_plan,
    build_cleanup_chunks,
    build_clean_transcript_prompt,
    build_client_notes_prompt,
    load_transcript_units,
    run_post_transcription_notes,
)
from transcribe.models import AudioSourceMode
from transcribe.runtime_defaults import DEFAULT_SESSION_NOTES_MODEL


class FakePromptRuntime:
    """Simple fake runtime for deterministic notes-generation tests."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.checked_models: list[str] = []
        self.prompts: list[str] = []
        self.request_options: list[PromptRequestOptions | None] = []

    def ensure_model_available(self, model: str) -> None:
        self.checked_models.append(model)

    def run_prompt(
        self,
        *,
        model: str,
        prompt: str,
        on_text_delta=None,
        request_options: PromptRequestOptions | None = None,
    ) -> str:
        self.checked_models.append(model)
        self.prompts.append(prompt)
        self.request_options.append(request_options)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        text = str(response)
        if on_text_delta is not None and text:
            on_text_delta(text)
        return text


class FakeRuntimeFactory:
    """Record whether the pipeline retried on CPU."""

    def __init__(self, runtimes_by_cpu: dict[bool, list[FakePromptRuntime]]) -> None:
        self._runtimes_by_cpu = runtimes_by_cpu
        self.calls: list[bool] = []

    @contextlib.contextmanager
    def __call__(self, *, cpu_only: bool = False):
        self.calls.append(cpu_only)
        runtime = self._runtimes_by_cpu[cpu_only].pop(0)
        yield runtime


def test_build_clean_transcript_prompt_embeds_rough_transcript() -> None:
    prompt = build_clean_transcript_prompt("rough transcript")
    assert "rough transcript" in prompt
    assert "repairing a rough ASR transcript into a readable transcript" in prompt
    assert "Remove only text that is clearly ASR garbage" in prompt
    assert "Return only the cleaned transcript text" in prompt


def test_build_client_notes_prompt_embeds_template_and_transcript() -> None:
    prompt = build_client_notes_prompt("SYSTEM PROMPT", "clean transcript")
    assert "SYSTEM PROMPT" in prompt
    assert "clean transcript" in prompt
    assert "Output only the finished client note" in prompt


def test_load_transcript_units_uses_structured_session_json_when_available(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text("flat transcript\n", encoding="utf-8")
    (tmp_path / "transcript.json").write_text(
        """
        {
          "final_segments": [
            {"selected_source": "mic", "text": "hello there"},
            {"selected_source": "mic", "text": "how are you"},
            {"selected_source": "speakers", "text": "i am okay"},
            {"selected_source": "speakers", "text": "thank you"}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    units = load_transcript_units(transcript_path)

    assert units == [
        "Speaker A: hello there how are you",
        "Speaker B: i am okay thank you",
    ]


def test_build_cleanup_chunks_splits_long_transcripts_by_word_budget() -> None:
    chunks = build_cleanup_chunks(
        [
            "Speaker A: " + " ".join(f"word{i}" for i in range(12)),
            "Speaker B: short reply",
        ],
        max_words=5,
    )

    assert len(chunks) >= 3
    assert chunks[0].startswith("Speaker A:")
    assert any("Speaker B: short reply" in chunk for chunk in chunks)


def test_build_notes_execution_plan_skips_cleanup_for_structured_transcript() -> None:
    transcript_units = [
        "Speaker A: Client reports improved sleep and lower anxiety after taking time off. "
        "Client says morning routines, reduced overtime, and clearer boundaries have helped noticeably. "
        "Client also reports fewer early-morning awakenings and steadier energy during the day.",
        "Speaker B: Therapist reflected the improvement and asked what helped most. "
        "Therapist also reviewed how the client handled work stress during the week. "
        "Therapist highlighted the difference between this week and the prior month.",
        "Speaker A: Client identified boundaries at work and daily walks as the main changes. "
        "Client describes fewer spirals, better rest, and more patience at home. "
        "Client says evening routines felt calmer and less rushed this week.",
        "Speaker B: Therapist reinforced the observed progress and asked how the client wants to maintain it. "
        "Therapist summarized the practical habits that seemed most effective. "
        "Therapist also reflected the client's growing confidence in using those habits.",
        "Speaker A: Client reports the session felt clarifying and says the plan is to continue the same routines. "
        "Client adds that communication with their partner felt calmer this week. "
        "Client reports feeling more hopeful about sustaining the recent changes.",
    ]

    plan = build_notes_execution_plan(
        model="qwen3.5:2b-q4_K_M",
        transcript_units=transcript_units,
        prompt_template="WRITE THE NOTE",
    )

    assert plan.cleanup_required is False
    assert plan.cleanup_chunks == ()
    assert plan.cleanup_request.max_tokens == 384
    assert plan.notes_request.max_tokens == 768
    assert plan.llama_launch.context_tokens >= 16_384


def test_run_post_transcription_notes_writes_clean_transcript_and_notes(tmp_path: Path) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text("um client says things\n", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    runtime = FakePromptRuntime(["Clean transcript", "Client note"])
    runtime_factory = FakeRuntimeFactory({False: [runtime], True: []})

    result = run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=transcript_path,
            output_dir=tmp_path / "notes_out",
            model="qwen3.5:4b-q4_K_M",
            prompt_path=prompt_path,
        ),
        runtime_factory=runtime_factory,
    )

    assert result.clean_transcript_path.read_text(encoding="utf-8") == "Clean transcript\n"
    assert result.client_notes_path.read_text(encoding="utf-8") == "Client note\n"
    assert runtime.checked_models[0] == "qwen3.5:4b-q4_K_M"
    assert "um client says things" in runtime.prompts[0]
    assert "WRITE THE NOTE" in runtime.prompts[1]
    assert "Clean transcript" in runtime.prompts[1]
    assert runtime.request_options[0] == PromptRequestOptions(max_tokens=384, context_tokens=8_192)
    assert runtime.request_options[1] == PromptRequestOptions(max_tokens=1_024, context_tokens=16_384)
    assert runtime_factory.calls == [False]


def test_run_post_transcription_notes_uses_multiple_cleanup_prompts_for_long_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text(
        "\n".join(" ".join(f"word{index}_{i}" for i in range(450)) for index in range(3)),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    runtime = FakePromptRuntime(
        [
            "Cleaned chunk one",
            "Cleaned chunk two",
            "Cleaned chunk three",
            "Client note",
        ]
    )
    runtime_factory = FakeRuntimeFactory({False: [runtime], True: []})

    result = run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=transcript_path,
            output_dir=tmp_path / "notes_out",
            prompt_path=prompt_path,
        ),
        runtime_factory=runtime_factory,
    )

    assert result.clean_transcript_path.read_text(encoding="utf-8") == (
        "Cleaned chunk one\n\nCleaned chunk two\n\nCleaned chunk three\n"
    )
    assert len(runtime.prompts) == 4
    assert "Cleaned chunk three" in runtime.prompts[-1]


def test_run_post_transcription_notes_reports_progress_events(tmp_path: Path) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text(
        "\n".join(" ".join(f"word{index}_{i}" for i in range(450)) for index in range(3)),
        encoding="utf-8",
    )
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    runtime = FakePromptRuntime(
        [
            "Cleaned chunk one",
            "Cleaned chunk two",
            "Cleaned chunk three",
            "Client note",
        ]
    )
    runtime_factory = FakeRuntimeFactory({False: [runtime], True: []})
    progress_events: list[tuple[str, dict[str, object]]] = []

    run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=transcript_path,
            output_dir=tmp_path / "notes_out",
            prompt_path=prompt_path,
        ),
        runtime_factory=runtime_factory,
        progress_callback=lambda event, fields: progress_events.append((event, fields)),
    )

    event_names = [event for event, _ in progress_events]
    assert event_names[0] == "notes_started"
    assert event_names.count("clean_transcript_chunk_started") == 3
    assert "clean_transcript_ready" in event_names
    assert "client_notes_started" in event_names
    assert "client_notes_ready" in event_names


def test_run_post_transcription_notes_retries_on_cpu_after_gpu_error(tmp_path: Path) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text("rough text", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    gpu_runtime = FakePromptRuntime([NotesGpuRuntimeError("cuda out of memory")])
    cpu_runtime = FakePromptRuntime(["Clean transcript", "Client note"])
    runtime_factory = FakeRuntimeFactory({False: [gpu_runtime], True: [cpu_runtime]})

    result = run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=transcript_path,
            output_dir=tmp_path / "notes_out",
            prompt_path=prompt_path,
        ),
        runtime_factory=runtime_factory,
    )

    assert result.cpu_fallback_used is True
    assert result.clean_transcript_path.read_text(encoding="utf-8") == "Clean transcript\n"
    assert result.client_notes_path.read_text(encoding="utf-8") == "Client note\n"
    assert runtime_factory.calls == [False, True]


def test_run_post_transcription_notes_falls_back_to_raw_cleanup_chunk_when_model_returns_empty(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text("rough text", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    runtime = FakePromptRuntime(["", "", "Client note"])
    runtime_factory = FakeRuntimeFactory({False: [runtime], True: []})
    progress_events: list[tuple[str, dict[str, object]]] = []

    result = run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=transcript_path,
            output_dir=tmp_path / "notes_out",
            prompt_path=prompt_path,
        ),
        runtime_factory=runtime_factory,
        progress_callback=lambda event, fields: progress_events.append((event, fields)),
    )

    assert result.clean_transcript_path.read_text(encoding="utf-8") == "rough text\n"
    assert result.client_notes_path.read_text(encoding="utf-8") == "Client note\n"
    assert any(event == "clean_transcript_chunk_fallback" for event, _fields in progress_events)


def test_run_post_transcription_notes_uses_limited_content_fallback_note_when_model_returns_empty(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text("One, two, three.\nBrief fragment.\n", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    runtime = FakePromptRuntime(["", "", "", "", "", ""])
    runtime_factory = FakeRuntimeFactory({False: [runtime], True: []})
    progress_events: list[tuple[str, dict[str, object]]] = []

    result = run_post_transcription_notes(
        SessionNotesConfig(
            transcript_path=transcript_path,
            output_dir=tmp_path / "notes_out",
            prompt_path=prompt_path,
        ),
        runtime_factory=runtime_factory,
        progress_callback=lambda event, fields: progress_events.append((event, fields)),
    )

    note_text = result.client_notes_path.read_text(encoding="utf-8")
    assert "Transcript contained limited clear client-reported clinical content" in note_text
    assert any(event == "client_notes_retrying" for event, _fields in progress_events)
    assert any(event == "client_notes_fallback" for event, _fields in progress_events)


def test_llama_cpp_runtime_retries_when_model_is_still_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    responses = [
        notes_module.NotesRuntimeError(
            '{"error":{"message":"Loading model","type":"unavailable_error","code":503}}'
        ),
        {
            "choices": [
                {
                    "message": {
                        "content": "Client note",
                    }
                }
            ]
        },
    ]

    def fake_request(*, host: str, path: str, payload: dict[str, object] | None, timeout_sec: float, method: str = "POST") -> dict[str, object]:
        _ = (host, path, payload, timeout_sec, method)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(notes_module, "_llama_server_request", fake_request)
    monkeypatch.setattr(notes_module.time, "sleep", lambda seconds: None)

    runtime = LlamaCppRuntimeSession(
        binary_path=Path("llama-server.exe"),
        model_path=Path("model.gguf"),
        host="127.0.0.1:1234",
    )

    result = runtime.run_prompt(model="ignored", prompt="hello")

    assert result == "Client note"
    assert responses == []


def test_llama_cpp_runtime_rejects_reasoning_only_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    def fake_request(*, host: str, path: str, payload: dict[str, object] | None, timeout_sec: float, method: str = "POST") -> dict[str, object]:
        _ = (host, path, payload, timeout_sec, method)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": "thinking",
                    }
                }
            ]
        }

    monkeypatch.setattr(notes_module, "_llama_server_request", fake_request)

    runtime = LlamaCppRuntimeSession(
        binary_path=Path("llama-server.exe"),
        model_path=Path("model.gguf"),
        host="127.0.0.1:1234",
    )

    with pytest.raises(RuntimeError, match="reasoning content without a final answer"):
        runtime.run_prompt(model="ignored", prompt="hello")


def test_temporary_llama_cpp_runtime_forces_cpu_layers_for_packaged_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    observed: dict[str, object] = {}

    class _FakeProcess:
        def __init__(self) -> None:
            self.stderr = None

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

    def fake_popen(command, *, env, stdout, stderr, text, encoding, errors, creationflags):
        observed["command"] = list(command)
        observed["env"] = dict(env)
        observed["stdout"] = stdout
        observed["stderr"] = stderr
        observed["text"] = text
        observed["encoding"] = encoding
        observed["errors"] = errors
        observed["creationflags"] = creationflags
        return _FakeProcess()

    monkeypatch.setattr(notes_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(notes_module, "_loopback_host_for_free_port", lambda: "127.0.0.1:8080")
    monkeypatch.setattr(notes_module, "_wait_for_llama_cpp_runtime_ready", lambda *, process, host: None)

    with notes_module._temporary_llama_cpp_runtime(
        executable=Path("llama-server.exe"),
        model_path=Path("model.gguf"),
        cpu_only=False,
    ):
        pass

    command = observed["command"]
    assert isinstance(command, list)
    assert "--reasoning" in command
    assert command[command.index("--reasoning") + 1] == "off"
    assert "--reasoning-format" in command
    assert command[command.index("--reasoning-format") + 1] == "none"
    assert "--n-gpu-layers" in command
    assert command[command.index("--n-gpu-layers") + 1] == "0"
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"
    assert observed["creationflags"] == notes_module._subprocess_creationflags_no_window()


def test_temporary_llama_cpp_runtime_uses_gpu_layers_when_backend_is_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transcribe.notes as notes_module

    observed: dict[str, object] = {}

    runtime_dir = tmp_path / "llm"
    runtime_dir.mkdir()
    executable = runtime_dir / "llama-server.exe"
    executable.write_text("", encoding="utf-8")
    (runtime_dir / "ggml-cuda.dll").write_text("", encoding="utf-8")

    class _FakeProcess:
        stderr = None

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

    def fake_popen(command, *, env, stdout, stderr, text, encoding, errors, creationflags):
        observed["command"] = list(command)
        return _FakeProcess()

    monkeypatch.setattr(notes_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(notes_module, "_loopback_host_for_free_port", lambda: "127.0.0.1:8080")
    monkeypatch.setattr(notes_module, "_wait_for_llama_cpp_runtime_ready", lambda *, process, host: None)

    with notes_module._temporary_llama_cpp_runtime(
        executable=executable,
        model_path=Path("model.gguf"),
        cpu_only=False,
    ):
        pass

    command = observed["command"]
    assert "--n-gpu-layers" in command
    assert command[command.index("--n-gpu-layers") + 1] == "auto"


def test_shared_llama_cpp_runtime_reuses_warm_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    starts: list[int] = []

    class _FakeProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            _ = timeout
            return 0

    def fake_start_llama_cpp_runtime(*, executable: Path, model_path: Path, cpu_only: bool, launch_config):
        _ = (executable, model_path, cpu_only, launch_config)
        starts.append(1)
        return notes_module.LlamaCppRuntimeSession(
            binary_path=Path("llama-server.exe"),
            model_path=Path("model.gguf"),
            host="127.0.0.1:8080",
            server_process=_FakeProcess(),
            launch_config=launch_config,
        )

    monkeypatch.setattr(notes_module, "_start_llama_cpp_runtime", fake_start_llama_cpp_runtime)
    notes_module._shutdown_shared_llama_cpp_runtimes()

    launch_config = notes_module.LlamaCppLaunchConfig(
        context_tokens=16_384,
        threads=4,
        threads_batch=4,
    )

    with notes_module._shared_llama_cpp_runtime(
        executable=Path("llama-server.exe"),
        model_path=Path("model.gguf"),
        cpu_only=False,
        launch_config=launch_config,
    ):
        pass

    with notes_module._shared_llama_cpp_runtime(
        executable=Path("llama-server.exe"),
        model_path=Path("model.gguf"),
        cpu_only=False,
        launch_config=launch_config,
    ):
        pass

    assert len(starts) == 1
    notes_module._shutdown_shared_llama_cpp_runtimes()


def test_resolve_llama_cpp_executable_prefers_existing_primary_path(tmp_path: Path) -> None:
    from transcribe.notes import _resolve_llama_cpp_executable

    install_root = tmp_path / "app"
    runtime_dir = install_root / "runtime" / "llm"
    runtime_dir.mkdir(parents=True)
    primary_binary = runtime_dir / "llama-server.exe"
    primary_binary.write_text("", encoding="utf-8")

    runtime_paths = SimpleNamespace(
        install_root=install_root,
        notes_runtime_binary=primary_binary,
    )

    assert _resolve_llama_cpp_executable(runtime_paths) == primary_binary.resolve()


def test_resolve_llama_cpp_executable_falls_back_to_staged_runtime(tmp_path: Path) -> None:
    from transcribe.notes import _resolve_llama_cpp_executable

    install_root = tmp_path / "repo"
    staged_runtime_dir = install_root / "build" / "windows_standalone" / "0.2.2" / "stage" / "runtime" / "llm"
    staged_runtime_dir.mkdir(parents=True)
    staged_binary = staged_runtime_dir / "llama-server.exe"
    staged_binary.write_text("", encoding="utf-8")

    runtime_paths = SimpleNamespace(
        install_root=install_root,
        notes_runtime_binary=install_root / "runtime" / "llm" / "llama-server.exe",
    )

    assert _resolve_llama_cpp_executable(runtime_paths) == staged_binary.resolve()


def test_run_ollama_command_uses_replace_for_text_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import transcribe.notes as notes_module

    observed: dict[str, object] = {}

    def fake_run(argv, *, env, input, capture_output, text, encoding, errors, timeout, check):
        observed["argv"] = list(argv)
        observed["env"] = dict(env)
        observed["input"] = input
        observed["capture_output"] = capture_output
        observed["text"] = text
        observed["encoding"] = encoding
        observed["errors"] = errors
        observed["timeout"] = timeout
        observed["check"] = check
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(notes_module, "_ollama_executable", lambda: "ollama")
    monkeypatch.setattr(notes_module.subprocess, "run", fake_run)

    result = notes_module._run_ollama_command(
        ["list"],
        env={"OLLAMA_HOST": "127.0.0.1:11434"},
        timeout_sec=20.0,
    )

    assert result.stdout == "ok"
    assert observed["text"] is True
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "replace"


def test_ollama_runtime_uses_http_chat_payload_with_request_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    observed: dict[str, object] = {}

    def fake_request(*, host: str, path: str, payload: dict[str, object] | None, timeout_sec: float, method: str = "POST") -> dict[str, object]:
        observed["host"] = host
        observed["path"] = path
        observed["payload"] = payload
        observed["timeout_sec"] = timeout_sec
        observed["method"] = method
        return {"message": {"content": "Client note"}}

    monkeypatch.setattr(notes_module, "_ollama_request", fake_request)

    runtime = notes_module.OllamaRuntimeSession(env={"OLLAMA_HOST": "127.0.0.1:11434"}, host="127.0.0.1:11434")
    result = runtime.run_prompt(
        model="qwen3.5:4b-q4_K_M",
        prompt="hello",
        request_options=PromptRequestOptions(max_tokens=512, context_tokens=12_288),
    )

    assert result == "Client note"
    assert observed["path"] == "/api/chat"
    assert observed["method"] == "POST"
    assert observed["payload"]["keep_alive"] == "10m"
    assert observed["payload"]["think"] is False
    assert observed["payload"]["options"] == {"num_ctx": 12_288, "num_predict": 512}


def test_llama_server_stream_request_rejects_reasoning_only_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    class _FakeResponse:
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n'
            yield b"data: [DONE]\n"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = (exc_type, exc, tb)
            return False

    monkeypatch.setattr(notes_module.urllib_request, "urlopen", lambda request, timeout: _FakeResponse())

    with pytest.raises(RuntimeError, match="reasoning content without a final answer"):
        notes_module._llama_server_stream_request_once(
            host="127.0.0.1:1234",
            path="/v1/chat/completions",
            payload={"stream": True},
            timeout_sec=30.0,
            on_text_delta=lambda text: None,
        )


def test_base_ollama_env_limits_parallelism() -> None:
    import transcribe.notes as notes_module

    env = notes_module._base_ollama_env(host="127.0.0.1:11434")

    assert env["OLLAMA_HOST"] == "127.0.0.1:11434"
    assert env["OLLAMA_NUM_PARALLEL"] == "1"
    assert env["OLLAMA_KEEP_ALIVE"] == "10m"


def test_wait_for_llama_cpp_runtime_ready_retries_while_model_is_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    responses = [
        notes_module.NotesRuntimeError(
            '{"error":{"message":"Loading model","type":"unavailable_error","code":503}}'
        ),
        {},
    ]
    sleeps: list[float] = []

    class _FakeProcess:
        stderr = None

        def poll(self) -> None:
            return None

    def fake_request(*, host: str, path: str, payload: dict[str, object] | None, timeout_sec: float, method: str = "POST") -> dict[str, object]:
        _ = (host, path, payload, timeout_sec, method)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(notes_module, "_llama_server_request", fake_request)
    monkeypatch.setattr(notes_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    notes_module._wait_for_llama_cpp_runtime_ready(
        process=_FakeProcess(),
        host="127.0.0.1:1234",
    )

    assert sleeps == [0.25]
    assert responses == []


def test_run_post_transcription_notes_rejects_empty_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "rough_transcript.txt"
    transcript_path.write_text(" \n", encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("WRITE THE NOTE", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Transcript is empty"):
        run_post_transcription_notes(
            SessionNotesConfig(
                transcript_path=transcript_path,
                output_dir=tmp_path / "notes_out",
                prompt_path=prompt_path,
            ),
            runtime_factory=FakeRuntimeFactory({False: [], True: []}),
        )


def test_cli_parser_accepts_notes_run_command() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "notes",
            "run",
            "--transcript",
            "rough.txt",
        ]
    )
    assert args.command == "notes"
    assert args.notes_command == "run"
    assert args.transcript == Path("rough.txt")
    assert args.notes_model == DEFAULT_SESSION_NOTES_MODEL
    assert args.notes_runtime == "auto"


def test_cli_parser_allows_disabling_session_notes() -> None:
    from transcribe.cli import build_parser

    args = build_parser().parse_args(
        [
            "session",
            "run",
            "--fixture",
            "--no-notes",
        ]
    )
    assert args.notes is False


def test_run_session_runs_notes_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    import transcribe.cli as cli_module
    import transcribe.transcription_runtime as transcription_runtime
    import transcribe.live.session as live_session_module
    import transcribe.notes as notes_module

    monkeypatch.setattr(cli_module, "load_and_configure_logging", lambda args: None)

    observed: dict[str, SessionNotesConfig] = {}
    released: dict[str, str] = {}

    def fake_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None):
        _ = (config, use_fixture, debug, progress_callback)
        return live_session_module.LiveSessionResult(
            session_dir=tmp_path / "live-test",
            events_path=tmp_path / "live-test" / "events.jsonl",
            transcript_json_path=tmp_path / "live-test" / "transcript.json",
            transcript_txt_path=tmp_path / "live-test" / "transcript.txt",
            final_segment_count=1,
            partial_event_count=0,
            sample_rate_hz=16_000,
            sample_rate_hz_requested=16_000,
            total_audio_sec=4.0,
            total_inference_sec=0.2,
            source_selection_counts={"mic": 1},
            interrupted=True,
        )

    def fake_run_post_transcription_notes(
        config: SessionNotesConfig,
        *,
        progress_callback=None,
    ) -> SessionNotesResult:
        observed["config"] = config
        if progress_callback is not None:
            progress_callback(
                "notes_started",
                {"model": config.model, "cleanup_chunk_count": 2},
            )
            progress_callback("clean_transcript_chunk_started", {"chunk_index": 1, "chunk_count": 2})
            progress_callback("clean_transcript_ready", {})
            progress_callback("client_notes_started", {})
        return SessionNotesResult(
            transcript_path=config.transcript_path,
            clean_transcript_path=config.output_dir / "clean_transcript.txt",
            client_notes_path=config.output_dir / "client_notes.txt",
            model=config.model,
            cpu_fallback_used=False,
            clean_duration_sec=0.1,
            notes_duration_sec=0.2,
        )

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", fake_runner)
    monkeypatch.setattr(notes_module, "run_post_transcription_notes", fake_run_post_transcription_notes)
    monkeypatch.setattr(
        transcription_runtime,
        "release_transcription_runtime_resources",
        lambda transcription_model: released.setdefault("model", transcription_model) or 1,
    )

    args = argparse.Namespace(
        config=None,
        log_level=None,
        debug=False,
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        duration_sec=0.0,
        chunk_overlap_sec=0.75,
        stitch_overlap_text=True,
        chunk_sec=4.0,
        partial_interval_sec=0.0,
        mode=AudioSourceMode.BOTH,
        mic_device=None,
        speaker_device=None,
        single_device_per_source=False,
        strict_sources=False,
        out=tmp_path,
        session_id="live-test",
        max_model_ram_gb=8.0,
        fixture=False,
        notes=True,
        notes_model=DEFAULT_SESSION_NOTES_MODEL,
        notes_runtime="llama_cpp",
    )
    rc = cli_module.run_session(args)
    captured = capsys.readouterr()

    assert rc == 0
    assert released["model"] == "nvidia/parakeet-tdt-0.6b-v3"
    assert observed["config"].transcript_path == tmp_path / "live-test" / "transcript.txt"
    assert observed["config"].output_dir == tmp_path / "live-test"
    assert observed["config"].model == DEFAULT_SESSION_NOTES_MODEL
    assert observed["config"].runtime == "llama_cpp"
    assert "Preparing notes: releasing transcription model resources..." in captured.out
    assert "Post-session notes: cleaning transcript" in captured.out
    assert "Cleanup pass 1/2..." in captured.out
    assert "Clean transcript ready." in captured.out
    assert "Writing client notes..." in captured.out
    assert "Clean transcript:" in captured.out
    assert "Client notes:" in captured.out


def test_run_session_skips_notes_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import transcribe.cli as cli_module
    import transcribe.transcription_runtime as transcription_runtime
    import transcribe.live.session as live_session_module
    import transcribe.notes as notes_module

    monkeypatch.setattr(cli_module, "load_and_configure_logging", lambda args: None)

    release_calls = {"count": 0}

    def fake_runner(config, *, use_fixture: bool = False, debug: bool = False, progress_callback=None):
        _ = (config, use_fixture, debug, progress_callback)
        return live_session_module.LiveSessionResult(
            session_dir=tmp_path / "live-test",
            events_path=tmp_path / "live-test" / "events.jsonl",
            transcript_json_path=tmp_path / "live-test" / "transcript.json",
            transcript_txt_path=tmp_path / "live-test" / "transcript.txt",
            final_segment_count=1,
            partial_event_count=0,
            sample_rate_hz=16_000,
            sample_rate_hz_requested=16_000,
            total_audio_sec=4.0,
            total_inference_sec=0.2,
            source_selection_counts={"mic": 1},
            interrupted=True,
        )

    def fail_run_post_transcription_notes(
        config: SessionNotesConfig,
        *,
        progress_callback=None,
    ) -> SessionNotesResult:
        _ = progress_callback
        raise AssertionError(f"notes should not run when disabled: {config}")

    monkeypatch.setattr(live_session_module, "run_live_transcription_session", fake_runner)
    monkeypatch.setattr(notes_module, "run_post_transcription_notes", fail_run_post_transcription_notes)
    monkeypatch.setattr(
        transcription_runtime,
        "release_transcription_runtime_resources",
        lambda transcription_model: release_calls.__setitem__("count", release_calls["count"] + 1) or 1,
    )

    args = argparse.Namespace(
        config=None,
        log_level=None,
        debug=False,
        transcription_model="nvidia/parakeet-tdt-0.6b-v3",
        duration_sec=0.0,
        chunk_overlap_sec=0.75,
        stitch_overlap_text=True,
        chunk_sec=4.0,
        partial_interval_sec=0.0,
        mode=AudioSourceMode.BOTH,
        mic_device=None,
        speaker_device=None,
        single_device_per_source=False,
        strict_sources=False,
        out=tmp_path,
        session_id="live-test",
        max_model_ram_gb=8.0,
        fixture=False,
        notes=False,
        notes_model=DEFAULT_SESSION_NOTES_MODEL,
        notes_runtime="auto",
    )

    assert cli_module.run_session(args) == 0
    assert release_calls["count"] == 0


def test_run_notes_command_uses_requested_transcript(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    import transcribe.cli as cli_module
    import transcribe.notes as notes_module

    monkeypatch.setattr(cli_module, "load_and_configure_logging", lambda args: None)

    rough_transcript = tmp_path / "rough.txt"
    rough_transcript.write_text("rough", encoding="utf-8")
    observed: dict[str, SessionNotesConfig] = {}

    def fake_run_post_transcription_notes(
        config: SessionNotesConfig,
        *,
        progress_callback=None,
    ) -> SessionNotesResult:
        observed["config"] = config
        if progress_callback is not None:
            progress_callback(
                "notes_started",
                {"model": config.model, "cleanup_chunk_count": 1},
            )
            progress_callback("notes_cpu_fallback", {})
            progress_callback("client_notes_started", {})
        return SessionNotesResult(
            transcript_path=config.transcript_path,
            clean_transcript_path=config.output_dir / "clean_transcript.txt",
            client_notes_path=config.output_dir / "client_notes.txt",
            model=config.model,
            cpu_fallback_used=True,
            clean_duration_sec=0.1,
            notes_duration_sec=0.2,
        )

    monkeypatch.setattr(notes_module, "run_post_transcription_notes", fake_run_post_transcription_notes)

    rc = cli_module.run_notes(
        argparse.Namespace(
            config=None,
            log_level=None,
            debug=False,
            transcript=rough_transcript,
            out_dir=None,
            notes_model=DEFAULT_SESSION_NOTES_MODEL,
            notes_runtime="llama_cpp",
        )
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert observed["config"].transcript_path == rough_transcript
    assert observed["config"].output_dir == tmp_path
    assert observed["config"].runtime == "llama_cpp"
    assert "Post-session notes: cleaning transcript" in captured.out
    assert "retrying on CPU" in captured.out


def test_default_runtime_factory_uses_llama_cpp_when_auto_resolves_to_packaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    observed: list[bool] = []

    @contextlib.contextmanager
    def fake_llama_runtime(*, model: str, cpu_only: bool = False):
        assert model == DEFAULT_SESSION_NOTES_MODEL
        observed.append(cpu_only)
        yield FakePromptRuntime(["unused"])

    monkeypatch.setattr(notes_module, "default_notes_runtime", lambda: "llama_cpp")
    monkeypatch.setattr(notes_module, "open_llama_cpp_runtime", fake_llama_runtime)

    factory = _default_runtime_factory(
        SessionNotesConfig(
            transcript_path=Path("rough.txt"),
            output_dir=Path("out"),
        )
    )

    with factory(cpu_only=True):
        pass

    assert observed == [True]


def test_default_runtime_factory_falls_back_to_llama_cpp_when_ollama_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transcribe.notes as notes_module

    observed: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def fake_ollama_runtime(*, cpu_only: bool = False):
        observed.append(("ollama", cpu_only))
        raise notes_module.NotesRuntimeError(
            f"Session notes model {DEFAULT_SESSION_NOTES_MODEL!r} is not installed locally in Ollama."
        )
        yield

    @contextlib.contextmanager
    def fake_llama_runtime(*, model: str, cpu_only: bool = False):
        assert model == DEFAULT_SESSION_NOTES_MODEL
        observed.append(("llama_cpp", cpu_only))
        yield FakePromptRuntime(["unused"])

    monkeypatch.setattr(notes_module, "default_notes_runtime", lambda: "ollama")
    monkeypatch.setattr(notes_module, "open_ollama_runtime", fake_ollama_runtime)
    monkeypatch.setattr(notes_module, "open_llama_cpp_runtime", fake_llama_runtime)

    factory = _default_runtime_factory(
        SessionNotesConfig(
            transcript_path=Path("rough.txt"),
            output_dir=Path("out"),
        )
    )

    with factory(cpu_only=False):
        pass

    assert observed == [("ollama", False), ("llama_cpp", False)]


