When you encounter a new problem that required significant searching to resolve or a developer intervention, add a brief summary and the solution to the end of this file.

This project uses uv; `uv run transcribe`
Use python 3.13 best practices and numpy style docstrings. 
Before adding a new dependency, explain the need and ask the developer.

Benchmark harness wrappers can temporarily mirror helper functions into `transcribe.transcription_runtime` so benchmark-specific monkeypatches affect delegated calls. Always restore the runtime module's original helper functions after the delegated call finishes, or later tests can accidentally reuse stale benchmark fixtures.

Installer-driven packaged model downloads need the CLI to honor `TRANSCRIBE_ALLOW_NETWORK`; otherwise `transcribe models install` fails during setup because the offline socket guard is installed too early. Fix this by checking `network_access_allowed()` before installing the guard, and launch installer-time model downloads with `TRANSCRIBE_ALLOW_NETWORK=1`.

Windows standalone builds should resolve PyInstaller package roots from installed distribution metadata, not by synthesizing fallback import names from hyphenated distribution names. Use `top_level.txt` when available and otherwise derive import roots from `distribution.files`, while still copying distribution metadata separately via the original hyphenated package name.

Windows standalone builds can emit undecodable bytes on stdout/stderr during long PyInstaller runs, especially on Windows console code pages. `run_command()` should pass `errors="replace"` to `subprocess.run(..., text=True)` and treat `CompletedProcess.stdout`/`stderr` as optional values before calling string methods, or the build can fail in logging with `UnicodeDecodeError` and follow-on `AttributeError` instead of surfacing the real subprocess result.

Windows standalone packaging should bootstrap Inno Setup 6 automatically when `ISCC.exe` is missing. Reuse the build downloads directory to cache the official installer, install it silently for the current user into a predictable location such as `%LOCALAPPDATA%\Programs\Inno Setup 6`, and only fail if the bootstrap install does not produce `ISCC.exe`.

Frozen Windows builds should use a windowed PyInstaller launcher (`console=False`) for the main desktop executable, or Start menu / double-click launches will behave like a console app instead of a GUI app. Inno shortcuts should also set `WorkingDir={app}` so launched processes resolve bundled assets relative to the install root.

NeMo packaged runtimes can fail inside installed one-file builds if `lightning_fabric/version.info` is not bundled. Force PyInstaller to collect `lightning_fabric` package data in addition to `lightning`, so Parakeet/Lightning imports can resolve bundled metadata from `_MEIPASS`.

Packaged Windows notes generation can fail right after stopping a session because the private bundled `llama.cpp` server may answer HTTP before its GGUF model finishes loading, returning `{"error":{"message":"Loading model","type":"unavailable_error","code":503}}`. Treat that response as a transient startup state and retry chat-completion requests until the model is ready or a bounded timeout is reached.

Windows standalone capture startup can regress if COM initialization wraps the top-level `soundcard` import or device enumeration path. Keep explicit COM initialization scoped to the SoundCard recorder worker thread that touches WASAPI streams, and leave `load_soundcard()` plus backend `open()` discovery/import behavior unchanged so packaged builds can still import `soundcard` successfully.

Bundled Windows notes generation uses the CPU-only llama.cpp runtime archive, so the packaged `llama-server` launcher must force `--n-gpu-layers 0`. Requesting GPU offload in the packaged notes runtime can leave the server stuck returning `loading model` 503 responses instead of ever serving completions.

The Windows standalone builder should not assume the latest `ggml-org/llama.cpp` release still publishes a `win-cpu-x64` archive. When the latest release only exposes Windows x64 CUDA-branded zip assets, accept those as the runtime bundle and rely on the packaged notes launcher to force CPU execution at runtime.

Packaged Windows note generation can hit extreme RAM use if the completed ASR session still holds model/runtime allocations while the notes model is starting. Start ASR cleanup immediately after live transcription returns, let that release continue in a background thread while notes preparation begins, and make the runtime release path aggressively drop cached model references, call best-effort unload hooks, and flush allocator caches before loading the notes model.

Long-running UI tasks should not lock the page-navigation buttons. Keep page switching, especially access to the Logs tab and back to the active workflow tab, available during background work, and stream notes-phase progress into the session output panel so packaged note generation does not look frozen while cleanup and client-note generation run.

Notes prompt-path resolution should match the actual packaged and repo layout. Prefer `./clinical_note_synthesis_llm_prompt.md` at the install root, and only fall back to `./prompts/clinical_note_synthesis_llm_prompt.md` for compatibility, or packaged notes startup will fail with a missing-file error even though the prompt was bundled correctly.

Notes runtime `auto` should not behave like a hard dependency on one backend. In development, prefer Ollama but fall back to llama.cpp if Ollama is unavailable; in packaged mode, prefer llama.cpp but still allow fallback when bootstrap-time runtime availability makes that the only working local option.

Windows notes/runtime subprocesses can emit undecodable bytes during startup on cp1252 consoles, which crashes background `subprocess.Popen(..., text=True)` reader threads with `UnicodeDecodeError`. In `transcribe.notes`, pass decode-safe text options such as `encoding="utf-8", errors="replace"` to both long-lived `Popen` calls and Ollama `subprocess.run` calls so runtime startup failures surface normally instead of crashing the UI thread machinery.

Windows live transcription can appear to silently stop when capture read timeouts keep repeating after a recorder thread dies or device recovery keeps failing. Instrument the capture backends with diagnostics snapshots (timeout streaks, dropped callback frames, stream state, runtime errors, recovery counters) and have `run_live_transcription_session()` emit `capture_timeout` / `capture_resumed` progress events plus persist `capture_diagnostics` in `transcript.json` so exploratory runs show whether audio capture actually stalled, recovered, or failed to recover.

Installer-time packaged model downloads should run `Transcribe.exe` directly per selected model instead of spawning `cmd.exe` with a chained command. Set `TRANSCRIBE_ALLOW_NETWORK` inside the Inno Setup process before each hidden `Exec(...)` call, and use a custom installer progress page to show which model is currently being installed so setup does not sit at 100% with a visible shell window.

Tk UI background-task dispatch must not let page callback exceptions kill `_poll_messages()`. If a models-page progress/result handler receives an unexpected payload and raises, catch that failure inside the polling loop, log and surface it with `messagebox.showerror`, always reschedule polling in `finally`, and validate models-page result payload types explicitly instead of relying on `assert` so the app does not freeze in a stuck busy state.

Packaged Windows UI startup can become multi-minute if the standalone builder emits a single huge PyInstaller one-file `Transcribe.exe`. The frozen process stays invisible until the bootloader finishes unpacking, so build the app as an onedir bundle instead (`EXE(..., exclude_binaries=True)` plus `COLLECT(...)`), stage the full bundle contents into the installer source tree, and have Inno Setup copy the staged directory recursively instead of only `Transcribe.exe`.

Packaged Parakeet sessions can fail or appear frozen across repeated runs if a canceled live-session preload keeps restoring the NeMo model in the background while a later session starts a second restore, and some runtimes fail inside the TDT CUDA-graph decoder path. Serialize NeMo ASR model initialization behind a shared lock so later sessions reuse the in-flight load, but keep CUDA-graph decoding enabled by default and only fall back by calling available `disable_cuda_graphs()` hooks plus forcing `use_cuda_graph_decoder = False` if an actual Parakeet decoder/runtime failure occurs.

Repeated Tk UI tests can flake under the uv-managed Windows Python when each test creates and destroys its own `tk.Tk()` root, surfacing intermittent `tcl_findLibrary` / `init.tcl` lookup errors even though Tk works overall. Keep one hidden module-scoped Tk host root for the UI test file, create per-test `tk.Toplevel()` windows on that shared interpreter, and cancel the app's repeating `_poll_messages()` `after()` callback during shutdown before destroying the window.

Packaged-mode notes launches can fail in source-checkout or staged-run workflows if `resolve_app_runtime_paths()` points at an app root without `runtime/llm/llama-server.exe`, even though a staged Windows runtime exists under `build/windows_standalone/*/stage/runtime/llm`. Fix this by resolving the llama.cpp executable from the normal packaged path first, then falling back to `_internal/runtime/llm`, nearby staged build runtime directories, and finally `PATH` before declaring the runtime unavailable.

Sparse or low-signal transcripts can make the notes model return an empty cleanup chunk and then an empty final note, especially after cleanup already fell back to raw text. Harden `run_post_transcription_notes()` by retrying client-note generation with an explicit non-empty sparse-transcript prompt and, if the transcript is still short/fragmentary, emitting a deterministic limited-content SOAP fallback note instead of failing with `Client notes generation returned no content`.

Bundled llama.cpp notes runs can appear to return zero output even after long generation when reasoning mode is left on and the server emits `reasoning_content` while `message.content` stays empty. Launch notes-time `llama-server` with `--reasoning off --reasoning-format none`, and treat reasoning-only chat/stream responses as explicit runtime errors so the app surfaces the real protocol issue instead of silently treating it as an empty note.
