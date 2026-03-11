When you encounter a new problem that required significant searching to resolve or a developer intervention, add a brief summary and the solution to the end of this file.

This project uses uv; `uv run transcribe`
Use python 3.13 best practices and numpy style docstrings. 
Before adding a new dependency, explain the need and ask the developer.

Guidance from past stumbling blocks:
- Keep non-audio commands free of eager audio/backend imports.
- Windows audio capture on this machine was more reliable with soundcard (native WASAPI mic/loopback enumeration) than sounddevice/PortAudio, which exposed duplicate MME/DirectSound/WDM-KS endpoints and silent streams. Use `transcribe.audio.windows_capture` with `soundcard` for Windows, keep Linux on `sounddevice`, and treat WASAPI `0x80070005` as a Windows microphone privacy/access issue with an explicit remediation message instead of a generic device error.
- soundcard on Windows currently calls NumPy binary romstring() inside soundcard.mediafoundation, which breaks on NumPy 2 with 'The binary mode of fromstring is removed'. Patch soundcard.mediafoundation.numpy.fromstring to delegate binary inputs to 
umpy.frombuffer during backend import so live capture keeps working without waiting on an upstream release.

- Live sessions can begin recording before the ASR model finishes loading. Keep the startup progress order as `capture_ready -> loading_model(capture_active=True) -> model_ready(buffered_audio_sec=...) -> transcribing_started`, and have CLI messaging explain buffering/catch-up so non-debug output stays clear.
- The Tkinter UI runs as one long-lived process. Any workflow that calls `configure_runtime()` permanently installs the outbound network guard for that process, so networked tasks like packaged model install or `init-bench` must run before offline workflows or after restarting `transcribe-ui`.
- The Windows packaged launcher can stay as a single `transcribe.exe`: dispatch to the GUI when started with no args, but keep CLI behavior for any arguments. Keep `transcribe.packaged_ui` explicitly included in standalone build imports so packaged GUI launches are not dropped by the bundler.
- The UI network toggle has two layers: skip the socket guard and relax HF offline env flags. It supports the safe flow of enabling network first, downloading/caching models, then turning network back off; once an offline task installs the socket guard, re-enabling network still requires restarting the UI process.
- The UI network toggle must do more than skip `install_outbound_network_guard()`: HF-backed runtime helpers also pass `local_files_only=True` in live-session and benchmark flows, so they need to derive an effective offline mode from the same process-level network toggle or first-run model downloads will still fail.
- The `Allow Network Access` UI toggle now survives restarts via `data/ui-preferences.json` (or the packaged data root). Persist both the network opt-in and expanded-options state so the required restart flow is actually usable and the app reopens showing that network access is still enabled.
