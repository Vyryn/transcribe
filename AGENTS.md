`uv run python -m transcribe.cli ...` appeared to hang indefinitely because `transcribe.cli` imported audio modules at import time, and `transcribe.audio.linux_capture` imported `sounddevice` at module load; in this sandboxed `.venv`, `import sounddevice` blocks.

Remedy implemented:
- Moved audio-heavy imports in `transcribe/cli.py` into command handlers (`_run_capture`, `_run_devices`, `_run_benchmark`) so compliance commands do not import audio code.
- Changed `transcribe/audio/linux_capture.py` to lazy-load `sounddevice` via `_load_sounddevice()` only when real audio capture/listing is needed.

Guidance:
- Keep non-audio commands free of eager audio/backend imports.
- If a command seems hung, test `.venv/bin/python3 -c "import transcribe.cli"` and isolate import-time side effects before suspecting infinite loops.
