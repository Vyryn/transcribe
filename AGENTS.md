When you encounter a new problem that required significant searching to resolve or a developer intervention, add a brief summary and the solution to the end of this file.

This project uses uv; `uv run transcribe`
Use python 3.13 best practices and numpy style docstrings. 
Before adding a new dependency, explain the need and ask the developer.

Benchmark harness wrappers can temporarily mirror helper functions into `transcribe.transcription_runtime` so benchmark-specific monkeypatches affect delegated calls. Always restore the runtime module's original helper functions after the delegated call finishes, or later tests can accidentally reuse stale benchmark fixtures.

Installer-driven packaged model downloads need the CLI to honor `TRANSCRIBE_ALLOW_NETWORK`; otherwise `transcribe models install` fails during setup because the offline socket guard is installed too early. Fix this by checking `network_access_allowed()` before installing the guard, and launch installer-time model downloads with `TRANSCRIBE_ALLOW_NETWORK=1`.

Windows standalone builds should resolve PyInstaller package roots from installed distribution metadata, not by synthesizing fallback import names from hyphenated distribution names. Use `top_level.txt` when available and otherwise derive import roots from `distribution.files`, while still copying distribution metadata separately via the original hyphenated package name.

Windows standalone builds can emit undecodable bytes on stdout/stderr during long PyInstaller runs, especially on Windows console code pages. `run_command()` should pass `errors="replace"` to `subprocess.run(..., text=True)` and treat `CompletedProcess.stdout`/`stderr` as optional values before calling string methods, or the build can fail in logging with `UnicodeDecodeError` and follow-on `AttributeError` instead of surfacing the real subprocess result.

Windows standalone packaging should bootstrap Inno Setup 6 automatically when `ISCC.exe` is missing. Reuse the build downloads directory to cache the official installer, install it silently for the current user into a predictable location such as `%LOCALAPPDATA%\Programs\Inno Setup 6`, and only fail if the bootstrap install does not produce `ISCC.exe`.
