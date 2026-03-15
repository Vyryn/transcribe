When you encounter a new problem that required significant searching to resolve or a developer intervention, add a brief summary and the solution to the end of this file.

This project uses uv; `uv run transcribe`
Use python 3.13 best practices and numpy style docstrings. 
Before adding a new dependency, explain the need and ask the developer.

Benchmark harness wrappers can temporarily mirror helper functions into `transcribe.transcription_runtime` so benchmark-specific monkeypatches affect delegated calls. Always restore the runtime module's original helper functions after the delegated call finishes, or later tests can accidentally reuse stale benchmark fixtures.

Installer-driven packaged model downloads need the CLI to honor `TRANSCRIBE_ALLOW_NETWORK`; otherwise `transcribe models install` fails during setup because the offline socket guard is installed too early. Fix this by checking `network_access_allowed()` before installing the guard, and launch installer-time model downloads with `TRANSCRIBE_ALLOW_NETWORK=1`.
