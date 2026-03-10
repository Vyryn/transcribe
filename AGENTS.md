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
