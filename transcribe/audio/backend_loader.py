from __future__ import annotations

import sys

from transcribe.audio.interfaces import AudioBackend


def open_audio_backend(*, use_fixture: bool) -> AudioBackend:
    """Create the active platform capture backend lazily."""
    if sys.platform.startswith("linux"):
        from transcribe.audio.linux_capture import LinuxAudioCaptureBackend

        return LinuxAudioCaptureBackend(use_fixture=use_fixture)
    if sys.platform == "win32":
        from transcribe.audio.windows_capture import WindowsAudioCaptureBackend

        return WindowsAudioCaptureBackend(use_fixture=use_fixture)
    raise RuntimeError(f"Unsupported capture platform: {sys.platform}")
