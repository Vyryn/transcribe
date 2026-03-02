from __future__ import annotations

import wave
from pathlib import Path


class Pcm16MonoWavWriter:
    """Context-managed WAV writer for mono PCM16 streams."""

    def __init__(self, path: Path, *, sample_rate_hz: int) -> None:
        """Open a WAV file for PCM16 mono frame writes.

        Parameters
        ----------
        path : Path
            Output WAV file path.
        sample_rate_hz : int
            Sampling rate for the file header.
        """
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = wave.open(str(path), "wb")
        self._handle.setnchannels(1)
        self._handle.setsampwidth(2)
        self._handle.setframerate(sample_rate_hz)

    def write(self, pcm16: bytes) -> None:
        """Append raw PCM16 mono frame bytes to the file.

        Parameters
        ----------
        pcm16 : bytes
            Raw PCM16 mono audio frame bytes.
        """
        self._handle.writeframesraw(pcm16)

    def close(self) -> None:
        """Finalize and close the WAV file handle."""
        self._handle.writeframes(b"")
        self._handle.close()

    def __enter__(self) -> "Pcm16MonoWavWriter":
        """Return context-managed writer instance."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close the writer when exiting context manager scope."""
        self.close()
