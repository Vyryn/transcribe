from __future__ import annotations

from collections.abc import Mapping

from transcribe.audio.linux_capture import LinuxAudioCaptureBackend

_WINDOWS_GENERIC_BACKEND_DEVICE_NAMES = {
    "primary sound capture driver",
    "windows directsound",
    "mme",
}

_WINDOWS_SPEAKER_STRONG_MARKERS = (
    "loopback",
    "stereo mix",
    "what u hear",
    "wave out mix",
    "mixage stéréo",
)
_WINDOWS_SPEAKER_WEAK_MARKERS = (
    "speaker",
    "speakers",
    "output",
    "headphones",
    "headset earphone",
    "monitor",
    "playback",
    "digital output",
)
_WINDOWS_MIC_STRONG_MARKERS = (
    "microphone",
    "mic",
    "array",
    "input",
    "headset mic",
    "usb mic",
)


class WindowsAudioCaptureBackend(LinuxAudioCaptureBackend):
    """Windows capture backend using lazy PortAudio access with loopback heuristics."""

    @staticmethod
    def _is_generic_backend_device(device_name_lower: str) -> bool:
        """Return True when name matches a generic Windows routing backend."""
        return device_name_lower in _WINDOWS_GENERIC_BACKEND_DEVICE_NAMES

    @staticmethod
    def _score_device_for_role(*, name: str, require_monitor: bool) -> float:
        """Score a Windows device name for microphone or loopback suitability."""
        score = 0.0
        is_generic = WindowsAudioCaptureBackend._is_generic_backend_device(name)

        has_speaker_strong = any(marker in name for marker in _WINDOWS_SPEAKER_STRONG_MARKERS)
        has_speaker_weak = any(marker in name for marker in _WINDOWS_SPEAKER_WEAK_MARKERS)
        has_mic_strong = any(marker in name for marker in _WINDOWS_MIC_STRONG_MARKERS)

        if require_monitor:
            if has_speaker_strong:
                score += 12.0
            if has_speaker_weak:
                score += 4.0
            if has_mic_strong:
                score -= 9.0
            if is_generic:
                score += 0.5
            return score

        if has_mic_strong:
            score += 10.0
        if has_speaker_strong:
            score -= 10.0
        elif has_speaker_weak:
            score -= 4.0
        if is_generic:
            score -= 2.0
        return score

    def find_candidate_devices(
        self,
        sd: object,
        *,
        require_monitor: bool,
    ) -> list[int]:
        """Return ranked Windows capture candidates with loopback-friendly scoring."""
        scored_candidates: list[tuple[float, int]] = []
        weak_candidates: list[int] = []
        for index, raw_device in enumerate(sd.query_devices()):  # type: ignore[attr-defined]
            if not isinstance(raw_device, Mapping):
                continue
            name = self._normalize_device_name(raw_device)
            input_channels = int(raw_device.get("max_input_channels", 0))
            if input_channels < 1:
                continue

            score = self._score_device_for_role(name=name, require_monitor=require_monitor)
            if score > 0:
                scored_candidates.append((score + min(float(input_channels), 4.0) * 0.01, index))
                continue
            if require_monitor and any(marker in name for marker in _WINDOWS_SPEAKER_WEAK_MARKERS):
                weak_candidates.append(index)

        if scored_candidates:
            scored_candidates.sort(key=lambda item: (-item[0], item[1]))
            return [index for _, index in scored_candidates]
        if require_monitor:
            return weak_candidates
        return []
