from __future__ import annotations

import struct


def resample_pcm16_mono_linear(
    pcm16: bytes,
    *,
    source_rate_hz: int,
    target_rate_hz: int,
) -> bytes:
    """Resample mono PCM16 bytes to target sample rate via linear interpolation."""
    if not pcm16 or source_rate_hz <= 0 or target_rate_hz <= 0 or source_rate_hz == target_rate_hz:
        return pcm16

    input_samples = [sample for (sample,) in struct.iter_unpack("<h", pcm16)]
    input_count = len(input_samples)
    if input_count <= 1:
        return pcm16

    if source_rate_hz % target_rate_hz == 0:
        step = source_rate_hz // target_rate_hz
        if step > 1:
            reduced = input_samples[::step]
            return struct.pack(f"<{len(reduced)}h", *reduced)

    output_count = max(1, int(round(input_count * (target_rate_hz / source_rate_hz))))
    if output_count == input_count:
        return pcm16

    output = bytearray(output_count * 2)
    position_scale = (input_count - 1) / max(1, output_count - 1)
    for output_index in range(output_count):
        source_position = output_index * position_scale
        left_index = int(source_position)
        right_index = min(left_index + 1, input_count - 1)
        interpolation = source_position - left_index
        sample_value = ((1.0 - interpolation) * input_samples[left_index]) + (interpolation * input_samples[right_index])
        sample_int = int(round(sample_value))
        if sample_int > 32_767:
            sample_int = 32_767
        elif sample_int < -32_768:
            sample_int = -32_768
        struct.pack_into("<h", output, output_index * 2, sample_int)
    return bytes(output)
