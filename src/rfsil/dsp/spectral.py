from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class SpectrogramResult:
    """Frequency-domain representation of a complex IQ signal."""

    frequencies_hz: Float64Array
    times_s: Float64Array
    power_db: Float32Array


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer parameter."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated


def compute_spectrogram(
    samples: ArrayLike,
    sample_rate_hz: float,
    window_size: int = 256,
    hop_size: int = 64,
    fft_size: int | None = None,
    dynamic_range_db: float = 100.0,
) -> SpectrogramResult:
    """Compute a normalized short-time Fourier spectrogram of complex IQ data.

    A Hann window is applied to overlapping signal frames. Each frame is
    Fourier transformed, frequency shifted, converted to relative decibels,
    and clipped to the requested dynamic range. The maximum spectral power is
    normalized to 0 dB.

    Args:
        samples: One-dimensional complex IQ signal.
        sample_rate_hz: IQ sample rate in hertz.
        window_size: Number of IQ samples in each analysis frame.
        hop_size: Number of IQ samples between consecutive frames.
        fft_size: Fourier-transform length. Defaults to window_size.
        dynamic_range_db: Minimum displayed power relative to the peak.

    Returns:
        SpectrogramResult containing frequency bins, frame-center times, and a
        frequency-by-time relative-power matrix.

    Raises:
        ValueError: If the signal or configuration is invalid.
    """
    iq = np.asarray(samples, dtype=np.complex64)

    if iq.ndim != 1:
        raise ValueError("samples must be a one-dimensional IQ array.")

    if iq.size == 0:
        raise ValueError("samples must not be empty.")

    if not np.all(np.isfinite(iq)):
        raise ValueError("samples must contain only finite values.")

    signal_power = float(np.mean(np.abs(iq.astype(np.complex128)) ** 2))

    if signal_power <= 0.0:
        raise ValueError("samples must have positive signal power.")

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite.")

    validated_window_size = _validate_positive_integer(
        window_size,
        "window_size",
    )
    validated_hop_size = _validate_positive_integer(
        hop_size,
        "hop_size",
    )

    if validated_window_size > iq.size:
        raise ValueError("window_size must not exceed the signal length.")

    if validated_hop_size > validated_window_size:
        raise ValueError("hop_size must not exceed window_size.")

    if fft_size is None:
        validated_fft_size = validated_window_size
    else:
        validated_fft_size = _validate_positive_integer(
            fft_size,
            "fft_size",
        )

    if validated_fft_size < validated_window_size:
        raise ValueError("fft_size must be at least as large as window_size.")

    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0.0:
        raise ValueError("dynamic_range_db must be positive and finite.")

    frame_count = (
        1
        + (iq.size - validated_window_size)
        // validated_hop_size
    )

    frame_starts = (
        np.arange(frame_count, dtype=np.int64)
        * validated_hop_size
    )
    frame_offsets = np.arange(validated_window_size, dtype=np.int64)
    frame_indices = frame_starts[:, np.newaxis] + frame_offsets[np.newaxis, :]

    frames = iq[frame_indices].astype(np.complex128)
    window = np.hanning(validated_window_size).astype(np.float64)
    windowed_frames = frames * window[np.newaxis, :]

    spectrum = np.fft.fft(
        windowed_frames,
        n=validated_fft_size,
        axis=1,
    )
    spectrum = np.fft.fftshift(spectrum, axes=1)

    window_energy = float(np.sum(window**2))
    power = np.abs(spectrum) ** 2 / window_energy
    reference_power = float(np.max(power))

    tiny = np.finfo(np.float64).tiny
    relative_power_db = 10.0 * np.log10(
        np.maximum(power, tiny) / reference_power
    )
    relative_power_db = np.maximum(
        relative_power_db,
        -float(dynamic_range_db),
    )

    frequencies_hz = np.fft.fftshift(
        np.fft.fftfreq(
            validated_fft_size,
            d=1.0 / sample_rate_hz,
        )
    ).astype(np.float64)

    frame_centers = (
        frame_starts.astype(np.float64)
        + (validated_window_size - 1) / 2.0
    )
    times_s = frame_centers / sample_rate_hz

    return SpectrogramResult(
        frequencies_hz=frequencies_hz,
        times_s=times_s,
        power_db=relative_power_db.T.astype(np.float32),
    )


__all__ = [
    "SpectrogramResult",
    "compute_spectrogram",
]
