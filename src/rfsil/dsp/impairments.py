from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from rfsil.dsp.modulations import ComplexArray


def _validate_iq_samples(samples: ArrayLike) -> ComplexArray:
    """Convert and validate a one-dimensional complex IQ signal."""
    iq = np.asarray(samples, dtype=np.complex64)

    if iq.ndim != 1:
        raise ValueError("samples must be a one-dimensional IQ array.")

    if iq.size == 0:
        raise ValueError("samples must not be empty.")

    if not np.all(np.isfinite(iq)):
        raise ValueError("samples must contain only finite values.")

    return iq


def add_awgn(
    samples: ArrayLike,
    snr_db: float,
    seed: int | None = None,
) -> ComplexArray:
    """Add complex white Gaussian noise at a requested signal-to-noise ratio."""
    iq = _validate_iq_samples(samples)

    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    signal_power = float(np.mean(np.abs(iq.astype(np.complex128)) ** 2))

    if signal_power <= 0.0:
        raise ValueError("samples must have positive signal power.")

    linear_snr = float(np.power(10.0, snr_db / 10.0))

    if not np.isfinite(linear_snr) or linear_snr <= 0.0:
        raise ValueError("snr_db produces an invalid linear SNR.")

    noise_power = signal_power / linear_snr
    component_std = np.sqrt(noise_power / 2.0)

    rng = np.random.default_rng(seed)

    noise = (
        rng.normal(0.0, component_std, size=iq.shape)
        + 1j * rng.normal(0.0, component_std, size=iq.shape)
    ).astype(np.complex64)

    return (iq + noise).astype(np.complex64)


def apply_phase_offset(
    samples: ArrayLike,
    phase_offset_rad: float,
) -> ComplexArray:
    """Apply a constant carrier phase offset to IQ samples."""
    iq = _validate_iq_samples(samples)

    if not np.isfinite(phase_offset_rad):
        raise ValueError("phase_offset_rad must be finite.")

    phase_rotation = np.complex64(np.exp(1j * phase_offset_rad))

    return (iq * phase_rotation).astype(np.complex64)


def apply_amplitude_scaling(
    samples: ArrayLike,
    amplitude_scale: float,
) -> ComplexArray:
    """Apply a positive linear amplitude scale to complex IQ samples."""
    iq = _validate_iq_samples(samples)

    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
        raise ValueError("amplitude_scale must be positive and finite.")

    return (iq * np.float32(amplitude_scale)).astype(np.complex64)


def apply_time_shift(
    samples: ArrayLike,
    shift_samples: int,
) -> ComplexArray:
    """Apply a zero-padded integer time shift to IQ samples.

    Positive shifts delay the signal by inserting zeros at the beginning.
    Negative shifts advance the signal by inserting zeros at the end. The
    output always has the same length as the input and does not wrap samples
    around circularly.

    Args:
        samples: One-dimensional complex IQ signal.
        shift_samples: Integer sample displacement. Positive values delay the
            signal and negative values advance it.

    Returns:
        Shifted complex IQ samples with the same shape and ``complex64`` dtype.

    Raises:
        ValueError: If the signal is invalid or shift_samples is not an integer.
    """
    iq = _validate_iq_samples(samples)

    if isinstance(shift_samples, bool) or not isinstance(shift_samples, Integral):
        raise ValueError("shift_samples must be an integer.")

    shift = int(shift_samples)
    shifted = np.zeros_like(iq)

    if shift == 0:
        return iq.copy()

    if abs(shift) >= iq.size:
        return shifted

    if shift > 0:
        shifted[shift:] = iq[:-shift]
    else:
        advance = -shift
        shifted[:-advance] = iq[advance:]

    return shifted


def apply_frequency_offset(
    samples: ArrayLike,
    frequency_offset_hz: float,
    sample_rate_hz: float,
    initial_phase_rad: float = 0.0,
) -> ComplexArray:
    """Apply a carrier frequency and initial phase offset to IQ samples."""
    iq = _validate_iq_samples(samples)

    if not np.isfinite(frequency_offset_hz):
        raise ValueError("frequency_offset_hz must be finite.")

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite.")

    if not np.isfinite(initial_phase_rad):
        raise ValueError("initial_phase_rad must be finite.")

    sample_indices = np.arange(iq.size, dtype=np.float64)
    phase = (
        2.0 * np.pi * frequency_offset_hz * sample_indices / sample_rate_hz
        + initial_phase_rad
    )

    rotating_phasor = np.exp(1j * phase).astype(np.complex64)

    return (iq * rotating_phasor).astype(np.complex64)


__all__ = [
    "add_awgn",
    "apply_amplitude_scaling",
    "apply_frequency_offset",
    "apply_phase_offset",
    "apply_time_shift",
]
