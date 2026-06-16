from __future__ import annotations

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
    """Add complex white Gaussian noise at a requested signal-to-noise ratio.

    Signal and noise power are measured in the complex baseband domain. Noise
    power is divided equally between the in-phase and quadrature components.

    Args:
        samples: One-dimensional complex IQ signal.
        snr_db: Requested signal-to-noise ratio in decibels.
        seed: Optional random seed for reproducibility.

    Returns:
        Noisy complex IQ samples with dtype ``complex64``.

    Raises:
        ValueError: If the input signal or requested SNR is invalid.
    """
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


def apply_frequency_offset(
    samples: ArrayLike,
    frequency_offset_hz: float,
    sample_rate_hz: float,
    initial_phase_rad: float = 0.0,
) -> ComplexArray:
    """Apply a carrier frequency and initial phase offset to IQ samples.

    The complex baseband signal is multiplied by a rotating phasor:

        exp(j * (2 * pi * frequency_offset_hz * n / sample_rate_hz
                 + initial_phase_rad))

    Args:
        samples: One-dimensional complex IQ signal.
        frequency_offset_hz: Carrier frequency offset in hertz.
        sample_rate_hz: IQ sample rate in hertz.
        initial_phase_rad: Initial carrier phase offset in radians.

    Returns:
        Frequency-shifted complex IQ samples with dtype ``complex64``.

    Raises:
        ValueError: If the samples or parameters are invalid.
    """
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
    "apply_frequency_offset",
]
