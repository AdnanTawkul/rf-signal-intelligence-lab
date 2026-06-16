from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from rfsil.dsp.modulations import ComplexArray


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
        ValueError: If the signal is empty, multidimensional, has zero power, or
            the requested SNR is not finite.
    """
    iq = np.asarray(samples, dtype=np.complex64)

    if iq.ndim != 1:
        raise ValueError("samples must be a one-dimensional IQ array.")

    if iq.size == 0:
        raise ValueError("samples must not be empty.")

    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    signal_power = float(np.mean(np.abs(iq.astype(np.complex128)) ** 2))

    if not np.isfinite(signal_power) or signal_power <= 0.0:
        raise ValueError("samples must have positive finite signal power.")

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


__all__ = ["add_awgn"]
