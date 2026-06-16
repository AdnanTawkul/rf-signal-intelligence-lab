from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex64]
IntArray = NDArray[np.int_]


class Modulation(StrEnum):
    """Supported digital modulation schemes."""

    BPSK = "bpsk"
    QPSK = "qpsk"
    PSK8 = "8psk"
    QAM16 = "16qam"


@dataclass(frozen=True)
class IQSignal:
    """Container for generated complex IQ data and symbol labels."""

    samples: ComplexArray
    symbols: IntArray
    modulation: Modulation
    samples_per_symbol: int
    sample_rate_hz: float


def _psk_constellation(order: int) -> ComplexArray:
    """Create a unit-power phase-shift keying constellation."""
    phases = (
        2.0
        * np.pi
        * np.arange(order, dtype=np.float64)
        / float(order)
    )

    return np.exp(1j * phases).astype(np.complex64)


def _square_qam_constellation(levels: IntArray) -> ComplexArray:
    """Create a normalized square quadrature-amplitude constellation."""
    in_phase, quadrature = np.meshgrid(
        levels,
        levels[::-1],
    )

    points = (
        in_phase.astype(np.float32)
        + 1j * quadrature.astype(np.float32)
    ).reshape(-1)

    average_power = float(np.mean(np.abs(points) ** 2))
    normalized = points / np.sqrt(average_power)

    return normalized.astype(np.complex64)


def _constellation(modulation: Modulation) -> ComplexArray:
    """Return normalized constellation points for a modulation scheme."""
    if modulation == Modulation.BPSK:
        return np.array(
            [-1.0 + 0.0j, 1.0 + 0.0j],
            dtype=np.complex64,
        )

    if modulation == Modulation.QPSK:
        points = np.array(
            [
                1.0 + 1.0j,
                -1.0 + 1.0j,
                -1.0 - 1.0j,
                1.0 - 1.0j,
            ],
            dtype=np.complex64,
        )

        return (points / np.sqrt(2.0)).astype(np.complex64)

    if modulation == Modulation.PSK8:
        return _psk_constellation(order=8)

    if modulation == Modulation.QAM16:
        levels = np.array(
            [-3, -1, 1, 3],
            dtype=np.int_,
        )

        return _square_qam_constellation(levels)

    raise ValueError(f"Unsupported modulation: {modulation}")


def generate_iq_signal(
    modulation: Modulation | str,
    num_symbols: int,
    samples_per_symbol: int = 8,
    sample_rate_hz: float = 1_000_000.0,
    seed: int | None = None,
) -> IQSignal:
    """Generate a synthetic complex baseband IQ signal.

    The waveform currently uses rectangular pulse shaping. More realistic root
    raised cosine pulse shaping will be implemented as a separate DSP stage.

    Args:
        modulation: Supported modulation name.
        num_symbols: Number of random symbols to generate.
        samples_per_symbol: Number of IQ samples per symbol.
        sample_rate_hz: Baseband sample rate in hertz.
        seed: Optional random seed for reproducibility.

    Returns:
        IQSignal containing complex IQ samples and symbol indices.

    Raises:
        ValueError: If the modulation or generation parameters are invalid.
    """
    modulation = Modulation(modulation)

    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive.")

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive.")

    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite.")

    points = _constellation(modulation)
    rng = np.random.default_rng(seed)

    symbols = rng.integers(
        low=0,
        high=len(points),
        size=num_symbols,
        dtype=np.int_,
    )

    symbol_values = points[symbols]
    samples = np.repeat(
        symbol_values,
        samples_per_symbol,
    ).astype(np.complex64)

    return IQSignal(
        samples=samples,
        symbols=symbols,
        modulation=modulation,
        samples_per_symbol=samples_per_symbol,
        sample_rate_hz=float(sample_rate_hz),
    )


__all__ = [
    "ComplexArray",
    "IQSignal",
    "IntArray",
    "Modulation",
    "generate_iq_signal",
]
