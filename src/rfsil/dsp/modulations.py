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


@dataclass(frozen=True)
class IQSignal:
    """Container for generated complex IQ data and its discrete symbol labels."""

    samples: ComplexArray
    symbols: IntArray
    modulation: Modulation
    samples_per_symbol: int
    sample_rate_hz: float


def _constellation(modulation: Modulation) -> ComplexArray:
    """Return normalized constellation points for a modulation scheme."""
    if modulation == Modulation.BPSK:
        points = np.array([-1.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex64)

    elif modulation == Modulation.QPSK:
        points = np.array(
            [
                1.0 + 1.0j,
                -1.0 + 1.0j,
                -1.0 - 1.0j,
                1.0 - 1.0j,
            ],
            dtype=np.complex64,
        )
        points = points / np.sqrt(2.0)

    else:
        raise ValueError(f"Unsupported modulation: {modulation}")

    return points.astype(np.complex64)


def generate_iq_signal(
    modulation: Modulation | str,
    num_symbols: int,
    samples_per_symbol: int = 8,
    sample_rate_hz: float = 1_000_000.0,
    seed: int | None = None,
) -> IQSignal:
    """Generate a simple synthetic complex baseband IQ signal.

    The generated waveform uses rectangular pulse shaping. This is intentionally
    simple for the first milestone. Pulse shaping and channel impairments are
    added later as separate, testable DSP components.

    Args:
        modulation: Modulation name, currently 'bpsk' or 'qpsk'.
        num_symbols: Number of random modulation symbols to generate.
        samples_per_symbol: Number of IQ samples per discrete symbol.
        sample_rate_hz: Sample rate of the generated baseband signal.
        seed: Optional random seed for reproducibility.

    Returns:
        IQSignal containing repeated complex IQ samples and symbol labels.
    """
    modulation = Modulation(modulation)

    if num_symbols <= 0:
        raise ValueError("num_symbols must be positive.")

    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive.")

    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    points = _constellation(modulation)
    rng = np.random.default_rng(seed)

    symbols = rng.integers(
        low=0,
        high=len(points),
        size=num_symbols,
        dtype=np.int_,
    )

    symbol_values = points[symbols]
    samples = np.repeat(symbol_values, samples_per_symbol).astype(np.complex64)

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
