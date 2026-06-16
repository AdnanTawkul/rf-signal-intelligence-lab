from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import upfirdn

from rfsil.dsp.modulations import ComplexArray

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class PulseShapingResult:
    """Result of applying a transmit pulse-shaping filter."""

    samples: ComplexArray
    taps: FloatArray
    group_delay_samples: int
    samples_per_symbol: int
    rolloff: float
    span_symbols: int


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated


def design_root_raised_cosine_filter(
    samples_per_symbol: int,
    rolloff: float = 0.35,
    span_symbols: int = 8,
) -> FloatArray:
    """Design a unit-energy root-raised-cosine FIR filter.

    Args:
        samples_per_symbol: Number of output samples per modulation symbol.
        rolloff: Excess-bandwidth factor in the interval ``(0, 1]``.
        span_symbols: Total filter duration measured in symbols.

    Returns:
        Symmetric, odd-length, unit-energy FIR coefficients.

    Raises:
        ValueError: If any filter-design parameter is invalid.
    """
    sps = _validate_positive_integer(
        samples_per_symbol,
        "samples_per_symbol",
    )
    span = _validate_positive_integer(
        span_symbols,
        "span_symbols",
    )

    if sps < 2:
        raise ValueError("samples_per_symbol must be at least 2.")

    if not np.isfinite(rolloff) or not 0.0 < rolloff <= 1.0:
        raise ValueError("rolloff must be finite and in the interval (0, 1].")

    if (span * sps) % 2 != 0:
        raise ValueError(
            "span_symbols * samples_per_symbol must be even."
        )

    time_symbols = (
        np.arange(
            -(span * sps) // 2,
            (span * sps) // 2 + 1,
            dtype=np.float64,
        )
        / float(sps)
    )

    taps = np.empty_like(time_symbols)
    zero_mask = np.isclose(time_symbols, 0.0)
    singular_time = 1.0 / (4.0 * rolloff)
    singular_mask = np.isclose(
        np.abs(time_symbols),
        singular_time,
    )
    regular_mask = ~(zero_mask | singular_mask)

    taps[zero_mask] = (
        1.0
        + rolloff * (4.0 / np.pi - 1.0)
    )

    taps[singular_mask] = (
        rolloff
        / np.sqrt(2.0)
        * (
            (1.0 + 2.0 / np.pi)
            * np.sin(np.pi / (4.0 * rolloff))
            + (1.0 - 2.0 / np.pi)
            * np.cos(np.pi / (4.0 * rolloff))
        )
    )

    regular_time = time_symbols[regular_mask]
    numerator = (
        np.sin(np.pi * regular_time * (1.0 - rolloff))
        + 4.0
        * rolloff
        * regular_time
        * np.cos(np.pi * regular_time * (1.0 + rolloff))
    )
    denominator = (
        np.pi
        * regular_time
        * (1.0 - (4.0 * rolloff * regular_time) ** 2)
    )
    taps[regular_mask] = numerator / denominator

    energy = float(np.sum(taps**2))

    if not np.isfinite(energy) or energy <= 0.0:
        raise RuntimeError("Failed to create a valid RRC filter.")

    taps /= np.sqrt(energy)

    return taps.astype(np.float32)


def apply_root_raised_cosine(
    symbols: ArrayLike,
    samples_per_symbol: int,
    rolloff: float = 0.35,
    span_symbols: int = 8,
) -> PulseShapingResult:
    """Upsample and pulse-shape complex symbols with an RRC filter.

    The full convolution output is returned, including filter transients.
    ``group_delay_samples`` identifies the delay introduced by the symmetric
    FIR filter.

    Args:
        symbols: One-dimensional complex modulation-symbol sequence.
        samples_per_symbol: Number of output IQ samples per symbol.
        rolloff: RRC excess-bandwidth factor.
        span_symbols: Total RRC filter duration measured in symbols.

    Returns:
        PulseShapingResult containing the shaped IQ waveform and filter data.

    Raises:
        ValueError: If the symbol sequence or filter parameters are invalid.
    """
    symbol_array = np.asarray(symbols, dtype=np.complex64)

    if symbol_array.ndim != 1:
        raise ValueError("symbols must be a one-dimensional array.")

    if symbol_array.size == 0:
        raise ValueError("symbols must not be empty.")

    if not np.all(np.isfinite(symbol_array)):
        raise ValueError("symbols must contain only finite values.")

    taps = design_root_raised_cosine_filter(
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        span_symbols=span_symbols,
    )

    shaped_samples = upfirdn(
        taps,
        symbol_array,
        up=int(samples_per_symbol),
    ).astype(np.complex64)

    group_delay_samples = (len(taps) - 1) // 2

    return PulseShapingResult(
        samples=shaped_samples,
        taps=taps,
        group_delay_samples=group_delay_samples,
        samples_per_symbol=int(samples_per_symbol),
        rolloff=float(rolloff),
        span_symbols=int(span_symbols),
    )


__all__ = [
    "FloatArray",
    "PulseShapingResult",
    "apply_root_raised_cosine",
    "design_root_raised_cosine_filter",
]
