from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.pulse_shaping import (
    apply_root_raised_cosine,
    design_root_raised_cosine_filter,
)


def test_rrc_filter_has_expected_length() -> None:
    taps = design_root_raised_cosine_filter(
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
    )

    assert len(taps) == 8 * 8 + 1


def test_rrc_filter_is_symmetric() -> None:
    taps = design_root_raised_cosine_filter(
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
    )

    np.testing.assert_allclose(
        taps,
        taps[::-1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_rrc_filter_has_unit_energy() -> None:
    taps = design_root_raised_cosine_filter(
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
    )

    energy = float(np.sum(taps.astype(np.float64) ** 2))

    assert energy == pytest.approx(1.0, abs=1e-6)


def test_apply_rrc_preserves_complex64_dtype() -> None:
    symbols = np.array(
        [1.0 + 1.0j, -1.0 + 1.0j, 1.0 - 1.0j],
        dtype=np.complex64,
    )

    result = apply_root_raised_cosine(
        symbols,
        samples_per_symbol=8,
    )

    assert result.samples.dtype == np.complex64
    assert result.taps.dtype == np.float32


def test_apply_rrc_has_expected_output_length() -> None:
    symbols = np.ones(32, dtype=np.complex64)
    samples_per_symbol = 8
    span_symbols = 8

    result = apply_root_raised_cosine(
        symbols,
        samples_per_symbol=samples_per_symbol,
        span_symbols=span_symbols,
    )

    expected_tap_count = span_symbols * samples_per_symbol + 1
    expected_length = (
        (len(symbols) - 1) * samples_per_symbol
        + expected_tap_count
    )

    assert len(result.samples) == expected_length


def test_apply_rrc_reports_correct_group_delay() -> None:
    result = apply_root_raised_cosine(
        np.ones(16, dtype=np.complex64),
        samples_per_symbol=8,
        span_symbols=8,
    )

    assert result.group_delay_samples == 32


def test_single_impulse_reproduces_filter_taps() -> None:
    result = apply_root_raised_cosine(
        np.array([1.0 + 0.0j], dtype=np.complex64),
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
    )

    np.testing.assert_allclose(
        result.samples.real,
        result.taps,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.samples.imag,
        np.zeros_like(result.taps),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "invalid_symbols",
    [
        np.array([], dtype=np.complex64),
        np.ones((4, 4), dtype=np.complex64),
        np.array(
            [1.0 + 0.0j, complex(float("nan"), 0.0)],
            dtype=np.complex64,
        ),
    ],
)
def test_apply_rrc_rejects_invalid_symbols(
    invalid_symbols: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        apply_root_raised_cosine(
            invalid_symbols,
            samples_per_symbol=8,
        )


@pytest.mark.parametrize(
    ("samples_per_symbol", "rolloff", "span_symbols"),
    [
        (1, 0.35, 8),
        (0, 0.35, 8),
        (8, 0.0, 8),
        (8, 1.1, 8),
        (8, float("nan"), 8),
        (8, 0.35, 0),
        (3, 0.35, 3),
    ],
)
def test_rrc_rejects_invalid_parameters(
    samples_per_symbol: int,
    rolloff: float,
    span_symbols: int,
) -> None:
    with pytest.raises(ValueError):
        design_root_raised_cosine_filter(
            samples_per_symbol=samples_per_symbol,
            rolloff=rolloff,
            span_symbols=span_symbols,
        )
