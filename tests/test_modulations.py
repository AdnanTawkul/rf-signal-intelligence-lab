from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.modulations import Modulation, generate_iq_signal


def test_generate_bpsk_signal_shape_and_dtype() -> None:
    signal = generate_iq_signal(
        modulation=Modulation.BPSK,
        num_symbols=16,
        samples_per_symbol=4,
        sample_rate_hz=1_000_000.0,
        seed=123,
    )

    assert signal.modulation == Modulation.BPSK
    assert signal.samples.shape == (64,)
    assert signal.symbols.shape == (16,)
    assert signal.samples.dtype == np.complex64
    assert signal.samples_per_symbol == 4
    assert signal.sample_rate_hz == 1_000_000.0


@pytest.mark.parametrize(
    ("modulation", "expected_point_count"),
    [
        (Modulation.BPSK, 2),
        (Modulation.QPSK, 4),
        (Modulation.PSK8, 8),
        (Modulation.QAM16, 16),
    ],
)
def test_supported_modulation_has_expected_constellation_size(
    modulation: Modulation,
    expected_point_count: int,
) -> None:
    signal = generate_iq_signal(
        modulation=modulation,
        num_symbols=4_096,
        samples_per_symbol=1,
        seed=42,
    )

    unique_points = np.unique(signal.samples)

    assert len(unique_points) == expected_point_count


@pytest.mark.parametrize(
    "modulation",
    [
        Modulation.BPSK,
        Modulation.QPSK,
        Modulation.PSK8,
        Modulation.QAM16,
    ],
)
def test_constellation_has_unit_average_power(
    modulation: Modulation,
) -> None:
    signal = generate_iq_signal(
        modulation=modulation,
        num_symbols=4_096,
        samples_per_symbol=1,
        seed=42,
    )

    unique_points = np.unique(signal.samples)
    average_power = float(np.mean(np.abs(unique_points) ** 2))

    assert average_power == pytest.approx(1.0, abs=1e-6)


def test_generate_iq_signal_is_reproducible_with_seed() -> None:
    signal_a = generate_iq_signal(
        "16qam",
        num_symbols=32,
        samples_per_symbol=8,
        seed=7,
    )
    signal_b = generate_iq_signal(
        "16qam",
        num_symbols=32,
        samples_per_symbol=8,
        seed=7,
    )

    np.testing.assert_array_equal(signal_a.symbols, signal_b.symbols)
    np.testing.assert_array_equal(signal_a.samples, signal_b.samples)


@pytest.mark.parametrize(
    ("num_symbols", "samples_per_symbol", "sample_rate_hz"),
    [
        (0, 8, 1_000_000.0),
        (16, 0, 1_000_000.0),
        (16, 8, 0.0),
    ],
)
def test_generate_iq_signal_rejects_invalid_parameters(
    num_symbols: int,
    samples_per_symbol: int,
    sample_rate_hz: float,
) -> None:
    with pytest.raises(ValueError):
        generate_iq_signal(
            modulation="bpsk",
            num_symbols=num_symbols,
            samples_per_symbol=samples_per_symbol,
            sample_rate_hz=sample_rate_hz,
        )


def test_generate_iq_signal_rejects_unknown_modulation() -> None:
    with pytest.raises(ValueError):
        generate_iq_signal(
            modulation="unknown",
            num_symbols=16,
            samples_per_symbol=8,
        )
