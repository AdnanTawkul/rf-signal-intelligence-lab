from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.impairments import add_awgn, apply_frequency_offset


def test_add_awgn_preserves_shape_and_dtype() -> None:
    clean = np.ones(1024, dtype=np.complex64)

    noisy = add_awgn(clean, snr_db=10.0, seed=42)

    assert noisy.shape == clean.shape
    assert noisy.dtype == np.complex64
    assert not np.array_equal(noisy, clean)


def test_add_awgn_is_reproducible() -> None:
    clean = np.ones(1024, dtype=np.complex64)

    noisy_a = add_awgn(clean, snr_db=5.0, seed=7)
    noisy_b = add_awgn(clean, snr_db=5.0, seed=7)

    np.testing.assert_array_equal(noisy_a, noisy_b)


def test_add_awgn_matches_requested_snr() -> None:
    clean = np.ones(200_000, dtype=np.complex64)
    requested_snr_db = 10.0

    noisy = add_awgn(clean, snr_db=requested_snr_db, seed=123)
    noise = noisy - clean

    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    measured_snr_db = 10.0 * np.log10(signal_power / noise_power)

    assert measured_snr_db == pytest.approx(requested_snr_db, abs=0.15)


@pytest.mark.parametrize(
    "invalid_samples",
    [
        np.array([], dtype=np.complex64),
        np.zeros(128, dtype=np.complex64),
        np.ones((8, 8), dtype=np.complex64),
    ],
)
def test_add_awgn_rejects_invalid_samples(invalid_samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        add_awgn(invalid_samples, snr_db=10.0)


@pytest.mark.parametrize("invalid_snr", [float("nan"), float("inf"), float("-inf")])
def test_add_awgn_rejects_nonfinite_snr(invalid_snr: float) -> None:
    clean = np.ones(128, dtype=np.complex64)

    with pytest.raises(ValueError):
        add_awgn(clean, snr_db=invalid_snr)


def test_frequency_offset_preserves_shape_dtype_and_magnitude() -> None:
    samples = np.array(
        [1.0 + 0.0j, 0.5 + 0.5j, -1.0 + 0.25j],
        dtype=np.complex64,
    )

    shifted = apply_frequency_offset(
        samples,
        frequency_offset_hz=1_000.0,
        sample_rate_hz=100_000.0,
    )

    assert shifted.shape == samples.shape
    assert shifted.dtype == np.complex64
    np.testing.assert_allclose(
        np.abs(shifted),
        np.abs(samples),
        rtol=1e-6,
        atol=1e-6,
    )


def test_frequency_offset_produces_known_quarter_turns() -> None:
    samples = np.ones(4, dtype=np.complex64)

    shifted = apply_frequency_offset(
        samples,
        frequency_offset_hz=1.0,
        sample_rate_hz=4.0,
    )

    expected = np.array(
        [1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j],
        dtype=np.complex64,
    )

    np.testing.assert_allclose(shifted, expected, atol=1e-6)


def test_frequency_offset_applies_initial_phase() -> None:
    samples = np.ones(8, dtype=np.complex64)

    shifted = apply_frequency_offset(
        samples,
        frequency_offset_hz=0.0,
        sample_rate_hz=1_000.0,
        initial_phase_rad=np.pi / 2.0,
    )

    expected = np.full(8, 1j, dtype=np.complex64)
    np.testing.assert_allclose(shifted, expected, atol=1e-6)


def test_zero_frequency_and_phase_offset_preserves_signal() -> None:
    samples = np.array(
        [1.0 + 1.0j, -0.5 + 0.25j, 0.1 - 0.8j],
        dtype=np.complex64,
    )

    shifted = apply_frequency_offset(
        samples,
        frequency_offset_hz=0.0,
        sample_rate_hz=1_000_000.0,
    )

    np.testing.assert_array_equal(shifted, samples)


@pytest.mark.parametrize(
    "invalid_samples",
    [
        np.array([], dtype=np.complex64),
        np.ones((4, 4), dtype=np.complex64),
        np.array([1.0 + 0.0j, complex(float("nan"), 0.0)], dtype=np.complex64),
    ],
)
def test_frequency_offset_rejects_invalid_samples(invalid_samples: np.ndarray) -> None:
    with pytest.raises(ValueError):
        apply_frequency_offset(
            invalid_samples,
            frequency_offset_hz=1_000.0,
            sample_rate_hz=1_000_000.0,
        )


@pytest.mark.parametrize(
    ("frequency_offset_hz", "sample_rate_hz", "initial_phase_rad"),
    [
        (float("nan"), 1_000_000.0, 0.0),
        (float("inf"), 1_000_000.0, 0.0),
        (1_000.0, 0.0, 0.0),
        (1_000.0, -1_000_000.0, 0.0),
        (1_000.0, float("inf"), 0.0),
        (1_000.0, 1_000_000.0, float("nan")),
    ],
)
def test_frequency_offset_rejects_invalid_parameters(
    frequency_offset_hz: float,
    sample_rate_hz: float,
    initial_phase_rad: float,
) -> None:
    samples = np.ones(128, dtype=np.complex64)

    with pytest.raises(ValueError):
        apply_frequency_offset(
            samples,
            frequency_offset_hz=frequency_offset_hz,
            sample_rate_hz=sample_rate_hz,
            initial_phase_rad=initial_phase_rad,
        )
