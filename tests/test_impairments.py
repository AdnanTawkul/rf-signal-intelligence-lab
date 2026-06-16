from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.impairments import add_awgn


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
