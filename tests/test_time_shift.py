from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.impairments import apply_time_shift


def test_time_shift_preserves_shape_and_dtype() -> None:
    samples = np.ones(32, dtype=np.complex64)

    shifted = apply_time_shift(samples, shift_samples=5)

    assert shifted.shape == samples.shape
    assert shifted.dtype == np.complex64


def test_positive_time_shift_delays_with_zero_padding() -> None:
    samples = np.array(
        [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j],
        dtype=np.complex64,
    )

    shifted = apply_time_shift(samples, shift_samples=2)

    expected = np.array(
        [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 2.0j],
        dtype=np.complex64,
    )

    np.testing.assert_array_equal(shifted, expected)


def test_negative_time_shift_advances_with_zero_padding() -> None:
    samples = np.array(
        [1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j, 4.0 + 4.0j],
        dtype=np.complex64,
    )

    shifted = apply_time_shift(samples, shift_samples=-2)

    expected = np.array(
        [3.0 + 3.0j, 4.0 + 4.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        dtype=np.complex64,
    )

    np.testing.assert_array_equal(shifted, expected)


def test_zero_time_shift_preserves_signal() -> None:
    samples = np.array(
        [1.0 + 1.0j, -0.5 + 0.25j, 0.1 - 0.8j],
        dtype=np.complex64,
    )

    shifted = apply_time_shift(samples, shift_samples=0)

    np.testing.assert_array_equal(shifted, samples)
    assert shifted is not samples


@pytest.mark.parametrize("shift_samples", [4, -4, 10, -10])
def test_shift_equal_to_or_larger_than_signal_returns_zeros(
    shift_samples: int,
) -> None:
    samples = np.ones(4, dtype=np.complex64)

    shifted = apply_time_shift(samples, shift_samples=shift_samples)

    np.testing.assert_array_equal(
        shifted,
        np.zeros_like(samples),
    )


@pytest.mark.parametrize(
    "invalid_samples",
    [
        np.array([], dtype=np.complex64),
        np.ones((4, 4), dtype=np.complex64),
        np.array([1.0 + 0.0j, complex(float("nan"), 0.0)], dtype=np.complex64),
    ],
)
def test_time_shift_rejects_invalid_samples(
    invalid_samples: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        apply_time_shift(invalid_samples, shift_samples=2)


@pytest.mark.parametrize(
    "invalid_shift",
    [
        1.5,
        float("nan"),
        "2",
        True,
    ],
)
def test_time_shift_rejects_noninteger_shift(
    invalid_shift: object,
) -> None:
    samples = np.ones(16, dtype=np.complex64)

    with pytest.raises(ValueError):
        apply_time_shift(samples, shift_samples=invalid_shift)  # type: ignore[arg-type]
