from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.impairments import (
    apply_tapped_delay_line,
)


def test_identity_channel_preserves_signal() -> None:
    samples = np.asarray(
        [
            1.0 + 2.0j,
            -0.5 + 0.25j,
            3.0 - 1.0j,
        ],
        dtype=np.complex64,
    )

    result = apply_tapped_delay_line(
        samples,
        tap_delays_samples=[0],
        tap_gains=[1.0 + 0.0j],
    )

    np.testing.assert_array_equal(
        result,
        samples,
    )


def test_two_tap_channel_uses_zero_padding() -> None:
    samples = np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.complex64,
    )

    result = apply_tapped_delay_line(
        samples,
        tap_delays_samples=[0, 2],
        tap_gains=[1.0, 0.5],
        normalize_tap_power=False,
    )

    expected = np.asarray(
        [1.0, 2.0, 3.5, 5.0],
        dtype=np.complex64,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_complex_tap_rotates_signal() -> None:
    samples = np.asarray(
        [1.0, 1.0],
        dtype=np.complex64,
    )

    result = apply_tapped_delay_line(
        samples,
        tap_delays_samples=[0],
        tap_gains=[1.0j],
        normalize_tap_power=False,
    )

    expected = np.asarray(
        [1.0j, 1.0j],
        dtype=np.complex64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_tap_power_normalization() -> None:
    samples = np.asarray(
        [1.0, 2.0, 3.0],
        dtype=np.complex64,
    )

    result = apply_tapped_delay_line(
        samples,
        tap_delays_samples=[0, 1],
        tap_gains=[2.0, 0.0],
        normalize_tap_power=True,
    )

    np.testing.assert_allclose(
        result,
        samples,
    )


def test_output_preserves_shape_and_dtype() -> None:
    samples = np.asarray(
        [1.0 + 1.0j] * 16,
        dtype=np.complex64,
    )

    result = apply_tapped_delay_line(
        samples,
        tap_delays_samples=[0, 3, 7],
        tap_gains=[1.0, 0.4j, -0.2],
    )

    assert result.shape == samples.shape
    assert result.dtype == np.complex64


def test_duplicate_delays_are_summed() -> None:
    samples = np.asarray(
        [1.0, 2.0],
        dtype=np.complex64,
    )

    result = apply_tapped_delay_line(
        samples,
        tap_delays_samples=[0, 0],
        tap_gains=[0.5, 0.25],
        normalize_tap_power=False,
    )

    expected = samples * np.complex64(0.75)

    np.testing.assert_allclose(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "delays",
    [
        [],
        [-1],
        [0.5],
        [True],
    ],
)
def test_invalid_delays_are_rejected(
    delays: object,
) -> None:
    with pytest.raises(ValueError):
        apply_tapped_delay_line(
            np.ones(4, dtype=np.complex64),
            tap_delays_samples=delays,
            tap_gains=[1.0],
        )


def test_delay_must_be_smaller_than_signal() -> None:
    with pytest.raises(ValueError):
        apply_tapped_delay_line(
            np.ones(4, dtype=np.complex64),
            tap_delays_samples=[4],
            tap_gains=[1.0],
        )


def test_delay_and_gain_lengths_must_match() -> None:
    with pytest.raises(ValueError):
        apply_tapped_delay_line(
            np.ones(4, dtype=np.complex64),
            tap_delays_samples=[0, 1],
            tap_gains=[1.0],
        )


@pytest.mark.parametrize(
    "gain",
    [
        np.nan + 0.0j,
        np.inf + 1.0j,
    ],
)
def test_nonfinite_gain_is_rejected(
    gain: complex,
) -> None:
    with pytest.raises(ValueError):
        apply_tapped_delay_line(
            np.ones(4, dtype=np.complex64),
            tap_delays_samples=[0],
            tap_gains=[gain],
        )


def test_all_zero_taps_are_rejected() -> None:
    with pytest.raises(ValueError):
        apply_tapped_delay_line(
            np.ones(4, dtype=np.complex64),
            tap_delays_samples=[0, 1],
            tap_gains=[0.0, 0.0],
        )


@pytest.mark.parametrize(
    "value",
    [
        1,
        "true",
    ],
)
def test_nonboolean_normalization_is_rejected(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        apply_tapped_delay_line(
            np.ones(4, dtype=np.complex64),
            tap_delays_samples=[0],
            tap_gains=[1.0],
            normalize_tap_power=value,
        )
