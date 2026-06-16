from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.impairments import apply_flat_rayleigh_fading


def test_flat_rayleigh_fading_preserves_shape_and_dtype() -> None:
    samples = np.ones(128, dtype=np.complex64)

    faded = apply_flat_rayleigh_fading(samples, seed=42)

    assert faded.shape == samples.shape
    assert faded.dtype == np.complex64


def test_flat_rayleigh_fading_is_reproducible() -> None:
    samples = np.array(
        [1.0 + 1.0j, -0.5 + 0.25j, 0.1 - 0.8j],
        dtype=np.complex64,
    )

    faded_a = apply_flat_rayleigh_fading(samples, seed=7)
    faded_b = apply_flat_rayleigh_fading(samples, seed=7)

    np.testing.assert_array_equal(faded_a, faded_b)


def test_flat_rayleigh_fading_changes_with_seed() -> None:
    samples = np.ones(128, dtype=np.complex64)

    faded_a = apply_flat_rayleigh_fading(samples, seed=1)
    faded_b = apply_flat_rayleigh_fading(samples, seed=2)

    assert not np.array_equal(faded_a, faded_b)


def test_flat_rayleigh_fading_uses_one_coefficient_per_block() -> None:
    samples = np.array(
        [1.0 + 0.5j, -0.5 + 1.0j, 0.25 - 0.75j],
        dtype=np.complex64,
    )

    faded = apply_flat_rayleigh_fading(samples, seed=123)
    measured_coefficients = faded / samples

    np.testing.assert_allclose(
        measured_coefficients,
        np.full_like(measured_coefficients, measured_coefficients[0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_flat_rayleigh_fading_matches_seeded_coefficient() -> None:
    samples = np.ones(16, dtype=np.complex64)
    seed = 123

    rng = np.random.default_rng(seed)
    expected_coefficient = np.complex64(
        (
            rng.normal(0.0, 1.0)
            + 1j * rng.normal(0.0, 1.0)
        )
        / np.sqrt(2.0)
    )

    faded = apply_flat_rayleigh_fading(samples, seed=seed)

    np.testing.assert_allclose(
        faded,
        np.full_like(samples, expected_coefficient),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "invalid_samples",
    [
        np.array([], dtype=np.complex64),
        np.ones((4, 4), dtype=np.complex64),
        np.array(
            [1.0 + 0.0j, complex(float("nan"), 0.0)],
            dtype=np.complex64,
        ),
    ],
)
def test_flat_rayleigh_fading_rejects_invalid_samples(
    invalid_samples: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        apply_flat_rayleigh_fading(invalid_samples, seed=42)
