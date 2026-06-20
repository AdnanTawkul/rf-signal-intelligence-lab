from __future__ import annotations

import math

import numpy as np
import pytest

from rfsil.evaluation.iq_channel_features import (
    DEFAULT_AUTOCORRELATION_LAGS,
    compute_iq_channel_features,
)


def complex_to_iq(
    samples: np.ndarray,
) -> np.ndarray:
    return np.stack(
        (
            samples.real,
            samples.imag,
        ),
        axis=-2,
    ).astype(np.float64)


def tone(
    sample_count: int = 512,
    *,
    cycles: int = 32,
) -> np.ndarray:
    phase = (
        2.0
        * np.pi
        * cycles
        * np.arange(sample_count)
        / sample_count
    )

    return np.exp(1j * phase)


def feature_value(
    result,
    name: str,
    example: int = 0,
) -> float:
    index = result.feature_names.index(
        name
    )
    return float(
        result.values[
            example,
            index,
        ]
    )


def test_default_feature_shape() -> None:
    examples = np.stack(
        (
            complex_to_iq(tone()),
            complex_to_iq(
                tone(cycles=48)
            ),
        )
    )

    result = compute_iq_channel_features(
        examples
    )

    assert result.example_count == 2
    assert result.feature_count == 21
    assert result.values.shape == (
        2,
        21,
    )
    assert len(
        set(result.feature_names)
    ) == result.feature_count
    assert np.all(
        np.isfinite(result.values)
    )


def test_single_example_is_supported() -> None:
    result = compute_iq_channel_features(
        complex_to_iq(tone())
    )

    assert result.values.shape == (
        1,
        21,
    )


def test_feature_metadata() -> None:
    result = compute_iq_channel_features(
        complex_to_iq(tone())
    )
    metadata = result.to_dict()

    assert metadata["example_count"] == 1
    assert metadata["feature_count"] == 21
    assert len(
        metadata["feature_names"]
    ) == 21


def test_gain_and_global_phase_invariance() -> None:
    samples = tone()
    transformed = (
        4.2
        * samples
        * np.exp(1j * 0.73)
    )

    original = compute_iq_channel_features(
        complex_to_iq(samples)
    )
    changed = compute_iq_channel_features(
        complex_to_iq(transformed)
    )

    np.testing.assert_allclose(
        original.values,
        changed.values,
        rtol=1e-9,
        atol=1e-9,
    )


def test_tone_has_constant_envelope() -> None:
    result = compute_iq_channel_features(
        complex_to_iq(tone())
    )

    assert feature_value(
        result,
        "amplitude_coefficient_of_variation",
    ) < 1e-10


def test_tone_autocorrelation_is_high() -> None:
    result = compute_iq_channel_features(
        complex_to_iq(tone())
    )

    for lag in (
        DEFAULT_AUTOCORRELATION_LAGS
    ):
        assert feature_value(
            result,
            f"autocorrelation_abs_lag_{lag}",
        ) > 0.99


def test_noise_has_higher_spectral_entropy() -> None:
    generator = np.random.default_rng(
        2026
    )
    noise = (
        generator.normal(size=2_048)
        + 1j
        * generator.normal(size=2_048)
    )

    tone_result = (
        compute_iq_channel_features(
            complex_to_iq(
                tone(
                    sample_count=2_048
                )
            )
        )
    )
    noise_result = (
        compute_iq_channel_features(
            complex_to_iq(noise)
        )
    )

    assert feature_value(
        noise_result,
        "spectral_entropy",
    ) > feature_value(
        tone_result,
        "spectral_entropy",
    )


def test_noise_has_higher_spectral_flatness() -> None:
    generator = np.random.default_rng(
        2027
    )
    noise = (
        generator.normal(size=2_048)
        + 1j
        * generator.normal(size=2_048)
    )

    tone_result = (
        compute_iq_channel_features(
            complex_to_iq(
                tone(
                    sample_count=2_048
                )
            )
        )
    )
    noise_result = (
        compute_iq_channel_features(
            complex_to_iq(noise)
        )
    )

    assert feature_value(
        noise_result,
        "spectral_flatness",
    ) > feature_value(
        tone_result,
        "spectral_flatness",
    )


def test_tone_has_lower_spectral_occupancy() -> None:
    generator = np.random.default_rng(
        2028
    )
    noise = (
        generator.normal(size=2_048)
        + 1j
        * generator.normal(size=2_048)
    )

    tone_result = (
        compute_iq_channel_features(
            complex_to_iq(
                tone(
                    sample_count=2_048
                )
            )
        )
    )
    noise_result = (
        compute_iq_channel_features(
            complex_to_iq(noise)
        )
    )

    assert feature_value(
        tone_result,
        "spectral_occupancy_fraction",
    ) < feature_value(
        noise_result,
        "spectral_occupancy_fraction",
    )


def test_custom_autocorrelation_lags() -> None:
    result = compute_iq_channel_features(
        complex_to_iq(tone()),
        autocorrelation_lags=(
            3,
            7,
        ),
    )

    assert result.feature_count == 19
    assert (
        "autocorrelation_abs_lag_3"
        in result.feature_names
    )
    assert (
        "autocorrelation_abs_lag_7"
        in result.feature_names
    )


def test_differential_phase_matches_tone() -> None:
    samples = tone(
        sample_count=512,
        cycles=64,
    )
    result = compute_iq_channel_features(
        complex_to_iq(samples)
    )

    expected = abs(
        2.0
        * math.pi
        * 64
        / 512
    ) / math.pi

    assert feature_value(
        result,
        "dphase_mean_abs_normalized",
    ) == pytest.approx(
        expected,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    "invalid_iq",
    (
        [],
        [1.0, 2.0],
        np.zeros((3, 128)),
        np.zeros((2, 2, 2, 2)),
        np.full(
            (2, 128),
            float("nan"),
        ),
        np.ones(
            (2, 128),
            dtype=np.bool_,
        ),
        np.ones(
            (2, 128),
            dtype=np.complex64,
        ),
        np.zeros((2, 128)),
    ),
)
def test_rejects_invalid_iq(
    invalid_iq: object,
) -> None:
    with pytest.raises(ValueError):
        compute_iq_channel_features(
            invalid_iq
        )


@pytest.mark.parametrize(
    "invalid_lags",
    (
        [],
        [0],
        [-1],
        [True],
        [1.5],
        [1, 1],
        [512],
    ),
)
def test_rejects_invalid_lags(
    invalid_lags: object,
) -> None:
    with pytest.raises(ValueError):
        compute_iq_channel_features(
            complex_to_iq(tone()),
            autocorrelation_lags=(
                invalid_lags
            ),
        )


@pytest.mark.parametrize(
    "invalid_fraction",
    (
        0.0,
        1.0,
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_rejects_invalid_occupancy_fraction(
    invalid_fraction: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="occupancy_fraction",
    ):
        compute_iq_channel_features(
            complex_to_iq(tone()),
            occupancy_fraction=(
                invalid_fraction
            ),
        )


@pytest.mark.parametrize(
    "invalid_epsilon",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ),
)
def test_rejects_invalid_epsilon(
    invalid_epsilon: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="epsilon",
    ):
        compute_iq_channel_features(
            complex_to_iq(tone()),
            epsilon=invalid_epsilon,
        )
