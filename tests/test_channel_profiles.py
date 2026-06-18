from __future__ import annotations

import numpy as np
import pytest

from rfsil.dsp.channel_profiles import (
    MILD_MULTIPATH_PROFILE,
    MODERATE_MULTIPATH_PROFILE,
    SEVERE_MULTIPATH_PROFILE,
    MultipathChannelProfile,
    get_multipath_profile,
    sample_multipath_tap_gains,
)


@pytest.mark.parametrize(
    ("profile", "expected_name", "expected_taps"),
    [
        (
            MILD_MULTIPATH_PROFILE,
            "mild",
            3,
        ),
        (
            MODERATE_MULTIPATH_PROFILE,
            "moderate",
            4,
        ),
        (
            SEVERE_MULTIPATH_PROFILE,
            "severe",
            5,
        ),
    ],
)
def test_predefined_profiles_are_valid(
    profile: MultipathChannelProfile,
    expected_name: str,
    expected_taps: int,
) -> None:
    assert profile.name == expected_name
    assert (
        len(profile.tap_delays_samples)
        == expected_taps
    )
    assert (
        len(profile.average_powers_db)
        == expected_taps
    )
    assert profile.tap_delays_samples[0] == 0


def test_profile_lookup_is_case_insensitive() -> None:
    assert (
        get_multipath_profile("  MiLd  ")
        is MILD_MULTIPATH_PROFILE
    )


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError):
        get_multipath_profile("unknown")


def test_sampling_is_reproducible() -> None:
    first = sample_multipath_tap_gains(
        "moderate",
        seed=2026,
    )
    second = sample_multipath_tap_gains(
        "moderate",
        seed=2026,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_different_seeds_change_gains() -> None:
    first = sample_multipath_tap_gains(
        "moderate",
        seed=2026,
    )
    second = sample_multipath_tap_gains(
        "moderate",
        seed=2027,
    )

    assert not np.array_equal(
        first,
        second,
    )


def test_sampled_gains_have_expected_shape_and_dtype() -> None:
    gains = sample_multipath_tap_gains(
        "severe",
        seed=2026,
    )

    assert gains.shape == (5,)
    assert gains.dtype == np.complex64
    assert np.all(np.isfinite(gains))


def test_sampled_gains_are_normalized() -> None:
    gains = sample_multipath_tap_gains(
        "mild",
        seed=2026,
        normalize_total_power=True,
    )

    total_power = float(
        np.sum(
            np.abs(
                gains.astype(np.complex128)
            )
            ** 2
        )
    )

    assert total_power == pytest.approx(
        1.0,
        rel=1e-6,
        abs=1e-6,
    )


def test_sampling_without_normalization_is_finite() -> None:
    gains = sample_multipath_tap_gains(
        "mild",
        seed=2026,
        normalize_total_power=False,
    )

    total_power = float(
        np.sum(
            np.abs(
                gains.astype(np.complex128)
            )
            ** 2
        )
    )

    assert np.all(np.isfinite(gains))
    assert total_power > 0.0


def test_custom_profile_can_be_sampled() -> None:
    profile = MultipathChannelProfile(
        name="custom",
        tap_delays_samples=(0, 4),
        average_powers_db=(0.0, -10.0),
    )

    gains = sample_multipath_tap_gains(
        profile,
        seed=2026,
    )

    assert gains.shape == (2,)


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {
            "name": "",
            "tap_delays_samples": (0,),
            "average_powers_db": (0.0,),
        },
        {
            "name": "invalid",
            "tap_delays_samples": (),
            "average_powers_db": (),
        },
        {
            "name": "invalid",
            "tap_delays_samples": (0, 1),
            "average_powers_db": (0.0,),
        },
        {
            "name": "invalid",
            "tap_delays_samples": (1, 2),
            "average_powers_db": (0.0, -3.0),
        },
        {
            "name": "invalid",
            "tap_delays_samples": (0, -1),
            "average_powers_db": (0.0, -3.0),
        },
        {
            "name": "invalid",
            "tap_delays_samples": (0, 0),
            "average_powers_db": (0.0, -3.0),
        },
        {
            "name": "invalid",
            "tap_delays_samples": (0, 1),
            "average_powers_db": (0.0, np.nan),
        },
    ],
)
def test_invalid_profiles_are_rejected(
    keyword_arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MultipathChannelProfile(
            **keyword_arguments,
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
        sample_multipath_tap_gains(
            "mild",
            seed=2026,
            normalize_total_power=value,
        )
