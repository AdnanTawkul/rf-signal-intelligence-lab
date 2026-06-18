from __future__ import annotations

import numpy as np
import pytest

from rfsil.data.synthetic import (
    SyntheticExampleConfig,
    generate_synthetic_example,
)


def create_configuration(
    multipath_profile: object = None,
) -> SyntheticExampleConfig:
    """Create a controlled noise-free test configuration."""
    return SyntheticExampleConfig(
        sample_count=256,
        sample_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
        snr_db=None,
        frequency_offset_hz=0.0,
        phase_offset_rad=0.0,
        amplitude_scale=1.0,
        time_shift_samples=0,
        apply_rayleigh_fading=False,
        multipath_profile=multipath_profile,
    )


def test_default_and_explicit_none_match() -> None:
    default_configuration = (
        SyntheticExampleConfig(
            sample_count=256,
            snr_db=None,
        )
    )
    explicit_configuration = (
        SyntheticExampleConfig(
            sample_count=256,
            snr_db=None,
            multipath_profile=None,
        )
    )

    default_example = generate_synthetic_example(
        modulation="qpsk",
        configuration=default_configuration,
        seed=2026,
    )
    explicit_example = generate_synthetic_example(
        modulation="qpsk",
        configuration=explicit_configuration,
        seed=2026,
    )

    np.testing.assert_array_equal(
        default_example.samples,
        explicit_example.samples,
    )


@pytest.mark.parametrize(
    "profile_name",
    [
        "mild",
        "moderate",
        "severe",
    ],
)
def test_predefined_profiles_generate_valid_examples(
    profile_name: str,
) -> None:
    example = generate_synthetic_example(
        modulation="qpsk",
        configuration=create_configuration(
            profile_name
        ),
        seed=2026,
    )

    assert example.samples.shape == (256,)
    assert example.samples.dtype == np.complex64
    assert np.all(np.isfinite(example.samples))
    assert (
        example.configuration.multipath_profile
        == profile_name
    )


def test_multipath_generation_is_reproducible() -> None:
    configuration = create_configuration(
        "moderate"
    )

    first = generate_synthetic_example(
        modulation="8psk",
        configuration=configuration,
        seed=2026,
    )
    second = generate_synthetic_example(
        modulation="8psk",
        configuration=configuration,
        seed=2026,
    )

    np.testing.assert_array_equal(
        first.samples,
        second.samples,
    )


def test_different_seeds_change_output() -> None:
    configuration = create_configuration(
        "moderate"
    )

    first = generate_synthetic_example(
        modulation="8psk",
        configuration=configuration,
        seed=2026,
    )
    second = generate_synthetic_example(
        modulation="8psk",
        configuration=configuration,
        seed=2027,
    )

    assert not np.array_equal(
        first.samples,
        second.samples,
    )


def test_multipath_changes_controlled_waveform() -> None:
    clean = generate_synthetic_example(
        modulation="qpsk",
        configuration=create_configuration(),
        seed=2026,
    )
    multipath = generate_synthetic_example(
        modulation="qpsk",
        configuration=create_configuration(
            "mild"
        ),
        seed=2026,
    )

    assert not np.array_equal(
        clean.samples,
        multipath.samples,
    )


def test_multipath_preserves_output_shape_and_dtype() -> None:
    example = generate_synthetic_example(
        modulation="16qam",
        configuration=create_configuration(
            "severe"
        ),
        seed=2026,
    )

    assert example.samples.shape == (256,)
    assert example.samples.dtype == np.complex64


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown multipath profile",
    ):
        generate_synthetic_example(
            modulation="qpsk",
            configuration=create_configuration(
                "unknown"
            ),
            seed=2026,
        )


@pytest.mark.parametrize(
    "profile_value",
    [
        1,
        True,
    ],
)
def test_nonstring_profile_is_rejected(
    profile_value: object,
) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_example(
            modulation="qpsk",
            configuration=create_configuration(
                profile_value
            ),
            seed=2026,
        )
