from __future__ import annotations

import numpy as np
import pytest

from rfsil.data.dataset import (
    DatasetGenerationConfig,
    build_dataset_split,
)


def create_configuration(
    *,
    multipath_profile: str | None = None,
    multipath_distribution: object = None,
) -> DatasetGenerationConfig:
    """Create a small deterministic dataset configuration."""
    return DatasetGenerationConfig(
        dataset_name="multipath_distribution_test",
        sample_count=256,
        sample_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
        snr_values_db=(8.0,),
        frequency_offset_range_hz=(0.0, 0.0),
        phase_offset_range_rad=(0.0, 0.0),
        amplitude_scale_range=(1.0, 1.0),
        time_shift_range_samples=(0, 0),
        rayleigh_probability=0.0,
        multipath_profile=multipath_profile,
        multipath_distribution=multipath_distribution,
    )


def test_default_distribution_is_none() -> None:
    configuration = create_configuration()

    assert (
        configuration.multipath_distribution
        is None
    )


def test_distribution_names_are_normalized() -> None:
    configuration = create_configuration(
        multipath_distribution={
            " NONE ": 0.5,
            "MiLd": 0.5,
        }
    )

    assert configuration.multipath_distribution == {
        "none": 0.5,
        "mild": 0.5,
    }


def test_fixed_profile_and_distribution_are_exclusive() -> None:
    with pytest.raises(
        ValueError,
        match="mutually exclusive",
    ):
        create_configuration(
            multipath_profile="mild",
            multipath_distribution={
                "none": 0.5,
                "mild": 0.5,
            },
        )


@pytest.mark.parametrize(
    "distribution",
    [
        {},
        [],
        {"none": 0.0},
        {"none": 0.7},
        {
            "none": -0.1,
            "mild": 1.1,
        },
        {
            "none": np.nan,
            "mild": 0.0,
        },
        {"none": True},
        {"unknown": 1.0},
        {
            "MILD": 0.5,
            " mild ": 0.5,
        },
    ],
)
def test_invalid_distributions_are_rejected(
    distribution: object,
) -> None:
    with pytest.raises(ValueError):
        create_configuration(
            multipath_distribution=distribution
        )


def test_none_only_matches_clean_dataset() -> None:
    clean = build_dataset_split(
        configuration=create_configuration(),
        examples_per_class_per_snr=2,
        seed=2026,
    )
    distributed = build_dataset_split(
        configuration=create_configuration(
            multipath_distribution={
                "none": 1.0,
            }
        ),
        examples_per_class_per_snr=2,
        seed=2026,
    )

    np.testing.assert_array_equal(
        clean.iq,
        distributed.iq,
    )
    np.testing.assert_array_equal(
        clean.example_seed,
        distributed.example_seed,
    )


def test_mild_only_matches_fixed_profile() -> None:
    fixed = build_dataset_split(
        configuration=create_configuration(
            multipath_profile="mild"
        ),
        examples_per_class_per_snr=2,
        seed=2026,
    )
    distributed = build_dataset_split(
        configuration=create_configuration(
            multipath_distribution={
                "mild": 1.0,
            }
        ),
        examples_per_class_per_snr=2,
        seed=2026,
    )

    np.testing.assert_array_equal(
        fixed.iq,
        distributed.iq,
    )
    np.testing.assert_array_equal(
        fixed.example_seed,
        distributed.example_seed,
    )


def test_mixed_distribution_is_reproducible() -> None:
    configuration = create_configuration(
        multipath_distribution={
            "none": 0.25,
            "mild": 0.25,
            "moderate": 0.25,
            "severe": 0.25,
        }
    )

    first = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=4,
        seed=2026,
    )
    second = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=4,
        seed=2026,
    )

    np.testing.assert_array_equal(
        first.iq,
        second.iq,
    )
    np.testing.assert_array_equal(
        first.example_seed,
        second.example_seed,
    )


def test_mixed_distribution_preserves_metadata_pairing() -> None:
    clean = build_dataset_split(
        configuration=create_configuration(),
        examples_per_class_per_snr=4,
        seed=2026,
    )
    mixed = build_dataset_split(
        configuration=create_configuration(
            multipath_distribution={
                "none": 0.25,
                "mild": 0.25,
                "moderate": 0.25,
                "severe": 0.25,
            }
        ),
        examples_per_class_per_snr=4,
        seed=2026,
    )

    for clean_values, mixed_values in (
        (clean.labels, mixed.labels),
        (clean.snr_db, mixed.snr_db),
        (
            clean.frequency_offset_hz,
            mixed.frequency_offset_hz,
        ),
        (
            clean.phase_offset_rad,
            mixed.phase_offset_rad,
        ),
        (
            clean.amplitude_scale,
            mixed.amplitude_scale,
        ),
        (
            clean.time_shift_samples,
            mixed.time_shift_samples,
        ),
        (
            clean.rayleigh_fading,
            mixed.rayleigh_fading,
        ),
        (
            clean.example_seed,
            mixed.example_seed,
        ),
    ):
        np.testing.assert_array_equal(
            clean_values,
            mixed_values,
        )


def test_mixed_multipath_changes_iq() -> None:
    clean = build_dataset_split(
        configuration=create_configuration(),
        examples_per_class_per_snr=2,
        seed=2026,
    )
    mixed = build_dataset_split(
        configuration=create_configuration(
            multipath_distribution={
                "mild": 0.5,
                "severe": 0.5,
            }
        ),
        examples_per_class_per_snr=2,
        seed=2026,
    )

    assert not np.array_equal(
        clean.iq,
        mixed.iq,
    )
