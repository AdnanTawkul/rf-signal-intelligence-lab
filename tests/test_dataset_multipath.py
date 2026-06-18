from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfsil.data.dataset import (
    DatasetGenerationConfig,
    build_dataset_split,
    write_dataset_manifest,
)


def create_configuration(
    multipath_profile: object = None,
) -> DatasetGenerationConfig:
    """Create a small deterministic dataset configuration."""
    return DatasetGenerationConfig(
        dataset_name="multipath_test",
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
    )


def test_default_profile_is_none() -> None:
    configuration = DatasetGenerationConfig(
        dataset_name="default_test",
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
    )

    assert configuration.multipath_profile is None


@pytest.mark.parametrize(
    "profile_name",
    [
        "mild",
        "moderate",
        "severe",
    ],
)
def test_predefined_profiles_are_accepted(
    profile_name: str,
) -> None:
    configuration = create_configuration(
        profile_name
    )

    assert (
        configuration.multipath_profile
        == profile_name
    )


@pytest.mark.parametrize(
    "profile_value",
    [
        "unknown",
        1,
        True,
    ],
)
def test_invalid_profiles_are_rejected(
    profile_value: object,
) -> None:
    with pytest.raises(ValueError):
        create_configuration(profile_value)


def test_multipath_dataset_split_is_valid() -> None:
    dataset = build_dataset_split(
        configuration=create_configuration(
            "moderate"
        ),
        examples_per_class_per_snr=1,
        seed=2026,
    )

    assert dataset.iq.shape == (4, 2, 256)
    assert dataset.iq.dtype == np.float32
    assert np.all(np.isfinite(dataset.iq))
    assert dataset.labels.shape == (4,)
    assert dataset.snr_db.shape == (4,)


def test_multipath_dataset_is_reproducible() -> None:
    configuration = create_configuration(
        "severe"
    )

    first = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=2,
        seed=2026,
    )
    second = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=2,
        seed=2026,
    )

    np.testing.assert_array_equal(
        first.iq,
        second.iq,
    )
    np.testing.assert_array_equal(
        first.labels,
        second.labels,
    )
    np.testing.assert_array_equal(
        first.example_seed,
        second.example_seed,
    )


def test_multipath_changes_same_seed_dataset() -> None:
    clean = build_dataset_split(
        configuration=create_configuration(),
        examples_per_class_per_snr=1,
        seed=2026,
    )
    multipath = build_dataset_split(
        configuration=create_configuration(
            "mild"
        ),
        examples_per_class_per_snr=1,
        seed=2026,
    )

    np.testing.assert_array_equal(
        clean.labels,
        multipath.labels,
    )
    np.testing.assert_array_equal(
        clean.example_seed,
        multipath.example_seed,
    )

    assert not np.array_equal(
        clean.iq,
        multipath.iq,
    )


def test_default_and_explicit_none_match() -> None:
    default_configuration = (
        DatasetGenerationConfig(
            dataset_name="default_test",
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
        )
    )

    explicit_none = create_configuration(None)

    default_split = build_dataset_split(
        configuration=default_configuration,
        examples_per_class_per_snr=1,
        seed=2026,
    )
    explicit_split = build_dataset_split(
        configuration=explicit_none,
        examples_per_class_per_snr=1,
        seed=2026,
    )

    np.testing.assert_array_equal(
        default_split.iq,
        explicit_split.iq,
    )
    np.testing.assert_array_equal(
        default_split.labels,
        explicit_split.labels,
    )


def test_manifest_records_multipath_profile(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"

    write_dataset_manifest(
        configuration=create_configuration(
            "moderate"
        ),
        split_files={
            "test": tmp_path / "test.npz",
        },
        split_sizes={
            "test": 4,
        },
        output_path=manifest_path,
    )

    content = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        content["configuration"][
            "multipath_profile"
        ]
        == "moderate"
    )
