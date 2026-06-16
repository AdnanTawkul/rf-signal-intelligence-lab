from __future__ import annotations

import numpy as np

from rfsil.data.dataset import (
    DatasetGenerationConfig,
    build_dataset_split,
    load_dataset_split,
    save_dataset_split,
)


def create_test_configuration() -> DatasetGenerationConfig:
    """Create a small balanced dataset configuration for testing."""
    return DatasetGenerationConfig(
        dataset_name="test_dataset",
        sample_count=256,
        sample_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
        snr_values_db=(-6.0, 6.0),
        frequency_offset_range_hz=(-5_000.0, 5_000.0),
        phase_offset_range_rad=(-1.0, 1.0),
        amplitude_scale_range=(0.8, 1.2),
        time_shift_range_samples=(-4, 4),
        rayleigh_probability=0.25,
    )


def test_dataset_split_has_expected_shapes_and_dtypes() -> None:
    configuration = create_test_configuration()

    dataset = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=2,
        seed=42,
    )

    expected_examples = 4 * 2 * 2

    assert dataset.iq.shape == (
        expected_examples,
        2,
        configuration.sample_count,
    )
    assert dataset.labels.shape == (expected_examples,)
    assert dataset.snr_db.shape == (expected_examples,)
    assert dataset.iq.dtype == np.float32
    assert dataset.labels.dtype == np.int64
    assert dataset.snr_db.dtype == np.float32


def test_dataset_is_balanced_by_label_and_snr() -> None:
    configuration = create_test_configuration()

    dataset = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=3,
        seed=42,
    )

    for label in range(4):
        for snr_value in configuration.snr_values_db:
            matching = (
                (dataset.labels == label)
                & np.isclose(dataset.snr_db, snr_value)
            )

            assert int(np.count_nonzero(matching)) == 3


def test_dataset_generation_is_reproducible() -> None:
    configuration = create_test_configuration()

    dataset_a = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=2,
        seed=123,
    )
    dataset_b = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=2,
        seed=123,
    )

    np.testing.assert_array_equal(dataset_a.iq, dataset_b.iq)
    np.testing.assert_array_equal(dataset_a.labels, dataset_b.labels)
    np.testing.assert_array_equal(dataset_a.snr_db, dataset_b.snr_db)
    np.testing.assert_array_equal(
        dataset_a.example_seed,
        dataset_b.example_seed,
    )


def test_different_dataset_seeds_produce_different_iq() -> None:
    configuration = create_test_configuration()

    dataset_a = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=1,
        seed=1,
    )
    dataset_b = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=1,
        seed=2,
    )

    assert not np.array_equal(dataset_a.iq, dataset_b.iq)


def test_dataset_save_and_load_round_trip(tmp_path) -> None:
    configuration = create_test_configuration()

    original = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=1,
        seed=42,
    )

    output_path = tmp_path / "split.npz"
    save_dataset_split(original, output_path)
    loaded = load_dataset_split(output_path)

    np.testing.assert_array_equal(loaded.iq, original.iq)
    np.testing.assert_array_equal(loaded.labels, original.labels)
    np.testing.assert_array_equal(loaded.snr_db, original.snr_db)
    np.testing.assert_array_equal(
        loaded.frequency_offset_hz,
        original.frequency_offset_hz,
    )
    np.testing.assert_array_equal(
        loaded.rayleigh_fading,
        original.rayleigh_fading,
    )
