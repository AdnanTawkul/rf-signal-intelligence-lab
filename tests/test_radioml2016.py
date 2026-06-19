from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from rfsil.data.dataset import (
    save_dataset_split,
)
from rfsil.data.radioml2016 import (
    build_radioml2016_four_class_splits,
    load_radioml2016_dictionary,
)
from rfsil.data.torch_dataset import (
    NPZIQDataset,
)


def create_groups(
    examples_per_group: int = 10,
) -> dict[tuple[str, int], np.ndarray]:
    groups: dict[
        tuple[str, int],
        np.ndarray,
    ] = {}

    modulation_names = (
        "BPSK",
        "QPSK",
        "8PSK",
        "QAM16",
    )
    snr_values = (
        -2,
        2,
    )

    for class_index, modulation in enumerate(
        modulation_names
    ):
        for snr_index, snr in enumerate(
            snr_values
        ):
            value = float(
                class_index * 10
                + snr_index
            )

            groups[
                (modulation, snr)
            ] = np.full(
                (
                    examples_per_group,
                    2,
                    128,
                ),
                value,
                dtype=np.float32,
            )

    return groups


def test_loads_expected_numpy_dictionary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "radioml.pkl"
    groups = create_groups()

    with path.open("wb") as file:
        pickle.dump(
            groups,
            file,
            protocol=2,
        )

    loaded = load_radioml2016_dictionary(
        path
    )

    assert loaded.keys() == groups.keys()

    for key in groups:
        np.testing.assert_array_equal(
            loaded[key],
            groups[key],
        )


def test_restricted_loader_blocks_other_globals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blocked.pkl"

    with path.open("wb") as file:
        pickle.dump(
            datetime(2026, 1, 1),
            file,
            protocol=2,
        )

    with pytest.raises(
        pickle.UnpicklingError,
        match="Blocked unsupported pickle global",
    ):
        load_radioml2016_dictionary(path)


def test_four_class_splits_are_balanced() -> None:
    splits, snr_values = (
        build_radioml2016_four_class_splits(
            create_groups(),
            split_counts={
                "train": 6,
                "validation": 2,
                "test": 2,
            },
            seed=2026,
        )
    )

    assert snr_values == (-2, 2)

    expected_sizes = {
        "train": 48,
        "validation": 16,
        "test": 16,
    }

    examples_per_group = {
        "train": 6,
        "validation": 2,
        "test": 2,
    }

    for split_name, split in splits.items():
        assert (
            split.iq.shape
            == (
                expected_sizes[split_name],
                2,
                128,
            )
        )

        for label in range(4):
            for snr in snr_values:
                count = int(
                    np.sum(
                        (split.labels == label)
                        & (split.snr_db == snr)
                    )
                )

                assert count == (
                    examples_per_group[
                        split_name
                    ]
                )


def test_conversion_is_reproducible() -> None:
    first, _ = (
        build_radioml2016_four_class_splits(
            create_groups(),
            split_counts={
                "train": 6,
                "validation": 2,
                "test": 2,
            },
            seed=2026,
        )
    )
    second, _ = (
        build_radioml2016_four_class_splits(
            create_groups(),
            split_counts={
                "train": 6,
                "validation": 2,
                "test": 2,
            },
            seed=2026,
        )
    )

    for split_name in first:
        np.testing.assert_array_equal(
            first[split_name].iq,
            second[split_name].iq,
        )
        np.testing.assert_array_equal(
            first[split_name].labels,
            second[split_name].labels,
        )
        np.testing.assert_array_equal(
            first[split_name].example_seed,
            second[split_name].example_seed,
        )


def test_neutral_compatibility_metadata() -> None:
    splits, _ = (
        build_radioml2016_four_class_splits(
            create_groups(),
            split_counts={
                "train": 6,
                "validation": 2,
                "test": 2,
            },
            seed=2026,
        )
    )

    split = splits["test"]

    assert np.all(
        split.frequency_offset_hz == 0.0
    )
    assert np.all(
        split.phase_offset_rad == 0.0
    )
    assert np.all(
        split.amplitude_scale == 1.0
    )
    assert np.all(
        split.time_shift_samples == 0
    )
    assert not np.any(
        split.rayleigh_fading
    )


def test_converted_split_loads_with_torch_dataset(
    tmp_path: Path,
) -> None:
    splits, _ = (
        build_radioml2016_four_class_splits(
            create_groups(),
            split_counts={
                "train": 6,
                "validation": 2,
                "test": 2,
            },
            seed=2026,
        )
    )

    output_path = tmp_path / "test.npz"
    save_dataset_split(
        splits["test"],
        output_path,
    )

    dataset = NPZIQDataset(output_path)

    assert len(dataset) == 16
    assert dataset.iq.shape == (
        16,
        2,
        128,
    )
    assert dataset.labels.dtype.is_floating_point is False


def test_source_examples_are_disjoint_and_labels_match() -> None:
    groups = create_groups()

    expected_labels = {
        "BPSK": 0,
        "QPSK": 1,
        "8PSK": 2,
        "QAM16": 3,
    }

    expected_markers: dict[
        str,
        set[int],
    ] = {}

    for source_name, label in expected_labels.items():
        key = (
            source_name,
            -2,
        )
        markers: set[int] = set()

        for example_index in range(
            groups[key].shape[0]
        ):
            marker = (
                label * 100
                + example_index
            )
            groups[key][
                example_index
            ].fill(float(marker))
            markers.add(marker)

        expected_markers[source_name] = markers

    splits, _ = (
        build_radioml2016_four_class_splits(
            groups,
            split_counts={
                "train": 6,
                "validation": 2,
                "test": 2,
            },
            seed=2026,
        )
    )

    for source_name, label in expected_labels.items():
        values_by_split: dict[
            str,
            set[int],
        ] = {}

        for split_name, split in splits.items():
            selected = (
                (split.labels == label)
                & (split.snr_db == -2.0)
            )

            values = {
                int(value)
                for value in split.iq[
                    selected,
                    0,
                    0,
                ]
            }

            values_by_split[
                split_name
            ] = values

        assert values_by_split[
            "train"
        ].isdisjoint(
            values_by_split["validation"]
        )
        assert values_by_split[
            "train"
        ].isdisjoint(
            values_by_split["test"]
        )
        assert values_by_split[
            "validation"
        ].isdisjoint(
            values_by_split["test"]
        )

        combined = set().union(
            *values_by_split.values()
        )

        assert combined == (
            expected_markers[source_name]
        )
