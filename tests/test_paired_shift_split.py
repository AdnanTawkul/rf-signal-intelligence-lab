from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.paired_shift_split import (
    create_paired_shift_split,
)


def example_metadata(
    *,
    examples_per_stratum: int = 10,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    labels = []
    snr_values = []

    for label in range(4):
        for snr in (-4.0, 0.0, 8.0):
            labels.extend(
                [label]
                * examples_per_stratum
            )
            snr_values.extend(
                [snr]
                * examples_per_stratum
            )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )
    snr_array = np.asarray(
        snr_values,
        dtype=np.float64,
    )
    seeds = np.arange(
        10_000,
        10_000 + labels_array.size,
        dtype=np.uint64,
    )

    return (
        labels_array,
        snr_array,
        seeds,
    )


def test_split_is_deterministic() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    first = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
        split_seed=2026,
    )
    second = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
        split_seed=2026,
    )

    np.testing.assert_array_equal(
        first.development_indices,
        second.development_indices,
    )
    np.testing.assert_array_equal(
        first.test_indices,
        second.test_indices,
    )


def test_different_seed_changes_split() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    first = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
        split_seed=2026,
    )
    second = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
        split_seed=2027,
    )

    assert not np.array_equal(
        first.development_indices,
        second.development_indices,
    )


def test_split_is_complete_and_disjoint() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    split = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
    )

    assert np.intersect1d(
        split.development_indices,
        split.test_indices,
    ).size == 0

    combined = np.sort(
        np.concatenate(
            (
                split.development_indices,
                split.test_indices,
            )
        )
    )

    np.testing.assert_array_equal(
        combined,
        np.arange(labels.size),
    )


def test_each_stratum_is_balanced() -> None:
    labels, snr_db, seeds = (
        example_metadata(
            examples_per_stratum=10
        )
    )

    split = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
        development_fraction=0.4,
    )

    for label in range(4):
        for snr in (-4.0, 0.0, 8.0):
            stratum = np.flatnonzero(
                (labels == label)
                & (snr_db == snr)
            )
            selected = np.intersect1d(
                stratum,
                split.development_indices,
            )

            assert selected.size == 4


def test_summary() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    split = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
        development_fraction=0.5,
    )
    summary = split.summary()

    assert summary[
        "development_count"
    ] == 60
    assert summary["test_count"] == 60
    assert summary["example_count"] == 120
    assert summary["stratum_count"] == 12


def test_same_metadata_reuses_indices() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    clean_split = create_paired_shift_split(
        labels,
        snr_db,
        seeds,
    )
    shifted_split = create_paired_shift_split(
        labels.copy(),
        snr_db.copy(),
        seeds.copy(),
    )

    np.testing.assert_array_equal(
        clean_split.development_indices,
        shifted_split.development_indices,
    )


def test_rejects_duplicate_seeds() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )
    seeds[1] = seeds[0]

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        create_paired_shift_split(
            labels,
            snr_db,
            seeds,
        )


def test_rejects_single_example_stratum() -> None:
    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        create_paired_shift_split(
            labels=np.asarray(
                [0, 1, 1],
                dtype=np.int64,
            ),
            snr_db=np.asarray(
                [0.0, 0.0, 0.0],
            ),
            example_seed=np.asarray(
                [1, 2, 3],
                dtype=np.uint64,
            ),
        )


@pytest.mark.parametrize(
    "fraction",
    (
        0.0,
        1.0,
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_rejects_invalid_fraction(
    fraction: object,
) -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    with pytest.raises(
        ValueError,
        match="development_fraction",
    ):
        create_paired_shift_split(
            labels,
            snr_db,
            seeds,
            development_fraction=fraction,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("split_seed", -1),
        ("split_seed", True),
        ("snr_decimals", -1),
        ("snr_decimals", 1.5),
    ),
)
def test_rejects_invalid_integer_options(
    name: str,
    value: object,
) -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )
    options = {
        "split_seed": 2026,
        "snr_decimals": 6,
    }
    options[name] = value

    with pytest.raises(ValueError):
        create_paired_shift_split(
            labels,
            snr_db,
            seeds,
            **options,
        )


@pytest.mark.parametrize(
    "labels",
    (
        [],
        [0.0, 1.0],
        [True, False],
        [[0, 1]],
        [-1, 0],
    ),
)
def test_rejects_invalid_labels(
    labels: object,
) -> None:
    with pytest.raises(ValueError):
        create_paired_shift_split(
            labels,
            snr_db=[0.0, 0.0],
            example_seed=[1, 2],
        )


def test_rejects_metadata_length_mismatch() -> None:
    labels, snr_db, seeds = (
        example_metadata()
    )

    with pytest.raises(ValueError):
        create_paired_shift_split(
            labels,
            snr_db[:-1],
            seeds,
        )
