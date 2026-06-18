from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from rfsil.data.stratified_subset import (
    create_class_snr_stratified_subset,
    select_class_snr_stratified_indices,
)


def create_metadata() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    labels: list[int] = []
    snr_values: list[float] = []

    for label in range(2):
        for snr in (-4.0, 0.0, 4.0):
            labels.extend([label] * 5)
            snr_values.extend([snr] * 5)

    return (
        np.asarray(labels, dtype=np.int64),
        np.asarray(
            snr_values,
            dtype=np.float32,
        ),
    )


def test_selection_has_requested_count_per_stratum() -> None:
    labels, snr_db = create_metadata()

    indices = select_class_snr_stratified_indices(
        labels,
        snr_db,
        examples_per_stratum=2,
        seed=2026,
    )

    assert indices.shape == (12,)

    selected_labels = labels[indices]
    selected_snr = snr_db[indices]

    for label in np.unique(labels):
        for snr in np.unique(snr_db):
            count = np.count_nonzero(
                (selected_labels == label)
                & (selected_snr == snr)
            )

            assert count == 2


def test_selection_is_reproducible() -> None:
    labels, snr_db = create_metadata()

    first = select_class_snr_stratified_indices(
        labels,
        snr_db,
        examples_per_stratum=3,
        seed=2026,
    )
    second = select_class_snr_stratified_indices(
        labels,
        snr_db,
        examples_per_stratum=3,
        seed=2026,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_different_seeds_change_selection() -> None:
    labels, snr_db = create_metadata()

    first = select_class_snr_stratified_indices(
        labels,
        snr_db,
        examples_per_stratum=2,
        seed=2026,
    )
    second = select_class_snr_stratified_indices(
        labels,
        snr_db,
        examples_per_stratum=2,
        seed=2027,
    )

    assert not np.array_equal(first, second)


def test_selection_contains_no_duplicates() -> None:
    labels, snr_db = create_metadata()

    indices = select_class_snr_stratified_indices(
        labels,
        snr_db,
        examples_per_stratum=4,
        seed=2026,
    )

    assert (
        np.unique(indices).shape[0]
        == indices.shape[0]
    )


def test_insufficient_examples_are_rejected() -> None:
    labels, snr_db = create_metadata()

    with pytest.raises(ValueError):
        select_class_snr_stratified_indices(
            labels,
            snr_db,
            examples_per_stratum=6,
            seed=2026,
        )


@pytest.mark.parametrize(
    "examples_per_stratum",
    [0, -1, 1.5, True],
)
def test_invalid_example_count_is_rejected(
    examples_per_stratum: object,
) -> None:
    labels, snr_db = create_metadata()

    with pytest.raises(ValueError):
        select_class_snr_stratified_indices(
            labels,
            snr_db,
            examples_per_stratum=examples_per_stratum,
            seed=2026,
        )


def test_non_integer_labels_are_rejected() -> None:
    labels, snr_db = create_metadata()

    with pytest.raises(TypeError):
        select_class_snr_stratified_indices(
            labels.astype(np.float32),
            snr_db,
            examples_per_stratum=2,
            seed=2026,
        )


def test_non_numeric_snr_is_rejected() -> None:
    labels, snr_db = create_metadata()

    with pytest.raises(TypeError):
        select_class_snr_stratified_indices(
            labels,
            snr_db.astype(str),
            examples_per_stratum=2,
            seed=2026,
        )


def test_mismatched_lengths_are_rejected() -> None:
    labels, snr_db = create_metadata()

    with pytest.raises(ValueError):
        select_class_snr_stratified_indices(
            labels[:-1],
            snr_db,
            examples_per_stratum=2,
            seed=2026,
        )


def test_non_finite_snr_is_rejected() -> None:
    labels, snr_db = create_metadata()
    snr_db[0] = np.nan

    with pytest.raises(ValueError):
        select_class_snr_stratified_indices(
            labels,
            snr_db,
            examples_per_stratum=2,
            seed=2026,
        )


@pytest.mark.parametrize(
    "seed",
    [True, 1.5],
)
def test_invalid_seed_is_rejected(
    seed: object,
) -> None:
    labels, snr_db = create_metadata()

    with pytest.raises(ValueError):
        select_class_snr_stratified_indices(
            labels,
            snr_db,
            examples_per_stratum=2,
            seed=seed,
        )


class TinyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self) -> None:
        labels, snr_db = create_metadata()

        self.labels = torch.from_numpy(labels)
        self.snr_db = torch.from_numpy(snr_db)
        self.iq = torch.randn(
            labels.shape[0],
            2,
            16,
        )

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "iq": self.iq[index],
            "label": self.labels[index],
            "snr_db": self.snr_db[index],
        }


def test_pytorch_subset_has_expected_length() -> None:
    dataset = TinyDataset()

    subset = create_class_snr_stratified_subset(
        dataset,
        examples_per_stratum=2,
        seed=2026,
    )

    assert len(subset) == 12
