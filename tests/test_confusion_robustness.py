from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.confusion_robustness import (
    compare_condition_confusions,
    load_condition_confusions,
)


def write_predictions(
    directory: Path,
    seed: int,
    labels: tuple[int, ...],
    predictions: tuple[int, ...],
    snr_db: tuple[float, ...] | None = None,
) -> None:
    """Write one prediction artifact."""
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if snr_db is None:
        snr_db = tuple(
            0.0
            for _ in labels
        )

    np.savez_compressed(
        directory / f"seed_{seed}.npz",
        labels=np.asarray(
            labels,
            dtype=np.int64,
        ),
        predictions=np.asarray(
            predictions,
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            snr_db,
            dtype=np.float32,
        ),
    )


def create_condition(
    tmp_path: Path,
    name: str,
    predictions: tuple[int, ...],
):
    """Create and load one two-seed condition."""
    directory = tmp_path / name
    labels = (0, 0, 1, 1)

    for seed in (2026, 2027):
        write_predictions(
            directory,
            seed,
            labels,
            predictions,
        )

    return load_condition_confusions(
        condition=name,
        predictions_directory=directory,
        seeds=(2026, 2027),
        class_names=("A", "B"),
    )


def test_load_condition_shapes(
    tmp_path: Path,
) -> None:
    summary = create_condition(
        tmp_path,
        "clean",
        (0, 0, 1, 1),
    )

    assert summary.seeds == (
        2026,
        2027,
    )
    assert summary.labels.shape == (4,)
    assert (
        summary.confusion_matrices.shape
        == (2, 2, 2)
    )
    assert (
        summary
        .normalized_confusion_matrices
        .shape
        == (2, 2, 2)
    )


def test_pooled_confusion_is_correct(
    tmp_path: Path,
) -> None:
    clean = create_condition(
        tmp_path,
        "clean",
        (0, 1, 1, 0),
    )

    result = compare_condition_confusions(
        {"clean": clean}
    )

    matrix = result["conditions"]["clean"][
        "pooled_confusion_matrix"
    ]

    assert matrix == [
        [2, 2],
        [2, 2],
    ]


def test_dominant_error_is_reported(
    tmp_path: Path,
) -> None:
    clean = create_condition(
        tmp_path,
        "clean",
        (1, 1, 1, 1),
    )

    result = compare_condition_confusions(
        {"clean": clean}
    )

    error = result["conditions"]["clean"][
        "dominant_errors"
    ]["A"]

    assert error["predicted_class"] == "B"
    assert error["rate"] == pytest.approx(
        1.0
    )


def test_paired_drop_is_computed(
    tmp_path: Path,
) -> None:
    clean = create_condition(
        tmp_path,
        "clean",
        (0, 0, 1, 1),
    )
    mild = create_condition(
        tmp_path,
        "mild",
        (1, 0, 0, 1),
    )

    result = compare_condition_confusions(
        {
            "clean": clean,
            "mild": mild,
        }
    )

    drop = result["conditions"]["mild"][
        "paired_drop_from_reference"
    ]

    assert drop["mean"] == pytest.approx(
        0.5
    )


def test_mismatched_labels_across_seeds_fail(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "condition"

    write_predictions(
        directory,
        2026,
        (0, 0, 1, 1),
        (0, 0, 1, 1),
    )
    write_predictions(
        directory,
        2027,
        (0, 1, 1, 1),
        (0, 1, 1, 1),
    )

    with pytest.raises(AssertionError):
        load_condition_confusions(
            "condition",
            directory,
            (2026, 2027),
            ("A", "B"),
        )


def test_mismatched_snr_across_seeds_fail(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "condition"

    write_predictions(
        directory,
        2026,
        (0, 1),
        (0, 1),
        (-4.0, 0.0),
    )
    write_predictions(
        directory,
        2027,
        (0, 1),
        (0, 1),
        (0.0, 4.0),
    )

    with pytest.raises(AssertionError):
        load_condition_confusions(
            "condition",
            directory,
            (2026, 2027),
            ("A", "B"),
        )


def test_mismatched_condition_metadata_fail(
    tmp_path: Path,
) -> None:
    clean = create_condition(
        tmp_path,
        "clean",
        (0, 0, 1, 1),
    )

    directory = tmp_path / "other"

    for seed in (2026, 2027):
        write_predictions(
            directory,
            seed,
            (0, 1, 1, 1),
            (0, 1, 1, 1),
        )

    other = load_condition_confusions(
        "other",
        directory,
        (2026, 2027),
        ("A", "B"),
    )

    with pytest.raises(ValueError):
        compare_condition_confusions(
            {
                "clean": clean,
                "other": other,
            }
        )


def test_missing_seed_file_fails(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "condition"

    write_predictions(
        directory,
        2026,
        (0, 1),
        (0, 1),
    )

    with pytest.raises(FileNotFoundError):
        load_condition_confusions(
            "condition",
            directory,
            (2026, 2027),
            ("A", "B"),
        )
