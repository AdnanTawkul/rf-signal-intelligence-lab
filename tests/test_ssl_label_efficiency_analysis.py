from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.ssl_label_efficiency_analysis import (
    build_paired_change_matrix,
    pool_confusion_matrices,
    summarize_selection,
)


def make_validation() -> dict[str, object]:
    return {
        "fractions": {
            "labels_001pct": {
                "methods": {
                    "random": {
                        "mean_validation_accuracy": 0.70,
                        "validation_accuracy_standard_deviation": 0.02,
                    },
                    "simclr": {
                        "mean_validation_accuracy": 0.75,
                        "validation_accuracy_standard_deviation": 0.01,
                    },
                }
            }
        }
    }


def make_held_out() -> dict[str, object]:
    conditions = {}

    for condition, random, simclr in (
        ("clean", 0.70, 0.75),
        ("mild", 0.60, 0.61),
        ("moderate", 0.50, 0.48),
        ("severe", 0.40, 0.37),
    ):
        conditions[condition] = {
            "methods": {
                "random": {
                    "mean": random,
                    "standard_deviation": 0.01,
                },
                "simclr": {
                    "mean": simclr,
                    "standard_deviation": 0.02,
                },
            },
            "paired_changes_vs_random": {
                "simclr": {
                    "mean": simclr - random,
                    "standard_deviation": 0.01,
                    "seeds_improved": 3,
                }
            },
        }

    return {
        "fractions": {
            "labels_001pct": {
                "conditions": conditions,
                "macro_condition_average": {
                    "methods": {
                        "random": {
                            "mean": 0.55,
                            "standard_deviation": 0.01,
                        },
                        "simclr": {
                            "mean": 0.5525,
                            "standard_deviation": 0.02,
                        },
                    },
                    "paired_changes_vs_random": {
                        "simclr": {
                            "mean": 0.0025,
                            "standard_deviation": 0.01,
                            "seeds_improved": 3,
                        }
                    },
                },
            }
        }
    }


def test_summarizes_selection() -> None:
    summary = summarize_selection(
        validation_aggregate=make_validation(),
        held_out_aggregate=make_held_out(),
        fraction_identifier="labels_001pct",
        method="simclr",
        conditions=(
            "clean",
            "mild",
            "moderate",
            "severe",
        ),
    )

    assert (
        summary.validation_accuracy
        == pytest.approx(0.75)
    )
    assert (
        summary.condition_metrics[
            "clean"
        ]["change_vs_random"]
        == pytest.approx(0.05)
    )
    assert (
        summary.macro_change_vs_random
        == pytest.approx(0.0025)
    )


def test_builds_paired_change_matrix() -> None:
    labels, matrix = (
        build_paired_change_matrix(
            aggregate=make_held_out(),
            fractions=(
                "labels_001pct",
            ),
            conditions=(
                "clean",
                "mild",
                "moderate",
                "severe",
            ),
            methods=(
                "random",
                "simclr",
            ),
        )
    )

    assert labels == [
        "clean: simclr",
        "mild: simclr",
        "moderate: simclr",
        "severe: simclr",
    ]
    assert matrix.shape == (4, 1)
    assert matrix[0, 0] == pytest.approx(
        0.05
    )


def write_metrics(
    path: Path,
    *,
    class_names: list[str],
    confusion: list[list[int]],
    accuracy: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "confusion_matrix": confusion,
                "overall_accuracy": accuracy,
            }
        ),
        encoding="utf-8",
    )


def test_pools_confusion_matrices(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_metrics(
        first,
        class_names=["A", "B"],
        confusion=[
            [8, 2],
            [1, 9],
        ],
        accuracy=0.85,
    )
    write_metrics(
        second,
        class_names=["A", "B"],
        confusion=[
            [7, 3],
            [2, 8],
        ],
        accuracy=0.75,
    )

    pooled = pool_confusion_matrices(
        [first, second]
    )

    assert pooled[
        "confusion_matrix"
    ] == [
        [15, 5],
        [3, 17],
    ]

    normalized = np.asarray(
        pooled[
            "normalized_confusion_matrix"
        ]
    )

    assert normalized[0, 0] == pytest.approx(
        0.75
    )
    assert normalized[1, 1] == pytest.approx(
        0.85
    )
    assert pooled[
        "mean_accuracy"
    ] == pytest.approx(0.80)


def test_rejects_mismatched_classes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_metrics(
        first,
        class_names=["A", "B"],
        confusion=[
            [1, 0],
            [0, 1],
        ],
        accuracy=1.0,
    )
    write_metrics(
        second,
        class_names=["A", "C"],
        confusion=[
            [1, 0],
            [0, 1],
        ],
        accuracy=1.0,
    )

    with pytest.raises(
        ValueError,
        match="Class names differ",
    ):
        pool_confusion_matrices(
            [first, second]
        )
