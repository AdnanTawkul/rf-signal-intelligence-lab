from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.classification import (
    evaluate_predictions,
)
from rfsil.evaluation.prediction_artifacts import (
    load_prediction_results,
)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ConditionConfusionSummary:
    """Per-seed confusion results for one channel condition."""

    condition: str
    seeds: tuple[int, ...]
    class_names: tuple[str, ...]
    labels: Int64Array
    snr_db: Float32Array
    overall_accuracy: Float64Array
    confusion_matrices: Int64Array
    normalized_confusion_matrices: Float64Array


def _validate_condition(value: object) -> str:
    """Validate a channel-condition name."""
    if not isinstance(value, str):
        raise ValueError(
            "condition must be a string."
        )

    validated = value.strip()

    if not validated:
        raise ValueError(
            "condition must not be empty."
        )

    return validated


def _validate_seeds(
    values: Sequence[object],
) -> tuple[int, ...]:
    """Validate a nonempty sequence of unique seeds."""
    if not values:
        raise ValueError(
            "seeds must not be empty."
        )

    seeds: list[int] = []

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise ValueError(
                "Every seed must be an integer."
            )

        seeds.append(int(value))

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Seeds must not contain duplicates."
        )

    return tuple(seeds)


def _validate_class_names(
    values: Sequence[object],
) -> tuple[str, ...]:
    """Validate class labels."""
    if len(values) < 2:
        raise ValueError(
            "At least two class names are required."
        )

    names: list[str] = []

    for value in values:
        if not isinstance(value, str):
            raise ValueError(
                "Every class name must be a string."
            )

        name = value.strip()

        if not name:
            raise ValueError(
                "Class names must not be empty."
            )

        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError(
            "Class names must not contain duplicates."
        )

    return tuple(names)


def load_condition_confusions(
    condition: str,
    predictions_directory: str | Path,
    seeds: Sequence[int],
    class_names: Sequence[str],
) -> ConditionConfusionSummary:
    """Load and evaluate all per-seed prediction artifacts."""
    selected_condition = _validate_condition(
        condition
    )
    selected_seeds = _validate_seeds(
        tuple(seeds)
    )
    selected_class_names = _validate_class_names(
        tuple(class_names)
    )

    directory = Path(predictions_directory)

    if not directory.is_dir():
        raise FileNotFoundError(
            "Predictions directory does not exist: "
            f"{directory}"
        )

    reference_labels: Int64Array | None = None
    reference_snr: Float32Array | None = None

    overall_accuracy: list[float] = []
    confusion_matrices: list[Int64Array] = []
    normalized_matrices: list[Float64Array] = []

    for seed in selected_seeds:
        path = directory / f"seed_{seed}.npz"
        predictions = load_prediction_results(path)

        if reference_labels is None:
            reference_labels = (
                predictions.labels.copy()
            )
            reference_snr = (
                predictions.snr_db.copy()
            )
        else:
            np.testing.assert_array_equal(
                predictions.labels,
                reference_labels,
                err_msg=(
                    "Labels differ across prediction "
                    "artifacts."
                ),
            )
            np.testing.assert_array_equal(
                predictions.snr_db,
                reference_snr,
                err_msg=(
                    "SNR values differ across prediction "
                    "artifacts."
                ),
            )

        evaluation = evaluate_predictions(
            labels=predictions.labels,
            predictions=predictions.predictions,
            snr_db=predictions.snr_db,
            num_classes=len(selected_class_names),
        )

        overall_accuracy.append(
            evaluation.accuracy
        )
        confusion_matrices.append(
            evaluation.confusion_matrix
        )
        normalized_matrices.append(
            evaluation.normalized_confusion_matrix.astype(
                np.float64
            )
        )

    if (
        reference_labels is None
        or reference_snr is None
    ):
        raise RuntimeError(
            "No prediction artifacts were loaded."
        )

    return ConditionConfusionSummary(
        condition=selected_condition,
        seeds=selected_seeds,
        class_names=selected_class_names,
        labels=reference_labels,
        snr_db=reference_snr,
        overall_accuracy=np.asarray(
            overall_accuracy,
            dtype=np.float64,
        ),
        confusion_matrices=np.stack(
            confusion_matrices,
        ).astype(np.int64),
        normalized_confusion_matrices=np.stack(
            normalized_matrices,
        ).astype(np.float64),
    )


def _normalize_confusion(
    confusion: Int64Array,
) -> Float64Array:
    """Normalize one confusion matrix by true-class row."""
    totals = confusion.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        confusion,
        totals,
        out=np.zeros_like(
            confusion,
            dtype=np.float64,
        ),
        where=totals > 0,
    )


def compare_condition_confusions(
    conditions: Mapping[
        str,
        ConditionConfusionSummary,
    ],
    reference_condition: str = "clean",
) -> dict[str, Any]:
    """Compare pooled confusion behavior across conditions."""
    if not conditions:
        raise ValueError(
            "conditions must not be empty."
        )

    if reference_condition not in conditions:
        raise ValueError(
            "reference_condition must exist in "
            "conditions."
        )

    reference = conditions[
        reference_condition
    ]

    for name, summary in conditions.items():
        if summary.seeds != reference.seeds:
            raise ValueError(
                f"Condition {name!r} does not use "
                "the same seeds."
            )

        if (
            summary.class_names
            != reference.class_names
        ):
            raise ValueError(
                f"Condition {name!r} does not use "
                "the same class names."
            )

        if not np.array_equal(
            summary.labels,
            reference.labels,
        ):
            raise ValueError(
                f"Condition {name!r} does not use "
                "the same labels."
            )

        if not np.array_equal(
            summary.snr_db,
            reference.snr_db,
        ):
            raise ValueError(
                f"Condition {name!r} does not use "
                "the same SNR metadata."
            )

    reference_accuracy = (
        reference.overall_accuracy
    )
    result_conditions: dict[str, Any] = {}

    for name, summary in conditions.items():
        pooled_confusion = np.sum(
            summary.confusion_matrices,
            axis=0,
            dtype=np.int64,
        )
        pooled_normalized = _normalize_confusion(
            pooled_confusion
        )
        normalized_mean = np.mean(
            summary.normalized_confusion_matrices,
            axis=0,
        )
        normalized_std = np.std(
            summary.normalized_confusion_matrices,
            axis=0,
        )
        paired_drop = (
            reference_accuracy
            - summary.overall_accuracy
        )

        dominant_errors: dict[
            str,
            dict[str, Any],
        ] = {}

        for class_index, class_name in enumerate(
            summary.class_names
        ):
            error_rates = (
                pooled_normalized[
                    class_index
                ].copy()
            )
            error_rates[class_index] = -1.0

            predicted_index = int(
                np.argmax(error_rates)
            )

            dominant_errors[class_name] = {
                "predicted_class": (
                    summary.class_names[
                        predicted_index
                    ]
                ),
                "rate": float(
                    max(
                        error_rates[
                            predicted_index
                        ],
                        0.0,
                    )
                ),
            }

        result_conditions[name] = {
            "overall_mean": float(
                np.mean(
                    summary.overall_accuracy
                )
            ),
            "overall_standard_deviation": float(
                np.std(
                    summary.overall_accuracy
                )
            ),
            "paired_drop_from_reference": {
                "mean": float(
                    np.mean(paired_drop)
                ),
                "standard_deviation": float(
                    np.std(paired_drop)
                ),
                "per_seed": {
                    str(seed): float(drop)
                    for seed, drop in zip(
                        summary.seeds,
                        paired_drop,
                        strict=True,
                    )
                },
            },
            "pooled_confusion_matrix": (
                pooled_confusion.tolist()
            ),
            "pooled_normalized_confusion_matrix": (
                pooled_normalized.tolist()
            ),
            "mean_normalized_confusion_matrix": (
                normalized_mean.tolist()
            ),
            "standard_deviation_normalized_confusion_matrix": (
                normalized_std.tolist()
            ),
            "class_accuracy": {
                class_name: float(
                    pooled_normalized[
                        index,
                        index,
                    ]
                )
                for index, class_name
                in enumerate(
                    summary.class_names
                )
            },
            "dominant_errors": dominant_errors,
        }

    return {
        "format_version": 1,
        "reference_condition": (
            reference_condition
        ),
        "seeds": list(reference.seeds),
        "class_names": list(
            reference.class_names
        ),
        "example_count_per_seed": int(
            reference.labels.size
        ),
        "conditions": result_conditions,
    }


__all__ = [
    "ConditionConfusionSummary",
    "compare_condition_confusions",
    "load_condition_confusions",
]
