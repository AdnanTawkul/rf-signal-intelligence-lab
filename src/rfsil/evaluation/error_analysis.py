from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassSNRAnalysis:
    """Per-class classification performance at each evaluated SNR level."""

    snr_values_db: Float32Array
    accuracy: Float32Array
    counts: Int64Array
    error_counts: Int64Array


def _validate_num_classes(num_classes: object) -> int:
    """Validate and return the number of classes."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise ValueError("num_classes must be an integer.")

    validated = int(num_classes)

    if validated < 2:
        raise ValueError("num_classes must be at least 2.")

    return validated


def compute_class_snr_analysis(
    labels: NDArray[np.integer],
    predictions: NDArray[np.integer],
    snr_db: NDArray[np.floating],
    num_classes: int,
) -> ClassSNRAnalysis:
    """Compute accuracy and error counts for every class-SNR combination.

    Rows correspond to true classes. Columns correspond to sorted SNR levels.
    Missing class-SNR combinations receive ``NaN`` accuracy rather than a
    misleading zero.

    Args:
        labels: Ground-truth integer class labels.
        predictions: Predicted integer class labels.
        snr_db: SNR value associated with every example.
        num_classes: Total number of classification classes.

    Returns:
        ClassSNRAnalysis containing accuracy, counts, and errors.

    Raises:
        ValueError: If arrays, labels, predictions, or class count are invalid.
    """
    validated_num_classes = _validate_num_classes(num_classes)

    label_array = np.asarray(labels, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    snr_array = np.asarray(snr_db, dtype=np.float32)

    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional.")

    if prediction_array.ndim != 1:
        raise ValueError("predictions must be one-dimensional.")

    if snr_array.ndim != 1:
        raise ValueError("snr_db must be one-dimensional.")

    if label_array.size == 0:
        raise ValueError("Analysis arrays must not be empty.")

    if not (
        label_array.shape
        == prediction_array.shape
        == snr_array.shape
    ):
        raise ValueError(
            "labels, predictions, and snr_db must have matching shapes."
        )

    if not np.all(np.isfinite(snr_array)):
        raise ValueError("snr_db must contain only finite values.")

    if np.any(label_array < 0) or np.any(
        label_array >= validated_num_classes
    ):
        raise ValueError("labels contain an out-of-range class index.")

    if np.any(prediction_array < 0) or np.any(
        prediction_array >= validated_num_classes
    ):
        raise ValueError("predictions contain an out-of-range class index.")

    snr_values = np.unique(snr_array).astype(np.float32)
    shape = (
        validated_num_classes,
        len(snr_values),
    )

    accuracy = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )
    counts = np.zeros(
        shape,
        dtype=np.int64,
    )
    error_counts = np.zeros(
        shape,
        dtype=np.int64,
    )

    for class_index in range(validated_num_classes):
        for snr_index, snr_value in enumerate(snr_values):
            matching = (
                (label_array == class_index)
                & np.isclose(snr_array, snr_value)
            )

            group_count = int(np.count_nonzero(matching))
            counts[class_index, snr_index] = group_count

            if group_count == 0:
                continue

            correct_count = int(
                np.count_nonzero(
                    prediction_array[matching]
                    == label_array[matching]
                )
            )

            accuracy[class_index, snr_index] = (
                correct_count / group_count
            )
            error_counts[class_index, snr_index] = (
                group_count - correct_count
            )

    return ClassSNRAnalysis(
        snr_values_db=snr_values,
        accuracy=accuracy,
        counts=counts,
        error_counts=error_counts,
    )


__all__ = [
    "ClassSNRAnalysis",
    "compute_class_snr_analysis",
]
