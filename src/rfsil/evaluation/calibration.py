from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Statistics for one confidence interval."""

    index: int
    lower_bound: float
    upper_bound: float
    example_count: int
    accuracy: float | None
    mean_confidence: float | None
    absolute_gap: float | None


@dataclass(frozen=True, slots=True)
class CalibrationEvaluation:
    """Confidence-calibration metrics for one prediction set."""

    example_count: int
    class_count: int
    bin_count: int
    accuracy: float
    mean_confidence: float
    expected_calibration_error: float
    maximum_calibration_error: float
    negative_log_likelihood: float
    brier_score: float
    bins: tuple[CalibrationBin, ...]


def _validate_bin_count(
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            "bin_count must be a positive integer."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            "bin_count must be a positive integer."
        )

    return validated


def _validate_labels(
    labels: object,
    *,
    example_count: int,
    class_count: int,
) -> Int64Array:
    raw = np.asarray(labels)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "labels must contain integers."
        )

    validated = np.asarray(
        raw,
        dtype=np.int64,
    )

    if validated.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if validated.shape[0] != example_count:
        raise ValueError(
            "labels and probabilities must "
            "contain the same number of examples."
        )

    if np.any(validated < 0) or np.any(
        validated >= class_count
    ):
        raise ValueError(
            "labels contain an out-of-range "
            "class index."
        )

    return validated


def _validate_probabilities(
    probabilities: object,
) -> Float64Array:
    raw = np.asarray(probabilities)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "probabilities must contain "
            "real numeric values."
        )

    validated = np.asarray(
        raw,
        dtype=np.float64,
    )

    if validated.ndim != 2:
        raise ValueError(
            "probabilities must have shape "
            "[examples, classes]."
        )

    if validated.shape[0] <= 0:
        raise ValueError(
            "probabilities must not be empty."
        )

    if validated.shape[1] < 2:
        raise ValueError(
            "probabilities must contain at least "
            "two classes."
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            "probabilities must contain only "
            "finite values."
        )

    if np.any(validated < 0.0) or np.any(
        validated > 1.0
    ):
        raise ValueError(
            "probabilities must be between "
            "zero and one."
        )

    row_sums = validated.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "Every probability row must sum "
            "to one."
        )

    return validated


def probabilities_from_logits(
    logits: object,
) -> Float64Array:
    """Convert finite class logits to probabilities."""
    raw = np.asarray(logits)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "logits must contain real numeric values."
        )

    values = np.asarray(
        raw,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[examples, classes]."
        )

    if values.shape[0] <= 0:
        raise ValueError(
            "logits must not be empty."
        )

    if values.shape[1] < 2:
        raise ValueError(
            "logits must contain at least "
            "two classes."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "logits must contain only "
            "finite values."
        )

    shifted = values - np.max(
        values,
        axis=1,
        keepdims=True,
    )
    exponentials = np.exp(shifted)

    return exponentials / exponentials.sum(
        axis=1,
        keepdims=True,
    )


def evaluate_calibration(
    labels: object,
    probabilities: object,
    *,
    bin_count: int = 15,
) -> CalibrationEvaluation:
    """Compute top-label calibration metrics."""
    validated_probabilities = (
        _validate_probabilities(
            probabilities
        )
    )
    validated_bin_count = (
        _validate_bin_count(bin_count)
    )

    example_count = int(
        validated_probabilities.shape[0]
    )
    class_count = int(
        validated_probabilities.shape[1]
    )
    validated_labels = _validate_labels(
        labels,
        example_count=example_count,
        class_count=class_count,
    )

    predicted_indices = np.argmax(
        validated_probabilities,
        axis=1,
    )
    confidences = np.max(
        validated_probabilities,
        axis=1,
    )
    correctness = (
        predicted_indices == validated_labels
    )

    accuracy = float(
        np.mean(correctness)
    )
    mean_confidence = float(
        np.mean(confidences)
    )

    boundaries = np.linspace(
        0.0,
        1.0,
        validated_bin_count + 1,
        dtype=np.float64,
    )
    bin_indices = np.searchsorted(
        boundaries[1:-1],
        confidences,
        side="right",
    )

    bins: list[CalibrationBin] = []
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0

    for index in range(
        validated_bin_count
    ):
        matching = bin_indices == index
        count = int(
            np.count_nonzero(matching)
        )

        if count == 0:
            bins.append(
                CalibrationBin(
                    index=index,
                    lower_bound=float(
                        boundaries[index]
                    ),
                    upper_bound=float(
                        boundaries[index + 1]
                    ),
                    example_count=0,
                    accuracy=None,
                    mean_confidence=None,
                    absolute_gap=None,
                )
            )
            continue

        bin_accuracy = float(
            np.mean(correctness[matching])
        )
        bin_confidence = float(
            np.mean(confidences[matching])
        )
        gap = abs(
            bin_accuracy - bin_confidence
        )

        expected_calibration_error += (
            count / example_count
        ) * gap
        maximum_calibration_error = max(
            maximum_calibration_error,
            gap,
        )

        bins.append(
            CalibrationBin(
                index=index,
                lower_bound=float(
                    boundaries[index]
                ),
                upper_bound=float(
                    boundaries[index + 1]
                ),
                example_count=count,
                accuracy=bin_accuracy,
                mean_confidence=bin_confidence,
                absolute_gap=gap,
            )
        )

    row_indices = np.arange(
        example_count
    )
    true_probabilities = (
        validated_probabilities[
            row_indices,
            validated_labels,
        ]
    )
    safe_true_probabilities = np.clip(
        true_probabilities,
        np.finfo(np.float64).tiny,
        1.0,
    )
    negative_log_likelihood = float(
        -np.mean(
            np.log(
                safe_true_probabilities
            )
        )
    )

    targets = np.zeros_like(
        validated_probabilities
    )
    targets[
        row_indices,
        validated_labels,
    ] = 1.0

    brier_score = float(
        np.mean(
            np.sum(
                (
                    validated_probabilities
                    - targets
                )
                ** 2,
                axis=1,
            )
        )
    )

    return CalibrationEvaluation(
        example_count=example_count,
        class_count=class_count,
        bin_count=validated_bin_count,
        accuracy=accuracy,
        mean_confidence=mean_confidence,
        expected_calibration_error=float(
            expected_calibration_error
        ),
        maximum_calibration_error=float(
            maximum_calibration_error
        ),
        negative_log_likelihood=(
            negative_log_likelihood
        ),
        brier_score=brier_score,
        bins=tuple(bins),
    )


__all__ = [
    "CalibrationBin",
    "CalibrationEvaluation",
    "evaluate_calibration",
    "probabilities_from_logits",
]
