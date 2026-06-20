from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.calibration import (
    probabilities_from_logits,
)

Float64Array = NDArray[np.float64]

SHIFT_SCORE_NAMES = (
    "msp_uncertainty",
    "predictive_entropy",
    "negative_logit_margin",
    "energy",
)


@dataclass(frozen=True, slots=True)
class ShiftDetectionMetrics:
    """Binary clean-versus-shift detection metrics."""

    auroc: float
    average_precision: float
    fpr_at_target_tpr: float
    target_tpr: float
    clean_count: int
    shifted_count: int
    clean_mean: float
    shifted_mean: float
    clean_std: float
    shifted_std: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to JSON-compatible data."""
        return asdict(self)


def _validate_logits(
    value: object,
) -> Float64Array:
    raw = np.asarray(value)

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

    logits = np.asarray(
        raw,
        dtype=np.float64,
    )

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[examples, classes]."
        )

    if logits.shape[0] <= 0:
        raise ValueError(
            "logits must not be empty."
        )

    if logits.shape[1] < 2:
        raise ValueError(
            "logits must contain at least "
            "two classes."
        )

    if not np.all(np.isfinite(logits)):
        raise ValueError(
            "logits must contain only "
            "finite values."
        )

    return np.ascontiguousarray(logits)


def _validate_probabilities(
    value: object,
    *,
    example_count: int,
    class_count: int,
) -> Float64Array:
    raw = np.asarray(value)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "probabilities must contain real "
            "numeric values."
        )

    probabilities = np.asarray(
        raw,
        dtype=np.float64,
    )

    if probabilities.shape != (
        example_count,
        class_count,
    ):
        raise ValueError(
            "probabilities must match the "
            "logit shape."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(
            "probabilities must contain only "
            "finite values."
        )

    if np.any(probabilities < 0.0):
        raise ValueError(
            "probabilities must be non-negative."
        )

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(
            "probability rows must sum to one."
        )

    return np.ascontiguousarray(
        probabilities
    )


def _validate_score_vector(
    value: object,
    *,
    name: str,
) -> Float64Array:
    raw = np.asarray(value)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            f"{name} must contain real "
            "numeric values."
        )

    scores = np.asarray(
        raw,
        dtype=np.float64,
    )

    if scores.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if scores.size <= 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    return np.ascontiguousarray(scores)


def _validate_target_tpr(
    value: object,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            "target_tpr must be a finite "
            "number in (0, 1]."
        )

    target = float(value)

    if (
        not math.isfinite(target)
        or target <= 0.0
        or target > 1.0
    ):
        raise ValueError(
            "target_tpr must be a finite "
            "number in (0, 1]."
        )

    return target


def compute_shift_scores(
    logits: object,
    *,
    probabilities: object | None = None,
) -> dict[str, Float64Array]:
    """Compute output-only channel-shift scores.

    Every returned score uses the same direction:
    a larger value is intended to indicate a more
    shift-like or uncertain example.
    """
    validated_logits = _validate_logits(
        logits
    )

    if probabilities is None:
        validated_probabilities = np.asarray(
            probabilities_from_logits(
                validated_logits
            ),
            dtype=np.float64,
        )
    else:
        validated_probabilities = (
            _validate_probabilities(
                probabilities,
                example_count=int(
                    validated_logits.shape[0]
                ),
                class_count=int(
                    validated_logits.shape[1]
                ),
            )
        )

    maximum_probability = np.max(
        validated_probabilities,
        axis=1,
    )

    predictive_entropy = -np.sum(
        validated_probabilities
        * np.log(
            np.clip(
                validated_probabilities,
                1e-12,
                1.0,
            )
        ),
        axis=1,
    )

    sorted_logits = np.sort(
        validated_logits,
        axis=1,
    )
    logit_margin = (
        sorted_logits[:, -1]
        - sorted_logits[:, -2]
    )

    maximum_logit = np.max(
        validated_logits,
        axis=1,
        keepdims=True,
    )
    log_sum_exp = (
        maximum_logit[:, 0]
        + np.log(
            np.sum(
                np.exp(
                    validated_logits
                    - maximum_logit
                ),
                axis=1,
            )
        )
    )

    return {
        "msp_uncertainty": np.ascontiguousarray(
            1.0 - maximum_probability
        ),
        "predictive_entropy": (
            np.ascontiguousarray(
                predictive_entropy
            )
        ),
        "negative_logit_margin": (
            np.ascontiguousarray(
                -logit_margin
            )
        ),
        "energy": np.ascontiguousarray(
            -log_sum_exp
        ),
    }


def _average_ranks(
    values: Float64Array,
) -> Float64Array:
    order = np.argsort(
        values,
        kind="stable",
    )
    sorted_values = values[order]
    ranks = np.empty(
        values.size,
        dtype=np.float64,
    )

    start = 0

    while start < values.size:
        end = start + 1

        while (
            end < values.size
            and sorted_values[end]
            == sorted_values[start]
        ):
            end += 1

        average_rank = 0.5 * (
            (start + 1) + end
        )
        ranks[order[start:end]] = (
            average_rank
        )
        start = end

    return ranks


def _auroc(
    labels: NDArray[np.bool_],
    scores: Float64Array,
) -> float:
    positive_count = int(
        np.count_nonzero(labels)
    )
    negative_count = int(
        labels.size - positive_count
    )

    ranks = _average_ranks(scores)
    positive_rank_sum = float(
        np.sum(ranks[labels])
    )
    mann_whitney = (
        positive_rank_sum
        - positive_count
        * (positive_count + 1)
        / 2.0
    )

    return float(
        mann_whitney
        / (
            positive_count
            * negative_count
        )
    )


def _threshold_statistics(
    labels: NDArray[np.bool_],
    scores: Float64Array,
) -> tuple[
    Float64Array,
    Float64Array,
    Float64Array,
]:
    order = np.argsort(
        -scores,
        kind="stable",
    )
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    true_positives = np.cumsum(
        sorted_labels,
        dtype=np.int64,
    )
    false_positives = (
        np.arange(
            1,
            labels.size + 1,
            dtype=np.int64,
        )
        - true_positives
    )

    distinct_ends = np.concatenate(
        (
            np.flatnonzero(
                np.diff(sorted_scores) != 0.0
            ),
            np.asarray(
                [labels.size - 1],
                dtype=np.int64,
            ),
        )
    )

    positive_count = int(
        true_positives[-1]
    )
    negative_count = int(
        labels.size - positive_count
    )

    recall = (
        true_positives[distinct_ends]
        / positive_count
    )
    precision = (
        true_positives[distinct_ends]
        / (
            true_positives[distinct_ends]
            + false_positives[distinct_ends]
        )
    )
    false_positive_rate = (
        false_positives[distinct_ends]
        / negative_count
    )

    return (
        recall.astype(
            np.float64,
            copy=False,
        ),
        precision.astype(
            np.float64,
            copy=False,
        ),
        false_positive_rate.astype(
            np.float64,
            copy=False,
        ),
    )


def _average_precision(
    labels: NDArray[np.bool_],
    scores: Float64Array,
) -> float:
    recall, precision, _ = (
        _threshold_statistics(
            labels,
            scores,
        )
    )
    recall_change = np.diff(
        np.concatenate(
            (
                np.asarray(
                    [0.0],
                    dtype=np.float64,
                ),
                recall,
            )
        )
    )

    return float(
        np.sum(
            recall_change
            * precision
        )
    )


def _fpr_at_target_tpr(
    labels: NDArray[np.bool_],
    scores: Float64Array,
    *,
    target_tpr: float,
) -> float:
    recall, _, false_positive_rate = (
        _threshold_statistics(
            labels,
            scores,
        )
    )

    matching = np.flatnonzero(
        recall >= target_tpr
    )

    if matching.size == 0:
        return 1.0

    return float(
        false_positive_rate[
            matching[0]
        ]
    )


def evaluate_shift_detection(
    clean_scores: object,
    shifted_scores: object,
    *,
    target_tpr: object = 0.95,
) -> ShiftDetectionMetrics:
    """Evaluate one clean-versus-shift score."""
    clean = _validate_score_vector(
        clean_scores,
        name="clean_scores",
    )
    shifted = _validate_score_vector(
        shifted_scores,
        name="shifted_scores",
    )
    validated_target_tpr = (
        _validate_target_tpr(target_tpr)
    )

    labels = np.concatenate(
        (
            np.zeros(
                clean.size,
                dtype=np.bool_,
            ),
            np.ones(
                shifted.size,
                dtype=np.bool_,
            ),
        )
    )
    scores = np.concatenate(
        (
            clean,
            shifted,
        )
    )

    return ShiftDetectionMetrics(
        auroc=_auroc(
            labels,
            scores,
        ),
        average_precision=(
            _average_precision(
                labels,
                scores,
            )
        ),
        fpr_at_target_tpr=(
            _fpr_at_target_tpr(
                labels,
                scores,
                target_tpr=(
                    validated_target_tpr
                ),
            )
        ),
        target_tpr=validated_target_tpr,
        clean_count=int(clean.size),
        shifted_count=int(shifted.size),
        clean_mean=float(
            np.mean(clean)
        ),
        shifted_mean=float(
            np.mean(shifted)
        ),
        clean_std=float(
            np.std(clean)
        ),
        shifted_std=float(
            np.std(shifted)
        ),
    )


def evaluate_shift_score_sets(
    clean_scores: object,
    shifted_scores: object,
    *,
    target_tpr: object = 0.95,
) -> dict[str, ShiftDetectionMetrics]:
    """Evaluate matching mappings of shift scores."""
    if not isinstance(
        clean_scores,
        Mapping,
    ) or not isinstance(
        shifted_scores,
        Mapping,
    ):
        raise ValueError(
            "clean_scores and shifted_scores "
            "must be mappings."
        )

    clean_names = tuple(clean_scores)
    shifted_names = tuple(shifted_scores)

    if not clean_names:
        raise ValueError(
            "Score mappings must not be empty."
        )

    if set(clean_names) != set(
        shifted_names
    ):
        raise ValueError(
            "Clean and shifted score mappings "
            "must contain identical names."
        )

    return {
        str(name): evaluate_shift_detection(
            clean_scores[name],
            shifted_scores[name],
            target_tpr=target_tpr,
        )
        for name in clean_names
    }


__all__ = [
    "SHIFT_SCORE_NAMES",
    "ShiftDetectionMetrics",
    "compute_shift_scores",
    "evaluate_shift_detection",
    "evaluate_shift_score_sets",
]
