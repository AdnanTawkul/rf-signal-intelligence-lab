from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.channel_shift import (
    ShiftDetectionMetrics,
    evaluate_shift_detection,
)
from rfsil.evaluation.paired_shift_split import (
    PairedShiftSplit,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]


@dataclass(frozen=True, slots=True)
class StandardizedLinearShiftDetector:
    """Standardized L2-regularized linear detector."""

    feature_names: tuple[str, ...]
    feature_mean: Float64Array
    feature_scale: Float64Array
    coefficients: Float64Array
    intercept: float
    l2_strength: float

    @property
    def feature_count(self) -> int:
        """Return the detector feature count."""
        return len(self.feature_names)

    def decision_function(
        self,
        values: object,
    ) -> Float64Array:
        """Compute shift scores for feature rows."""
        matrix = _validate_matrix(
            values,
            name="values",
        )

        if matrix.shape[1] != self.feature_count:
            raise ValueError(
                "values feature count does not "
                "match the detector."
            )

        standardized = (
            matrix - self.feature_mean
        ) / self.feature_scale

        scores = (
            standardized @ self.coefficients
            + self.intercept
        )

        return np.ascontiguousarray(
            scores,
            dtype=np.float64,
        )

    def flipped(
        self,
    ) -> StandardizedLinearShiftDetector:
        """Return a detector with reversed score direction."""
        return StandardizedLinearShiftDetector(
            feature_names=self.feature_names,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            coefficients=-self.coefficients,
            intercept=-self.intercept,
            l2_strength=self.l2_strength,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible model parameters."""
        return {
            "feature_names": list(
                self.feature_names
            ),
            "feature_count": self.feature_count,
            "feature_mean": (
                self.feature_mean.tolist()
            ),
            "feature_scale": (
                self.feature_scale.tolist()
            ),
            "coefficients": (
                self.coefficients.tolist()
            ),
            "intercept": self.intercept,
            "l2_strength": self.l2_strength,
        }


@dataclass(frozen=True, slots=True)
class LinearDetectorCandidate:
    """Cross-validation result for one L2 strength."""

    l2_strength: float
    fold_count: int
    mean_auroc: float
    mean_average_precision: float
    mean_fpr_at_target_tpr: float
    minimum_auroc: float
    fold_metrics: tuple[
        ShiftDetectionMetrics,
        ...,
    ]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible candidate data."""
        return {
            "l2_strength": self.l2_strength,
            "fold_count": self.fold_count,
            "mean_auroc": self.mean_auroc,
            "mean_average_precision": (
                self.mean_average_precision
            ),
            "mean_fpr_at_target_tpr": (
                self.mean_fpr_at_target_tpr
            ),
            "minimum_auroc": self.minimum_auroc,
            "fold_metrics": [
                metrics.to_dict()
                for metrics in self.fold_metrics
            ],
        }


@dataclass(frozen=True, slots=True)
class LinearShiftConditionResult:
    """Frozen detector result for one condition."""

    condition: str
    metrics: ShiftDetectionMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible condition data."""
        return {
            "condition": self.condition,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class LinearShiftDetectorAnalysis:
    """Development-selected linear detector analysis."""

    detector: StandardizedLinearShiftDetector
    split: PairedShiftSplit
    selected_l2_strength: float
    target_tpr: float
    development_metrics: ShiftDetectionMetrics
    candidates: tuple[
        LinearDetectorCandidate,
        ...,
    ]
    condition_results: tuple[
        LinearShiftConditionResult,
        ...,
    ]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible analysis data."""
        return {
            "detector": self.detector.to_dict(),
            "split": self.split.summary(),
            "selected_l2_strength": (
                self.selected_l2_strength
            ),
            "target_tpr": self.target_tpr,
            "development_metrics": (
                self.development_metrics.to_dict()
            ),
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
            "condition_results": [
                result.to_dict()
                for result
                in self.condition_results
            ],
        }


def _validate_matrix(
    value: object,
    *,
    name: str,
) -> Float64Array:
    raw = np.asarray(value)

    if (
        raw.ndim != 2
        or raw.shape[0] <= 0
        or raw.shape[1] <= 0
    ):
        raise ValueError(
            f"{name} must have shape "
            "[examples, features]."
        )

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

    matrix = np.asarray(
        raw,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    return np.ascontiguousarray(matrix)


def _validate_feature_names(
    value: object,
    *,
    feature_count: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "feature_names must be a sequence."
        )

    try:
        raw_names = tuple(value)
    except TypeError as error:
        raise ValueError(
            "feature_names must be a sequence."
        ) from error

    if len(raw_names) != feature_count:
        raise ValueError(
            "feature_names length must match "
            "the feature count."
        )

    names = []

    for raw_name in raw_names:
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
        ):
            raise ValueError(
                "Every feature name must be "
                "a non-empty string."
            )

        names.append(raw_name.strip())

    if len(set(names)) != len(names):
        raise ValueError(
            "feature_names must be unique."
        )

    return tuple(names)


def _validate_shifted_values(
    value: object,
    *,
    expected_shape: tuple[int, int],
) -> dict[str, Float64Array]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "shifted_values must be a mapping."
        )

    if not value:
        raise ValueError(
            "shifted_values must not be empty."
        )

    output = {}

    for raw_condition, raw_matrix in (
        value.items()
    ):
        if (
            not isinstance(raw_condition, str)
            or not raw_condition.strip()
        ):
            raise ValueError(
                "Condition names must be "
                "non-empty strings."
            )

        condition = raw_condition.strip()
        matrix = _validate_matrix(
            raw_matrix,
            name=(
                f"shifted_values[{condition!r}]"
            ),
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "Every shifted matrix must "
                "match the clean shape."
            )

        if condition in output:
            raise ValueError(
                "Condition names must be unique."
            )

        output[condition] = matrix

    return {
        condition: output[condition]
        for condition in sorted(output)
    }


def _validate_example_seed(
    value: object,
    *,
    example_count: int,
) -> UInt64Array:
    raw = np.asarray(value)

    if raw.shape != (example_count,):
        raise ValueError(
            "example_seed must match the "
            "example count."
        )

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "example_seed must use an "
            "integer dtype."
        )

    if (
        np.issubdtype(
            raw.dtype,
            np.signedinteger,
        )
        and np.any(raw < 0)
    ):
        raise ValueError(
            "example_seed must not contain "
            "negative values."
        )

    seeds = np.asarray(
        raw,
        dtype=np.uint64,
    )

    if np.unique(seeds).size != seeds.size:
        raise ValueError(
            "example_seed values must be unique."
        )

    return np.ascontiguousarray(seeds)


def _validate_split(
    value: object,
    *,
    example_count: int,
) -> PairedShiftSplit:
    if not isinstance(
        value,
        PairedShiftSplit,
    ):
        raise ValueError(
            "split must be a PairedShiftSplit."
        )

    development = np.asarray(
        value.development_indices,
        dtype=np.int64,
    )
    test = np.asarray(
        value.test_indices,
        dtype=np.int64,
    )

    combined = np.sort(
        np.concatenate(
            (
                development,
                test,
            )
        )
    )

    if not np.array_equal(
        combined,
        np.arange(
            example_count,
            dtype=np.int64,
        ),
    ):
        raise ValueError(
            "Split indices must cover every "
            "example exactly once."
        )

    if np.intersect1d(
        development,
        test,
    ).size:
        raise ValueError(
            "Development and test indices "
            "must be disjoint."
        )

    return value


def _validate_positive_float(
    value: object,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be positive "
            "and finite."
        )

    number = float(value)

    if (
        not math.isfinite(number)
        or number <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive "
            "and finite."
        )

    return number


def _validate_positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            f"{name} must be a positive "
            "integer."
        )

    return int(value)


def _group_hash(
    group: int,
    *,
    cv_seed: int,
) -> int:
    payload = (
        f"{cv_seed}:{group}"
    ).encode("ascii")

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"rfsil-cv",
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )


def create_grouped_fold_assignments(
    groups: object,
    *,
    fold_count: object = 5,
    cv_seed: object = 2026,
) -> Int64Array:
    """Assign identical groups to identical CV folds."""
    raw = np.asarray(groups)

    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(
            "groups must be a non-empty "
            "one-dimensional integer array."
        )

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "groups must use an integer dtype."
        )

    validated_fold_count = (
        _validate_positive_integer(
            fold_count,
            name="fold_count",
        )
    )
    validated_cv_seed = (
        _validate_positive_integer(
            cv_seed,
            name="cv_seed",
        )
    )

    converted = np.asarray(
        raw,
        dtype=np.uint64,
    )
    unique_groups = np.unique(converted)

    if unique_groups.size < (
        validated_fold_count
    ):
        raise ValueError(
            "fold_count must not exceed the "
            "number of unique groups."
        )

    ordered_groups = sorted(
        (
            int(group)
            for group in unique_groups
        ),
        key=lambda group: (
            _group_hash(
                group,
                cv_seed=validated_cv_seed,
            ),
            group,
        ),
    )

    group_to_fold = {
        group: index % validated_fold_count
        for index, group
        in enumerate(ordered_groups)
    }

    assignments = np.asarray(
        [
            group_to_fold[int(group)]
            for group in converted
        ],
        dtype=np.int64,
    )

    return np.ascontiguousarray(assignments)


def _balanced_weights(
    labels: Int64Array,
) -> Float64Array:
    clean_mask = labels == -1
    shifted_mask = labels == 1

    clean_count = int(
        np.count_nonzero(clean_mask)
    )
    shifted_count = int(
        np.count_nonzero(shifted_mask)
    )

    if clean_count <= 0 or shifted_count <= 0:
        raise ValueError(
            "Training labels must contain both "
            "clean and shifted examples."
        )

    weights = np.empty(
        labels.size,
        dtype=np.float64,
    )
    weights[clean_mask] = (
        0.5 / clean_count
    )
    weights[shifted_mask] = (
        0.5 / shifted_count
    )

    return weights


def _fit_detector(
    values: Float64Array,
    labels: Int64Array,
    feature_names: tuple[str, ...],
    *,
    l2_strength: float,
    epsilon: float = 1e-12,
) -> StandardizedLinearShiftDetector:
    weights = _balanced_weights(labels)
    weight_sum = float(np.sum(weights))

    feature_mean = np.sum(
        values * weights[:, np.newaxis],
        axis=0,
    ) / weight_sum

    centered = (
        values - feature_mean
    )
    variance = np.sum(
        centered**2
        * weights[:, np.newaxis],
        axis=0,
    ) / weight_sum
    feature_scale = np.sqrt(
        np.maximum(
            variance,
            epsilon,
        )
    )

    standardized = (
        centered / feature_scale
    )
    design = np.column_stack(
        (
            standardized,
            np.ones(
                standardized.shape[0],
                dtype=np.float64,
            ),
        )
    )

    square_root_weights = np.sqrt(
        weights
    )
    weighted_design = (
        design
        * square_root_weights[
            :,
            np.newaxis,
        ]
    )
    weighted_targets = (
        labels.astype(np.float64)
        * square_root_weights
    )

    regularization = np.zeros(
        (
            design.shape[1],
            design.shape[1],
        ),
        dtype=np.float64,
    )
    regularization[
        :-1,
        :-1,
    ] = (
        np.eye(
            standardized.shape[1],
            dtype=np.float64,
        )
        * l2_strength
    )

    system = (
        weighted_design.T
        @ weighted_design
        + regularization
    )
    right_hand_side = (
        weighted_design.T
        @ weighted_targets
    )

    solution = np.linalg.solve(
        system,
        right_hand_side,
    )

    return StandardizedLinearShiftDetector(
        feature_names=feature_names,
        feature_mean=np.ascontiguousarray(
            feature_mean
        ),
        feature_scale=np.ascontiguousarray(
            feature_scale
        ),
        coefficients=np.ascontiguousarray(
            solution[:-1]
        ),
        intercept=float(solution[-1]),
        l2_strength=l2_strength,
    )


def _stack_rows(
    clean: Float64Array,
    shifted: Mapping[str, Float64Array],
    indices: Int64Array,
    example_seed: UInt64Array,
) -> tuple[
    Float64Array,
    Int64Array,
    UInt64Array,
]:
    matrices = [
        clean[indices],
    ]
    labels = [
        np.full(
            indices.size,
            -1,
            dtype=np.int64,
        ),
    ]
    groups = [
        example_seed[indices],
    ]

    for matrix in shifted.values():
        matrices.append(
            matrix[indices]
        )
        labels.append(
            np.full(
                indices.size,
                1,
                dtype=np.int64,
            )
        )
        groups.append(
            example_seed[indices]
        )

    return (
        np.ascontiguousarray(
            np.concatenate(
                matrices,
                axis=0,
            )
        ),
        np.ascontiguousarray(
            np.concatenate(labels)
        ),
        np.ascontiguousarray(
            np.concatenate(groups)
        ),
    )


def analyze_linear_iq_shift_detection(
    clean_values: object,
    shifted_values: object,
    feature_names: object,
    example_seed: object,
    split: object,
    *,
    l2_strengths: Sequence[object],
    fold_count: object = 5,
    cv_seed: object = 2026,
    target_tpr: object = 0.95,
) -> LinearShiftDetectorAnalysis:
    """Select and evaluate a linear IQ shift detector."""
    clean = _validate_matrix(
        clean_values,
        name="clean_values",
    )
    names = _validate_feature_names(
        feature_names,
        feature_count=int(
            clean.shape[1]
        ),
    )
    shifted = _validate_shifted_values(
        shifted_values,
        expected_shape=clean.shape,
    )
    seeds = _validate_example_seed(
        example_seed,
        example_count=int(
            clean.shape[0]
        ),
    )
    validated_split = _validate_split(
        split,
        example_count=int(
            clean.shape[0]
        ),
    )
    validated_fold_count = (
        _validate_positive_integer(
            fold_count,
            name="fold_count",
        )
    )
    validated_cv_seed = (
        _validate_positive_integer(
            cv_seed,
            name="cv_seed",
        )
    )
    validated_target_tpr = (
        _validate_positive_float(
            target_tpr,
            name="target_tpr",
        )
    )

    if validated_target_tpr > 1.0:
        raise ValueError(
            "target_tpr must not exceed 1."
        )

    strengths = tuple(
        _validate_positive_float(
            strength,
            name="l2 strength",
        )
        for strength in l2_strengths
    )

    if not strengths:
        raise ValueError(
            "l2_strengths must not be empty."
        )

    if len(set(strengths)) != len(
        strengths
    ):
        raise ValueError(
            "l2_strengths must be unique."
        )

    development_values, development_labels, (
        development_groups
    ) = _stack_rows(
        clean,
        shifted,
        validated_split.development_indices,
        seeds,
    )

    fold_assignments = (
        create_grouped_fold_assignments(
            development_groups,
            fold_count=(
                validated_fold_count
            ),
            cv_seed=validated_cv_seed,
        )
    )

    candidates = []

    for l2_strength in strengths:
        fold_metrics = []

        for fold in range(
            validated_fold_count
        ):
            validation_mask = (
                fold_assignments == fold
            )
            training_mask = (
                ~validation_mask
            )

            detector = _fit_detector(
                development_values[
                    training_mask
                ],
                development_labels[
                    training_mask
                ],
                names,
                l2_strength=l2_strength,
            )
            validation_scores = (
                detector.decision_function(
                    development_values[
                        validation_mask
                    ]
                )
            )
            validation_labels = (
                development_labels[
                    validation_mask
                ]
            )

            fold_metrics.append(
                evaluate_shift_detection(
                    validation_scores[
                        validation_labels == -1
                    ],
                    validation_scores[
                        validation_labels == 1
                    ],
                    target_tpr=(
                        validated_target_tpr
                    ),
                )
            )

        auroc_values = np.asarray(
            [
                metrics.auroc
                for metrics in fold_metrics
            ],
            dtype=np.float64,
        )
        average_precision_values = (
            np.asarray(
                [
                    metrics.average_precision
                    for metrics in fold_metrics
                ],
                dtype=np.float64,
            )
        )
        fpr_values = np.asarray(
            [
                metrics.fpr_at_target_tpr
                for metrics in fold_metrics
            ],
            dtype=np.float64,
        )

        candidates.append(
            LinearDetectorCandidate(
                l2_strength=l2_strength,
                fold_count=(
                    validated_fold_count
                ),
                mean_auroc=float(
                    np.mean(auroc_values)
                ),
                mean_average_precision=(
                    float(
                        np.mean(
                            average_precision_values
                        )
                    )
                ),
                mean_fpr_at_target_tpr=(
                    float(np.mean(fpr_values))
                ),
                minimum_auroc=float(
                    np.min(auroc_values)
                ),
                fold_metrics=tuple(
                    fold_metrics
                ),
            )
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate.mean_auroc,
            candidate.mean_fpr_at_target_tpr,
            -candidate.minimum_auroc,
            candidate.l2_strength,
        )
    )
    selected = candidates[0]

    detector = _fit_detector(
        development_values,
        development_labels,
        names,
        l2_strength=(
            selected.l2_strength
        ),
    )

    development_scores = (
        detector.decision_function(
            development_values
        )
    )
    development_metrics = (
        evaluate_shift_detection(
            development_scores[
                development_labels == -1
            ],
            development_scores[
                development_labels == 1
            ],
            target_tpr=(
                validated_target_tpr
            ),
        )
    )

    if development_metrics.auroc < 0.5:
        detector = detector.flipped()
        development_scores = (
            detector.decision_function(
                development_values
            )
        )
        development_metrics = (
            evaluate_shift_detection(
                development_scores[
                    development_labels == -1
                ],
                development_scores[
                    development_labels == 1
                ],
                target_tpr=(
                    validated_target_tpr
                ),
            )
        )

    test_indices = (
        validated_split.test_indices
    )
    clean_test_scores = (
        detector.decision_function(
            clean[test_indices]
        )
    )

    condition_results = []

    for condition, matrix in shifted.items():
        shifted_scores = (
            detector.decision_function(
                matrix[test_indices]
            )
        )

        condition_results.append(
            LinearShiftConditionResult(
                condition=condition,
                metrics=(
                    evaluate_shift_detection(
                        clean_test_scores,
                        shifted_scores,
                        target_tpr=(
                            validated_target_tpr
                        ),
                    )
                ),
            )
        )

    return LinearShiftDetectorAnalysis(
        detector=detector,
        split=validated_split,
        selected_l2_strength=(
            selected.l2_strength
        ),
        target_tpr=validated_target_tpr,
        development_metrics=(
            development_metrics
        ),
        candidates=tuple(candidates),
        condition_results=tuple(
            condition_results
        ),
    )


__all__ = [
    "LinearDetectorCandidate",
    "LinearShiftConditionResult",
    "LinearShiftDetectorAnalysis",
    "StandardizedLinearShiftDetector",
    "analyze_linear_iq_shift_detection",
    "create_grouped_fold_assignments",
]
