from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class IQFeatureDirectionSelection:
    """Direction selected using development data only."""

    feature_name: str
    multiplier: int
    direction: str
    raw_development_metrics: ShiftDetectionMetrics
    directed_development_metrics: (
        ShiftDetectionMetrics
    )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible selection data."""
        return {
            "feature_name": self.feature_name,
            "multiplier": self.multiplier,
            "direction": self.direction,
            "raw_development_metrics": (
                self.raw_development_metrics.to_dict()
            ),
            "directed_development_metrics": (
                self.directed_development_metrics.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class IQFeatureConditionResult:
    """One feature evaluated on one shifted condition."""

    feature_name: str
    condition: str
    multiplier: int
    direction: str
    raw_test_metrics: ShiftDetectionMetrics
    directed_test_metrics: ShiftDetectionMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible test results."""
        return {
            "feature_name": self.feature_name,
            "condition": self.condition,
            "multiplier": self.multiplier,
            "direction": self.direction,
            "raw_test_metrics": (
                self.raw_test_metrics.to_dict()
            ),
            "directed_test_metrics": (
                self.directed_test_metrics.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class IQFeatureDetectionAnalysis:
    """Complete individual-feature detection analysis."""

    feature_names: tuple[str, ...]
    conditions: tuple[str, ...]
    split: PairedShiftSplit
    target_tpr: float
    selections: tuple[
        IQFeatureDirectionSelection,
        ...,
    ]
    condition_results: tuple[
        IQFeatureConditionResult,
        ...,
    ]
    feature_summaries: tuple[
        dict[str, Any],
        ...,
    ]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible analysis data."""
        return {
            "feature_count": len(
                self.feature_names
            ),
            "condition_count": len(
                self.conditions
            ),
            "feature_names": list(
                self.feature_names
            ),
            "conditions": list(
                self.conditions
            ),
            "target_tpr": self.target_tpr,
            "direction_selection": (
                "balanced_pooled_shifted_development"
            ),
            "split": self.split.summary(),
            "selections": [
                selection.to_dict()
                for selection in self.selections
            ],
            "condition_results": [
                result.to_dict()
                for result
                in self.condition_results
            ],
            "feature_summaries": list(
                self.feature_summaries
            ),
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


def _validate_shifted_matrices(
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

    matrices = {}

    for condition, raw_matrix in value.items():
        if (
            not isinstance(condition, str)
            or not condition.strip()
        ):
            raise ValueError(
                "Every condition name must be "
                "a non-empty string."
            )

        matrix = _validate_matrix(
            raw_matrix,
            name=(
                f"shifted_values[{condition!r}]"
            ),
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "Every shifted feature matrix "
                "must match the clean shape."
            )

        matrices[condition.strip()] = matrix

    if len(matrices) != len(value):
        raise ValueError(
            "Shifted condition names must "
            "be unique after trimming."
        )

    return {
        condition: matrices[condition]
        for condition in sorted(matrices)
    }


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

    for name, indices in (
        ("development_indices", development),
        ("test_indices", test),
    ):
        if (
            indices.ndim != 1
            or indices.size == 0
        ):
            raise ValueError(
                f"{name} must be non-empty "
                "and one-dimensional."
            )

        if (
            np.any(indices < 0)
            or np.any(
                indices >= example_count
            )
        ):
            raise ValueError(
                f"{name} contains an "
                "out-of-range index."
            )

        if np.unique(indices).size != (
            indices.size
        ):
            raise ValueError(
                f"{name} contains duplicates."
            )

    if np.intersect1d(
        development,
        test,
    ).size:
        raise ValueError(
            "Development and test indices "
            "must be disjoint."
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
            "Development and test indices "
            "must cover every example."
        )

    return value


def select_iq_feature_directions(
    clean_values: object,
    shifted_values: object,
    feature_names: object,
    split: object,
    *,
    target_tpr: object = 0.95,
) -> tuple[
    IQFeatureDirectionSelection,
    ...,
]:
    """Select feature direction on pooled development data."""
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
    shifted = _validate_shifted_matrices(
        shifted_values,
        expected_shape=clean.shape,
    )
    validated_split = _validate_split(
        split,
        example_count=int(
            clean.shape[0]
        ),
    )

    development_indices = (
        validated_split.development_indices
    )
    condition_count = len(shifted)

    selections = []

    for feature_index, feature_name in (
        enumerate(names)
    ):
        clean_development = clean[
            development_indices,
            feature_index,
        ]

        pooled_shifted = np.concatenate(
            [
                matrix[
                    development_indices,
                    feature_index,
                ]
                for matrix in shifted.values()
            ]
        )

        balanced_clean = np.tile(
            clean_development,
            condition_count,
        )

        raw_metrics = (
            evaluate_shift_detection(
                balanced_clean,
                pooled_shifted,
                target_tpr=target_tpr,
            )
        )

        multiplier = (
            1
            if raw_metrics.auroc >= 0.5
            else -1
        )
        direction = (
            "larger_is_shift_like"
            if multiplier == 1
            else "smaller_is_shift_like"
        )

        directed_metrics = (
            evaluate_shift_detection(
                multiplier * balanced_clean,
                multiplier * pooled_shifted,
                target_tpr=target_tpr,
            )
        )

        selections.append(
            IQFeatureDirectionSelection(
                feature_name=feature_name,
                multiplier=multiplier,
                direction=direction,
                raw_development_metrics=(
                    raw_metrics
                ),
                directed_development_metrics=(
                    directed_metrics
                ),
            )
        )

    return tuple(selections)


def evaluate_iq_feature_directions(
    clean_values: object,
    shifted_values: object,
    feature_names: object,
    split: object,
    selections: Sequence[
        IQFeatureDirectionSelection
    ],
    *,
    target_tpr: object = 0.95,
) -> tuple[
    IQFeatureConditionResult,
    ...,
]:
    """Evaluate frozen feature directions on test data."""
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
    shifted = _validate_shifted_matrices(
        shifted_values,
        expected_shape=clean.shape,
    )
    validated_split = _validate_split(
        split,
        example_count=int(
            clean.shape[0]
        ),
    )

    selection_tuple = tuple(selections)

    if len(selection_tuple) != len(names):
        raise ValueError(
            "There must be one direction "
            "selection per feature."
        )

    selection_by_name = {}

    for selection in selection_tuple:
        if not isinstance(
            selection,
            IQFeatureDirectionSelection,
        ):
            raise ValueError(
                "Every selection must be an "
                "IQFeatureDirectionSelection."
            )

        if selection.multiplier not in (
            -1,
            1,
        ):
            raise ValueError(
                "Every direction multiplier "
                "must be -1 or 1."
            )

        if (
            selection.feature_name
            in selection_by_name
        ):
            raise ValueError(
                "Feature direction selections "
                "must be unique."
            )

        selection_by_name[
            selection.feature_name
        ] = selection

    if set(selection_by_name) != set(
        names
    ):
        raise ValueError(
            "Direction selections must match "
            "the feature names."
        )

    test_indices = (
        validated_split.test_indices
    )
    results = []

    for condition, matrix in shifted.items():
        for feature_index, feature_name in (
            enumerate(names)
        ):
            selection = selection_by_name[
                feature_name
            ]
            clean_test = clean[
                test_indices,
                feature_index,
            ]
            shifted_test = matrix[
                test_indices,
                feature_index,
            ]

            raw_metrics = (
                evaluate_shift_detection(
                    clean_test,
                    shifted_test,
                    target_tpr=target_tpr,
                )
            )
            directed_metrics = (
                evaluate_shift_detection(
                    (
                        selection.multiplier
                        * clean_test
                    ),
                    (
                        selection.multiplier
                        * shifted_test
                    ),
                    target_tpr=target_tpr,
                )
            )

            results.append(
                IQFeatureConditionResult(
                    feature_name=feature_name,
                    condition=condition,
                    multiplier=(
                        selection.multiplier
                    ),
                    direction=(
                        selection.direction
                    ),
                    raw_test_metrics=(
                        raw_metrics
                    ),
                    directed_test_metrics=(
                        directed_metrics
                    ),
                )
            )

    return tuple(results)


def summarize_iq_feature_results(
    selections: Sequence[
        IQFeatureDirectionSelection
    ],
    results: Sequence[
        IQFeatureConditionResult
    ],
) -> tuple[dict[str, Any], ...]:
    """Aggregate test metrics across conditions."""
    result_tuple = tuple(results)
    summaries = []

    for selection in selections:
        feature_results = [
            result
            for result in result_tuple
            if (
                result.feature_name
                == selection.feature_name
            )
        ]

        if not feature_results:
            raise ValueError(
                "Every selected feature must "
                "have condition results."
            )

        auroc = np.asarray(
            [
                result
                .directed_test_metrics
                .auroc
                for result
                in feature_results
            ],
            dtype=np.float64,
        )
        average_precision = np.asarray(
            [
                result
                .directed_test_metrics
                .average_precision
                for result
                in feature_results
            ],
            dtype=np.float64,
        )
        fpr95 = np.asarray(
            [
                result
                .directed_test_metrics
                .fpr_at_target_tpr
                for result
                in feature_results
            ],
            dtype=np.float64,
        )

        summaries.append(
            {
                "feature_name": (
                    selection.feature_name
                ),
                "multiplier": (
                    selection.multiplier
                ),
                "direction": (
                    selection.direction
                ),
                "condition_count": len(
                    feature_results
                ),
                "development_auroc": (
                    selection
                    .directed_development_metrics
                    .auroc
                ),
                "mean_test_auroc": float(
                    np.mean(auroc)
                ),
                "minimum_test_auroc": float(
                    np.min(auroc)
                ),
                "maximum_test_auroc": float(
                    np.max(auroc)
                ),
                "mean_test_average_precision": (
                    float(
                        np.mean(
                            average_precision
                        )
                    )
                ),
                "mean_test_fpr_at_target_tpr": (
                    float(np.mean(fpr95))
                ),
                "maximum_test_fpr_at_target_tpr": (
                    float(np.max(fpr95))
                ),
            }
        )

    summaries.sort(
        key=lambda summary: (
            -summary["mean_test_auroc"],
            summary[
                "mean_test_fpr_at_target_tpr"
            ],
            summary["feature_name"],
        )
    )

    return tuple(summaries)


def analyze_iq_feature_detection(
    clean_values: object,
    shifted_values: object,
    feature_names: object,
    split: object,
    *,
    target_tpr: object = 0.95,
) -> IQFeatureDetectionAnalysis:
    """Run direction selection and held-out evaluation."""
    selections = select_iq_feature_directions(
        clean_values,
        shifted_values,
        feature_names,
        split,
        target_tpr=target_tpr,
    )
    results = evaluate_iq_feature_directions(
        clean_values,
        shifted_values,
        feature_names,
        split,
        selections,
        target_tpr=target_tpr,
    )
    summaries = summarize_iq_feature_results(
        selections,
        results,
    )

    return IQFeatureDetectionAnalysis(
        feature_names=tuple(
            selection.feature_name
            for selection in selections
        ),
        conditions=tuple(
            sorted(
                {
                    result.condition
                    for result in results
                }
            )
        ),
        split=split,
        target_tpr=float(target_tpr),
        selections=selections,
        condition_results=results,
        feature_summaries=summaries,
    )


__all__ = [
    "IQFeatureConditionResult",
    "IQFeatureDetectionAnalysis",
    "IQFeatureDirectionSelection",
    "analyze_iq_feature_detection",
    "evaluate_iq_feature_directions",
    "select_iq_feature_directions",
    "summarize_iq_feature_results",
]
