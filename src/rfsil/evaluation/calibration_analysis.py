from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

from rfsil.evaluation.calibration import (
    evaluate_calibration,
)
from rfsil.evaluation.calibration_artifacts import (
    CalibrationPredictionArtifact,
)
from rfsil.evaluation.temperature_scaling import (
    probabilities_with_temperature,
)

DEFAULT_COVERAGES = (
    1.0,
    0.95,
    0.90,
    0.80,
    0.50,
)

CALIBRATION_METRICS = (
    "accuracy",
    "mean_confidence",
    "expected_calibration_error",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "brier_score",
)

LOWER_IS_BETTER = frozenset(
    {
        "expected_calibration_error",
        "maximum_calibration_error",
        "negative_log_likelihood",
        "brier_score",
    }
)


@dataclass(frozen=True, slots=True)
class SelectiveAccuracyPoint:
    """Accuracy after retaining the most confident examples."""

    requested_coverage: float
    actual_coverage: float
    selected_count: int
    confidence_threshold: float
    accuracy: float
    risk: float
    mean_confidence: float


def _validate_coverages(
    coverages: object,
) -> tuple[float, ...]:
    if isinstance(
        coverages,
        (str, bytes),
    ):
        raise ValueError(
            "coverages must be a non-empty "
            "sequence."
        )

    try:
        raw_values = tuple(coverages)
    except TypeError as error:
        raise ValueError(
            "coverages must be a non-empty "
            "sequence."
        ) from error

    if not raw_values:
        raise ValueError(
            "coverages must be a non-empty "
            "sequence."
        )

    validated: list[float] = []

    for value in raw_values:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
        ):
            raise ValueError(
                "Every coverage must be a "
                "finite number in (0, 1]."
            )

        coverage = float(value)

        if (
            not math.isfinite(coverage)
            or coverage <= 0.0
            or coverage > 1.0
        ):
            raise ValueError(
                "Every coverage must be a "
                "finite number in (0, 1]."
            )

        validated.append(coverage)

    if len(set(validated)) != len(validated):
        raise ValueError(
            "Coverage values must be unique."
        )

    return tuple(validated)


def selective_accuracy_curve(
    labels: object,
    probabilities: object,
    *,
    coverages: object = DEFAULT_COVERAGES,
) -> tuple[SelectiveAccuracyPoint, ...]:
    """Compute top-confidence coverage and accuracy."""
    validated_coverages = (
        _validate_coverages(coverages)
    )

    evaluation = evaluate_calibration(
        labels=labels,
        probabilities=probabilities,
        bin_count=1,
    )

    label_array = np.asarray(
        labels,
        dtype=np.int64,
    )
    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = np.argmax(
        probability_array,
        axis=1,
    )
    confidences = np.max(
        probability_array,
        axis=1,
    )
    correctness = (
        predictions == label_array
    )

    order = np.argsort(
        -confidences,
        kind="stable",
    )
    example_count = evaluation.example_count
    points: list[SelectiveAccuracyPoint] = []

    for requested_coverage in (
        validated_coverages
    ):
        selected_count = min(
            example_count,
            max(
                1,
                math.ceil(
                    requested_coverage
                    * example_count
                ),
            ),
        )
        selected_indices = order[
            :selected_count
        ]
        selected_confidences = confidences[
            selected_indices
        ]
        selected_accuracy = float(
            np.mean(
                correctness[
                    selected_indices
                ]
            )
        )

        points.append(
            SelectiveAccuracyPoint(
                requested_coverage=(
                    requested_coverage
                ),
                actual_coverage=(
                    selected_count
                    / example_count
                ),
                selected_count=(
                    selected_count
                ),
                confidence_threshold=float(
                    selected_confidences[-1]
                ),
                accuracy=selected_accuracy,
                risk=1.0 - selected_accuracy,
                mean_confidence=float(
                    np.mean(
                        selected_confidences
                    )
                ),
            )
        )

    return tuple(points)


def evaluate_temperature_transfer(
    artifact: CalibrationPredictionArtifact,
    *,
    temperature: float,
    bin_count: int = 15,
    coverages: object = DEFAULT_COVERAGES,
) -> dict[str, Any]:
    """Apply one frozen temperature to an artifact."""
    baseline_probabilities = np.asarray(
        artifact.probabilities,
        dtype=np.float64,
    )
    calibrated_probabilities = (
        probabilities_with_temperature(
            artifact.logits,
            temperature,
        )
    )

    baseline_predictions = np.argmax(
        baseline_probabilities,
        axis=1,
    )
    calibrated_predictions = np.argmax(
        calibrated_probabilities,
        axis=1,
    )

    if not np.array_equal(
        baseline_predictions,
        artifact.predictions,
    ):
        raise ValueError(
            "Artifact probability argmax does "
            "not match stored predictions."
        )

    if not np.array_equal(
        calibrated_predictions,
        artifact.predictions,
    ):
        raise RuntimeError(
            "Temperature scaling changed "
            "predicted classes."
        )

    baseline = evaluate_calibration(
        labels=artifact.labels,
        probabilities=(
            baseline_probabilities
        ),
        bin_count=bin_count,
    )
    calibrated = evaluate_calibration(
        labels=artifact.labels,
        probabilities=(
            calibrated_probabilities
        ),
        bin_count=bin_count,
    )

    if baseline.accuracy != calibrated.accuracy:
        raise RuntimeError(
            "Temperature scaling changed "
            "classification accuracy."
        )

    baseline_selective = (
        selective_accuracy_curve(
            labels=artifact.labels,
            probabilities=(
                baseline_probabilities
            ),
            coverages=coverages,
        )
    )
    calibrated_selective = (
        selective_accuracy_curve(
            labels=artifact.labels,
            probabilities=(
                calibrated_probabilities
            ),
            coverages=coverages,
        )
    )

    baseline_mapping = asdict(baseline)
    calibrated_mapping = asdict(
        calibrated
    )

    metric_changes = {
        metric: (
            float(
                calibrated_mapping[metric]
            )
            - float(
                baseline_mapping[metric]
            )
        )
        for metric in CALIBRATION_METRICS
    }

    return {
        "temperature": float(temperature),
        "example_count": (
            artifact.example_count
        ),
        "class_count": artifact.class_count,
        "class_names": (
            list(artifact.class_names)
            if artifact.class_names
            is not None
            else None
        ),
        "accuracy_preserved": True,
        "baseline": baseline_mapping,
        "calibrated": calibrated_mapping,
        "metric_changes": metric_changes,
        "selective_accuracy": {
            "baseline": [
                asdict(point)
                for point in baseline_selective
            ],
            "calibrated": [
                asdict(point)
                for point in calibrated_selective
            ],
        },
    }


def _numeric_summary(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(array)),
        "std": float(
            np.std(
                array,
                ddof=1,
            )
            if array.size > 1
            else 0.0
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _comparison_counts(
    baseline: list[float],
    calibrated: list[float],
) -> dict[str, int]:
    improved = 0
    worsened = 0
    tied = 0

    for before, after in zip(
        baseline,
        calibrated,
        strict=True,
    ):
        if np.isclose(
            before,
            after,
            rtol=0.0,
            atol=1e-12,
        ):
            tied += 1
        elif after < before:
            improved += 1
        else:
            worsened += 1

    return {
        "improved": improved,
        "worsened": worsened,
        "tied": tied,
    }


def aggregate_temperature_transfer_records(
    records: object,
) -> dict[str, Any]:
    """Aggregate calibration comparisons across seeds."""
    if (
        isinstance(records, (str, bytes))
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError(
            "records must be a non-empty list."
        )

    grouped: dict[
        tuple[str, str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(
                "Every record must be a mapping."
            )

        key = (
            str(
                record[
                    "fraction_identifier"
                ]
            ),
            str(record["method"]),
            str(record["condition"]),
        )
        grouped[key].append(record)

    aggregate_groups: list[
        dict[str, Any]
    ] = []

    for (
        fraction_identifier,
        method,
        condition,
    ), group_records in sorted(
        grouped.items()
    ):
        metric_summaries: dict[
            str,
            Any,
        ] = {}

        for metric in CALIBRATION_METRICS:
            baseline_values = [
                float(
                    record["baseline"][
                        metric
                    ]
                )
                for record in group_records
            ]
            calibrated_values = [
                float(
                    record["calibrated"][
                        metric
                    ]
                )
                for record in group_records
            ]
            changes = [
                after - before
                for before, after in zip(
                    baseline_values,
                    calibrated_values,
                    strict=True,
                )
            ]

            metric_summary = {
                "baseline": (
                    _numeric_summary(
                        baseline_values
                    )
                ),
                "calibrated": (
                    _numeric_summary(
                        calibrated_values
                    )
                ),
                "change": (
                    _numeric_summary(
                        changes
                    )
                ),
            }

            if metric in LOWER_IS_BETTER:
                metric_summary[
                    "comparison_counts"
                ] = _comparison_counts(
                    baseline_values,
                    calibrated_values,
                )

            metric_summaries[metric] = (
                metric_summary
            )

        first_selective = group_records[0][
            "selective_accuracy"
        ]["baseline"]

        selective_summaries: list[
            dict[str, Any]
        ] = []

        for point_index, first_point in enumerate(
            first_selective
        ):
            requested_coverage = float(
                first_point[
                    "requested_coverage"
                ]
            )

            baseline_accuracy = []
            calibrated_accuracy = []
            baseline_risk = []
            calibrated_risk = []

            for record in group_records:
                baseline_point = record[
                    "selective_accuracy"
                ]["baseline"][point_index]
                calibrated_point = record[
                    "selective_accuracy"
                ]["calibrated"][point_index]

                if not np.isclose(
                    float(
                        baseline_point[
                            "requested_coverage"
                        ]
                    ),
                    requested_coverage,
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError(
                        "Selective coverage grids "
                        "do not match."
                    )

                baseline_accuracy.append(
                    float(
                        baseline_point[
                            "accuracy"
                        ]
                    )
                )
                calibrated_accuracy.append(
                    float(
                        calibrated_point[
                            "accuracy"
                        ]
                    )
                )
                baseline_risk.append(
                    float(
                        baseline_point[
                            "risk"
                        ]
                    )
                )
                calibrated_risk.append(
                    float(
                        calibrated_point[
                            "risk"
                        ]
                    )
                )

            selective_summaries.append(
                {
                    "requested_coverage": (
                        requested_coverage
                    ),
                    "baseline_accuracy": (
                        _numeric_summary(
                            baseline_accuracy
                        )
                    ),
                    "calibrated_accuracy": (
                        _numeric_summary(
                            calibrated_accuracy
                        )
                    ),
                    "accuracy_change": (
                        _numeric_summary(
                            [
                                after - before
                                for before, after
                                in zip(
                                    baseline_accuracy,
                                    calibrated_accuracy,
                                    strict=True,
                                )
                            ]
                        )
                    ),
                    "baseline_risk": (
                        _numeric_summary(
                            baseline_risk
                        )
                    ),
                    "calibrated_risk": (
                        _numeric_summary(
                            calibrated_risk
                        )
                    ),
                }
            )

        aggregate_groups.append(
            {
                "fraction_identifier": (
                    fraction_identifier
                ),
                "method": method,
                "condition": condition,
                "run_count": len(
                    group_records
                ),
                "seeds": sorted(
                    int(record["seed"])
                    for record
                    in group_records
                ),
                "temperature": (
                    _numeric_summary(
                        [
                            float(
                                record[
                                    "temperature"
                                ]
                            )
                            for record
                            in group_records
                        ]
                    )
                ),
                "metrics": metric_summaries,
                "selective_accuracy": (
                    selective_summaries
                ),
            }
        )

    return {
        "record_count": len(records),
        "group_count": len(
            aggregate_groups
        ),
        "groups": aggregate_groups,
    }


__all__ = [
    "CALIBRATION_METRICS",
    "DEFAULT_COVERAGES",
    "LOWER_IS_BETTER",
    "SelectiveAccuracyPoint",
    "aggregate_temperature_transfer_records",
    "evaluate_temperature_transfer",
    "selective_accuracy_curve",
]
