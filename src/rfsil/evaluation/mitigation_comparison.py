from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.channel_robustness import (
    SeedSweepMetrics,
)
from rfsil.evaluation.confusion_robustness import (
    ConditionConfusionSummary,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


def _pooled_normalized_confusion(
    summary: ConditionConfusionSummary,
) -> Float64Array:
    """Pool seed confusion matrices and normalize true-class rows."""
    pooled = np.sum(
        summary.confusion_matrices,
        axis=0,
        dtype=np.int64,
    )
    row_totals = pooled.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        pooled,
        row_totals,
        out=np.zeros_like(
            pooled,
            dtype=np.float64,
        ),
        where=row_totals > 0,
    )


def _validate_condition_order(
    condition_order: Sequence[str],
) -> tuple[str, ...]:
    """Validate an ordered nonempty condition sequence."""
    if not condition_order:
        raise ValueError(
            "condition_order must not be empty."
        )

    validated: list[str] = []

    for value in condition_order:
        if not isinstance(value, str):
            raise ValueError(
                "Every condition name must be a string."
            )

        condition = value.strip()

        if not condition:
            raise ValueError(
                "Condition names must not be empty."
            )

        validated.append(condition)

    if len(validated) != len(set(validated)):
        raise ValueError(
            "condition_order must not contain duplicates."
        )

    return tuple(validated)


def _validate_model_pair(
    *,
    condition: str,
    original_metrics: SeedSweepMetrics,
    mitigated_metrics: SeedSweepMetrics,
    original_confusion: ConditionConfusionSummary,
    mitigated_confusion: ConditionConfusionSummary,
) -> None:
    """Validate paired model results for one test condition."""
    if original_metrics.seeds != mitigated_metrics.seeds:
        raise ValueError(
            f"Condition {condition!r} does not use "
            "matching training seeds."
        )

    if (
        original_metrics.class_names
        != mitigated_metrics.class_names
    ):
        raise ValueError(
            f"Condition {condition!r} does not use "
            "matching class names."
        )

    if (
        original_metrics.snr_values_db
        != mitigated_metrics.snr_values_db
    ):
        raise ValueError(
            f"Condition {condition!r} does not use "
            "matching SNR values."
        )

    if original_confusion.seeds != original_metrics.seeds:
        raise ValueError(
            f"Original confusion results for {condition!r} "
            "do not match the metric seeds."
        )

    if mitigated_confusion.seeds != mitigated_metrics.seeds:
        raise ValueError(
            f"Mitigated confusion results for {condition!r} "
            "do not match the metric seeds."
        )

    if (
        original_confusion.class_names
        != original_metrics.class_names
    ):
        raise ValueError(
            f"Original confusion results for {condition!r} "
            "do not match the metric classes."
        )

    if (
        mitigated_confusion.class_names
        != mitigated_metrics.class_names
    ):
        raise ValueError(
            f"Mitigated confusion results for {condition!r} "
            "do not match the metric classes."
        )

    if not np.array_equal(
        original_confusion.labels,
        mitigated_confusion.labels,
    ):
        raise ValueError(
            f"Condition {condition!r} does not use "
            "paired labels."
        )

    if not np.array_equal(
        original_confusion.snr_db,
        mitigated_confusion.snr_db,
    ):
        raise ValueError(
            f"Condition {condition!r} does not use "
            "paired SNR metadata."
        )


def compare_model_families(
    *,
    original_metrics: Mapping[
        str,
        SeedSweepMetrics,
    ],
    mitigated_metrics: Mapping[
        str,
        SeedSweepMetrics,
    ],
    original_confusions: Mapping[
        str,
        ConditionConfusionSummary,
    ],
    mitigated_confusions: Mapping[
        str,
        ConditionConfusionSummary,
    ],
    condition_order: Sequence[str],
    target_class: str = "16QAM",
) -> dict[str, Any]:
    """Compare original and mitigated models on paired conditions."""
    conditions = _validate_condition_order(
        condition_order
    )

    for condition in conditions:
        for name, values in (
            ("original_metrics", original_metrics),
            ("mitigated_metrics", mitigated_metrics),
            (
                "original_confusions",
                original_confusions,
            ),
            (
                "mitigated_confusions",
                mitigated_confusions,
            ),
        ):
            if condition not in values:
                raise ValueError(
                    f"{name} is missing condition "
                    f"{condition!r}."
                )

    first = original_metrics[conditions[0]]

    if target_class not in first.class_names:
        raise ValueError(
            f"target_class {target_class!r} is not "
            "present in class_names."
        )

    target_index = first.class_names.index(
        target_class
    )
    result_conditions: dict[str, Any] = {}

    for condition in conditions:
        original = original_metrics[condition]
        mitigated = mitigated_metrics[condition]
        original_confusion = original_confusions[
            condition
        ]
        mitigated_confusion = mitigated_confusions[
            condition
        ]

        _validate_model_pair(
            condition=condition,
            original_metrics=original,
            mitigated_metrics=mitigated,
            original_confusion=original_confusion,
            mitigated_confusion=mitigated_confusion,
        )

        if original.seeds != first.seeds:
            raise ValueError(
                "All conditions must use the same seeds."
            )

        if original.class_names != first.class_names:
            raise ValueError(
                "All conditions must use the same classes."
            )

        if (
            original.snr_values_db
            != first.snr_values_db
        ):
            raise ValueError(
                "All conditions must use the same "
                "SNR values."
            )

        paired_improvement = (
            mitigated.overall_accuracy
            - original.overall_accuracy
        )

        original_class_mean = np.mean(
            original.class_accuracy,
            axis=0,
        )
        mitigated_class_mean = np.mean(
            mitigated.class_accuracy,
            axis=0,
        )
        class_improvement = (
            mitigated_class_mean
            - original_class_mean
        )

        original_snr_mean = np.mean(
            original.accuracy_by_snr,
            axis=0,
        )
        mitigated_snr_mean = np.mean(
            mitigated.accuracy_by_snr,
            axis=0,
        )
        snr_improvement = (
            mitigated_snr_mean
            - original_snr_mean
        )

        original_matrix = (
            _pooled_normalized_confusion(
                original_confusion
            )
        )
        mitigated_matrix = (
            _pooled_normalized_confusion(
                mitigated_confusion
            )
        )

        target_errors: dict[str, Any] = {}

        for source_index, class_name in enumerate(
            first.class_names
        ):
            if source_index == target_index:
                continue

            original_rate = float(
                original_matrix[
                    source_index,
                    target_index,
                ]
            )
            mitigated_rate = float(
                mitigated_matrix[
                    source_index,
                    target_index,
                ]
            )

            target_errors[class_name] = {
                "original_rate": original_rate,
                "mitigated_rate": mitigated_rate,
                "absolute_reduction": (
                    original_rate
                    - mitigated_rate
                ),
            }

        result_conditions[condition] = {
            "original": {
                "overall_mean": float(
                    np.mean(
                        original.overall_accuracy
                    )
                ),
                "overall_standard_deviation": float(
                    np.std(
                        original.overall_accuracy
                    )
                ),
                "class_accuracy_mean": {
                    class_name: float(value)
                    for class_name, value in zip(
                        first.class_names,
                        original_class_mean,
                        strict=True,
                    )
                },
                "accuracy_by_snr_mean": {
                    str(snr): float(value)
                    for snr, value in zip(
                        first.snr_values_db,
                        original_snr_mean,
                        strict=True,
                    )
                },
            },
            "mitigated": {
                "overall_mean": float(
                    np.mean(
                        mitigated.overall_accuracy
                    )
                ),
                "overall_standard_deviation": float(
                    np.std(
                        mitigated.overall_accuracy
                    )
                ),
                "class_accuracy_mean": {
                    class_name: float(value)
                    for class_name, value in zip(
                        first.class_names,
                        mitigated_class_mean,
                        strict=True,
                    )
                },
                "accuracy_by_snr_mean": {
                    str(snr): float(value)
                    for snr, value in zip(
                        first.snr_values_db,
                        mitigated_snr_mean,
                        strict=True,
                    )
                },
            },
            "improvement": {
                "overall_mean": float(
                    np.mean(paired_improvement)
                ),
                "overall_standard_deviation": float(
                    np.std(paired_improvement)
                ),
                "per_seed": {
                    str(seed): float(value)
                    for seed, value in zip(
                        first.seeds,
                        paired_improvement,
                        strict=True,
                    )
                },
                "class_accuracy": {
                    class_name: float(value)
                    for class_name, value in zip(
                        first.class_names,
                        class_improvement,
                        strict=True,
                    )
                },
                "accuracy_by_snr": {
                    str(snr): float(value)
                    for snr, value in zip(
                        first.snr_values_db,
                        snr_improvement,
                        strict=True,
                    )
                },
            },
            "pooled_normalized_confusion": {
                "original": original_matrix.tolist(),
                "mitigated": (
                    mitigated_matrix.tolist()
                ),
                "difference": (
                    mitigated_matrix
                    - original_matrix
                ).tolist(),
            },
            "misclassification_to_target": (
                target_errors
            ),
        }

    return {
        "format_version": 1,
        "condition_order": list(conditions),
        "seeds": list(first.seeds),
        "class_names": list(first.class_names),
        "snr_values_db": list(
            first.snr_values_db
        ),
        "target_class": target_class,
        "conditions": result_conditions,
    }


__all__ = [
    "compare_model_families",
]
