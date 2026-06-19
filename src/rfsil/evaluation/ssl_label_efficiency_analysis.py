from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SelectionSummary:
    """Metrics for one selected label-efficiency system."""

    fraction_identifier: str
    method: str
    validation_accuracy: float
    validation_standard_deviation: float
    condition_metrics: dict[str, dict[str, object]]
    macro_condition_accuracy: float
    macro_condition_standard_deviation: float
    macro_change_vs_random: float
    macro_change_standard_deviation: float
    macro_seeds_improved: int


def require_mapping(
    value: object,
    name: str,
) -> Mapping[str, Any]:
    """Validate one mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping."
        )

    return value


def require_sequence(
    value: object,
    name: str,
) -> Sequence[object]:
    """Validate a non-string sequence."""
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(
            f"{name} must be a sequence."
        )

    return value


def load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    """Load one UTF-8 JSON mapping."""
    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            f"JSON content must be a mapping: {path}"
        )

    return content


def validate_accuracy(
    value: object,
    name: str,
) -> float:
    """Validate a finite accuracy value."""
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be numeric."
        )

    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be numeric."
        ) from error

    if (
        not math.isfinite(validated)
        or validated < 0.0
        or validated > 1.0
    ):
        raise ValueError(
            f"{name} must be between zero and one."
        )

    return validated


def validation_method_metrics(
    aggregate: Mapping[str, Any],
    *,
    fraction_identifier: str,
    method: str,
) -> Mapping[str, Any]:
    """Return validation metrics for one method."""
    fractions = require_mapping(
        aggregate.get("fractions"),
        "validation fractions",
    )
    fraction = require_mapping(
        fractions.get(fraction_identifier),
        (
            "validation fraction "
            f"{fraction_identifier}"
        ),
    )
    methods = require_mapping(
        fraction.get("methods"),
        "validation methods",
    )

    return require_mapping(
        methods.get(method),
        f"validation method {method}",
    )


def held_out_method_metrics(
    aggregate: Mapping[str, Any],
    *,
    fraction_identifier: str,
    condition: str,
    method: str,
) -> Mapping[str, Any]:
    """Return held-out metrics for one method."""
    fractions = require_mapping(
        aggregate.get("fractions"),
        "held-out fractions",
    )
    fraction = require_mapping(
        fractions.get(fraction_identifier),
        f"held-out fraction {fraction_identifier}",
    )
    conditions = require_mapping(
        fraction.get("conditions"),
        "held-out conditions",
    )
    condition_content = require_mapping(
        conditions.get(condition),
        f"held-out condition {condition}",
    )
    methods = require_mapping(
        condition_content.get("methods"),
        "held-out methods",
    )

    return require_mapping(
        methods.get(method),
        f"held-out method {method}",
    )


def held_out_paired_metrics(
    aggregate: Mapping[str, Any],
    *,
    fraction_identifier: str,
    condition: str,
    method: str,
) -> Mapping[str, Any] | None:
    """Return paired change versus random."""
    if method == "random":
        return None

    fractions = require_mapping(
        aggregate.get("fractions"),
        "held-out fractions",
    )
    fraction = require_mapping(
        fractions.get(fraction_identifier),
        f"held-out fraction {fraction_identifier}",
    )
    conditions = require_mapping(
        fraction.get("conditions"),
        "held-out conditions",
    )
    condition_content = require_mapping(
        conditions.get(condition),
        f"held-out condition {condition}",
    )
    paired = require_mapping(
        condition_content.get(
            "paired_changes_vs_random"
        ),
        "paired changes",
    )

    return require_mapping(
        paired.get(method),
        f"paired method {method}",
    )


def summarize_selection(
    *,
    validation_aggregate: Mapping[str, Any],
    held_out_aggregate: Mapping[str, Any],
    fraction_identifier: str,
    method: str,
    conditions: Sequence[str],
) -> SelectionSummary:
    """Summarize one validation-selected system."""
    validation = validation_method_metrics(
        validation_aggregate,
        fraction_identifier=fraction_identifier,
        method=method,
    )

    validation_accuracy = validate_accuracy(
        validation.get(
            "mean_validation_accuracy"
        ),
        "mean_validation_accuracy",
    )
    validation_std = float(
        validation.get(
            "validation_accuracy_standard_deviation"
        )
    )

    condition_metrics = {}

    for condition in conditions:
        method_metrics = (
            held_out_method_metrics(
                held_out_aggregate,
                fraction_identifier=(
                    fraction_identifier
                ),
                condition=condition,
                method=method,
            )
        )
        paired = held_out_paired_metrics(
            held_out_aggregate,
            fraction_identifier=(
                fraction_identifier
            ),
            condition=condition,
            method=method,
        )

        condition_metrics[condition] = {
            "mean_accuracy": validate_accuracy(
                method_metrics.get("mean"),
                (
                    f"{condition} mean "
                    "accuracy"
                ),
            ),
            "standard_deviation": float(
                method_metrics.get(
                    "standard_deviation"
                )
            ),
            "change_vs_random": (
                0.0
                if paired is None
                else float(
                    paired.get("mean")
                )
            ),
            "change_standard_deviation": (
                0.0
                if paired is None
                else float(
                    paired.get(
                        "standard_deviation"
                    )
                )
            ),
            "seeds_improved": (
                None
                if paired is None
                else int(
                    paired.get(
                        "seeds_improved"
                    )
                )
            ),
        }

    fractions = require_mapping(
        held_out_aggregate.get("fractions"),
        "held-out fractions",
    )
    fraction = require_mapping(
        fractions.get(fraction_identifier),
        f"held-out fraction {fraction_identifier}",
    )
    macro = require_mapping(
        fraction.get(
            "macro_condition_average"
        ),
        "macro condition average",
    )
    macro_methods = require_mapping(
        macro.get("methods"),
        "macro methods",
    )
    macro_method = require_mapping(
        macro_methods.get(method),
        f"macro method {method}",
    )

    if method == "random":
        macro_change = 0.0
        macro_change_std = 0.0
        macro_improved = 0
    else:
        macro_paired = require_mapping(
            macro.get(
                "paired_changes_vs_random"
            ),
            "macro paired changes",
        )
        macro_method_paired = require_mapping(
            macro_paired.get(method),
            f"macro paired method {method}",
        )
        macro_change = float(
            macro_method_paired.get("mean")
        )
        macro_change_std = float(
            macro_method_paired.get(
                "standard_deviation"
            )
        )
        macro_improved = int(
            macro_method_paired.get(
                "seeds_improved"
            )
        )

    return SelectionSummary(
        fraction_identifier=(
            fraction_identifier
        ),
        method=method,
        validation_accuracy=(
            validation_accuracy
        ),
        validation_standard_deviation=(
            validation_std
        ),
        condition_metrics=(
            condition_metrics
        ),
        macro_condition_accuracy=(
            validate_accuracy(
                macro_method.get("mean"),
                "macro mean accuracy",
            )
        ),
        macro_condition_standard_deviation=(
            float(
                macro_method.get(
                    "standard_deviation"
                )
            )
        ),
        macro_change_vs_random=(
            macro_change
        ),
        macro_change_standard_deviation=(
            macro_change_std
        ),
        macro_seeds_improved=macro_improved,
    )


def build_paired_change_matrix(
    *,
    aggregate: Mapping[str, Any],
    fractions: Sequence[str],
    conditions: Sequence[str],
    methods: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Build SSL-minus-random changes."""
    rows = []
    values = []

    for condition in conditions:
        for method in methods:
            if method == "random":
                continue

            rows.append(
                f"{condition}: {method}"
            )
            row = []

            for fraction in fractions:
                paired = held_out_paired_metrics(
                    aggregate,
                    fraction_identifier=(
                        fraction
                    ),
                    condition=condition,
                    method=method,
                )

                if paired is None:
                    raise RuntimeError(
                        "Missing paired change."
                    )

                row.append(
                    float(paired.get("mean"))
                )

            values.append(row)

    return rows, np.asarray(
        values,
        dtype=np.float64,
    )


def pool_confusion_matrices(
    metrics_paths: Sequence[Path],
) -> dict[str, object]:
    """Pool raw confusion matrices across seeds."""
    if not metrics_paths:
        raise ValueError(
            "At least one metrics path is required."
        )

    class_names: list[str] | None = None
    pooled: np.ndarray | None = None
    accuracies = []

    for path in metrics_paths:
        metrics = load_json_mapping(path)

        current_names = [
            str(name)
            for name in require_sequence(
                metrics.get("class_names"),
                "class_names",
            )
        ]

        matrix = np.asarray(
            metrics.get("confusion_matrix"),
            dtype=np.int64,
        )

        if (
            matrix.ndim != 2
            or matrix.shape[0]
            != matrix.shape[1]
            or matrix.shape[0]
            != len(current_names)
        ):
            raise ValueError(
                f"Invalid confusion matrix: {path}"
            )

        if np.any(matrix < 0):
            raise ValueError(
                "Confusion values must be "
                "non-negative."
            )

        if class_names is None:
            class_names = current_names
            pooled = np.zeros_like(
                matrix,
                dtype=np.int64,
            )
        elif class_names != current_names:
            raise ValueError(
                "Class names differ across metrics."
            )

        if pooled is None:
            raise RuntimeError(
                "Pooled confusion was not created."
            )

        pooled += matrix
        accuracies.append(
            validate_accuracy(
                metrics.get(
                    "overall_accuracy"
                ),
                "overall_accuracy",
            )
        )

    if class_names is None or pooled is None:
        raise RuntimeError(
            "No confusion matrices were pooled."
        )

    row_totals = pooled.sum(
        axis=1,
        keepdims=True,
    )
    normalized = np.divide(
        pooled,
        row_totals,
        out=np.zeros_like(
            pooled,
            dtype=np.float64,
        ),
        where=row_totals != 0,
    )

    return {
        "class_names": class_names,
        "confusion_matrix": pooled.tolist(),
        "normalized_confusion_matrix": (
            normalized.tolist()
        ),
        "mean_accuracy": float(
            np.mean(accuracies)
        ),
        "standard_deviation": float(
            np.std(accuracies)
        ),
        "metrics_paths": [
            path.as_posix()
            for path in metrics_paths
        ],
    }


__all__ = [
    "SelectionSummary",
    "build_paired_change_matrix",
    "held_out_method_metrics",
    "held_out_paired_metrics",
    "load_json_mapping",
    "pool_confusion_matrices",
    "summarize_selection",
    "validation_method_metrics",
]
