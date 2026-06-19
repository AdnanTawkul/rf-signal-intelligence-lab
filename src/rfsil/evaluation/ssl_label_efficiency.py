from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rfsil.training.ssl_sweep_execution import (
    SweepRunExpectation,
)

_IDENTIFIER_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_]*$"
)


@dataclass(frozen=True, slots=True)
class TestCondition:
    """One untouched held-out evaluation condition."""

    identifier: str
    display_name: str
    test_path: Path


@dataclass(frozen=True, slots=True)
class CompletedTestEvaluation:
    """Validated metrics for one checkpoint and condition."""

    fraction_identifier: str
    method: str
    seed: int
    condition: str
    overall_accuracy: float
    metrics_path: Path


def _require_mapping(
    value: object,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping."
        )

    return value


def _resolve_path(
    value: object,
    *,
    project_root: Path,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{name} must be a non-empty path."
        )

    path = Path(value)

    if path.is_absolute():
        return path

    return project_root / path


def serialize_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    """Serialize a path relative to the project when possible."""
    resolved = path.resolve()
    root = project_root.resolve()

    try:
        return resolved.relative_to(
            root
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def validate_test_conditions(
    value: object,
    *,
    project_root: Path,
) -> tuple[TestCondition, ...]:
    """Validate configured held-out datasets."""
    mapping = _require_mapping(
        value,
        "conditions",
    )

    if not mapping:
        raise ValueError(
            "conditions must not be empty."
        )

    conditions: list[TestCondition] = []

    for identifier, raw_condition in mapping.items():
        normalized_identifier = str(
            identifier
        ).strip()

        if not _IDENTIFIER_PATTERN.fullmatch(
            normalized_identifier
        ):
            raise ValueError(
                "Condition identifiers must be "
                "lowercase identifiers."
            )

        condition = _require_mapping(
            raw_condition,
            f"conditions.{normalized_identifier}",
        )
        display_name = str(
            condition.get(
                "display_name",
                "",
            )
        ).strip()

        if not display_name:
            raise ValueError(
                "Condition display_name must "
                "not be empty."
            )

        test_path = _resolve_path(
            condition.get("test_path"),
            project_root=project_root,
            name=(
                f"conditions."
                f"{normalized_identifier}."
                "test_path"
            ),
        )

        if not test_path.is_file():
            raise FileNotFoundError(
                test_path
            )

        conditions.append(
            TestCondition(
                identifier=(
                    normalized_identifier
                ),
                display_name=display_name,
                test_path=test_path,
            )
        )

    return tuple(conditions)


def build_metrics_path(
    *,
    output_root: Path,
    expectation: SweepRunExpectation,
    condition: TestCondition,
) -> Path:
    """Build one deterministic metrics path."""
    return (
        output_root
        / condition.identifier
        / expectation.fraction_identifier
        / expectation.method
        / f"seed_{expectation.seed}"
        / "metrics.json"
    )


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    expectation: SweepRunExpectation,
) -> None:
    """Verify that a checkpoint matches its planned run."""
    if checkpoint.get("seed") != expectation.seed:
        raise ValueError(
            "Checkpoint seed does not match "
            "the planned run."
        )

    subset = _require_mapping(
        checkpoint.get("labeled_subset"),
        "checkpoint labeled_subset",
    )

    expected_subset = {
        "examples_per_class_snr": (
            expectation.examples_per_class_snr
        ),
        "subset_seed": (
            expectation.subset_seed
        ),
        "selected_training_examples": (
            expectation.selected_training_examples
        ),
    }

    for name, expected_value in (
        expected_subset.items()
    ):
        if subset.get(name) != expected_value:
            raise ValueError(
                "Checkpoint labeled-subset "
                f"field {name!r} does not match."
            )

    if (
        checkpoint.get("training_budget")
        != expectation.training_budget
    ):
        raise ValueError(
            "Checkpoint training budget does "
            "not match the planned run."
        )


def load_completed_test_evaluation(
    *,
    metrics_path: Path,
    expectation: SweepRunExpectation,
    condition: TestCondition,
    project_root: Path,
) -> CompletedTestEvaluation | None:
    """Validate an existing held-out evaluation."""
    if not metrics_path.is_file():
        return None

    content = json.loads(
        metrics_path.read_text(
            encoding="utf-8"
        )
    )
    metrics = _require_mapping(
        content,
        "test metrics",
    )

    expected_values = {
        "fraction_identifier": (
            expectation.fraction_identifier
        ),
        "method": expectation.method,
        "seed": expectation.seed,
        "condition": condition.identifier,
    }

    for name, expected_value in (
        expected_values.items()
    ):
        if metrics.get(name) != expected_value:
            raise ValueError(
                f"Completed metrics field "
                f"{name!r} does not match."
            )

    expected_checkpoint = (
        expectation.output_directory
        / "best_model.pt"
    ).resolve()
    recorded_checkpoint = _resolve_path(
        metrics.get("checkpoint_path"),
        project_root=project_root,
        name="checkpoint_path",
    ).resolve()

    if recorded_checkpoint != expected_checkpoint:
        raise ValueError(
            "Completed metrics checkpoint path "
            "does not match."
        )

    recorded_test_path = _resolve_path(
        metrics.get("test_path"),
        project_root=project_root,
        name="test_path",
    ).resolve()

    if (
        recorded_test_path
        != condition.test_path.resolve()
    ):
        raise ValueError(
            "Completed metrics test path "
            "does not match."
        )

    raw_accuracy = metrics.get(
        "overall_accuracy"
    )

    if isinstance(raw_accuracy, bool):
        raise ValueError(
            "overall_accuracy must be numeric."
        )

    try:
        overall_accuracy = float(
            raw_accuracy
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "overall_accuracy must be numeric."
        ) from error

    if (
        not math.isfinite(overall_accuracy)
        or overall_accuracy < 0.0
        or overall_accuracy > 1.0
    ):
        raise ValueError(
            "overall_accuracy must be between "
            "zero and one."
        )

    class_names = metrics.get(
        "class_names"
    )

    if (
        not isinstance(class_names, list)
        or not class_names
    ):
        raise ValueError(
            "class_names must be a non-empty list."
        )

    class_count = len(class_names)

    for name in (
        "confusion_matrix",
        "normalized_confusion_matrix",
    ):
        matrix = metrics.get(name)

        if (
            not isinstance(matrix, list)
            or len(matrix) != class_count
            or any(
                not isinstance(row, list)
                or len(row) != class_count
                for row in matrix
            )
        ):
            raise ValueError(
                f"{name} has an invalid shape."
            )

    return CompletedTestEvaluation(
        fraction_identifier=(
            expectation.fraction_identifier
        ),
        method=expectation.method,
        seed=expectation.seed,
        condition=condition.identifier,
        overall_accuracy=overall_accuracy,
        metrics_path=metrics_path,
    )


__all__ = [
    "CompletedTestEvaluation",
    "TestCondition",
    "build_metrics_path",
    "load_completed_test_evaluation",
    "serialize_path",
    "validate_checkpoint_metadata",
    "validate_test_conditions",
]
