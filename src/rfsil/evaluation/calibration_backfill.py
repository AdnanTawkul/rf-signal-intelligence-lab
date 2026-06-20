from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rfsil.evaluation.calibration_artifacts import (
    CalibrationPredictionArtifact,
    load_calibration_artifact,
)


def build_calibration_artifact_path(
    metrics_path: str | Path,
    *,
    filename: str = "calibration_predictions.npz",
) -> Path:
    """Build the artifact path beside one metrics file."""
    validated_filename = str(filename).strip()

    if (
        not validated_filename
        or Path(validated_filename).name
        != validated_filename
        or Path(validated_filename).suffix.lower()
        != ".npz"
    ):
        raise ValueError(
            "Calibration artifact filename must "
            "be one local .npz filename."
        )

    return (
        Path(metrics_path).parent
        / validated_filename
    )


def load_valid_calibration_artifact(
    path: str | Path,
    *,
    expected_example_count: int | None = None,
    expected_class_names: (
        tuple[str, ...] | None
    ) = None,
) -> CalibrationPredictionArtifact | None:
    """Load and validate an existing artifact."""
    artifact_path = Path(path)

    if not artifact_path.is_file():
        return None

    artifact = load_calibration_artifact(
        artifact_path
    )

    if (
        expected_example_count is not None
        and artifact.example_count
        != expected_example_count
    ):
        raise ValueError(
            "Calibration artifact example count "
            "does not match completed metrics."
        )

    if (
        expected_class_names is not None
        and artifact.class_names
        != expected_class_names
    ):
        raise ValueError(
            "Calibration artifact class names "
            "do not match completed metrics."
        )

    return artifact


def compute_artifact_accuracy(
    artifact: CalibrationPredictionArtifact,
) -> float:
    """Compute classification accuracy from an artifact."""
    return float(
        np.mean(
            artifact.labels
            == artifact.predictions
        )
    )


def validate_artifact_accuracy(
    artifact: CalibrationPredictionArtifact,
    *,
    expected_accuracy: float,
    absolute_tolerance: float = 1e-8,
) -> float:
    """Confirm artifact accuracy matches completed metrics."""
    if isinstance(
        expected_accuracy,
        (bool, np.bool_),
    ):
        raise ValueError(
            "expected_accuracy must be between "
            "zero and one."
        )

    try:
        validated_expected = float(
            expected_accuracy
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "expected_accuracy must be between "
            "zero and one."
        ) from error

    if (
        not math.isfinite(
            validated_expected
        )
        or not 0.0
        <= validated_expected
        <= 1.0
    ):
        raise ValueError(
            "expected_accuracy must be between "
            "zero and one."
        )

    if isinstance(
        absolute_tolerance,
        (bool, np.bool_),
    ):
        raise ValueError(
            "absolute_tolerance must be "
            "non-negative and finite."
        )

    try:
        validated_tolerance = float(
            absolute_tolerance
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "absolute_tolerance must be "
            "non-negative and finite."
        ) from error

    if (
        not math.isfinite(
            validated_tolerance
        )
        or validated_tolerance < 0.0
    ):
        raise ValueError(
            "absolute_tolerance must be "
            "non-negative and finite."
        )

    actual_accuracy = (
        compute_artifact_accuracy(
            artifact
        )
    )

    if not np.isclose(
        actual_accuracy,
        validated_expected,
        rtol=0.0,
        atol=validated_tolerance,
    ):
        raise ValueError(
            "Calibration artifact accuracy does "
            "not match completed metrics: "
            f"artifact={actual_accuracy:.10f}, "
            f"metrics={validated_expected:.10f}."
        )

    return actual_accuracy


__all__ = [
    "build_calibration_artifact_path",
    "compute_artifact_accuracy",
    "load_valid_calibration_artifact",
    "validate_artifact_accuracy",
]
