from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rfsil.evaluation.calibration import (
    evaluate_calibration,
)
from rfsil.evaluation.calibration_artifacts import (
    CalibrationPredictionArtifact,
)
from rfsil.evaluation.temperature_scaling import (
    fit_temperature,
    probabilities_with_temperature,
)

TEMPERATURE_SUMMARY_VERSION = 1


def fit_temperature_for_artifact(
    artifact: CalibrationPredictionArtifact,
    *,
    bin_count: int = 15,
    temperature_bounds: (
        tuple[float, float]
    ) = (0.05, 20.0),
    optimization_tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> dict[str, Any]:
    """Fit and evaluate one scalar temperature."""
    baseline_probabilities = np.asarray(
        artifact.probabilities,
        dtype=np.float64,
    )

    baseline_predictions = np.argmax(
        baseline_probabilities,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        baseline_predictions,
        artifact.predictions,
    ):
        raise ValueError(
            "Artifact predictions do not match "
            "its probability argmax."
        )

    result = fit_temperature(
        labels=artifact.labels,
        logits=artifact.logits,
        temperature_bounds=(
            temperature_bounds
        ),
        optimization_tolerance=(
            optimization_tolerance
        ),
        max_iterations=max_iterations,
    )

    calibrated_probabilities = (
        probabilities_with_temperature(
            artifact.logits,
            result.temperature,
        )
    )

    calibrated_predictions = np.argmax(
        calibrated_probabilities,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        calibrated_predictions,
        artifact.predictions,
    ):
        raise RuntimeError(
            "Positive scalar temperature changed "
            "the predicted classes."
        )

    baseline_evaluation = (
        evaluate_calibration(
            labels=artifact.labels,
            probabilities=(
                baseline_probabilities
            ),
            bin_count=bin_count,
        )
    )
    calibrated_evaluation = (
        evaluate_calibration(
            labels=artifact.labels,
            probabilities=(
                calibrated_probabilities
            ),
            bin_count=bin_count,
        )
    )

    if not np.isclose(
        baseline_evaluation.accuracy,
        calibrated_evaluation.accuracy,
        rtol=0.0,
        atol=0.0,
    ):
        raise RuntimeError(
            "Temperature scaling changed "
            "classification accuracy."
        )

    return {
        "format_version": (
            TEMPERATURE_SUMMARY_VERSION
        ),
        "temperature_scaling": (
            result.to_dict()
        ),
        "baseline_calibration": asdict(
            baseline_evaluation
        ),
        "calibrated_calibration": asdict(
            calibrated_evaluation
        ),
        "accuracy_preserved": True,
        "prediction_count": (
            artifact.example_count
        ),
        "class_count": artifact.class_count,
        "class_names": (
            list(artifact.class_names)
            if artifact.class_names
            is not None
            else None
        ),
    }


def save_temperature_summary(
    path: str | Path,
    summary: dict[str, Any],
) -> Path:
    """Save one temperature-calibration summary."""
    output_path = Path(path)

    validated = validate_temperature_summary(
        summary
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            validated,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


def load_temperature_summary(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate one summary."""
    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    content = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    return validate_temperature_summary(
        content
    )


def validate_temperature_summary(
    value: object,
) -> dict[str, Any]:
    """Validate saved summary structure."""
    if not isinstance(value, dict):
        raise ValueError(
            "Temperature summary must be "
            "a mapping."
        )

    version = value.get(
        "format_version"
    )

    if version != TEMPERATURE_SUMMARY_VERSION:
        raise ValueError(
            "Unsupported temperature summary "
            f"version: {version}."
        )

    scaling = value.get(
        "temperature_scaling"
    )

    if not isinstance(scaling, dict):
        raise ValueError(
            "temperature_scaling must be "
            "a mapping."
        )

    try:
        temperature = float(
            scaling["temperature"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Temperature summary does not "
            "contain a valid temperature."
        ) from error

    if (
        not np.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError(
            "Saved temperature must be "
            "positive and finite."
        )

    if value.get(
        "accuracy_preserved"
    ) is not True:
        raise ValueError(
            "Temperature summary must confirm "
            "accuracy preservation."
        )

    for key in (
        "baseline_calibration",
        "calibrated_calibration",
    ):
        if not isinstance(
            value.get(key),
            dict,
        ):
            raise ValueError(
                f"{key} must be a mapping."
            )

    return value


__all__ = [
    "TEMPERATURE_SUMMARY_VERSION",
    "fit_temperature_for_artifact",
    "load_temperature_summary",
    "save_temperature_summary",
    "validate_temperature_summary",
]
