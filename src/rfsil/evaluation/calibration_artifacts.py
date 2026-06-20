from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.calibration import (
    probabilities_from_logits,
)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]

CALIBRATION_ARTIFACT_VERSION = 2


@dataclass(frozen=True, slots=True)
class CalibrationPredictionArtifact:
    """Logit-preserving classification predictions."""

    labels: Int64Array
    predictions: Int64Array
    logits: Float32Array
    probabilities: Float32Array
    snr_db: Float64Array | None
    class_names: tuple[str, ...] | None
    format_version: int = (
        CALIBRATION_ARTIFACT_VERSION
    )

    @property
    def example_count(self) -> int:
        """Return the number of examples."""
        return int(self.labels.shape[0])

    @property
    def class_count(self) -> int:
        """Return the number of classes."""
        return int(self.logits.shape[1])


def _validate_integer_vector(
    value: object,
    *,
    name: str,
    expected_count: int | None = None,
) -> Int64Array:
    raw = np.asarray(value)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            f"{name} must contain integers."
        )

    validated = np.asarray(
        raw,
        dtype=np.int64,
    )

    if validated.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if validated.size <= 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if (
        expected_count is not None
        and validated.shape[0]
        != expected_count
    ):
        raise ValueError(
            f"{name} must contain "
            f"{expected_count} examples."
        )

    return np.ascontiguousarray(validated)


def _validate_real_matrix(
    value: object,
    *,
    name: str,
) -> Float32Array:
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

    validated = np.asarray(
        raw,
        dtype=np.float32,
    )

    if validated.ndim != 2:
        raise ValueError(
            f"{name} must have shape "
            "[examples, classes]."
        )

    if validated.shape[0] <= 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if validated.shape[1] < 2:
        raise ValueError(
            f"{name} must contain at least "
            "two classes."
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    return np.ascontiguousarray(validated)


def _validate_snr_db(
    value: object | None,
    *,
    expected_count: int,
) -> Float64Array | None:
    if value is None:
        return None

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
            "snr_db must contain real "
            "numeric values."
        )

    validated = np.asarray(
        raw,
        dtype=np.float64,
    )

    if validated.ndim != 1:
        raise ValueError(
            "snr_db must be one-dimensional."
        )

    if validated.shape[0] != expected_count:
        raise ValueError(
            "snr_db must contain one value "
            "per example."
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            "snr_db must contain only "
            "finite values."
        )

    return np.ascontiguousarray(validated)


def _validate_class_names(
    value: object | None,
    *,
    class_count: int,
) -> tuple[str, ...] | None:
    if value is None:
        return None

    raw_names = (
        value.tolist()
        if isinstance(value, np.ndarray)
        else value
    )

    if (
        isinstance(raw_names, (str, bytes))
        or not isinstance(
            raw_names,
            (list, tuple),
        )
    ):
        raise ValueError(
            "class_names must be a sequence "
            "of strings."
        )

    names = tuple(raw_names)

    if len(names) != class_count:
        raise ValueError(
            "class_names must match the "
            "class count."
        )

    if any(
        not isinstance(name, str)
        or not name.strip()
        for name in names
    ):
        raise ValueError(
            "Every class name must be a "
            "non-empty string."
        )

    normalized = tuple(
        name.strip()
        for name in names
    )

    if len(set(normalized)) != len(
        normalized
    ):
        raise ValueError(
            "class_names must be unique."
        )

    return normalized


def validate_calibration_artifact(
    *,
    labels: object,
    predictions: object,
    logits: object,
    probabilities: object,
    snr_db: object | None = None,
    class_names: object | None = None,
    format_version: int = (
        CALIBRATION_ARTIFACT_VERSION
    ),
) -> CalibrationPredictionArtifact:
    """Validate and normalize artifact arrays."""
    if format_version != (
        CALIBRATION_ARTIFACT_VERSION
    ):
        raise ValueError(
            "Unsupported calibration artifact "
            f"version: {format_version}."
        )

    validated_logits = _validate_real_matrix(
        logits,
        name="logits",
    )
    validated_probabilities = (
        _validate_real_matrix(
            probabilities,
            name="probabilities",
        )
    )

    if (
        validated_probabilities.shape
        != validated_logits.shape
    ):
        raise ValueError(
            "logits and probabilities must "
            "have the same shape."
        )

    if np.any(
        validated_probabilities < 0.0
    ) or np.any(
        validated_probabilities > 1.0
    ):
        raise ValueError(
            "probabilities must be between "
            "zero and one."
        )

    row_sums = validated_probabilities.sum(
        axis=1
    )

    if not np.allclose(
        row_sums,
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "Every probability row must "
            "sum to one."
        )

    expected_probabilities = (
        probabilities_from_logits(
            validated_logits
        )
    )

    if not np.allclose(
        validated_probabilities,
        expected_probabilities,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "probabilities are inconsistent "
            "with logits."
        )

    example_count = int(
        validated_logits.shape[0]
    )
    class_count = int(
        validated_logits.shape[1]
    )

    validated_labels = (
        _validate_integer_vector(
            labels,
            name="labels",
            expected_count=example_count,
        )
    )
    validated_predictions = (
        _validate_integer_vector(
            predictions,
            name="predictions",
            expected_count=example_count,
        )
    )

    for name, values in (
        ("labels", validated_labels),
        (
            "predictions",
            validated_predictions,
        ),
    ):
        if np.any(values < 0) or np.any(
            values >= class_count
        ):
            raise ValueError(
                f"{name} contain an "
                "out-of-range class index."
            )

    expected_predictions = np.argmax(
        validated_probabilities,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    if not np.array_equal(
        validated_predictions,
        expected_predictions,
    ):
        raise ValueError(
            "predictions are inconsistent "
            "with probabilities."
        )

    validated_snr_db = _validate_snr_db(
        snr_db,
        expected_count=example_count,
    )
    validated_class_names = (
        _validate_class_names(
            class_names,
            class_count=class_count,
        )
    )

    return CalibrationPredictionArtifact(
        labels=validated_labels,
        predictions=(
            validated_predictions
        ),
        logits=validated_logits,
        probabilities=(
            validated_probabilities
        ),
        snr_db=validated_snr_db,
        class_names=validated_class_names,
        format_version=format_version,
    )


def build_calibration_artifact(
    *,
    labels: object,
    logits: object,
    snr_db: object | None = None,
    class_names: object | None = None,
) -> CalibrationPredictionArtifact:
    """Build an artifact directly from logits."""
    validated_logits = _validate_real_matrix(
        logits,
        name="logits",
    )
    probabilities = (
        probabilities_from_logits(
            validated_logits
        ).astype(
            np.float32,
            copy=False,
        )
    )
    predictions = np.argmax(
        probabilities,
        axis=1,
    ).astype(
        np.int64,
        copy=False,
    )

    return validate_calibration_artifact(
        labels=labels,
        predictions=predictions,
        logits=validated_logits,
        probabilities=probabilities,
        snr_db=snr_db,
        class_names=class_names,
    )


def save_calibration_artifact(
    path: str | Path,
    artifact: CalibrationPredictionArtifact,
) -> Path:
    """Save a validated calibration artifact."""
    output_path = Path(path)

    validated = (
        validate_calibration_artifact(
            labels=artifact.labels,
            predictions=(
                artifact.predictions
            ),
            logits=artifact.logits,
            probabilities=(
                artifact.probabilities
            ),
            snr_db=artifact.snr_db,
            class_names=(
                artifact.class_names
            ),
            format_version=(
                artifact.format_version
            ),
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload: dict[str, Any] = {
        "format_version": np.asarray(
            validated.format_version,
            dtype=np.int64,
        ),
        "labels": validated.labels,
        "predictions": (
            validated.predictions
        ),
        "logits": validated.logits,
        "probabilities": (
            validated.probabilities
        ),
    }

    if validated.snr_db is not None:
        payload["snr_db"] = (
            validated.snr_db
        )

    if validated.class_names is not None:
        payload["class_names"] = np.asarray(
            validated.class_names,
            dtype=np.str_,
        )

    np.savez_compressed(
        output_path,
        **payload,
    )

    return output_path


def load_calibration_artifact(
    path: str | Path,
) -> CalibrationPredictionArtifact:
    """Load a versioned calibration artifact."""
    input_path = Path(path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(
        input_path,
        allow_pickle=False,
    ) as content:
        keys = set(content.files)

        if (
            "logits" not in keys
            or "probabilities" not in keys
        ):
            raise ValueError(
                "Legacy prediction artifact "
                "does not contain logits and "
                "probabilities."
            )

        required = {
            "format_version",
            "labels",
            "predictions",
            "logits",
            "probabilities",
        }
        missing = sorted(required - keys)

        if missing:
            raise ValueError(
                "Calibration artifact is "
                "missing required arrays: "
                + ", ".join(missing)
                + "."
            )

        version_array = np.asarray(
            content["format_version"]
        )

        if version_array.size != 1:
            raise ValueError(
                "format_version must contain "
                "one scalar value."
            )

        format_version = int(
            version_array.reshape(-1)[0]
        )

        snr_db = (
            np.asarray(content["snr_db"])
            if "snr_db" in keys
            else None
        )
        class_names = (
            np.asarray(
                content["class_names"]
            )
            if "class_names" in keys
            else None
        )

        return (
            validate_calibration_artifact(
                labels=np.asarray(
                    content["labels"]
                ),
                predictions=np.asarray(
                    content["predictions"]
                ),
                logits=np.asarray(
                    content["logits"]
                ),
                probabilities=np.asarray(
                    content[
                        "probabilities"
                    ]
                ),
                snr_db=snr_db,
                class_names=class_names,
                format_version=format_version,
            )
        )


__all__ = [
    "CALIBRATION_ARTIFACT_VERSION",
    "CalibrationPredictionArtifact",
    "build_calibration_artifact",
    "load_calibration_artifact",
    "save_calibration_artifact",
    "validate_calibration_artifact",
]
