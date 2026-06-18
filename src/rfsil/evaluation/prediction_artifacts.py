from __future__ import annotations

from pathlib import Path

import numpy as np

from rfsil.evaluation.classification import (
    PredictionResults,
)


def _validate_prediction_results(
    prediction_results: PredictionResults,
) -> PredictionResults:
    """Validate and standardize prediction arrays."""
    raw_labels = np.asarray(
        prediction_results.labels
    )
    raw_predictions = np.asarray(
        prediction_results.predictions
    )
    raw_snr_db = np.asarray(
        prediction_results.snr_db
    )

    if not np.issubdtype(
        raw_labels.dtype,
        np.integer,
    ):
        raise ValueError(
            "labels must contain integers."
        )

    if not np.issubdtype(
        raw_predictions.dtype,
        np.integer,
    ):
        raise ValueError(
            "predictions must contain integers."
        )

    if (
        np.issubdtype(raw_snr_db.dtype, np.bool_)
        or not np.issubdtype(
            raw_snr_db.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "snr_db must contain numeric values."
        )

    labels = np.asarray(
        raw_labels,
        dtype=np.int64,
    )
    predictions = np.asarray(
        raw_predictions,
        dtype=np.int64,
    )
    snr_db = np.asarray(
        raw_snr_db,
        dtype=np.float32,
    )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if predictions.ndim != 1:
        raise ValueError(
            "predictions must be one-dimensional."
        )

    if snr_db.ndim != 1:
        raise ValueError(
            "snr_db must be one-dimensional."
        )

    if labels.size == 0:
        raise ValueError(
            "Prediction arrays must not be empty."
        )

    if not (
        labels.shape
        == predictions.shape
        == snr_db.shape
    ):
        raise ValueError(
            "labels, predictions, and snr_db "
            "must have matching shapes."
        )

    if not np.all(np.isfinite(snr_db)):
        raise ValueError(
            "snr_db must contain only finite values."
        )

    return PredictionResults(
        labels=labels,
        predictions=predictions,
        snr_db=snr_db,
    )


def save_prediction_results(
    output_path: str | Path,
    prediction_results: PredictionResults,
) -> Path:
    """Save prediction results using the standard NPZ schema."""
    path = Path(output_path)

    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Prediction artifact path must use "
            "the .npz extension."
        )

    validated = _validate_prediction_results(
        prediction_results
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        labels=validated.labels,
        predictions=validated.predictions,
        snr_db=validated.snr_db,
    )

    return path


def load_prediction_results(
    input_path: str | Path,
) -> PredictionResults:
    """Load and validate a prediction NPZ artifact."""
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction artifact does not exist: "
            f"{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as content:
        required_keys = {
            "labels",
            "predictions",
            "snr_db",
        }
        missing_keys = required_keys - set(
            content.files
        )

        if missing_keys:
            raise ValueError(
                "Prediction artifact is missing keys: "
                + ", ".join(
                    sorted(missing_keys)
                )
            )

        prediction_results = PredictionResults(
            labels=content["labels"].copy(),
            predictions=(
                content["predictions"].copy()
            ),
            snr_db=content["snr_db"].copy(),
        )

    return _validate_prediction_results(
        prediction_results
    )


__all__ = [
    "load_prediction_results",
    "save_prediction_results",
]
