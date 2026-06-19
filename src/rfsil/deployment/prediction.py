from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from rfsil.deployment.inference import (
    BatchPrediction,
)
from rfsil.deployment.iq_io import LoadedIQ


def validate_top_k(
    value: object,
    *,
    class_count: int,
) -> int:
    """Validate a requested top-k count."""
    if class_count <= 0:
        raise ValueError(
            "class_count must be positive."
        )

    if value is None:
        return class_count

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            "top_k must be a positive integer."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            "top_k must be a positive integer."
        )

    if validated > class_count:
        raise ValueError(
            "top_k must not exceed the number "
            f"of classes ({class_count})."
        )

    return validated


def rank_probabilities(
    probabilities: Sequence[float]
    | np.ndarray,
    *,
    class_names: Sequence[str],
    top_k: int | None = None,
) -> list[dict[str, object]]:
    """Rank class probabilities in descending order."""
    names = tuple(
        str(name)
        for name in class_names
    )

    if not names:
        raise ValueError(
            "class_names must not be empty."
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "class_names must be unique."
        )

    values = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if (
        values.ndim != 1
        or values.shape[0] != len(names)
    ):
        raise ValueError(
            "Probability count must match "
            "the class-name count."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Probabilities must be finite."
        )

    if np.any(values < 0.0) or np.any(
        values > 1.0
    ):
        raise ValueError(
            "Probabilities must be between "
            "zero and one."
        )

    if not np.isclose(
        values.sum(),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "Probabilities must sum to one."
        )

    validated_top_k = validate_top_k(
        top_k,
        class_count=len(names),
    )

    order = np.argsort(
        -values,
        kind="stable",
    )[:validated_top_k]

    return [
        {
            "rank": rank,
            "class_index": int(index),
            "label": names[int(index)],
            "probability": float(
                values[int(index)]
            ),
        }
        for rank, index in enumerate(
            order,
            start=1,
        )
    ]


def _validate_input_scale(
    value: object,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "input_scale must be a positive "
            "finite number."
        )

    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "input_scale must be a positive "
            "finite number."
        ) from error

    if (
        not math.isfinite(validated)
        or validated <= 0.0
    ):
        raise ValueError(
            "input_scale must be a positive "
            "finite number."
        )

    return validated


def _validate_optional_metadata(
    values: np.ndarray | None,
    *,
    batch_size: int,
    name: str,
) -> None:
    if values is None:
        return

    if (
        values.ndim != 1
        or values.shape[0] != batch_size
    ):
        raise ValueError(
            f"{name} must match the loaded "
            "IQ batch size."
        )


def build_prediction_document(
    *,
    loaded: LoadedIQ,
    prediction: BatchPrediction,
    checkpoint_path: str | Path,
    device: str,
    input_scale: float,
    top_k: int | None = None,
    checkpoint_metadata: (
        Mapping[str, Any] | None
    ) = None,
) -> dict[str, object]:
    """Build a JSON-serializable prediction document."""
    if len(prediction) != loaded.batch_size:
        raise ValueError(
            "Prediction count does not match "
            "the loaded IQ batch size."
        )

    class_names = tuple(
        prediction.class_names
    )
    validated_top_k = validate_top_k(
        top_k,
        class_count=len(class_names),
    )
    validated_scale = _validate_input_scale(
        input_scale
    )

    if not isinstance(device, str) or not (
        device.strip()
    ):
        raise ValueError(
            "device must be a non-empty string."
        )

    _validate_optional_metadata(
        loaded.labels,
        batch_size=loaded.batch_size,
        name="labels",
    )
    _validate_optional_metadata(
        loaded.snr_db,
        batch_size=loaded.batch_size,
        name="snr_db",
    )

    if (
        prediction.logits.shape
        != (
            loaded.batch_size,
            len(class_names),
        )
    ):
        raise ValueError(
            "Logit shape does not match the "
            "loaded IQ batch."
        )

    if (
        prediction.probabilities.shape
        != (
            loaded.batch_size,
            len(class_names),
        )
    ):
        raise ValueError(
            "Probability shape does not match "
            "the loaded IQ batch."
        )

    predictions = []

    for position in range(
        loaded.batch_size
    ):
        predicted_index = int(
            prediction.predicted_indices[
                position
            ]
        )

        if not (
            0
            <= predicted_index
            < len(class_names)
        ):
            raise ValueError(
                "Predicted class index is "
                "out of range."
            )

        record: dict[str, object] = {
            "sample_index": int(
                loaded.sample_indices[
                    position
                ]
            ),
            "predicted_index": (
                predicted_index
            ),
            "predicted_label": (
                prediction.predicted_labels[
                    position
                ]
            ),
            "confidence": float(
                prediction.confidences[
                    position
                ]
            ),
            "logits": [
                float(value)
                for value in (
                    prediction.logits[
                        position
                    ]
                )
            ],
            "probabilities": [
                float(value)
                for value in (
                    prediction.probabilities[
                        position
                    ]
                )
            ],
            "top_k": rank_probabilities(
                prediction.probabilities[
                    position
                ],
                class_names=class_names,
                top_k=validated_top_k,
            ),
        }

        if loaded.labels is not None:
            true_index = int(
                loaded.labels[position]
            )

            if not (
                0
                <= true_index
                < len(class_names)
            ):
                raise ValueError(
                    "Ground-truth class index "
                    "is out of range."
                )

            record.update(
                {
                    "true_index": true_index,
                    "true_label": (
                        class_names[true_index]
                    ),
                    "correct": (
                        predicted_index
                        == true_index
                    ),
                }
            )

        if loaded.snr_db is not None:
            record["snr_db"] = float(
                loaded.snr_db[position]
            )

        predictions.append(record)

    metadata = dict(
        checkpoint_metadata or {}
    )

    return {
        "format_version": 1,
        "model": {
            "checkpoint_path": (
                Path(checkpoint_path)
                .resolve()
                .as_posix()
            ),
            "device": device.strip(),
            "input_scale": validated_scale,
            "class_names": list(
                class_names
            ),
            "checkpoint_metadata": metadata,
        },
        "input": {
            "source_path": (
                loaded.source_path
                .resolve()
                .as_posix()
            ),
            "array_key": loaded.array_key,
            "batch_size": (
                loaded.batch_size
            ),
            "channel_count": (
                loaded.channel_count
            ),
            "sample_count": (
                loaded.sample_count
            ),
            "sample_indices": [
                int(index)
                for index in (
                    loaded.sample_indices
                )
            ],
        },
        "configuration": {
            "top_k": validated_top_k,
        },
        "predictions": predictions,
    }


def write_prediction_document(
    output_path: str | Path,
    document: Mapping[str, object],
) -> Path:
    """Write one prediction document as UTF-8 JSON."""
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            document,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return path


__all__ = [
    "build_prediction_document",
    "rank_probabilities",
    "validate_top_k",
    "write_prediction_document",
]
