from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader

from rfsil.evaluation.calibration_artifacts import (
    CalibrationPredictionArtifact,
    build_calibration_artifact,
)

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassificationEvaluation:
    """Classification metrics for one evaluated dataset split."""

    accuracy: float
    confusion_matrix: Int64Array
    normalized_confusion_matrix: Float32Array
    class_accuracy: Float32Array
    snr_values_db: Float32Array
    snr_accuracy: Float32Array
    example_count: int


@dataclass(frozen=True, slots=True)
class PredictionResults:
    """Model predictions and associated ground-truth metadata."""

    labels: Int64Array
    predictions: Int64Array
    snr_db: Float32Array


def _validate_num_classes(
    num_classes: object,
) -> int:
    """Validate the requested number of classes."""
    if (
        isinstance(num_classes, bool)
        or not isinstance(
            num_classes,
            Integral,
        )
    ):
        raise ValueError(
            "num_classes must be an integer."
        )

    validated = int(num_classes)

    if validated < 2:
        raise ValueError(
            "num_classes must be at least 2."
        )

    return validated


def _validate_input_scale(
    input_scale: object,
) -> float:
    """Validate an evaluator-level input scale."""
    if isinstance(
        input_scale,
        (bool, np.bool_),
    ):
        raise ValueError(
            "input_scale must be a positive "
            "finite number."
        )

    try:
        validated = float(input_scale)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "input_scale must be a positive "
            "finite number."
        ) from error

    if (
        not np.isfinite(validated)
        or validated <= 0.0
    ):
        raise ValueError(
            "input_scale must be a positive "
            "finite number."
        )

    return validated


def evaluate_predictions(
    labels: NDArray[np.integer],
    predictions: NDArray[np.integer],
    snr_db: NDArray[np.floating],
    num_classes: int,
) -> ClassificationEvaluation:
    """Compute confusion, class, and SNR metrics."""
    validated_num_classes = (
        _validate_num_classes(num_classes)
    )

    label_array = np.asarray(
        labels,
        dtype=np.int64,
    )
    prediction_array = np.asarray(
        predictions,
        dtype=np.int64,
    )
    snr_array = np.asarray(
        snr_db,
        dtype=np.float32,
    )

    if label_array.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if prediction_array.ndim != 1:
        raise ValueError(
            "predictions must be one-dimensional."
        )

    if snr_array.ndim != 1:
        raise ValueError(
            "snr_db must be one-dimensional."
        )

    if label_array.size == 0:
        raise ValueError(
            "Evaluation arrays must not be empty."
        )

    if not (
        label_array.shape
        == prediction_array.shape
        == snr_array.shape
    ):
        raise ValueError(
            "labels, predictions, and snr_db "
            "must have matching shapes."
        )

    if not np.all(
        np.isfinite(snr_array)
    ):
        raise ValueError(
            "snr_db must contain only "
            "finite values."
        )

    if np.any(label_array < 0) or np.any(
        label_array
        >= validated_num_classes
    ):
        raise ValueError(
            "labels contain an out-of-range "
            "class index."
        )

    if np.any(
        prediction_array < 0
    ) or np.any(
        prediction_array
        >= validated_num_classes
    ):
        raise ValueError(
            "predictions contain an out-of-range "
            "class index."
        )

    confusion = np.zeros(
        (
            validated_num_classes,
            validated_num_classes,
        ),
        dtype=np.int64,
    )
    np.add.at(
        confusion,
        (
            label_array,
            prediction_array,
        ),
        1,
    )

    class_totals = confusion.sum(axis=1)
    class_accuracy = np.divide(
        np.diag(confusion),
        class_totals,
        out=np.zeros(
            validated_num_classes,
            dtype=np.float64,
        ),
        where=class_totals > 0,
    )

    normalized_confusion = np.divide(
        confusion,
        class_totals[:, np.newaxis],
        out=np.zeros_like(
            confusion,
            dtype=np.float64,
        ),
        where=(
            class_totals[:, np.newaxis]
            > 0
        ),
    )

    snr_values = np.unique(snr_array)
    snr_accuracy = np.empty(
        len(snr_values),
        dtype=np.float32,
    )

    for index, snr_value in enumerate(
        snr_values
    ):
        matching = np.isclose(
            snr_array,
            snr_value,
        )
        snr_accuracy[index] = np.mean(
            prediction_array[matching]
            == label_array[matching]
        )

    accuracy = float(
        np.mean(
            prediction_array
            == label_array
        )
    )

    return ClassificationEvaluation(
        accuracy=accuracy,
        confusion_matrix=confusion,
        normalized_confusion_matrix=(
            normalized_confusion.astype(
                np.float32
            )
        ),
        class_accuracy=(
            class_accuracy.astype(
                np.float32
            )
        ),
        snr_values_db=snr_values.astype(
            np.float32
        ),
        snr_accuracy=snr_accuracy,
        example_count=int(
            label_array.size
        ),
    )


def collect_calibration_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    input_scale: float = 1.0,
    class_names: object | None = None,
) -> CalibrationPredictionArtifact:
    """Collect labels, logits, probabilities, and SNR."""
    validated_input_scale = (
        _validate_input_scale(input_scale)
    )

    model.eval()

    collected_labels: list[Tensor] = []
    collected_logits: list[Tensor] = []
    collected_snr: list[Tensor] = []

    with torch.inference_mode():
        for batch in data_loader:
            if not isinstance(
                batch,
                Mapping,
            ):
                raise TypeError(
                    "DataLoader batches must "
                    "be mappings."
                )

            if (
                "iq" not in batch
                or "label" not in batch
                or "snr_db" not in batch
            ):
                raise KeyError(
                    "Batch must contain iq, label, "
                    "and snr_db tensors."
                )

            inputs = batch["iq"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if validated_input_scale != 1.0:
                inputs = (
                    inputs
                    * validated_input_scale
                )

            labels = batch["label"]
            snr_values = batch["snr_db"]

            logits = model(inputs)

            if not isinstance(
                logits,
                Tensor,
            ):
                raise TypeError(
                    "Model output must be "
                    "a tensor."
                )

            if logits.ndim != 2:
                raise ValueError(
                    "Model logits must have shape "
                    "[examples, classes]."
                )

            if logits.shape[0] != inputs.shape[0]:
                raise ValueError(
                    "Model logits must contain one "
                    "row per input example."
                )

            if logits.shape[1] < 2:
                raise ValueError(
                    "Model logits must contain at "
                    "least two classes."
                )

            collected_labels.append(
                labels.detach()
                .cpu()
                .to(torch.int64)
            )
            collected_logits.append(
                logits.detach()
                .cpu()
                .to(torch.float32)
            )
            collected_snr.append(
                snr_values.detach()
                .cpu()
                .to(torch.float32)
            )

    if not collected_labels:
        raise ValueError(
            "Evaluation DataLoader produced "
            "no examples."
        )

    labels_array = torch.cat(
        collected_labels
    ).numpy()
    logits_array = torch.cat(
        collected_logits
    ).numpy()
    snr_array = torch.cat(
        collected_snr
    ).numpy()

    return build_calibration_artifact(
        labels=labels_array,
        logits=logits_array,
        snr_db=snr_array,
        class_names=class_names,
    )


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    *,
    input_scale: float = 1.0,
) -> PredictionResults:
    """Collect the legacy prediction result schema."""
    artifact = (
        collect_calibration_predictions(
            model=model,
            data_loader=data_loader,
            device=device,
            input_scale=input_scale,
        )
    )

    if artifact.snr_db is None:
        raise RuntimeError(
            "Collected prediction artifact "
            "does not contain SNR values."
        )

    return PredictionResults(
        labels=artifact.labels,
        predictions=artifact.predictions,
        snr_db=np.asarray(
            artifact.snr_db,
            dtype=np.float32,
        ),
    )


__all__ = [
    "ClassificationEvaluation",
    "PredictionResults",
    "collect_calibration_predictions",
    "collect_predictions",
    "evaluate_predictions",
]
