from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from rfsil.models.model_factory import (
    ModelConfiguration,
    create_model_from_checkpoint,
)

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class WindowPrediction:
    """Prediction for one fixed-length IQ window."""

    predicted_index: int
    predicted_label: str
    confidence: float
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BatchPrediction:
    """Predictions for one batch of IQ windows."""

    class_names: tuple[str, ...]
    logits: Float32Array
    probabilities: Float32Array
    predicted_indices: Int64Array
    predicted_labels: tuple[str, ...]
    confidences: Float32Array

    def __len__(self) -> int:
        """Return the number of predicted windows."""
        return int(
            self.predicted_indices.shape[0]
        )

    def item(
        self,
        index: int,
    ) -> WindowPrediction:
        """Return one prediction from the batch."""
        if (
            isinstance(index, bool)
            or not isinstance(index, Integral)
        ):
            raise TypeError(
                "index must be an integer."
            )

        validated_index = int(index)

        if not 0 <= validated_index < len(self):
            raise IndexError(
                "prediction index is out of range."
            )

        return WindowPrediction(
            predicted_index=int(
                self.predicted_indices[
                    validated_index
                ]
            ),
            predicted_label=(
                self.predicted_labels[
                    validated_index
                ]
            ),
            confidence=float(
                self.confidences[
                    validated_index
                ]
            ),
            logits=tuple(
                float(value)
                for value in self.logits[
                    validated_index
                ]
            ),
            probabilities=tuple(
                float(value)
                for value in self.probabilities[
                    validated_index
                ]
            ),
        )


def resolve_device(
    value: str | torch.device,
) -> torch.device:
    """Resolve and validate an inference device."""
    if isinstance(value, torch.device):
        device = value
    elif isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == "auto":
            normalized = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        try:
            device = torch.device(normalized)
        except RuntimeError as error:
            raise ValueError(
                f"Invalid device: {value!r}."
            ) from error
    else:
        raise TypeError(
            "device must be a string or "
            "torch.device."
        )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is not "
            "available."
        )

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Only CPU and CUDA inference are "
            "currently supported."
        )

    return device


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


def _validate_expected_sample_count(
    value: object,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            "expected_sample_count must be "
            "a positive integer or None."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            "expected_sample_count must be "
            "a positive integer or None."
        )

    return validated


def _validate_class_names(
    value: object,
    *,
    expected_count: int,
) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(
            "Checkpoint class_names must be "
            "a sequence."
        )

    names = tuple(
        str(name).strip()
        for name in value
    )

    if len(names) != expected_count:
        raise ValueError(
            "Checkpoint class_names count does "
            "not match the model configuration."
        )

    if any(not name for name in names):
        raise ValueError(
            "Checkpoint class names must not "
            "be empty."
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Checkpoint class names must be "
            "unique."
        )

    return names


class IQInferenceEngine:
    """Checkpoint-backed RF IQ classifier."""

    def __init__(
        self,
        *,
        model: nn.Module,
        model_configuration: (
            ModelConfiguration
        ),
        class_names: Sequence[str],
        device: str | torch.device = "auto",
        input_scale: float = 1.0,
        expected_sample_count: (
            int | None
        ) = 2048,
        checkpoint_path: Path | None = None,
        checkpoint_metadata: (
            Mapping[str, Any] | None
        ) = None,
    ) -> None:
        """Create an inference engine."""
        self.device = resolve_device(device)
        self.input_scale = (
            _validate_input_scale(
                input_scale
            )
        )
        self.expected_sample_count = (
            _validate_expected_sample_count(
                expected_sample_count
            )
        )

        self.model_configuration = (
            model_configuration
        )
        self.class_names = (
            _validate_class_names(
                class_names,
                expected_count=(
                    model_configuration
                    .num_classes
                ),
            )
        )

        self.checkpoint_path = (
            checkpoint_path
        )
        self.checkpoint_metadata = dict(
            checkpoint_metadata or {}
        )

        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "auto",
        input_scale: float = 1.0,
        expected_sample_count: (
            int | None
        ) = 2048,
    ) -> IQInferenceEngine:
        """Load a strict inference engine."""
        path = Path(checkpoint_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{path}"
            )

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if not isinstance(
            checkpoint,
            Mapping,
        ):
            raise ValueError(
                "Checkpoint content must be "
                "a mapping."
            )

        state_dict = checkpoint.get(
            "model_state_dict"
        )

        if not isinstance(
            state_dict,
            Mapping,
        ):
            raise ValueError(
                "Checkpoint model_state_dict "
                "must be a mapping."
            )

        model, configuration = (
            create_model_from_checkpoint(
                checkpoint
            )
        )

        class_names = (
            _validate_class_names(
                checkpoint.get(
                    "class_names"
                ),
                expected_count=(
                    configuration.num_classes
                ),
            )
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        return cls(
            model=model,
            model_configuration=(
                configuration
            ),
            class_names=class_names,
            device=device,
            input_scale=input_scale,
            expected_sample_count=(
                expected_sample_count
            ),
            checkpoint_path=path,
            checkpoint_metadata={
                key: checkpoint.get(key)
                for key in (
                    "format_version",
                    "experiment_name",
                    "seed",
                    "best_epoch",
                    "best_validation_accuracy",
                    "initialization",
                )
                if key in checkpoint
            },
        )

    @property
    def in_channels(self) -> int:
        """Return the required channel count."""
        return int(
            self.model_configuration.in_channels
        )

    @property
    def num_classes(self) -> int:
        """Return the classifier class count."""
        return int(
            self.model_configuration.num_classes
        )

    def _prepare_inputs(
        self,
        inputs: np.ndarray | Tensor,
    ) -> Tensor:
        if isinstance(inputs, Tensor):
            tensor = inputs.detach()
        else:
            array = np.asarray(inputs)

            if not np.issubdtype(
                array.dtype,
                np.number,
            ):
                raise ValueError(
                    "IQ inputs must contain "
                    "numeric values."
                )

            tensor = torch.from_numpy(
                np.ascontiguousarray(array)
            )

        if tensor.is_complex():
            raise ValueError(
                "Complex IQ arrays must be "
                "converted to two real channels "
                "before inference."
            )

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError(
                "IQ inputs must have shape "
                "[channels, samples] or "
                "[batch, channels, samples]."
            )

        if tensor.shape[0] <= 0:
            raise ValueError(
                "IQ input batch must not be empty."
            )

        if tensor.shape[1] != self.in_channels:
            raise ValueError(
                "IQ channel count does not match "
                "the model configuration."
            )

        if (
            self.expected_sample_count
            is not None
            and tensor.shape[2]
            != self.expected_sample_count
        ):
            raise ValueError(
                "IQ sample count does not match "
                f"the expected value of "
                f"{self.expected_sample_count}."
            )

        tensor = tensor.to(
            dtype=torch.float32
        )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "IQ inputs must contain only "
                "finite values."
            )

        return tensor.to(
            device=self.device,
            non_blocking=(
                self.device.type == "cuda"
            ),
        )

    def predict_batch(
        self,
        inputs: np.ndarray | Tensor,
    ) -> BatchPrediction:
        """Predict one batch of IQ windows."""
        prepared = self._prepare_inputs(
            inputs
        )

        if self.input_scale != 1.0:
            prepared = (
                prepared * self.input_scale
            )

        with torch.inference_mode():
            logits = self.model(prepared)

            if (
                logits.ndim != 2
                or logits.shape[0]
                != prepared.shape[0]
                or logits.shape[1]
                != self.num_classes
            ):
                raise RuntimeError(
                    "Model returned logits with "
                    "an invalid shape."
                )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )
            confidences, indices = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )

        logits_array = (
            logits.detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        probabilities_array = (
            probabilities.detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        indices_array = (
            indices.detach()
            .cpu()
            .to(torch.int64)
            .numpy()
        )
        confidences_array = (
            confidences.detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

        return BatchPrediction(
            class_names=self.class_names,
            logits=logits_array,
            probabilities=(
                probabilities_array
            ),
            predicted_indices=(
                indices_array
            ),
            predicted_labels=tuple(
                self.class_names[int(index)]
                for index in indices_array
            ),
            confidences=confidences_array,
        )

    def predict_window(
        self,
        inputs: np.ndarray | Tensor,
    ) -> WindowPrediction:
        """Predict one IQ window."""
        if (
            isinstance(inputs, Tensor)
            and inputs.ndim == 3
            and inputs.shape[0] != 1
        ):
            raise ValueError(
                "predict_window accepts exactly "
                "one IQ window."
            )

        if not isinstance(inputs, Tensor):
            array = np.asarray(inputs)

            if (
                array.ndim == 3
                and array.shape[0] != 1
            ):
                raise ValueError(
                    "predict_window accepts "
                    "exactly one IQ window."
                )

        prediction = self.predict_batch(
            inputs
        )

        if len(prediction) != 1:
            raise ValueError(
                "predict_window accepts exactly "
                "one IQ window."
            )

        return prediction.item(0)


__all__ = [
    "BatchPrediction",
    "IQInferenceEngine",
    "WindowPrediction",
    "resolve_device",
]
