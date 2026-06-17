from __future__ import annotations

import math
import random
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


def _validate_positive_integer(
    value: object,
    name: str,
) -> int:
    """Validate and return a positive integer."""
    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise ValueError(f"{name} must be an integer.")

    validated_value = int(value)

    if validated_value <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated_value


@dataclass(frozen=True, slots=True)
class FrozenMLPProbeConfig:
    """Training configuration for a frozen-embedding MLP probe."""

    hidden_dimension: int = 128
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 2026

    def __post_init__(self) -> None:
        """Validate probe configuration."""
        _validate_positive_integer(
            self.hidden_dimension,
            "hidden_dimension",
        )
        _validate_positive_integer(
            self.epochs,
            "epochs",
        )
        _validate_positive_integer(
            self.batch_size,
            "batch_size",
        )

        if (
            not math.isfinite(self.dropout)
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError(
                "dropout must be finite and in [0, 1)."
            )

        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError(
                "weight_decay must be nonnegative and finite."
            )

        if isinstance(self.seed, bool) or not isinstance(
            self.seed,
            Integral,
        ):
            raise ValueError("seed must be an integer.")


@dataclass(frozen=True, slots=True)
class FrozenMLPProbeFit:
    """Result from fitting a frozen-embedding MLP probe."""

    model: FrozenMLPProbe
    best_epoch: int
    training_accuracy: float
    validation_accuracy: float
    best_validation_loss: float
    history: tuple[dict[str, float | int], ...]


class FrozenMLPProbe(nn.Module):
    """Standardized nonlinear classifier for frozen embeddings."""

    def __init__(
        self,
        input_dimension: int,
        num_classes: int,
        feature_mean: Tensor,
        feature_scale: Tensor,
        configuration: FrozenMLPProbeConfig | None = None,
    ) -> None:
        super().__init__()

        self.input_dimension = _validate_positive_integer(
            input_dimension,
            "input_dimension",
        )
        self.num_classes = _validate_positive_integer(
            num_classes,
            "num_classes",
        )

        if self.num_classes < 2:
            raise ValueError(
                "num_classes must be at least two."
            )

        selected_configuration = (
            configuration
            if configuration is not None
            else FrozenMLPProbeConfig()
        )

        if feature_mean.ndim != 1:
            raise ValueError(
                "feature_mean must be one-dimensional."
            )

        if feature_scale.ndim != 1:
            raise ValueError(
                "feature_scale must be one-dimensional."
            )

        if feature_mean.shape != feature_scale.shape:
            raise ValueError(
                "feature_mean and feature_scale must "
                "have identical shapes."
            )

        if feature_mean.shape[0] != self.input_dimension:
            raise ValueError(
                "Feature statistics do not match "
                "input_dimension."
            )

        if not torch.all(torch.isfinite(feature_mean)):
            raise ValueError(
                "feature_mean must contain finite values."
            )

        if not torch.all(torch.isfinite(feature_scale)):
            raise ValueError(
                "feature_scale must contain finite values."
            )

        if torch.any(feature_scale <= 0.0):
            raise ValueError(
                "feature_scale must contain positive values."
            )

        self.configuration = selected_configuration

        self.register_buffer(
            "feature_mean",
            feature_mean.to(
                dtype=torch.float32,
            ).reshape(1, -1),
        )
        self.register_buffer(
            "feature_scale",
            feature_scale.to(
                dtype=torch.float32,
            ).reshape(1, -1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                self.input_dimension,
                selected_configuration.hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(
                p=selected_configuration.dropout,
            ),
            nn.Linear(
                selected_configuration.hidden_dimension,
                self.num_classes,
            ),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return class logits for frozen embeddings."""
        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, embedding_dimension]."
            )

        if features.shape[1] != self.input_dimension:
            raise ValueError(
                "Feature dimension does not match "
                "the probe input dimension."
            )

        standardized = (
            features - self.feature_mean
        ) / self.feature_scale

        return self.classifier(standardized)


def _validate_feature_matrix(
    values: object,
    name: str,
) -> Float32Array:
    """Validate and convert a feature matrix."""
    array = np.asarray(values)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional matrix."
        )

    if array.shape[0] == 0:
        raise ValueError(
            f"{name} must contain at least one example."
        )

    if array.shape[1] == 0:
        raise ValueError(
            f"{name} must contain at least one feature."
        )

    if not np.issubdtype(
        array.dtype,
        np.floating,
    ):
        raise TypeError(
            f"{name} must use a floating-point dtype."
        )

    converted = np.asarray(
        array,
        dtype=np.float32,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return converted


def _validate_labels(
    values: object,
    example_count: int,
    num_classes: int,
    name: str,
) -> Int64Array:
    """Validate and convert class labels."""
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if array.shape[0] != example_count:
        raise ValueError(
            f"{name} length must match the feature count."
        )

    if not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise TypeError(
            f"{name} must use an integer dtype."
        )

    converted = np.asarray(
        array,
        dtype=np.int64,
    )

    if np.any(converted < 0):
        raise ValueError(
            f"{name} must not contain negative labels."
        )

    if np.any(converted >= num_classes):
        raise ValueError(
            f"{name} contains a label outside "
            "the configured class range."
        )

    return converted


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_loader(
    features: Float32Array,
    labels: Int64Array,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic embedding DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def _run_epoch(
    model: FrozenMLPProbe,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch."""
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    loss_function = nn.CrossEntropyLoss()

    accumulated_loss = 0.0
    correct_count = 0
    example_count = 0

    context = (
        torch.enable_grad()
        if training
        else torch.inference_mode()
    )

    with context:
        for features, labels in data_loader:
            features = features.to(
                device=device,
                dtype=torch.float32,
            )
            labels = labels.to(
                device=device,
                dtype=torch.int64,
            )

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = loss_function(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "MLP probe loss became non-finite."
                )

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = int(labels.shape[0])

            accumulated_loss += (
                float(loss.detach().item())
                * batch_size
            )
            correct_count += int(
                torch.count_nonzero(
                    torch.argmax(logits, dim=1)
                    == labels
                ).item()
            )
            example_count += batch_size

    if example_count == 0:
        raise ValueError(
            "MLP probe DataLoader produced no examples."
        )

    return (
        accumulated_loss / example_count,
        correct_count / example_count,
    )


def _state_dict_to_cpu(
    model: nn.Module,
) -> dict[str, Tensor]:
    """Copy a model state dictionary to CPU."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def fit_frozen_mlp_probe(
    train_features: object,
    train_labels: object,
    validation_features: object,
    validation_labels: object,
    num_classes: int,
    configuration: FrozenMLPProbeConfig | None = None,
    device: torch.device | None = None,
) -> FrozenMLPProbeFit:
    """Fit an MLP classifier on frozen encoder embeddings."""
    validated_num_classes = _validate_positive_integer(
        num_classes,
        "num_classes",
    )

    if validated_num_classes < 2:
        raise ValueError(
            "num_classes must be at least two."
        )

    selected_configuration = (
        configuration
        if configuration is not None
        else FrozenMLPProbeConfig()
    )

    train_feature_array = _validate_feature_matrix(
        train_features,
        "train_features",
    )
    validation_feature_array = (
        _validate_feature_matrix(
            validation_features,
            "validation_features",
        )
    )

    if (
        train_feature_array.shape[1]
        != validation_feature_array.shape[1]
    ):
        raise ValueError(
            "Training and validation feature dimensions "
            "must match."
        )

    train_label_array = _validate_labels(
        train_labels,
        train_feature_array.shape[0],
        validated_num_classes,
        "train_labels",
    )
    validation_label_array = _validate_labels(
        validation_labels,
        validation_feature_array.shape[0],
        validated_num_classes,
        "validation_labels",
    )

    observed_classes = np.unique(
        train_label_array
    )

    expected_classes = np.arange(
        validated_num_classes,
        dtype=np.int64,
    )

    if not np.array_equal(
        observed_classes,
        expected_classes,
    ):
        raise ValueError(
            "Training labels must contain every configured class."
        )

    feature_mean = train_feature_array.mean(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    feature_scale = train_feature_array.std(
        axis=0,
        dtype=np.float64,
    ).astype(np.float32)

    feature_scale = np.where(
        feature_scale < 1e-6,
        np.float32(1.0),
        feature_scale,
    ).astype(np.float32)

    _set_seed(selected_configuration.seed)

    selected_device = (
        device
        if device is not None
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    model = FrozenMLPProbe(
        input_dimension=train_feature_array.shape[1],
        num_classes=validated_num_classes,
        feature_mean=torch.from_numpy(feature_mean),
        feature_scale=torch.from_numpy(feature_scale),
        configuration=selected_configuration,
    ).to(selected_device)

    optimizer = AdamW(
        model.parameters(),
        lr=selected_configuration.learning_rate,
        weight_decay=(
            selected_configuration.weight_decay
        ),
    )

    train_loader = _create_loader(
        train_feature_array,
        train_label_array,
        selected_configuration.batch_size,
        shuffle=True,
        seed=selected_configuration.seed,
    )
    validation_loader = _create_loader(
        validation_feature_array,
        validation_label_array,
        selected_configuration.batch_size,
        shuffle=False,
        seed=selected_configuration.seed,
    )

    history: list[dict[str, float | int]] = []

    best_epoch = 0
    best_validation_accuracy = -1.0
    best_validation_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch in range(
        1,
        selected_configuration.epochs + 1,
    ):
        training_loss, training_accuracy = (
            _run_epoch(
                model,
                train_loader,
                selected_device,
                optimizer,
            )
        )
        (
            validation_loss,
            validation_accuracy,
        ) = _run_epoch(
            model,
            validation_loader,
            selected_device,
            optimizer=None,
        )

        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "training_accuracy": training_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": (
                    validation_accuracy
                ),
            }
        )

        improved_accuracy = (
            validation_accuracy
            > best_validation_accuracy
        )
        equal_accuracy_lower_loss = (
            math.isclose(
                validation_accuracy,
                best_validation_accuracy,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and validation_loss
            < best_validation_loss
        )

        if (
            improved_accuracy
            or equal_accuracy_lower_loss
        ):
            best_epoch = epoch
            best_validation_accuracy = (
                validation_accuracy
            )
            best_validation_loss = validation_loss
            best_state = _state_dict_to_cpu(model)

    if best_state is None:
        raise RuntimeError(
            "MLP probe training produced no checkpoint."
        )

    model.load_state_dict(best_state)
    model.to(selected_device)

    _, final_training_accuracy = _run_epoch(
        model,
        train_loader,
        selected_device,
        optimizer=None,
    )
    (
        final_validation_loss,
        final_validation_accuracy,
    ) = _run_epoch(
        model,
        validation_loader,
        selected_device,
        optimizer=None,
    )

    if not math.isclose(
        final_validation_accuracy,
        best_validation_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Restored MLP checkpoint does not match "
            "the selected validation accuracy."
        )

    model.to(torch.device("cpu"))
    model.eval()

    return FrozenMLPProbeFit(
        model=model,
        best_epoch=best_epoch,
        training_accuracy=final_training_accuracy,
        validation_accuracy=(
            final_validation_accuracy
        ),
        best_validation_loss=(
            final_validation_loss
        ),
        history=tuple(history),
    )


__all__ = [
    "FrozenMLPProbe",
    "FrozenMLPProbeConfig",
    "FrozenMLPProbeFit",
    "fit_frozen_mlp_probe",
]
