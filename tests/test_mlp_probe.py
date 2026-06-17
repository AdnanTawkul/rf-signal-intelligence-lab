from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rfsil.models.mlp_probe import (
    FrozenMLPProbe,
    FrozenMLPProbeConfig,
    fit_frozen_mlp_probe,
)


def create_features() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    generator = np.random.default_rng(2026)

    train_features = []
    train_labels = []
    validation_features = []
    validation_labels = []

    centers = np.asarray(
        [
            [-2.0, -2.0, 0.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )

    for class_index, center in enumerate(centers):
        train_features.append(
            generator.normal(
                loc=center,
                scale=0.5,
                size=(30, 4),
            ).astype(np.float32)
        )
        validation_features.append(
            generator.normal(
                loc=center,
                scale=0.5,
                size=(10, 4),
            ).astype(np.float32)
        )
        train_labels.extend([class_index] * 30)
        validation_labels.extend([class_index] * 10)

    return (
        np.concatenate(train_features),
        np.asarray(
            train_labels,
            dtype=np.int64,
        ),
        np.concatenate(validation_features),
        np.asarray(
            validation_labels,
            dtype=np.int64,
        ),
    )


def create_configuration() -> FrozenMLPProbeConfig:
    return FrozenMLPProbeConfig(
        hidden_dimension=16,
        dropout=0.0,
        epochs=25,
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=2026,
    )


def test_probe_forward_returns_expected_shape() -> None:
    model = FrozenMLPProbe(
        input_dimension=4,
        num_classes=3,
        feature_mean=torch.zeros(4),
        feature_scale=torch.ones(4),
        configuration=create_configuration(),
    )

    logits = model(torch.randn(5, 4))

    assert logits.shape == (5, 3)


def test_probe_fit_returns_finite_metrics() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = create_features()

    result = fit_frozen_mlp_probe(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        num_classes=3,
        configuration=create_configuration(),
        device=torch.device("cpu"),
    )

    assert 1 <= result.best_epoch <= 25
    assert math.isfinite(result.training_accuracy)
    assert math.isfinite(result.validation_accuracy)
    assert math.isfinite(result.best_validation_loss)
    assert 0.0 <= result.training_accuracy <= 1.0
    assert 0.0 <= result.validation_accuracy <= 1.0
    assert len(result.history) == 25


def test_probe_fit_is_reproducible() -> None:
    data = create_features()

    first = fit_frozen_mlp_probe(
        train_features=data[0],
        train_labels=data[1],
        validation_features=data[2],
        validation_labels=data[3],
        num_classes=3,
        configuration=create_configuration(),
        device=torch.device("cpu"),
    )
    second = fit_frozen_mlp_probe(
        train_features=data[0],
        train_labels=data[1],
        validation_features=data[2],
        validation_labels=data[3],
        num_classes=3,
        configuration=create_configuration(),
        device=torch.device("cpu"),
    )

    assert first.best_epoch == second.best_epoch
    assert (
        first.validation_accuracy
        == second.validation_accuracy
    )
    assert (
        first.training_accuracy
        == second.training_accuracy
    )


def test_probe_uses_training_statistics() -> None:
    data = create_features()

    result = fit_frozen_mlp_probe(
        train_features=data[0],
        train_labels=data[1],
        validation_features=data[2],
        validation_labels=data[3],
        num_classes=3,
        configuration=create_configuration(),
        device=torch.device("cpu"),
    )

    expected_mean = data[0].mean(axis=0)
    expected_scale = data[0].std(axis=0)
    expected_scale = np.where(
        expected_scale < 1e-6,
        1.0,
        expected_scale,
    )

    np.testing.assert_allclose(
        result.model.feature_mean.numpy()[0],
        expected_mean,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.model.feature_scale.numpy()[0],
        expected_scale,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {"hidden_dimension": 0},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
    ],
)
def test_invalid_configuration_is_rejected(
    keyword_arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        FrozenMLPProbeConfig(
            **keyword_arguments,
        )


@pytest.mark.parametrize(
    (
        "train_features",
        "train_labels",
        "validation_features",
        "validation_labels",
        "expected_exception",
    ),
    [
        (
            np.ones(4, dtype=np.float32),
            np.asarray([0, 1, 0, 1]),
            np.ones((4, 2), dtype=np.float32),
            np.asarray([0, 1, 0, 1]),
            ValueError,
        ),
        (
            np.ones((4, 2), dtype=np.int64),
            np.asarray([0, 1, 0, 1]),
            np.ones((4, 2), dtype=np.float32),
            np.asarray([0, 1, 0, 1]),
            TypeError,
        ),
        (
            np.ones((4, 2), dtype=np.float32),
            np.asarray([0, 1, 0]),
            np.ones((4, 2), dtype=np.float32),
            np.asarray([0, 1, 0, 1]),
            ValueError,
        ),
        (
            np.ones((4, 2), dtype=np.float32),
            np.asarray([0, 1, 2, 3]),
            np.ones((4, 3), dtype=np.float32),
            np.asarray([0, 1, 2, 0]),
            ValueError,
        ),
    ],
)
def test_invalid_fit_inputs_are_rejected(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        fit_frozen_mlp_probe(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            num_classes=3,
            configuration=create_configuration(),
            device=torch.device("cpu"),
        )
