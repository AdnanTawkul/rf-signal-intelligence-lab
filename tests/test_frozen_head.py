from __future__ import annotations

import numpy as np
import pytest

from rfsil.models.frozen_head import (
    fit_frozen_linear_head,
)
from rfsil.models.head_refit import (
    compute_linear_logits,
)


def create_multiclass_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    generator = np.random.default_rng(2026)

    centers = np.asarray(
        [
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, -3.0, 0.0],
        ],
        dtype=np.float64,
    )

    train_features = np.concatenate(
        [
            center
            + generator.normal(
                scale=0.25,
                size=(40, 3),
            )
            for center in centers
        ],
        axis=0,
    )
    train_labels = np.repeat(
        np.arange(4),
        40,
    )

    validation_features = np.concatenate(
        [
            center
            + generator.normal(
                scale=0.25,
                size=(20, 3),
            )
            for center in centers
        ],
        axis=0,
    )
    validation_labels = np.repeat(
        np.arange(4),
        20,
    )

    return (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )


def test_fit_returns_deployable_multiclass_parameters() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = create_multiclass_data()

    result = fit_frozen_linear_head(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        regularization_candidates=(0.01, 0.1, 1.0),
    )

    logits = compute_linear_logits(
        validation_features,
        result.parameters,
    )
    predictions = result.classes[
        np.argmax(logits, axis=1)
    ]

    assert result.regularization_c in {
        0.01,
        0.1,
        1.0,
    }
    assert result.parameters.weight.shape == (4, 3)
    assert result.parameters.bias.shape == (4,)
    assert result.validation_accuracy >= 0.95
    assert np.mean(
        predictions == validation_labels
    ) == pytest.approx(
        result.validation_accuracy
    )


def test_fit_supports_binary_sklearn_parameter_shape() -> None:
    generator = np.random.default_rng(2027)

    negative_train = generator.normal(
        loc=-2.0,
        scale=0.3,
        size=(50, 4),
    )
    positive_train = generator.normal(
        loc=2.0,
        scale=0.3,
        size=(50, 4),
    )
    train_features = np.concatenate(
        [negative_train, positive_train],
        axis=0,
    )
    train_labels = np.asarray(
        [3] * 50 + [7] * 50,
        dtype=np.int64,
    )

    negative_validation = generator.normal(
        loc=-2.0,
        scale=0.3,
        size=(20, 4),
    )
    positive_validation = generator.normal(
        loc=2.0,
        scale=0.3,
        size=(20, 4),
    )
    validation_features = np.concatenate(
        [
            negative_validation,
            positive_validation,
        ],
        axis=0,
    )
    validation_labels = np.asarray(
        [3] * 20 + [7] * 20,
        dtype=np.int64,
    )

    result = fit_frozen_linear_head(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        regularization_candidates=(0.1, 1.0),
    )

    assert result.parameters.weight.shape == (2, 4)
    assert result.parameters.bias.shape == (2,)
    np.testing.assert_array_equal(
        result.classes,
        np.asarray([3, 7]),
    )
    assert result.validation_accuracy == pytest.approx(1.0)


def test_fit_rejects_empty_candidates() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = create_multiclass_data()

    with pytest.raises(ValueError):
        fit_frozen_linear_head(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            regularization_candidates=(),
        )


def test_fit_rejects_nonpositive_candidate() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = create_multiclass_data()

    with pytest.raises(ValueError):
        fit_frozen_linear_head(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            regularization_candidates=(0.0, 1.0),
        )


def test_fit_rejects_feature_count_mismatch() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = create_multiclass_data()

    with pytest.raises(ValueError):
        fit_frozen_linear_head(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features[:, :2],
            validation_labels=validation_labels,
            regularization_candidates=(1.0,),
        )


def test_fit_rejects_nonfinite_features() -> None:
    (
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    ) = create_multiclass_data()

    train_features[0, 0] = np.nan

    with pytest.raises(ValueError):
        fit_frozen_linear_head(
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            regularization_candidates=(1.0,),
        )
