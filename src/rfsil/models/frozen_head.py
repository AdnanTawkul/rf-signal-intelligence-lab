from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rfsil.models.head_refit import (
    LinearHeadParameters,
    compute_linear_logits,
    convert_standardized_linear_head,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FrozenLinearHeadFit:
    """Result of validation-selected frozen linear-head fitting."""

    regularization_c: float
    validation_accuracy: float
    candidate_accuracies: tuple[tuple[float, float], ...]
    classes: Int64Array
    parameters: LinearHeadParameters


def _validate_feature_label_pair(
    features: NDArray[np.floating],
    labels: NDArray[np.integer],
    name: str,
) -> tuple[Float64Array, Int64Array]:
    """Validate one feature matrix and label vector."""
    feature_array = np.asarray(
        features,
        dtype=np.float64,
    )
    label_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    if feature_array.ndim != 2:
        raise ValueError(
            f"{name}_features must have shape "
            "[examples, features]."
        )

    if label_array.ndim != 1:
        raise ValueError(
            f"{name}_labels must have shape [examples]."
        )

    if feature_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"{name} feature and label counts must match."
        )

    if feature_array.shape[0] == 0:
        raise ValueError(
            f"{name} data must not be empty."
        )

    if feature_array.shape[1] == 0:
        raise ValueError(
            f"{name} features must not be empty."
        )

    if not np.all(np.isfinite(feature_array)):
        raise ValueError(
            f"{name}_features must contain finite values."
        )

    return feature_array, label_array


def _expand_binary_parameters(
    coefficients: Float64Array,
    intercept: Float64Array,
    class_count: int,
) -> tuple[Float64Array, Float64Array]:
    """Expand sklearn's binary parameter representation to two logits."""
    if class_count != 2:
        return coefficients, intercept

    if coefficients.shape[0] == 2:
        return coefficients, intercept

    if coefficients.shape[0] != 1 or intercept.shape != (1,):
        raise ValueError(
            "Unexpected binary logistic-regression parameter shape."
        )

    expanded_coefficients = np.concatenate(
        [
            np.zeros_like(coefficients),
            coefficients,
        ],
        axis=0,
    )
    expanded_intercept = np.asarray(
        [0.0, float(intercept[0])],
        dtype=np.float64,
    )

    return expanded_coefficients, expanded_intercept


def fit_frozen_linear_head(
    train_features: NDArray[np.floating],
    train_labels: NDArray[np.integer],
    validation_features: NDArray[np.floating],
    validation_labels: NDArray[np.integer],
    regularization_candidates: Iterable[float],
    max_iter: int = 5000,
    random_state: int = 2026,
) -> FrozenLinearHeadFit:
    """Fit and select a standardized logistic-regression classifier.

    Each regularization candidate is fitted on the training embeddings.
    Validation accuracy selects the winner. Ties prefer the smaller C,
    corresponding to stronger regularization.
    """
    train_feature_array, train_label_array = (
        _validate_feature_label_pair(
            train_features,
            train_labels,
            "train",
        )
    )
    validation_feature_array, validation_label_array = (
        _validate_feature_label_pair(
            validation_features,
            validation_labels,
            "validation",
        )
    )

    if (
        train_feature_array.shape[1]
        != validation_feature_array.shape[1]
    ):
        raise ValueError(
            "Training and validation feature counts must match."
        )

    if isinstance(max_iter, bool) or not isinstance(
        max_iter,
        Integral,
    ):
        raise ValueError("max_iter must be an integer.")

    if int(max_iter) <= 0:
        raise ValueError("max_iter must be positive.")

    candidates = tuple(
        float(value)
        for value in regularization_candidates
    )

    if not candidates:
        raise ValueError(
            "regularization_candidates must not be empty."
        )

    if len(candidates) != len(set(candidates)):
        raise ValueError(
            "regularization_candidates must be unique."
        )

    for candidate in candidates:
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ValueError(
                "Every regularization candidate must be "
                "positive and finite."
            )

    classes = np.unique(train_label_array)

    if len(classes) < 2:
        raise ValueError(
            "Training labels must contain at least two classes."
        )

    unknown_validation_classes = np.setdiff1d(
        np.unique(validation_label_array),
        classes,
    )

    if unknown_validation_classes.size:
        raise ValueError(
            "Validation labels contain classes absent from training."
        )

    best_pipeline = None
    best_candidate = 0.0
    best_accuracy = -1.0
    candidate_accuracies: list[tuple[float, float]] = []

    for candidate in sorted(candidates):
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=candidate,
                max_iter=int(max_iter),
                random_state=int(random_state),
            ),
        )

        pipeline.fit(
            train_feature_array,
            train_label_array,
        )

        validation_predictions = pipeline.predict(
            validation_feature_array
        )
        validation_accuracy = float(
            np.mean(
                validation_predictions
                == validation_label_array
            )
        )

        candidate_accuracies.append(
            (
                candidate,
                validation_accuracy,
            )
        )

        if validation_accuracy > best_accuracy:
            best_pipeline = pipeline
            best_candidate = candidate
            best_accuracy = validation_accuracy

    if best_pipeline is None:
        raise RuntimeError(
            "No frozen linear-head candidate was fitted."
        )

    scaler = best_pipeline.named_steps["standardscaler"]
    classifier = best_pipeline.named_steps[
        "logisticregression"
    ]

    classifier_classes = np.asarray(
        classifier.classes_,
        dtype=np.int64,
    )
    coefficients = np.asarray(
        classifier.coef_,
        dtype=np.float64,
    )
    intercept = np.asarray(
        classifier.intercept_,
        dtype=np.float64,
    )

    coefficients, intercept = _expand_binary_parameters(
        coefficients=coefficients,
        intercept=intercept,
        class_count=len(classifier_classes),
    )

    parameters = convert_standardized_linear_head(
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=np.asarray(
            scaler.mean_,
            dtype=np.float64,
        ),
        feature_scale=np.asarray(
            scaler.scale_,
            dtype=np.float64,
        ),
    )

    converted_logits = compute_linear_logits(
        validation_feature_array,
        parameters,
    )
    converted_predictions = classifier_classes[
        np.argmax(
            converted_logits,
            axis=1,
        )
    ]

    converted_accuracy = float(
        np.mean(
            converted_predictions
            == validation_label_array
        )
    )

    if not np.isclose(
        converted_accuracy,
        best_accuracy,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Converted linear parameters changed validation accuracy."
        )

    return FrozenLinearHeadFit(
        regularization_c=best_candidate,
        validation_accuracy=best_accuracy,
        candidate_accuracies=tuple(
            candidate_accuracies
        ),
        classes=classifier_classes,
        parameters=parameters,
    )


__all__ = [
    "FrozenLinearHeadFit",
    "fit_frozen_linear_head",
]
