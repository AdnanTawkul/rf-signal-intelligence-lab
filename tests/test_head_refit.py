from __future__ import annotations

import numpy as np
import pytest
import torch

from rfsil.models.baseline_cnn import BaselineIQCNN
from rfsil.models.head_refit import (
    LinearHeadParameters,
    apply_linear_head_parameters,
    compute_linear_logits,
    convert_standardized_linear_head,
)


def test_conversion_matches_standardized_logits() -> None:
    generator = np.random.default_rng(2026)

    features = generator.normal(
        size=(20, 8),
    )
    coefficients = generator.normal(
        size=(4, 8),
    )
    intercept = generator.normal(
        size=4,
    )
    feature_mean = generator.normal(
        size=8,
    )
    feature_scale = generator.uniform(
        low=0.2,
        high=3.0,
        size=8,
    )

    standardized_features = (
        features - feature_mean
    ) / feature_scale

    expected = (
        standardized_features @ coefficients.T
        + intercept
    )

    parameters = convert_standardized_linear_head(
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )

    actual = compute_linear_logits(
        features,
        parameters,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=1e-10,
    )


def test_identity_scaling_preserves_parameters() -> None:
    coefficients = np.asarray(
        [
            [1.0, 2.0],
            [-3.0, 4.0],
        ]
    )
    intercept = np.asarray([0.5, -0.25])

    parameters = convert_standardized_linear_head(
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=np.zeros(2),
        feature_scale=np.ones(2),
    )

    np.testing.assert_allclose(
        parameters.weight,
        coefficients,
    )
    np.testing.assert_allclose(
        parameters.bias,
        intercept,
    )


@pytest.mark.parametrize(
    (
        "coefficients",
        "intercept",
        "feature_mean",
        "feature_scale",
    ),
    [
        (
            np.ones((4, 8)),
            np.ones(3),
            np.ones(8),
            np.ones(8),
        ),
        (
            np.ones((4, 8)),
            np.ones(4),
            np.ones(7),
            np.ones(8),
        ),
        (
            np.ones((4, 8)),
            np.ones(4),
            np.ones(8),
            np.ones(7),
        ),
    ],
)
def test_conversion_rejects_shape_mismatches(
    coefficients: np.ndarray,
    intercept: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        convert_standardized_linear_head(
            coefficients=coefficients,
            intercept=intercept,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )


def test_conversion_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError):
        convert_standardized_linear_head(
            coefficients=np.ones((4, 8)),
            intercept=np.zeros(4),
            feature_mean=np.zeros(8),
            feature_scale=np.asarray(
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
            ),
        )


def test_conversion_rejects_nonfinite_values() -> None:
    coefficients = np.ones((4, 8))
    coefficients[0, 0] = np.nan

    with pytest.raises(ValueError):
        convert_standardized_linear_head(
            coefficients=coefficients,
            intercept=np.zeros(4),
            feature_mean=np.zeros(8),
            feature_scale=np.ones(8),
        )


def test_apply_updates_model_classifier() -> None:
    generator = np.random.default_rng(2026)

    weight = generator.normal(
        size=(4, 128),
    )
    bias = generator.normal(
        size=4,
    )

    parameters = LinearHeadParameters(
        weight=weight,
        bias=bias,
    )

    model = BaselineIQCNN()
    model.eval()

    apply_linear_head_parameters(
        model,
        parameters,
    )

    features = torch.randn(5, 128)

    with torch.inference_mode():
        actual = model.classifier(features)

    expected = torch.from_numpy(
        compute_linear_logits(
            features.numpy(),
            parameters,
        )
    ).to(dtype=actual.dtype)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


def test_apply_rejects_incompatible_head_shape() -> None:
    model = BaselineIQCNN()

    parameters = LinearHeadParameters(
        weight=np.ones((3, 128)),
        bias=np.zeros(3),
    )

    with pytest.raises(ValueError):
        apply_linear_head_parameters(
            model,
            parameters,
        )


def test_compute_logits_rejects_bad_feature_shape() -> None:
    parameters = LinearHeadParameters(
        weight=np.ones((4, 8)),
        bias=np.zeros(4),
    )

    with pytest.raises(ValueError):
        compute_linear_logits(
            np.ones((3, 7)),
            parameters,
        )
