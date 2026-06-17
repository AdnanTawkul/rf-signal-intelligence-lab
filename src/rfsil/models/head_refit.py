from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from rfsil.models.baseline_cnn import BaselineIQCNN

Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LinearHeadParameters:
    """Native linear-head parameters operating on unscaled embeddings."""

    weight: Float64Array
    bias: Float64Array


def convert_standardized_linear_head(
    coefficients: NDArray[np.floating],
    intercept: NDArray[np.floating],
    feature_mean: NDArray[np.floating],
    feature_scale: NDArray[np.floating],
) -> LinearHeadParameters:
    """Convert a standardized linear classifier to raw-feature parameters.

    A classifier trained on:

        z = (x - mean) / scale

    is converted so the returned parameters operate directly on x.
    """
    coefficient_array = np.asarray(
        coefficients,
        dtype=np.float64,
    )
    intercept_array = np.asarray(
        intercept,
        dtype=np.float64,
    )
    mean_array = np.asarray(
        feature_mean,
        dtype=np.float64,
    )
    scale_array = np.asarray(
        feature_scale,
        dtype=np.float64,
    )

    if coefficient_array.ndim != 2:
        raise ValueError(
            "coefficients must have shape [classes, features]."
        )

    if intercept_array.ndim != 1:
        raise ValueError(
            "intercept must have shape [classes]."
        )

    if mean_array.ndim != 1:
        raise ValueError(
            "feature_mean must have shape [features]."
        )

    if scale_array.ndim != 1:
        raise ValueError(
            "feature_scale must have shape [features]."
        )

    class_count, feature_count = coefficient_array.shape

    if intercept_array.shape != (class_count,):
        raise ValueError(
            "intercept length must match the number of classes."
        )

    if mean_array.shape != (feature_count,):
        raise ValueError(
            "feature_mean length must match the feature count."
        )

    if scale_array.shape != (feature_count,):
        raise ValueError(
            "feature_scale length must match the feature count."
        )

    for name, array in (
        ("coefficients", coefficient_array),
        ("intercept", intercept_array),
        ("feature_mean", mean_array),
        ("feature_scale", scale_array),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must contain only finite values."
            )

    if np.any(scale_array <= 0.0):
        raise ValueError(
            "feature_scale values must be strictly positive."
        )

    weight = coefficient_array / scale_array[np.newaxis, :]
    bias = intercept_array - weight @ mean_array

    return LinearHeadParameters(
        weight=weight.astype(
            np.float64,
            copy=False,
        ),
        bias=bias.astype(
            np.float64,
            copy=False,
        ),
    )


def compute_linear_logits(
    features: NDArray[np.floating],
    parameters: LinearHeadParameters,
) -> Float64Array:
    """Compute logits using converted raw-feature parameters."""
    feature_array = np.asarray(
        features,
        dtype=np.float64,
    )

    if feature_array.ndim != 2:
        raise ValueError(
            "features must have shape [examples, features]."
        )

    if feature_array.shape[1] != parameters.weight.shape[1]:
        raise ValueError(
            "Feature count does not match the linear-head weight."
        )

    return (
        feature_array @ parameters.weight.T
        + parameters.bias[np.newaxis, :]
    )


def apply_linear_head_parameters(
    model: BaselineIQCNN,
    parameters: LinearHeadParameters,
) -> None:
    """Replace the model's final linear classifier parameters."""
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError(
            "model.classifier must be an nn.Sequential module."
        )

    linear_layer = model.classifier[-1]

    if not isinstance(linear_layer, nn.Linear):
        raise TypeError(
            "The final classifier module must be nn.Linear."
        )

    if linear_layer.bias is None:
        raise ValueError(
            "The final classifier must include a bias parameter."
        )

    expected_weight_shape = tuple(
        linear_layer.weight.shape
    )
    expected_bias_shape = tuple(
        linear_layer.bias.shape
    )

    if parameters.weight.shape != expected_weight_shape:
        raise ValueError(
            "Converted weight shape does not match the model head: "
            f"expected {expected_weight_shape}, "
            f"received {parameters.weight.shape}."
        )

    if parameters.bias.shape != expected_bias_shape:
        raise ValueError(
            "Converted bias shape does not match the model head: "
            f"expected {expected_bias_shape}, "
            f"received {parameters.bias.shape}."
        )

    with torch.no_grad():
        linear_layer.weight.copy_(
            torch.as_tensor(
                parameters.weight,
                device=linear_layer.weight.device,
                dtype=linear_layer.weight.dtype,
            )
        )
        linear_layer.bias.copy_(
            torch.as_tensor(
                parameters.bias,
                device=linear_layer.bias.device,
                dtype=linear_layer.bias.dtype,
            )
        )


__all__ = [
    "LinearHeadParameters",
    "apply_linear_head_parameters",
    "compute_linear_logits",
    "convert_standardized_linear_head",
]
