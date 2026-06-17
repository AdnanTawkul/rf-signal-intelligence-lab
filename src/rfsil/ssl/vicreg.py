from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rfsil.models.baseline_cnn import BaselineIQCNN
from rfsil.ssl.contrastive import (
    ProjectionHead,
    ProjectionHeadConfig,
)


@dataclass(frozen=True, slots=True)
class VICRegLossConfig:
    """Configuration for the VICReg objective."""

    invariance_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0
    target_standard_deviation: float = 1.0
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        """Validate VICReg objective settings."""
        weights = (
            (
                "invariance_weight",
                self.invariance_weight,
            ),
            (
                "variance_weight",
                self.variance_weight,
            ),
            (
                "covariance_weight",
                self.covariance_weight,
            ),
        )

        for name, value in weights:
            if (
                not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{name} must be nonnegative and finite."
                )

        if not any(
            value > 0.0
            for _, value in weights
        ):
            raise ValueError(
                "At least one VICReg loss weight "
                "must be positive."
            )

        if (
            not math.isfinite(
                self.target_standard_deviation
            )
            or self.target_standard_deviation <= 0.0
        ):
            raise ValueError(
                "target_standard_deviation must be "
                "positive and finite."
            )

        if (
            not math.isfinite(self.epsilon)
            or self.epsilon <= 0.0
        ):
            raise ValueError(
                "epsilon must be positive and finite."
            )


@dataclass(frozen=True, slots=True)
class VICRegLossTerms:
    """Individual and total VICReg loss terms."""

    total: Tensor
    invariance: Tensor
    variance: Tensor
    covariance: Tensor


class VICRegModel(nn.Module):
    """CNN encoder with an unnormalized VICReg projector."""

    def __init__(
        self,
        encoder: BaselineIQCNN,
        projection_configuration: (
            ProjectionHeadConfig | None
        ) = None,
    ) -> None:
        super().__init__()

        if not isinstance(encoder, BaselineIQCNN):
            raise TypeError(
                "encoder must be a BaselineIQCNN."
            )

        embedding_dimension = int(
            encoder.configuration.channels[-1]
        )

        self.encoder = encoder
        self.embedding_dimension = (
            embedding_dimension
        )
        self.projection_head = ProjectionHead(
            input_dimension=embedding_dimension,
            configuration=projection_configuration,
        )

    def encode(self, inputs: Tensor) -> Tensor:
        """Return pooled CNN embeddings."""
        return self.encoder.extract_features(inputs)

    def project(self, features: Tensor) -> Tensor:
        """Return unnormalized VICReg projections."""
        return self.projection_head(features)

    def forward(self, inputs: Tensor) -> Tensor:
        """Encode IQ data and return raw projections."""
        return self.project(
            self.encode(inputs)
        )


def _validate_projection_pair(
    first_projections: Tensor,
    second_projections: Tensor,
) -> None:
    """Validate a pair of VICReg projection tensors."""
    if not isinstance(first_projections, Tensor):
        raise TypeError(
            "first_projections must be a torch.Tensor."
        )

    if not isinstance(second_projections, Tensor):
        raise TypeError(
            "second_projections must be a torch.Tensor."
        )

    if not torch.is_floating_point(
        first_projections
    ):
        raise TypeError(
            "first_projections must use a "
            "floating-point dtype."
        )

    if not torch.is_floating_point(
        second_projections
    ):
        raise TypeError(
            "second_projections must use a "
            "floating-point dtype."
        )

    if first_projections.ndim != 2:
        raise ValueError(
            "first_projections must have shape "
            "[batch, projection_dimension]."
        )

    if second_projections.ndim != 2:
        raise ValueError(
            "second_projections must have shape "
            "[batch, projection_dimension]."
        )

    if (
        first_projections.shape
        != second_projections.shape
    ):
        raise ValueError(
            "Projection tensors must have "
            "identical shapes."
        )

    if first_projections.shape[0] < 2:
        raise ValueError(
            "VICReg requires at least two examples."
        )

    if first_projections.shape[1] == 0:
        raise ValueError(
            "Projection dimension must be positive."
        )

    if (
        first_projections.device
        != second_projections.device
    ):
        raise ValueError(
            "Projection tensors must use "
            "the same device."
        )

    if (
        first_projections.dtype
        != second_projections.dtype
    ):
        raise ValueError(
            "Projection tensors must use "
            "the same dtype."
        )

    if not torch.all(
        torch.isfinite(first_projections)
    ):
        raise ValueError(
            "first_projections must contain "
            "only finite values."
        )

    if not torch.all(
        torch.isfinite(second_projections)
    ):
        raise ValueError(
            "second_projections must contain "
            "only finite values."
        )


def _variance_loss(
    projections: Tensor,
    configuration: VICRegLossConfig,
) -> Tensor:
    """Penalize dimensions with insufficient spread."""
    standard_deviation = torch.sqrt(
        projections.var(
            dim=0,
            unbiased=False,
        )
        + configuration.epsilon
    )

    return F.relu(
        configuration.target_standard_deviation
        - standard_deviation
    ).mean()


def _off_diagonal(
    matrix: Tensor,
) -> Tensor:
    """Return the flattened off-diagonal entries."""
    dimension = matrix.shape[0]

    if matrix.ndim != 2:
        raise ValueError(
            "matrix must be two-dimensional."
        )

    if matrix.shape[1] != dimension:
        raise ValueError(
            "matrix must be square."
        )

    return matrix.flatten()[:-1].view(
        dimension - 1,
        dimension + 1,
    )[:, 1:].flatten()


def _covariance_loss(
    projections: Tensor,
) -> Tensor:
    """Penalize covariance between projection dimensions."""
    batch_size = projections.shape[0]
    projection_dimension = projections.shape[1]

    centered = (
        projections
        - projections.mean(
            dim=0,
            keepdim=True,
        )
    )

    covariance_matrix = (
        centered.T @ centered
    ) / (batch_size - 1)

    return (
        _off_diagonal(
            covariance_matrix
        ).square().sum()
        / projection_dimension
    )


def compute_vicreg_loss(
    first_projections: Tensor,
    second_projections: Tensor,
    configuration: VICRegLossConfig | None = None,
) -> VICRegLossTerms:
    """Compute all VICReg objective terms."""
    _validate_projection_pair(
        first_projections,
        second_projections,
    )

    selected_configuration = (
        configuration
        if configuration is not None
        else VICRegLossConfig()
    )

    invariance = F.mse_loss(
        first_projections,
        second_projections,
    )

    variance = (
        _variance_loss(
            first_projections,
            selected_configuration,
        )
        + _variance_loss(
            second_projections,
            selected_configuration,
        )
    )

    covariance = (
        _covariance_loss(
            first_projections
        )
        + _covariance_loss(
            second_projections
        )
    )

    total = (
        selected_configuration.invariance_weight
        * invariance
        + selected_configuration.variance_weight
        * variance
        + selected_configuration.covariance_weight
        * covariance
    )

    if not torch.isfinite(total):
        raise RuntimeError(
            "VICReg loss became non-finite."
        )

    return VICRegLossTerms(
        total=total,
        invariance=invariance,
        variance=variance,
        covariance=covariance,
    )


__all__ = [
    "VICRegLossConfig",
    "VICRegLossTerms",
    "VICRegModel",
    "compute_vicreg_loss",
]
