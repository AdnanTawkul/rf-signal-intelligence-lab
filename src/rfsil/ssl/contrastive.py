from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rfsil.models.baseline_cnn import BaselineIQCNN


def _validate_positive_integer(
    value: object,
    name: str,
) -> int:
    """Validate and return a strictly positive integer."""
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
class ProjectionHeadConfig:
    """Configuration for the SimCLR projection head."""

    hidden_dimension: int = 256
    output_dimension: int = 128

    def __post_init__(self) -> None:
        """Validate projection-head dimensions."""
        _validate_positive_integer(
            self.hidden_dimension,
            "hidden_dimension",
        )
        _validate_positive_integer(
            self.output_dimension,
            "output_dimension",
        )


class ProjectionHead(nn.Module):
    """Two-layer nonlinear projection head."""

    def __init__(
        self,
        input_dimension: int,
        configuration: ProjectionHeadConfig | None = None,
    ) -> None:
        super().__init__()

        validated_input_dimension = (
            _validate_positive_integer(
                input_dimension,
                "input_dimension",
            )
        )
        selected_configuration = (
            configuration
            if configuration is not None
            else ProjectionHeadConfig()
        )

        self.input_dimension = validated_input_dimension
        self.configuration = selected_configuration

        self.layers = nn.Sequential(
            nn.Linear(
                validated_input_dimension,
                selected_configuration.hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                selected_configuration.hidden_dimension,
                selected_configuration.output_dimension,
            ),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Project encoder embeddings."""
        if features.ndim != 2:
            raise ValueError(
                "features must have shape "
                "[batch, embedding_dimension]."
            )

        if features.shape[1] != self.input_dimension:
            raise ValueError(
                "Feature dimension does not match "
                "the projection-head input dimension."
            )

        return self.layers(features)


class SimCLRModel(nn.Module):
    """CNN encoder with a normalized contrastive projection head."""

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
        self.embedding_dimension = embedding_dimension
        self.projection_head = ProjectionHead(
            input_dimension=embedding_dimension,
            configuration=projection_configuration,
        )

    def encode(self, inputs: Tensor) -> Tensor:
        """Return pooled CNN embeddings."""
        return self.encoder.extract_features(inputs)

    def project(self, features: Tensor) -> Tensor:
        """Return L2-normalized contrastive projections."""
        projections = self.projection_head(features)

        return F.normalize(
            projections,
            p=2.0,
            dim=1,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Encode IQ inputs and return normalized projections."""
        return self.project(
            self.encode(inputs)
        )


def nt_xent_loss(
    first_projections: Tensor,
    second_projections: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    """Compute the symmetric normalized temperature-scaled loss.

    The two input tensors must contain projections for two augmented
    views of the same batch in matching example order.
    """
    if not isinstance(first_projections, Tensor):
        raise TypeError(
            "first_projections must be a torch.Tensor."
        )

    if not isinstance(second_projections, Tensor):
        raise TypeError(
            "second_projections must be a torch.Tensor."
        )

    if not torch.is_floating_point(first_projections):
        raise TypeError(
            "first_projections must use a floating-point dtype."
        )

    if not torch.is_floating_point(second_projections):
        raise TypeError(
            "second_projections must use a floating-point dtype."
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
            "Projection tensors must have identical shapes."
        )

    if first_projections.shape[0] < 2:
        raise ValueError(
            "NT-Xent requires at least two examples."
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
            "Projection tensors must use the same device."
        )

    if (
        first_projections.dtype
        != second_projections.dtype
    ):
        raise ValueError(
            "Projection tensors must use the same dtype."
        )

    if not torch.all(
        torch.isfinite(first_projections)
    ):
        raise ValueError(
            "first_projections must contain finite values."
        )

    if not torch.all(
        torch.isfinite(second_projections)
    ):
        raise ValueError(
            "second_projections must contain finite values."
        )

    selected_temperature = float(temperature)

    if (
        not math.isfinite(selected_temperature)
        or selected_temperature <= 0.0
    ):
        raise ValueError(
            "temperature must be positive and finite."
        )

    first_normalized = F.normalize(
        first_projections,
        p=2.0,
        dim=1,
    )
    second_normalized = F.normalize(
        second_projections,
        p=2.0,
        dim=1,
    )

    batch_size = first_normalized.shape[0]

    representations = torch.cat(
        (
            first_normalized,
            second_normalized,
        ),
        dim=0,
    )

    logits = (
        representations
        @ representations.T
        / selected_temperature
    )

    total_count = 2 * batch_size

    self_similarity_mask = torch.eye(
        total_count,
        device=logits.device,
        dtype=torch.bool,
    )

    logits = logits.masked_fill(
        self_similarity_mask,
        torch.finfo(logits.dtype).min,
    )

    positive_indices = (
        torch.arange(
            total_count,
            device=logits.device,
        )
        + batch_size
    ) % total_count

    return F.cross_entropy(
        logits,
        positive_indices,
    )


__all__ = [
    "ProjectionHead",
    "ProjectionHeadConfig",
    "SimCLRModel",
    "nt_xent_loss",
]
