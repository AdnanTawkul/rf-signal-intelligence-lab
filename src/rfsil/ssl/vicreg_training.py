from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rfsil.ssl.augmentations import RandomIQAugmentation
from rfsil.ssl.vicreg import (
    VICRegLossConfig,
    VICRegModel,
    compute_vicreg_loss,
)


@dataclass(frozen=True, slots=True)
class VICRegEpochMetrics:
    """Aggregate metrics from one VICReg epoch."""

    total_loss: float
    invariance_loss: float
    variance_loss: float
    covariance_loss: float
    positive_cosine_similarity: float
    projection_standard_deviation: float
    example_count: int


def _move_iq_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> Tensor:
    """Move one IQ batch to the selected device."""
    if "iq" not in batch:
        raise KeyError(
            "VICReg batch must contain an 'iq' tensor."
        )

    inputs = batch["iq"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    if inputs.ndim != 3:
        raise ValueError(
            "Batch IQ tensor must have shape "
            "[batch, channels, samples]."
        )

    if inputs.shape[1] != 2:
        raise ValueError(
            "Batch IQ tensor must contain exactly "
            "two channels."
        )

    if inputs.shape[0] < 2:
        raise ValueError(
            "VICReg batches must contain at least "
            "two examples."
        )

    if inputs.shape[2] == 0:
        raise ValueError(
            "Batch IQ tensor must contain at least "
            "one sample."
        )

    if not torch.all(torch.isfinite(inputs)):
        raise ValueError(
            "Batch IQ tensor must contain only "
            "finite values."
        )

    return inputs


def _batch_diagnostics(
    first_projections: Tensor,
    second_projections: Tensor,
) -> tuple[float, float]:
    """Compute pair similarity and representation spread."""
    positive_similarity = F.cosine_similarity(
        first_projections,
        second_projections,
        dim=1,
    ).mean()

    combined_projections = torch.cat(
        (
            first_projections,
            second_projections,
        ),
        dim=0,
    )

    projection_standard_deviation = (
        combined_projections.std(
            dim=0,
            unbiased=False,
        ).mean()
    )

    if not torch.isfinite(positive_similarity):
        raise RuntimeError(
            "Positive-pair cosine similarity "
            "became non-finite."
        )

    if not torch.isfinite(
        projection_standard_deviation
    ):
        raise RuntimeError(
            "Projection standard deviation "
            "became non-finite."
        )

    return (
        float(positive_similarity.item()),
        float(
            projection_standard_deviation.item()
        ),
    )


def _finalize_metrics(
    accumulated_total_loss: float,
    accumulated_invariance_loss: float,
    accumulated_variance_loss: float,
    accumulated_covariance_loss: float,
    accumulated_positive_similarity: float,
    accumulated_projection_standard_deviation: float,
    example_count: int,
) -> VICRegEpochMetrics:
    """Create aggregate VICReg epoch metrics."""
    if example_count == 0:
        raise ValueError(
            "VICReg DataLoader produced no examples."
        )

    return VICRegEpochMetrics(
        total_loss=(
            accumulated_total_loss
            / example_count
        ),
        invariance_loss=(
            accumulated_invariance_loss
            / example_count
        ),
        variance_loss=(
            accumulated_variance_loss
            / example_count
        ),
        covariance_loss=(
            accumulated_covariance_loss
            / example_count
        ),
        positive_cosine_similarity=(
            accumulated_positive_similarity
            / example_count
        ),
        projection_standard_deviation=(
            accumulated_projection_standard_deviation
            / example_count
        ),
        example_count=example_count,
    )


def run_vicreg_training_epoch(
    model: VICRegModel,
    data_loader: DataLoader,
    optimizer: Optimizer,
    augmentation: RandomIQAugmentation,
    device: torch.device,
    loss_configuration: VICRegLossConfig | None = None,
    generator: torch.Generator | None = None,
) -> VICRegEpochMetrics:
    """Train a VICReg model for one complete epoch."""
    model.train()
    augmentation.train()

    accumulated_total_loss = 0.0
    accumulated_invariance_loss = 0.0
    accumulated_variance_loss = 0.0
    accumulated_covariance_loss = 0.0
    accumulated_positive_similarity = 0.0
    accumulated_projection_standard_deviation = 0.0
    example_count = 0

    for batch in data_loader:
        inputs = _move_iq_batch(
            batch,
            device,
        )

        first_view, second_view = (
            augmentation.create_views(
                inputs,
                generator=generator,
            )
        )

        optimizer.zero_grad(set_to_none=True)

        first_projections = model(first_view)
        second_projections = model(second_view)

        loss_terms = compute_vicreg_loss(
            first_projections,
            second_projections,
            configuration=loss_configuration,
        )

        if not torch.isfinite(loss_terms.total):
            raise RuntimeError(
                "VICReg training loss became non-finite."
            )

        loss_terms.total.backward()
        optimizer.step()

        batch_size = int(inputs.shape[0])

        (
            positive_similarity,
            projection_standard_deviation,
        ) = _batch_diagnostics(
            first_projections.detach(),
            second_projections.detach(),
        )

        accumulated_total_loss += (
            float(loss_terms.total.detach().item())
            * batch_size
        )
        accumulated_invariance_loss += (
            float(
                loss_terms.invariance.detach().item()
            )
            * batch_size
        )
        accumulated_variance_loss += (
            float(
                loss_terms.variance.detach().item()
            )
            * batch_size
        )
        accumulated_covariance_loss += (
            float(
                loss_terms.covariance.detach().item()
            )
            * batch_size
        )
        accumulated_positive_similarity += (
            positive_similarity
            * batch_size
        )
        accumulated_projection_standard_deviation += (
            projection_standard_deviation
            * batch_size
        )
        example_count += batch_size

    return _finalize_metrics(
        accumulated_total_loss=(
            accumulated_total_loss
        ),
        accumulated_invariance_loss=(
            accumulated_invariance_loss
        ),
        accumulated_variance_loss=(
            accumulated_variance_loss
        ),
        accumulated_covariance_loss=(
            accumulated_covariance_loss
        ),
        accumulated_positive_similarity=(
            accumulated_positive_similarity
        ),
        accumulated_projection_standard_deviation=(
            accumulated_projection_standard_deviation
        ),
        example_count=example_count,
    )


def run_vicreg_evaluation_epoch(
    model: VICRegModel,
    data_loader: DataLoader,
    augmentation: RandomIQAugmentation,
    device: torch.device,
    loss_configuration: VICRegLossConfig | None = None,
    generator: torch.Generator | None = None,
) -> VICRegEpochMetrics:
    """Evaluate a VICReg model without gradient updates."""
    model.eval()
    augmentation.eval()

    accumulated_total_loss = 0.0
    accumulated_invariance_loss = 0.0
    accumulated_variance_loss = 0.0
    accumulated_covariance_loss = 0.0
    accumulated_positive_similarity = 0.0
    accumulated_projection_standard_deviation = 0.0
    example_count = 0

    with torch.inference_mode():
        for batch in data_loader:
            inputs = _move_iq_batch(
                batch,
                device,
            )

            first_view, second_view = (
                augmentation.create_views(
                    inputs,
                    generator=generator,
                )
            )

            first_projections = model(first_view)
            second_projections = model(second_view)

            loss_terms = compute_vicreg_loss(
                first_projections,
                second_projections,
                configuration=loss_configuration,
            )

            if not torch.isfinite(
                loss_terms.total
            ):
                raise RuntimeError(
                    "VICReg evaluation loss "
                    "became non-finite."
                )

            batch_size = int(inputs.shape[0])

            (
                positive_similarity,
                projection_standard_deviation,
            ) = _batch_diagnostics(
                first_projections,
                second_projections,
            )

            accumulated_total_loss += (
                float(loss_terms.total.item())
                * batch_size
            )
            accumulated_invariance_loss += (
                float(
                    loss_terms.invariance.item()
                )
                * batch_size
            )
            accumulated_variance_loss += (
                float(
                    loss_terms.variance.item()
                )
                * batch_size
            )
            accumulated_covariance_loss += (
                float(
                    loss_terms.covariance.item()
                )
                * batch_size
            )
            accumulated_positive_similarity += (
                positive_similarity
                * batch_size
            )
            accumulated_projection_standard_deviation += (
                projection_standard_deviation
                * batch_size
            )
            example_count += batch_size

    return _finalize_metrics(
        accumulated_total_loss=(
            accumulated_total_loss
        ),
        accumulated_invariance_loss=(
            accumulated_invariance_loss
        ),
        accumulated_variance_loss=(
            accumulated_variance_loss
        ),
        accumulated_covariance_loss=(
            accumulated_covariance_loss
        ),
        accumulated_positive_similarity=(
            accumulated_positive_similarity
        ),
        accumulated_projection_standard_deviation=(
            accumulated_projection_standard_deviation
        ),
        example_count=example_count,
    )


__all__ = [
    "VICRegEpochMetrics",
    "run_vicreg_evaluation_epoch",
    "run_vicreg_training_epoch",
]
