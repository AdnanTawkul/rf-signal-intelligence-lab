from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from rfsil.ssl.augmentations import RandomIQAugmentation
from rfsil.ssl.contrastive import (
    SimCLRModel,
    nt_xent_loss,
)


@dataclass(frozen=True, slots=True)
class ContrastiveEpochMetrics:
    """Aggregate metrics from one contrastive-learning epoch."""

    loss: float
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
            "Contrastive batch must contain an 'iq' tensor."
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
            "Contrastive batches must contain at least "
            "two examples."
        )

    if inputs.shape[2] == 0:
        raise ValueError(
            "Batch IQ tensor must contain at least "
            "one sample."
        )

    if not torch.all(torch.isfinite(inputs)):
        raise ValueError(
            "Batch IQ tensor must contain only finite values."
        )

    return inputs


def _batch_diagnostics(
    first_projections: Tensor,
    second_projections: Tensor,
) -> tuple[float, float]:
    """Compute positive similarity and projection spread."""
    positive_similarity = (
        first_projections
        * second_projections
    ).sum(dim=1).mean()

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
            "Positive-pair cosine similarity became non-finite."
        )

    if not torch.isfinite(
        projection_standard_deviation
    ):
        raise RuntimeError(
            "Projection standard deviation became non-finite."
        )

    return (
        float(positive_similarity.item()),
        float(
            projection_standard_deviation.item()
        ),
    )


def _finalize_metrics(
    accumulated_loss: float,
    accumulated_positive_similarity: float,
    accumulated_projection_standard_deviation: float,
    example_count: int,
) -> ContrastiveEpochMetrics:
    """Create aggregate epoch metrics."""
    if example_count == 0:
        raise ValueError(
            "Contrastive DataLoader produced no examples."
        )

    return ContrastiveEpochMetrics(
        loss=accumulated_loss / example_count,
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


def run_contrastive_training_epoch(
    model: SimCLRModel,
    data_loader: DataLoader,
    optimizer: Optimizer,
    augmentation: RandomIQAugmentation,
    device: torch.device,
    temperature: float = 0.1,
    generator: torch.Generator | None = None,
) -> ContrastiveEpochMetrics:
    """Train a SimCLR model for one complete epoch."""
    model.train()
    augmentation.train()

    accumulated_loss = 0.0
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

        loss = nt_xent_loss(
            first_projections,
            second_projections,
            temperature=temperature,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Contrastive training loss became non-finite."
            )

        loss.backward()
        optimizer.step()

        batch_size = int(inputs.shape[0])

        (
            positive_similarity,
            projection_standard_deviation,
        ) = _batch_diagnostics(
            first_projections.detach(),
            second_projections.detach(),
        )

        accumulated_loss += (
            float(loss.detach().item())
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
        accumulated_loss=accumulated_loss,
        accumulated_positive_similarity=(
            accumulated_positive_similarity
        ),
        accumulated_projection_standard_deviation=(
            accumulated_projection_standard_deviation
        ),
        example_count=example_count,
    )


def run_contrastive_evaluation_epoch(
    model: SimCLRModel,
    data_loader: DataLoader,
    augmentation: RandomIQAugmentation,
    device: torch.device,
    temperature: float = 0.1,
    generator: torch.Generator | None = None,
) -> ContrastiveEpochMetrics:
    """Evaluate a SimCLR model without gradient updates."""
    model.eval()
    augmentation.eval()

    accumulated_loss = 0.0
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

            loss = nt_xent_loss(
                first_projections,
                second_projections,
                temperature=temperature,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Contrastive evaluation loss "
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

            accumulated_loss += (
                float(loss.item())
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
        accumulated_loss=accumulated_loss,
        accumulated_positive_similarity=(
            accumulated_positive_similarity
        ),
        accumulated_projection_standard_deviation=(
            accumulated_projection_standard_deviation
        ),
        example_count=example_count,
    )


__all__ = [
    "ContrastiveEpochMetrics",
    "run_contrastive_evaluation_epoch",
    "run_contrastive_training_epoch",
]
