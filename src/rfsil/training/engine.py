from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Aggregated loss and accuracy for one dataset pass."""

    loss: float
    accuracy: float
    example_count: int


def set_global_seed(
    seed: int,
    deterministic: bool = True,
) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=True,
        )
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def _move_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Move IQ inputs and labels to the selected device."""
    if "iq" not in batch or "label" not in batch:
        raise KeyError("Batch must contain 'iq' and 'label' tensors.")

    inputs = batch["iq"].to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    labels = batch["label"].to(
        device=device,
        dtype=torch.int64,
        non_blocking=True,
    )

    if inputs.ndim != 3:
        raise ValueError(
            "Batch IQ tensor must have shape [batch, channels, samples]."
        )

    if labels.ndim != 1:
        raise ValueError("Batch labels must have shape [batch].")

    if inputs.shape[0] != labels.shape[0]:
        raise ValueError("Input and label batch sizes must match.")

    return inputs, labels


def run_training_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Train a classification model for one complete epoch."""
    model.train()

    accumulated_loss = 0.0
    correct_predictions = 0
    example_count = 0

    for batch in data_loader:
        inputs, labels = _move_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = loss_function(logits, labels)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss became non-finite.")

        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        predictions = torch.argmax(logits.detach(), dim=1)

        accumulated_loss += float(loss.detach().item()) * batch_size
        correct_predictions += int(
            torch.count_nonzero(predictions == labels).item()
        )
        example_count += batch_size

    if example_count == 0:
        raise ValueError("Training DataLoader produced no examples.")

    return EpochMetrics(
        loss=accumulated_loss / example_count,
        accuracy=correct_predictions / example_count,
        example_count=example_count,
    )


def run_evaluation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate a classification model without gradient updates."""
    model.eval()

    accumulated_loss = 0.0
    correct_predictions = 0
    example_count = 0

    with torch.inference_mode():
        for batch in data_loader:
            inputs, labels = _move_batch(batch, device)

            logits = model(inputs)
            loss = loss_function(logits, labels)

            if not torch.isfinite(loss):
                raise RuntimeError("Evaluation loss became non-finite.")

            batch_size = int(labels.shape[0])
            predictions = torch.argmax(logits, dim=1)

            accumulated_loss += float(loss.item()) * batch_size
            correct_predictions += int(
                torch.count_nonzero(predictions == labels).item()
            )
            example_count += batch_size

    if example_count == 0:
        raise ValueError("Evaluation DataLoader produced no examples.")

    return EpochMetrics(
        loss=accumulated_loss / example_count,
        accuracy=correct_predictions / example_count,
        example_count=example_count,
    )


__all__ = [
    "EpochMetrics",
    "run_evaluation_epoch",
    "run_training_epoch",
    "set_global_seed",
]
