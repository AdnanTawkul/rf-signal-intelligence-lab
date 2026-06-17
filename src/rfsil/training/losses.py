from __future__ import annotations

import math
from numbers import Integral

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class ClassSNRWeightedCrossEntropyLoss(nn.Module):
    """Apply extra loss weight to one class at selected SNR levels.

    The weighted loss is divided by the sum of example weights. This keeps
    its overall scale comparable to ordinary mean cross-entropy.
    """

    def __init__(
        self,
        target_class_index: int,
        target_snr_values_db: tuple[float, ...],
        target_weight: float,
        snr_tolerance: float = 1e-4,
    ) -> None:
        super().__init__()

        if (
            isinstance(target_class_index, bool)
            or not isinstance(target_class_index, Integral)
        ):
            raise ValueError(
                "target_class_index must be an integer."
            )

        if int(target_class_index) < 0:
            raise ValueError(
                "target_class_index must not be negative."
            )

        if not target_snr_values_db:
            raise ValueError(
                "target_snr_values_db must not be empty."
            )

        validated_snr_values = tuple(
            float(value)
            for value in target_snr_values_db
        )

        if not all(
            math.isfinite(value)
            for value in validated_snr_values
        ):
            raise ValueError(
                "target_snr_values_db must contain finite values."
            )

        if (
            not math.isfinite(target_weight)
            or target_weight <= 0.0
        ):
            raise ValueError(
                "target_weight must be positive and finite."
            )

        if (
            not math.isfinite(snr_tolerance)
            or snr_tolerance <= 0.0
        ):
            raise ValueError(
                "snr_tolerance must be positive and finite."
            )

        self.target_class_index = int(target_class_index)
        self.target_snr_values_db = validated_snr_values
        self.target_weight = float(target_weight)
        self.snr_tolerance = float(snr_tolerance)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        snr_db: Tensor,
    ) -> Tensor:
        """Return weighted mean cross-entropy."""
        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape [batch, classes]."
            )

        if labels.ndim != 1:
            raise ValueError(
                "labels must have shape [batch]."
            )

        if snr_db.ndim != 1:
            raise ValueError(
                "snr_db must have shape [batch]."
            )

        if not (
            logits.shape[0]
            == labels.shape[0]
            == snr_db.shape[0]
        ):
            raise ValueError(
                "logits, labels, and snr_db batch sizes must match."
            )

        if logits.shape[0] == 0:
            raise ValueError("The batch must not be empty.")

        if self.target_class_index >= logits.shape[1]:
            raise ValueError(
                "target_class_index exceeds the number of classes."
            )

        if not torch.is_floating_point(logits):
            raise TypeError(
                "logits must use a floating-point dtype."
            )

        if not torch.is_floating_point(snr_db):
            raise TypeError(
                "snr_db must use a floating-point dtype."
            )

        if not torch.all(torch.isfinite(logits)):
            raise ValueError("logits must contain finite values.")

        if not torch.all(torch.isfinite(snr_db)):
            raise ValueError("snr_db must contain finite values.")

        per_example_loss = functional.cross_entropy(
            logits,
            labels,
            reduction="none",
        )

        target_class_mask = (
            labels == self.target_class_index
        )
        target_snr_mask = torch.zeros_like(
            target_class_mask,
            dtype=torch.bool,
        )

        for target_snr in self.target_snr_values_db:
            target_snr_mask |= torch.isclose(
                snr_db,
                torch.tensor(
                    target_snr,
                    device=snr_db.device,
                    dtype=snr_db.dtype,
                ),
                rtol=0.0,
                atol=self.snr_tolerance,
            )

        target_mask = (
            target_class_mask
            & target_snr_mask
        )

        example_weights = torch.ones_like(
            per_example_loss,
        )
        example_weights[target_mask] = self.target_weight

        return torch.sum(
            per_example_loss * example_weights
        ) / torch.sum(example_weights)


__all__ = [
    "ClassSNRWeightedCrossEntropyLoss",
]
