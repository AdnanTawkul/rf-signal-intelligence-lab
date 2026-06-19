from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

STATISTIC_NAMES = (
    "mean_absolute_correction",
    "correction_rms",
    "maximum_absolute_correction",
    "input_rms",
    "corrected_rms",
    "relative_correction_rms",
)


@dataclass(slots=True)
class CorrectionStatisticsAccumulator:
    """Accumulate correction statistics without storing all tensors."""

    example_count: int = 0
    value_count: int = 0
    absolute_sum: float = 0.0
    correction_square_sum: float = 0.0
    input_square_sum: float = 0.0
    corrected_square_sum: float = 0.0
    maximum_absolute_correction: float = 0.0

    def update(
        self,
        inputs: Tensor,
        corrections: Tensor,
    ) -> None:
        """Add one batch of raw inputs and predicted corrections."""
        if not isinstance(inputs, Tensor):
            raise TypeError(
                "inputs must be a torch.Tensor."
            )

        if not isinstance(corrections, Tensor):
            raise TypeError(
                "corrections must be a torch.Tensor."
            )

        if inputs.shape != corrections.shape:
            raise ValueError(
                "inputs and corrections must have "
                "the same shape."
            )

        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape "
                "[batch, 2, samples]."
            )

        if inputs.shape[1] != 2:
            raise ValueError(
                "inputs must contain exactly "
                "two IQ channels."
            )

        if inputs.shape[0] == 0 or inputs.shape[2] == 0:
            raise ValueError(
                "inputs must not be empty."
            )

        if not torch.is_floating_point(inputs):
            raise TypeError(
                "inputs must use a floating-point dtype."
            )

        if not torch.is_floating_point(corrections):
            raise TypeError(
                "corrections must use a floating-point dtype."
            )

        if not torch.all(torch.isfinite(inputs)):
            raise ValueError(
                "inputs must contain only finite values."
            )

        if not torch.all(
            torch.isfinite(corrections)
        ):
            raise ValueError(
                "corrections must contain only "
                "finite values."
            )

        detached_inputs = inputs.detach()
        detached_corrections = corrections.detach()
        corrected = (
            detached_inputs
            + detached_corrections
        )

        self.example_count += int(
            detached_inputs.shape[0]
        )
        self.value_count += int(
            detached_inputs.numel()
        )

        self.absolute_sum += float(
            detached_corrections.abs().sum().item()
        )
        self.correction_square_sum += float(
            detached_corrections.square().sum().item()
        )
        self.input_square_sum += float(
            detached_inputs.square().sum().item()
        )
        self.corrected_square_sum += float(
            corrected.square().sum().item()
        )

        batch_maximum = float(
            detached_corrections.abs().max().item()
        )
        self.maximum_absolute_correction = max(
            self.maximum_absolute_correction,
            batch_maximum,
        )

    def finalize(self) -> dict[str, int | float]:
        """Return finalized correction statistics."""
        if self.value_count == 0:
            raise ValueError(
                "No correction batches were accumulated."
            )

        denominator = float(self.value_count)

        correction_rms = math.sqrt(
            self.correction_square_sum
            / denominator
        )
        input_rms = math.sqrt(
            self.input_square_sum
            / denominator
        )
        corrected_rms = math.sqrt(
            self.corrected_square_sum
            / denominator
        )

        relative_correction_rms = (
            correction_rms / input_rms
            if input_rms > 0.0
            else float("nan")
        )

        return {
            "example_count": self.example_count,
            "value_count": self.value_count,
            "mean_absolute_correction": (
                self.absolute_sum
                / denominator
            ),
            "correction_rms": correction_rms,
            "maximum_absolute_correction": (
                self.maximum_absolute_correction
            ),
            "input_rms": input_rms,
            "corrected_rms": corrected_rms,
            "relative_correction_rms": (
                relative_correction_rms
            ),
        }


def aggregate_seed_correction_statistics(
    per_seed: Mapping[
        int,
        Mapping[str, int | float],
    ],
) -> dict[str, Any]:
    """Aggregate correction statistics across training seeds."""
    if not per_seed:
        raise ValueError(
            "per_seed must not be empty."
        )

    aggregate: dict[str, dict[str, float]] = {}

    for statistic_name in STATISTIC_NAMES:
        values = np.asarray(
            [
                float(
                    statistics[statistic_name]
                )
                for statistics in per_seed.values()
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{statistic_name} must contain "
                "only finite values."
            )

        aggregate[statistic_name] = {
            "mean": float(np.mean(values)),
            "standard_deviation": float(
                np.std(values)
            ),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "per_seed": {
            str(seed): dict(statistics)
            for seed, statistics in per_seed.items()
        },
        "aggregate": aggregate,
    }


__all__ = [
    "CorrectionStatisticsAccumulator",
    "STATISTIC_NAMES",
    "aggregate_seed_correction_statistics",
]
