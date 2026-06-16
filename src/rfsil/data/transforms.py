from __future__ import annotations

import math

import torch
from torch import Tensor


def normalize_iq_rms(
    inputs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Normalize each complex IQ example to unit average power.

    Supported input shapes are:

        [2, samples]
        [batch, 2, samples]

    Channel zero contains the in-phase component and channel one contains the
    quadrature component. The normalization preserves relative constellation
    geometry while removing global amplitude scaling.

    Args:
        inputs: Floating-point IQ tensor.
        epsilon: Minimum allowed average complex-signal power.

    Returns:
        IQ tensor with unit average complex power per example.

    Raises:
        TypeError: If inputs is not a floating-point tensor.
        ValueError: If the shape, values, or epsilon are invalid.
    """
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor.")

    if not torch.is_floating_point(inputs):
        raise TypeError("inputs must use a floating-point dtype.")

    if inputs.ndim not in (2, 3):
        raise ValueError(
            "inputs must have shape [2, samples] or "
            "[batch, 2, samples]."
        )

    if inputs.shape[-2] != 2:
        raise ValueError(
            "inputs must contain exactly two IQ channels."
        )

    if inputs.shape[-1] == 0:
        raise ValueError("inputs must contain at least one sample.")

    if not torch.all(torch.isfinite(inputs)):
        raise ValueError("inputs must contain only finite values.")

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite.")

    average_complex_power = (
        inputs.square()
        .sum(dim=-2)
        .mean(dim=-1)
    )

    if torch.any(average_complex_power <= epsilon):
        raise ValueError(
            "Every IQ example must have power greater than epsilon."
        )

    rms_scale = torch.sqrt(
        average_complex_power
    ).unsqueeze(-1).unsqueeze(-1)

    return inputs / rms_scale


__all__ = [
    "normalize_iq_rms",
]
