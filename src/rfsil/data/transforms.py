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



def create_channel_aware_iq_representation(
    inputs: Tensor,
    epsilon: float = 1e-8,
) -> Tensor:
    """Append normalized magnitude and differential phase to raw IQ.

    Supported input shapes are:

        [2, samples]
        [batch, 2, samples]

    The returned representation contains four channels:

        0. in-phase
        1. quadrature
        2. magnitude normalized to unit RMS per example
        3. wrapped differential phase divided by pi

    The first differential-phase sample is set to zero so the output keeps
    the original temporal length.
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
        raise ValueError(
            "inputs must contain at least one sample."
        )

    if not torch.all(torch.isfinite(inputs)):
        raise ValueError(
            "inputs must contain only finite values."
        )

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            "epsilon must be positive and finite."
        )

    in_phase = inputs[..., 0, :]
    quadrature = inputs[..., 1, :]

    magnitude = torch.sqrt(
        in_phase.square()
        + quadrature.square()
    )

    magnitude_rms = torch.sqrt(
        magnitude.square().mean(
            dim=-1,
            keepdim=True,
        )
    )

    if torch.any(magnitude_rms <= epsilon):
        raise ValueError(
            "Every IQ example must have power greater "
            "than epsilon."
        )

    normalized_magnitude = (
        magnitude / magnitude_rms
    )

    previous_in_phase = in_phase[..., :-1]
    previous_quadrature = quadrature[..., :-1]
    current_in_phase = in_phase[..., 1:]
    current_quadrature = quadrature[..., 1:]

    phase_product_real = (
        current_in_phase * previous_in_phase
        + current_quadrature
        * previous_quadrature
    )
    phase_product_imaginary = (
        current_quadrature * previous_in_phase
        - current_in_phase
        * previous_quadrature
    )

    phase_difference = torch.atan2(
        phase_product_imaginary,
        phase_product_real,
    ) / math.pi

    initial_phase_difference = torch.zeros_like(
        phase_difference[..., :1]
    )
    phase_difference = torch.cat(
        (
            initial_phase_difference,
            phase_difference,
        ),
        dim=-1,
    )

    return torch.cat(
        (
            inputs,
            normalized_magnitude.unsqueeze(-2),
            phase_difference.unsqueeze(-2),
        ),
        dim=-2,
    )


__all__ = [
    "create_channel_aware_iq_representation",
    "normalize_iq_rms",
]
