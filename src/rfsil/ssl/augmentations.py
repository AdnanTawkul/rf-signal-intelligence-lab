from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor, nn


def _validate_probability(
    value: float,
    name: str,
) -> float:
    """Validate and return a probability."""
    probability = float(value)

    if not math.isfinite(probability):
        raise ValueError(f"{name} must be finite.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{name} must be between zero and one."
        )

    return probability


@dataclass(frozen=True, slots=True)
class IQAugmentationConfig:
    """Configuration for stochastic RF IQ view generation."""

    phase_rotation_probability: float = 0.8
    max_phase_rotation_rad: float = math.pi

    amplitude_scale_probability: float = 0.5
    amplitude_scale_min: float = 0.8
    amplitude_scale_max: float = 1.25

    time_shift_probability: float = 0.5
    max_time_shift_samples: int = 64

    awgn_probability: float = 0.5
    awgn_snr_db_min: float = 18.0
    awgn_snr_db_max: float = 30.0

    def __post_init__(self) -> None:
        """Validate augmentation settings."""
        _validate_probability(
            self.phase_rotation_probability,
            "phase_rotation_probability",
        )
        _validate_probability(
            self.amplitude_scale_probability,
            "amplitude_scale_probability",
        )
        _validate_probability(
            self.time_shift_probability,
            "time_shift_probability",
        )
        _validate_probability(
            self.awgn_probability,
            "awgn_probability",
        )

        if (
            not math.isfinite(self.max_phase_rotation_rad)
            or self.max_phase_rotation_rad < 0.0
        ):
            raise ValueError(
                "max_phase_rotation_rad must be "
                "nonnegative and finite."
            )

        for name, value in (
            (
                "amplitude_scale_min",
                self.amplitude_scale_min,
            ),
            (
                "amplitude_scale_max",
                self.amplitude_scale_max,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be positive and finite."
                )

        if (
            self.amplitude_scale_min
            > self.amplitude_scale_max
        ):
            raise ValueError(
                "amplitude_scale_min must not exceed "
                "amplitude_scale_max."
            )

        if (
            isinstance(self.max_time_shift_samples, bool)
            or not isinstance(
                self.max_time_shift_samples,
                Integral,
            )
        ):
            raise ValueError(
                "max_time_shift_samples must be an integer."
            )

        if self.max_time_shift_samples < 0:
            raise ValueError(
                "max_time_shift_samples must be nonnegative."
            )

        for name, value in (
            ("awgn_snr_db_min", self.awgn_snr_db_min),
            ("awgn_snr_db_max", self.awgn_snr_db_max),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be positive and finite."
                )

        if self.awgn_snr_db_min > self.awgn_snr_db_max:
            raise ValueError(
                "awgn_snr_db_min must not exceed "
                "awgn_snr_db_max."
            )


def _validate_iq_tensor(
    inputs: Tensor,
) -> tuple[Tensor, bool]:
    """Validate IQ data and return a batched view."""
    if not isinstance(inputs, Tensor):
        raise TypeError(
            "inputs must be a torch.Tensor."
        )

    if not torch.is_floating_point(inputs):
        raise TypeError(
            "inputs must use a floating-point dtype."
        )

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

    single_example = inputs.ndim == 2

    return (
        inputs.unsqueeze(0)
        if single_example
        else inputs,
        single_example,
    )


def _random_mask(
    probability: float,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample one independent transform mask per example."""
    if probability <= 0.0:
        return torch.zeros(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

    if probability >= 1.0:
        return torch.ones(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

    return (
        torch.rand(
            batch_size,
            device=device,
            generator=generator,
        )
        < probability
    )


def _uniform_values(
    batch_size: int,
    minimum: float,
    maximum: float,
    reference: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample one uniformly distributed value per example."""
    random_values = torch.rand(
        batch_size,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )

    return minimum + (
        maximum - minimum
    ) * random_values


def _apply_phase_rotation(
    inputs: Tensor,
    configuration: IQAugmentationConfig,
    generator: torch.Generator | None,
) -> Tensor:
    """Apply an independent global phase rotation."""
    batch_size = inputs.shape[0]
    mask = _random_mask(
        configuration.phase_rotation_probability,
        batch_size,
        inputs.device,
        generator,
    )

    angles = _uniform_values(
        batch_size,
        -configuration.max_phase_rotation_rad,
        configuration.max_phase_rotation_rad,
        inputs,
        generator,
    )
    angles = torch.where(
        mask,
        angles,
        torch.zeros_like(angles),
    )

    cosine = torch.cos(angles).unsqueeze(-1)
    sine = torch.sin(angles).unsqueeze(-1)

    in_phase = inputs[:, 0, :]
    quadrature = inputs[:, 1, :]

    rotated_in_phase = (
        in_phase * cosine
        - quadrature * sine
    )
    rotated_quadrature = (
        in_phase * sine
        + quadrature * cosine
    )

    return torch.stack(
        (
            rotated_in_phase,
            rotated_quadrature,
        ),
        dim=1,
    )


def _apply_amplitude_scaling(
    inputs: Tensor,
    configuration: IQAugmentationConfig,
    generator: torch.Generator | None,
) -> Tensor:
    """Apply log-uniform global amplitude scaling."""
    batch_size = inputs.shape[0]
    mask = _random_mask(
        configuration.amplitude_scale_probability,
        batch_size,
        inputs.device,
        generator,
    )

    log_minimum = math.log(
        configuration.amplitude_scale_min
    )
    log_maximum = math.log(
        configuration.amplitude_scale_max
    )

    log_scales = _uniform_values(
        batch_size,
        log_minimum,
        log_maximum,
        inputs,
        generator,
    )
    scales = torch.exp(log_scales)
    scales = torch.where(
        mask,
        scales,
        torch.ones_like(scales),
    )

    return inputs * scales[:, None, None]


def _apply_time_shift(
    inputs: Tensor,
    configuration: IQAugmentationConfig,
    generator: torch.Generator | None,
) -> Tensor:
    """Apply a zero-padded integer time shift."""
    maximum_shift = int(
        configuration.max_time_shift_samples
    )

    if maximum_shift == 0:
        return inputs

    sample_count = inputs.shape[-1]

    if maximum_shift >= sample_count:
        raise ValueError(
            "max_time_shift_samples must be smaller "
            "than the IQ sample count."
        )

    batch_size = inputs.shape[0]
    mask = _random_mask(
        configuration.time_shift_probability,
        batch_size,
        inputs.device,
        generator,
    )

    sampled_shifts = torch.randint(
        low=-maximum_shift,
        high=maximum_shift + 1,
        size=(batch_size,),
        device=inputs.device,
        generator=generator,
    )
    shifts = torch.where(
        mask,
        sampled_shifts,
        torch.zeros_like(sampled_shifts),
    )

    destination_indices = torch.arange(
        sample_count,
        device=inputs.device,
    ).unsqueeze(0)

    source_indices = (
        destination_indices
        - shifts.unsqueeze(1)
    )

    valid_source = (
        (source_indices >= 0)
        & (source_indices < sample_count)
    )

    gather_indices = source_indices.clamp(
        min=0,
        max=sample_count - 1,
    )
    gather_indices = gather_indices.unsqueeze(1).expand(
        -1,
        inputs.shape[1],
        -1,
    )

    shifted = torch.gather(
        inputs,
        dim=2,
        index=gather_indices,
    )

    return shifted * valid_source.unsqueeze(1)


def _apply_awgn(
    inputs: Tensor,
    configuration: IQAugmentationConfig,
    generator: torch.Generator | None,
) -> Tensor:
    """Add mild AWGN relative to each example's current power."""
    batch_size = inputs.shape[0]
    mask = _random_mask(
        configuration.awgn_probability,
        batch_size,
        inputs.device,
        generator,
    )

    augmentation_snr_db = _uniform_values(
        batch_size,
        configuration.awgn_snr_db_min,
        configuration.awgn_snr_db_max,
        inputs,
        generator,
    )

    complex_signal_power = (
        inputs.square()
        .sum(dim=1)
        .mean(dim=1)
    )

    snr_linear = torch.pow(
        torch.full_like(
            augmentation_snr_db,
            10.0,
        ),
        augmentation_snr_db / 10.0,
    )

    complex_noise_power = (
        complex_signal_power
        / snr_linear
    )
    component_noise_std = torch.sqrt(
        complex_noise_power / 2.0
    )

    noise = torch.randn(
        inputs.shape,
        device=inputs.device,
        dtype=inputs.dtype,
        generator=generator,
    )
    noise = (
        noise
        * component_noise_std[:, None, None]
        * mask[:, None, None]
    )

    return inputs + noise


class RandomIQAugmentation(nn.Module):
    """Generate stochastic, label-preserving RF IQ views."""

    def __init__(
        self,
        configuration: IQAugmentationConfig | None = None,
    ) -> None:
        super().__init__()

        self.configuration = (
            configuration
            if configuration is not None
            else IQAugmentationConfig()
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate one augmented IQ view."""
        batch, single_example = _validate_iq_tensor(
            inputs
        )

        output = batch.clone()
        output = _apply_phase_rotation(
            output,
            self.configuration,
            generator,
        )
        output = _apply_amplitude_scaling(
            output,
            self.configuration,
            generator,
        )
        output = _apply_time_shift(
            output,
            self.configuration,
            generator,
        )
        output = _apply_awgn(
            output,
            self.configuration,
            generator,
        )

        return (
            output.squeeze(0)
            if single_example
            else output
        )

    def create_views(
        self,
        inputs: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Generate two independently augmented views."""
        first_view = self(
            inputs,
            generator=generator,
        )
        second_view = self(
            inputs,
            generator=generator,
        )

        return first_view, second_view


__all__ = [
    "IQAugmentationConfig",
    "RandomIQAugmentation",
]
