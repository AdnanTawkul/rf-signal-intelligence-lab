from __future__ import annotations

import math

import pytest
import torch

from rfsil.ssl.augmentations import (
    IQAugmentationConfig,
    RandomIQAugmentation,
)


def identity_configuration() -> IQAugmentationConfig:
    return IQAugmentationConfig(
        phase_rotation_probability=0.0,
        amplitude_scale_probability=0.0,
        time_shift_probability=0.0,
        max_time_shift_samples=0,
        awgn_probability=0.0,
    )


def test_identity_configuration_preserves_inputs() -> None:
    inputs = torch.randn(4, 2, 128)
    transform = RandomIQAugmentation(
        identity_configuration()
    )

    output = transform(inputs)

    torch.testing.assert_close(output, inputs)
    assert output.data_ptr() != inputs.data_ptr()


def test_single_example_shape_is_preserved() -> None:
    inputs = torch.randn(2, 128)
    transform = RandomIQAugmentation(
        IQAugmentationConfig(
            max_time_shift_samples=8,
        )
    )

    output = transform(inputs)

    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype
    assert torch.all(torch.isfinite(output))


def test_augmentation_is_reproducible_with_seed() -> None:
    inputs = torch.randn(5, 2, 128)
    transform = RandomIQAugmentation(
        IQAugmentationConfig(
            phase_rotation_probability=1.0,
            amplitude_scale_probability=1.0,
            time_shift_probability=1.0,
            max_time_shift_samples=8,
            awgn_probability=1.0,
        )
    )

    first_generator = torch.Generator().manual_seed(
        2026
    )
    second_generator = torch.Generator().manual_seed(
        2026
    )

    first = transform(
        inputs,
        generator=first_generator,
    )
    second = transform(
        inputs,
        generator=second_generator,
    )

    torch.testing.assert_close(first, second)


def test_two_views_are_independently_augmented() -> None:
    inputs = torch.randn(8, 2, 128)
    transform = RandomIQAugmentation(
        IQAugmentationConfig(
            phase_rotation_probability=1.0,
            amplitude_scale_probability=1.0,
            time_shift_probability=1.0,
            max_time_shift_samples=8,
            awgn_probability=1.0,
        )
    )
    generator = torch.Generator().manual_seed(2026)

    first, second = transform.create_views(
        inputs,
        generator=generator,
    )

    assert first.shape == inputs.shape
    assert second.shape == inputs.shape
    assert not torch.allclose(first, second)


def test_phase_rotation_preserves_complex_power() -> None:
    inputs = torch.randn(6, 2, 128)
    transform = RandomIQAugmentation(
        IQAugmentationConfig(
            phase_rotation_probability=1.0,
            max_phase_rotation_rad=math.pi,
            amplitude_scale_probability=0.0,
            time_shift_probability=0.0,
            max_time_shift_samples=0,
            awgn_probability=0.0,
        )
    )

    output = transform(
        inputs,
        generator=torch.Generator().manual_seed(
            2026
        ),
    )

    original_power = inputs.square().sum(dim=1)
    rotated_power = output.square().sum(dim=1)

    torch.testing.assert_close(
        rotated_power,
        original_power,
        rtol=1e-5,
        atol=1e-6,
    )


def test_time_shift_does_not_wrap_samples() -> None:
    inputs = torch.zeros(1, 2, 16)
    inputs[0, 0, 0] = 1.0

    transform = RandomIQAugmentation(
        IQAugmentationConfig(
            phase_rotation_probability=0.0,
            amplitude_scale_probability=0.0,
            time_shift_probability=1.0,
            max_time_shift_samples=1,
            awgn_probability=0.0,
        )
    )

    output = transform(
        inputs,
        generator=torch.Generator().manual_seed(
            2026
        ),
    )

    assert output[0, 0, -1].item() == 0.0
    assert torch.count_nonzero(output) <= 1


def test_awgn_changes_nonzero_inputs() -> None:
    inputs = torch.ones(4, 2, 128)

    transform = RandomIQAugmentation(
        IQAugmentationConfig(
            phase_rotation_probability=0.0,
            amplitude_scale_probability=0.0,
            time_shift_probability=0.0,
            max_time_shift_samples=0,
            awgn_probability=1.0,
            awgn_snr_db_min=20.0,
            awgn_snr_db_max=20.0,
        )
    )

    output = transform(
        inputs,
        generator=torch.Generator().manual_seed(
            2026
        ),
    )

    assert not torch.allclose(output, inputs)
    assert torch.all(torch.isfinite(output))


@pytest.mark.parametrize(
    "inputs",
    [
        torch.ones(2, 128, dtype=torch.int64),
        torch.ones(2, 3, 128),
        torch.ones(2, 2, 2, 128),
        torch.empty(2, 0),
    ],
)
def test_invalid_inputs_are_rejected(
    inputs: torch.Tensor,
) -> None:
    transform = RandomIQAugmentation(
        identity_configuration()
    )

    with pytest.raises(
        (TypeError, ValueError)
    ):
        transform(inputs)


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {
            "phase_rotation_probability": -0.1,
        },
        {
            "amplitude_scale_probability": 1.1,
        },
        {
            "max_phase_rotation_rad": -1.0,
        },
        {
            "amplitude_scale_min": 0.0,
        },
        {
            "amplitude_scale_min": 1.2,
            "amplitude_scale_max": 0.8,
        },
        {
            "max_time_shift_samples": -1,
        },
        {
            "awgn_snr_db_min": 0.0,
        },
        {
            "awgn_snr_db_min": 30.0,
            "awgn_snr_db_max": 20.0,
        },
    ],
)
def test_invalid_configuration_is_rejected(
    keyword_arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        IQAugmentationConfig(
            **keyword_arguments,
        )
