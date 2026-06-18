from __future__ import annotations

import math

import pytest
import torch

from rfsil.data.transforms import (
    create_channel_aware_iq_representation,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
    count_trainable_parameters,
)


def test_channel_aware_representation_shape() -> None:
    inputs = torch.randn(5, 2, 256)

    outputs = create_channel_aware_iq_representation(
        inputs
    )

    assert outputs.shape == (5, 4, 256)
    assert outputs.dtype == inputs.dtype


def test_channel_aware_representation_supports_one_example(
) -> None:
    inputs = torch.randn(2, 128)

    outputs = create_channel_aware_iq_representation(
        inputs
    )

    assert outputs.shape == (4, 128)


def test_original_iq_channels_are_preserved() -> None:
    inputs = torch.randn(3, 2, 128)

    outputs = create_channel_aware_iq_representation(
        inputs
    )

    torch.testing.assert_close(
        outputs[:, :2],
        inputs,
    )


def test_magnitude_channel_has_unit_rms() -> None:
    inputs = torch.randn(4, 2, 512)

    outputs = create_channel_aware_iq_representation(
        inputs
    )

    magnitude_rms = torch.sqrt(
        outputs[:, 2].square().mean(dim=-1)
    )

    torch.testing.assert_close(
        magnitude_rms,
        torch.ones_like(magnitude_rms),
        rtol=1e-5,
        atol=1e-6,
    )


def test_known_differential_phase() -> None:
    phases = (
        torch.arange(16, dtype=torch.float32)
        * (math.pi / 2.0)
    )
    inputs = torch.stack(
        (
            torch.cos(phases),
            torch.sin(phases),
        )
    )

    outputs = create_channel_aware_iq_representation(
        inputs
    )

    assert outputs[3, 0].item() == 0.0

    torch.testing.assert_close(
        outputs[3, 1:],
        torch.full(
            (15,),
            0.5,
            dtype=torch.float32,
        ),
        rtol=1e-5,
        atol=1e-5,
    )


def test_derived_channels_ignore_gain_and_global_phase(
) -> None:
    torch.manual_seed(2026)

    inputs = torch.randn(3, 2, 256)
    angle = 0.73

    cosine = math.cos(angle)
    sine = math.sin(angle)

    rotated = torch.stack(
        (
            inputs[:, 0] * cosine
            - inputs[:, 1] * sine,
            inputs[:, 0] * sine
            + inputs[:, 1] * cosine,
        ),
        dim=1,
    ) * 3.2

    original = create_channel_aware_iq_representation(
        inputs
    )
    transformed = (
        create_channel_aware_iq_representation(
            rotated
        )
    )

    torch.testing.assert_close(
        original[:, 2:],
        transformed[:, 2:],
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "invalid_inputs",
    [
        torch.randn(2, 3, 128),
        torch.randn(2, 128, 1, 1),
        torch.zeros(2, 128),
    ],
)
def test_channel_aware_representation_rejects_invalid_input(
    invalid_inputs: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        create_channel_aware_iq_representation(
            invalid_inputs
        )


def test_channel_aware_model_accepts_raw_iq() -> None:
    model = BaselineIQCNN(
        BaselineCNNConfig(
            input_representation=(
                "iq_magnitude_dphase"
            ),
            normalization="group",
        )
    )
    inputs = torch.randn(4, 2, 512)

    logits = model(inputs)

    assert logits.shape == (4, 4)


def test_channel_aware_model_parameter_count() -> None:
    model = BaselineIQCNN(
        BaselineCNNConfig(
            input_representation=(
                "iq_magnitude_dphase"
            )
        )
    )

    assert count_trainable_parameters(model) == 73_540


def test_default_representation_remains_iq() -> None:
    configuration = BaselineCNNConfig()

    assert configuration.input_representation == "iq"
    assert configuration.feature_channels == 2


def test_old_checkpoint_mapping_defaults_to_iq() -> None:
    configuration = BaselineCNNConfig.from_mapping(
        {
            "in_channels": 2,
            "num_classes": 4,
            "channels": [32, 64, 128],
            "kernel_size": 7,
            "dropout": 0.2,
            "normalize_input_rms": False,
            "normalization": "group",
            "group_norm_groups": 8,
        }
    )

    assert configuration.input_representation == "iq"


@pytest.mark.parametrize(
    "configuration",
    [
        {
            "input_representation": "unknown",
        },
        {
            "in_channels": 3,
            "input_representation": (
                "iq_magnitude_dphase"
            ),
        },
    ],
)
def test_invalid_channel_aware_configuration_is_rejected(
    configuration: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        BaselineCNNConfig(
            **configuration,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "include_magnitude",
        "include_differential_phase",
        "expected_channels",
    ),
    [
        (True, False, 3),
        (False, True, 3),
        (True, True, 4),
    ],
)
def test_derived_channel_ablation_shapes(
    include_magnitude: bool,
    include_differential_phase: bool,
    expected_channels: int,
) -> None:
    inputs = torch.randn(4, 2, 256)

    outputs = create_channel_aware_iq_representation(
        inputs,
        include_magnitude=include_magnitude,
        include_differential_phase=(
            include_differential_phase
        ),
    )

    assert outputs.shape == (
        4,
        expected_channels,
        256,
    )


def test_transform_rejects_no_derived_channels() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        create_channel_aware_iq_representation(
            torch.randn(2, 128),
            include_magnitude=False,
            include_differential_phase=False,
        )


@pytest.mark.parametrize(
    (
        "representation",
        "expected_channels",
        "expected_parameters",
    ),
    [
        ("iq", 2, 73_092),
        ("iq_magnitude", 3, 73_316),
        ("iq_dphase", 3, 73_316),
        ("iq_magnitude_dphase", 4, 73_540),
    ],
)
def test_representation_feature_channels(
    representation: str,
    expected_channels: int,
    expected_parameters: int,
) -> None:
    configuration = BaselineCNNConfig(
        input_representation=representation
    )
    model = BaselineIQCNN(configuration)

    assert (
        configuration.feature_channels
        == expected_channels
    )
    assert (
        count_trainable_parameters(model)
        == expected_parameters
    )

    logits = model(
        torch.randn(2, 2, 256)
    )

    assert logits.shape == (2, 4)
