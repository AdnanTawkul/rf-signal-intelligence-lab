from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineIQCNN,
    count_trainable_parameters,
)
from rfsil.models.model_factory import (
    create_model_from_checkpoint,
    create_model_from_mapping,
)
from rfsil.models.residual_equalizer_cnn import (
    ResidualEqualizerCNNConfig,
    ResidualEqualizerIQCNN,
)


def test_equalizer_starts_as_exact_identity() -> None:
    model = ResidualEqualizerIQCNN()
    inputs = torch.randn(4, 2, 256)

    corrected = model.equalize_iq(inputs)

    torch.testing.assert_close(
        corrected,
        inputs,
        rtol=0.0,
        atol=0.0,
    )


def test_equalizer_output_layer_is_zero_initialized() -> None:
    model = ResidualEqualizerIQCNN()

    assert torch.count_nonzero(
        model.equalizer.output.weight
    ).item() == 0


def test_equalizer_model_parameter_count() -> None:
    model = ResidualEqualizerIQCNN()

    assert (
        count_trainable_parameters(model)
        == 76_036
    )


def test_equalizer_forward_and_embedding_shapes() -> None:
    model = ResidualEqualizerIQCNN()
    inputs = torch.randn(3, 2, 256)

    embeddings = model.extract_features(inputs)
    logits = model(inputs)

    assert embeddings.shape == (3, 128)
    assert logits.shape == (3, 4)
    assert torch.all(torch.isfinite(logits))


def test_equalizer_output_receives_gradients() -> None:
    model = ResidualEqualizerIQCNN()
    inputs = torch.randn(3, 2, 256)

    loss = model(inputs).square().mean()
    loss.backward()

    gradient = (
        model.equalizer.output.weight.grad
    )

    assert gradient is not None
    assert torch.any(gradient != 0.0)
    assert torch.all(torch.isfinite(gradient))


@pytest.mark.parametrize(
    "inputs",
    [
        torch.randn(2, 3, 128),
        torch.randn(2, 128),
        torch.randn(2, 2, 0),
    ],
)
def test_equalizer_rejects_invalid_inputs(
    inputs: torch.Tensor,
) -> None:
    model = ResidualEqualizerIQCNN()

    with pytest.raises(ValueError):
        model(inputs)


@pytest.mark.parametrize(
    "content",
    [
        {
            "in_channels": 3,
        },
        {
            "equalizer_kernel_size": 8,
        },
        {
            "equalizer_hidden_channels": 18,
        },
        {
            "equalizer_normalization": "unknown",
        },
    ],
)
def test_invalid_equalizer_configuration(
    content: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ResidualEqualizerCNNConfig(
            **content,  # type: ignore[arg-type]
        )


def test_factory_builds_residual_equalizer() -> None:
    model, configuration = (
        create_model_from_mapping(
            {
                "model_type": (
                    "residual_equalizer_iq_cnn"
                ),
                "channels": [
                    32,
                    64,
                    128,
                ],
                "normalization": "group",
                "group_norm_groups": 8,
                "equalizer_hidden_channels": 16,
                "equalizer_kernel_size": 9,
                "equalizer_normalization": (
                    "group"
                ),
                "equalizer_group_norm_groups": 8,
            }
        )
    )

    assert isinstance(
        model,
        ResidualEqualizerIQCNN,
    )
    assert isinstance(
        configuration,
        ResidualEqualizerCNNConfig,
    )


def test_checkpoint_factory_reconstructs_equalizer() -> None:
    configuration = (
        ResidualEqualizerCNNConfig()
    )

    model, restored = (
        create_model_from_checkpoint(
            {
                "model_configuration": (
                    asdict(configuration)
                )
            }
        )
    )

    assert isinstance(
        model,
        ResidualEqualizerIQCNN,
    )
    assert restored == configuration


def test_factory_still_defaults_to_baseline() -> None:
    model, _ = create_model_from_mapping(
        {
            "in_channels": 2,
            "num_classes": 4,
            "channels": [
                32,
                64,
                128,
            ],
            "normalization": "group",
            "group_norm_groups": 8,
        }
    )

    assert isinstance(model, BaselineIQCNN)
