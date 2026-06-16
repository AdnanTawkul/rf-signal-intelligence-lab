from __future__ import annotations

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
    count_trainable_parameters,
)


def test_baseline_cnn_returns_expected_logits_shape() -> None:
    model = BaselineIQCNN()
    inputs = torch.randn(5, 2, 2_048)

    logits = model(inputs)

    assert logits.shape == (5, 4)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize(
    "sample_count",
    [
        256,
        2_048,
        4_096,
    ],
)
def test_baseline_cnn_accepts_variable_sample_counts(
    sample_count: int,
) -> None:
    model = BaselineIQCNN()
    inputs = torch.randn(3, 2, sample_count)

    logits = model(inputs)

    assert logits.shape == (3, 4)


def test_baseline_cnn_has_expected_parameter_count() -> None:
    model = BaselineIQCNN()

    assert count_trainable_parameters(model) == 73_092


def test_baseline_cnn_supports_backpropagation() -> None:
    model = BaselineIQCNN()
    inputs = torch.randn(
        2,
        2,
        256,
        requires_grad=True,
    )

    loss = model(inputs).sum()
    loss.backward()

    assert inputs.grad is not None

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        parameter.grad is not None
        for parameter in trainable_parameters
    )


@pytest.mark.parametrize(
    "invalid_inputs",
    [
        torch.randn(2, 2_048),
        torch.randn(2, 1, 2_048),
        torch.randn(2, 2, 4),
    ],
)
def test_baseline_cnn_rejects_invalid_input_shapes(
    invalid_inputs: torch.Tensor,
) -> None:
    model = BaselineIQCNN()

    with pytest.raises(ValueError):
        model(invalid_inputs)


@pytest.mark.parametrize(
    "invalid_configuration",
    [
        {"in_channels": 0},
        {"num_classes": 1},
        {"channels": ()},
        {"channels": (32, 0, 128)},
        {"kernel_size": 0},
        {"kernel_size": 6},
        {"dropout": -0.1},
        {"dropout": 1.0},
    ],
)
def test_baseline_cnn_config_rejects_invalid_values(
    invalid_configuration: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        BaselineCNNConfig(
            **invalid_configuration,  # type: ignore[arg-type]
        )
