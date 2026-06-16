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


def test_rms_normalized_model_is_invariant_to_positive_gain() -> None:
    torch.manual_seed(42)

    model = BaselineIQCNN(
        BaselineCNNConfig(
            normalize_input_rms=True,
        )
    )
    model.eval()

    inputs = torch.randn(4, 2, 512)

    with torch.inference_mode():
        original_logits = model(inputs)
        scaled_logits = model(inputs * 3.7)

    torch.testing.assert_close(
        original_logits,
        scaled_logits,
        rtol=1e-5,
        atol=1e-6,
    )


def test_rms_normalized_model_rejects_zero_power_input() -> None:
    model = BaselineIQCNN(
        BaselineCNNConfig(
            normalize_input_rms=True,
        )
    )

    with pytest.raises(ValueError):
        model(torch.zeros(2, 2, 256))


def test_baseline_config_rejects_nonboolean_rms_setting() -> None:
    with pytest.raises(ValueError):
        BaselineCNNConfig(
            normalize_input_rms=1,  # type: ignore[arg-type]
        )


def test_group_norm_model_returns_expected_logits_shape() -> None:
    model = BaselineIQCNN(
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
        )
    )
    inputs = torch.randn(5, 2, 512)

    logits = model(inputs)

    assert logits.shape == (5, 4)


def test_group_norm_preserves_trainable_parameter_count() -> None:
    model = BaselineIQCNN(
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
        )
    )

    assert count_trainable_parameters(model) == 73_092


def test_group_norm_has_identical_train_and_eval_behavior_without_dropout(
) -> None:
    torch.manual_seed(42)

    model = BaselineIQCNN(
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
            dropout=0.0,
        )
    )
    inputs = torch.randn(4, 2, 512)

    model.train()

    with torch.inference_mode():
        training_logits = model(inputs)

    model.eval()

    with torch.inference_mode():
        evaluation_logits = model(inputs)

    torch.testing.assert_close(
        training_logits,
        evaluation_logits,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "invalid_configuration",
    [
        {"normalization": "layer"},
        {"normalization": ""},
        {
            "normalization": "group",
            "group_norm_groups": 0,
        },
        {
            "normalization": "group",
            "group_norm_groups": 7,
        },
    ],
)
def test_normalization_configuration_rejects_invalid_values(
    invalid_configuration: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        BaselineCNNConfig(
            **invalid_configuration,  # type: ignore[arg-type]
        )
