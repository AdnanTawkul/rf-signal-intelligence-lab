from __future__ import annotations

import pytest
import torch

from rfsil.data.transforms import normalize_iq_rms


def complex_average_power(inputs: torch.Tensor) -> torch.Tensor:
    """Return average complex power for each IQ example."""
    return (
        inputs.square()
        .sum(dim=-2)
        .mean(dim=-1)
    )


def test_normalize_single_iq_example_to_unit_power() -> None:
    inputs = torch.randn(2, 2_048)

    normalized = normalize_iq_rms(inputs)

    assert complex_average_power(normalized).item() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_normalize_batch_independently() -> None:
    inputs = torch.randn(8, 2, 512)

    amplitude_scales = torch.tensor(
        [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
    ).reshape(-1, 1, 1)

    scaled_inputs = inputs * amplitude_scales
    normalized = normalize_iq_rms(scaled_inputs)

    torch.testing.assert_close(
        complex_average_power(normalized),
        torch.ones(8),
        rtol=1e-5,
        atol=1e-6,
    )


def test_normalization_preserves_relative_iq_geometry() -> None:
    inputs = torch.tensor(
        [
            [1.0, 3.0, -1.0, -3.0],
            [1.0, 1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )

    normalized = normalize_iq_rms(inputs)

    input_ratio = inputs[0, 1] / inputs[0, 0]
    normalized_ratio = normalized[0, 1] / normalized[0, 0]

    assert normalized_ratio.item() == pytest.approx(
        input_ratio.item(),
        abs=1e-6,
    )


def test_normalization_preserves_dtype_and_shape() -> None:
    inputs = torch.randn(
        4,
        2,
        256,
        dtype=torch.float64,
    )

    normalized = normalize_iq_rms(inputs)

    assert normalized.shape == inputs.shape
    assert normalized.dtype == torch.float64


def test_normalization_does_not_modify_input() -> None:
    inputs = torch.randn(2, 128)
    original = inputs.clone()

    normalize_iq_rms(inputs)

    torch.testing.assert_close(inputs, original)


@pytest.mark.parametrize(
    "invalid_inputs",
    [
        torch.randn(128),
        torch.randn(2, 2, 128, 1),
        torch.randn(4, 1, 128),
        torch.randn(4, 3, 128),
        torch.empty(2, 0),
    ],
)
def test_normalization_rejects_invalid_shapes(
    invalid_inputs: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        normalize_iq_rms(invalid_inputs)


def test_normalization_rejects_integer_tensor() -> None:
    inputs = torch.ones(
        2,
        128,
        dtype=torch.int64,
    )

    with pytest.raises(TypeError):
        normalize_iq_rms(inputs)


def test_normalization_rejects_zero_power() -> None:
    inputs = torch.zeros(2, 128)

    with pytest.raises(ValueError):
        normalize_iq_rms(inputs)


def test_normalization_rejects_nonfinite_values() -> None:
    inputs = torch.ones(2, 128)
    inputs[0, 0] = float("nan")

    with pytest.raises(ValueError):
        normalize_iq_rms(inputs)


@pytest.mark.parametrize(
    "invalid_epsilon",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_normalization_rejects_invalid_epsilon(
    invalid_epsilon: float,
) -> None:
    with pytest.raises(ValueError):
        normalize_iq_rms(
            torch.ones(2, 128),
            epsilon=invalid_epsilon,
        )
