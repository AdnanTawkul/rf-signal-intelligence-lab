from __future__ import annotations

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)
from rfsil.ssl.contrastive import (
    ProjectionHead,
    ProjectionHeadConfig,
    SimCLRModel,
    nt_xent_loss,
)


def create_encoder() -> BaselineIQCNN:
    return BaselineIQCNN(
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
        )
    )


def test_projection_head_returns_expected_shape() -> None:
    head = ProjectionHead(
        input_dimension=128,
        configuration=ProjectionHeadConfig(
            hidden_dimension=64,
            output_dimension=32,
        ),
    )
    features = torch.randn(5, 128)

    projections = head(features)

    assert projections.shape == (5, 32)


def test_simclr_forward_returns_unit_normalized_output() -> None:
    model = SimCLRModel(
        encoder=create_encoder(),
        projection_configuration=ProjectionHeadConfig(
            hidden_dimension=64,
            output_dimension=32,
        ),
    )
    model.eval()

    inputs = torch.randn(4, 2, 256)

    with torch.inference_mode():
        projections = model(inputs)

    norms = torch.linalg.vector_norm(
        projections,
        dim=1,
    )

    assert projections.shape == (4, 32)
    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        rtol=1e-5,
        atol=1e-6,
    )


def test_encode_matches_encoder_feature_interface() -> None:
    encoder = create_encoder()
    model = SimCLRModel(encoder=encoder)
    model.eval()

    inputs = torch.randn(3, 2, 256)

    with torch.inference_mode():
        expected = encoder.extract_features(inputs)
        actual = model.encode(inputs)

    torch.testing.assert_close(actual, expected)


def test_nt_xent_is_symmetric() -> None:
    first = torch.randn(8, 32)
    second = torch.randn(8, 32)

    first_loss = nt_xent_loss(
        first,
        second,
        temperature=0.2,
    )
    second_loss = nt_xent_loss(
        second,
        first,
        temperature=0.2,
    )

    torch.testing.assert_close(
        first_loss,
        second_loss,
        rtol=1e-6,
        atol=1e-6,
    )


def test_matching_pairs_have_lower_loss_than_shuffled() -> None:
    generator = torch.Generator().manual_seed(2026)

    first = torch.randn(
        32,
        64,
        generator=generator,
    )
    second = (
        first
        + 0.01
        * torch.randn(
            32,
            64,
            generator=generator,
        )
    )

    matching_loss = nt_xent_loss(
        first,
        second,
    )
    shuffled_loss = nt_xent_loss(
        first,
        second.roll(
            shifts=1,
            dims=0,
        ),
    )

    assert matching_loss < shuffled_loss


def test_nt_xent_supports_gradient_backpropagation() -> None:
    first = torch.randn(
        8,
        32,
        requires_grad=True,
    )
    second = torch.randn(
        8,
        32,
        requires_grad=True,
    )

    loss = nt_xent_loss(first, second)
    loss.backward()

    assert torch.isfinite(loss)
    assert first.grad is not None
    assert second.grad is not None
    assert torch.all(torch.isfinite(first.grad))
    assert torch.all(torch.isfinite(second.grad))


@pytest.mark.parametrize(
    (
        "first",
        "second",
        "temperature",
        "expected_exception",
    ),
    [
        (
            torch.randn(1, 16),
            torch.randn(1, 16),
            0.1,
            ValueError,
        ),
        (
            torch.randn(4, 16),
            torch.randn(4, 8),
            0.1,
            ValueError,
        ),
        (
            torch.randn(4, 16),
            torch.randn(4, 16),
            0.0,
            ValueError,
        ),
        (
            torch.ones(4, 16, dtype=torch.int64),
            torch.ones(4, 16, dtype=torch.int64),
            0.1,
            TypeError,
        ),
    ],
)
def test_nt_xent_rejects_invalid_inputs(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        nt_xent_loss(
            first,
            second,
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {
            "hidden_dimension": 0,
        },
        {
            "hidden_dimension": -1,
        },
        {
            "output_dimension": 0,
        },
        {
            "output_dimension": -1,
        },
        {
            "hidden_dimension": True,
        },
    ],
)
def test_projection_configuration_rejects_invalid_values(
    keyword_arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ProjectionHeadConfig(
            **keyword_arguments,
        )
