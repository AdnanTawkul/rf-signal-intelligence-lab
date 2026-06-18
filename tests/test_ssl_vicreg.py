from __future__ import annotations

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)
from rfsil.ssl.contrastive import (
    ProjectionHeadConfig,
)
from rfsil.ssl.vicreg import (
    VICRegLossConfig,
    VICRegModel,
    compute_vicreg_loss,
)


def create_encoder() -> BaselineIQCNN:
    return BaselineIQCNN(
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
        )
    )


def test_vicreg_model_returns_expected_shape() -> None:
    model = VICRegModel(
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

    assert projections.shape == (4, 32)


def test_vicreg_encode_matches_encoder() -> None:
    encoder = create_encoder()
    model = VICRegModel(encoder)
    model.eval()

    inputs = torch.randn(3, 2, 256)

    with torch.inference_mode():
        expected = encoder.extract_features(inputs)
        actual = model.encode(inputs)

    torch.testing.assert_close(
        actual,
        expected,
    )


def test_vicreg_loss_is_symmetric() -> None:
    first = torch.randn(16, 32)
    second = torch.randn(16, 32)

    first_terms = compute_vicreg_loss(
        first,
        second,
    )
    second_terms = compute_vicreg_loss(
        second,
        first,
    )

    torch.testing.assert_close(
        first_terms.total,
        second_terms.total,
    )


def test_identical_views_have_zero_invariance() -> None:
    projections = torch.randn(16, 32)

    terms = compute_vicreg_loss(
        projections,
        projections,
    )

    torch.testing.assert_close(
        terms.invariance,
        torch.zeros_like(terms.invariance),
    )


def test_collapsed_projections_receive_variance_penalty() -> None:
    projections = torch.zeros(16, 32)

    terms = compute_vicreg_loss(
        projections,
        projections,
    )

    assert terms.variance > 0.0


def test_total_matches_weighted_components() -> None:
    configuration = VICRegLossConfig(
        invariance_weight=2.0,
        variance_weight=3.0,
        covariance_weight=4.0,
    )
    first = torch.randn(16, 32)
    second = torch.randn(16, 32)

    terms = compute_vicreg_loss(
        first,
        second,
        configuration,
    )

    expected = (
        2.0 * terms.invariance
        + 3.0 * terms.variance
        + 4.0 * terms.covariance
    )

    torch.testing.assert_close(
        terms.total,
        expected,
    )


def test_vicreg_loss_supports_backpropagation() -> None:
    first = torch.randn(
        16,
        32,
        requires_grad=True,
    )
    second = torch.randn(
        16,
        32,
        requires_grad=True,
    )

    terms = compute_vicreg_loss(
        first,
        second,
    )
    terms.total.backward()

    assert first.grad is not None
    assert second.grad is not None
    assert torch.all(
        torch.isfinite(first.grad)
    )
    assert torch.all(
        torch.isfinite(second.grad)
    )


@pytest.mark.parametrize(
    (
        "first",
        "second",
        "expected_exception",
    ),
    [
        (
            torch.randn(1, 16),
            torch.randn(1, 16),
            ValueError,
        ),
        (
            torch.randn(4, 16),
            torch.randn(4, 8),
            ValueError,
        ),
        (
            torch.randn(4, 16, 1),
            torch.randn(4, 16, 1),
            ValueError,
        ),
        (
            torch.ones(
                4,
                16,
                dtype=torch.int64,
            ),
            torch.ones(
                4,
                16,
                dtype=torch.int64,
            ),
            TypeError,
        ),
        (
            torch.full(
                (4, 16),
                float("nan"),
            ),
            torch.randn(4, 16),
            ValueError,
        ),
    ],
)
def test_vicreg_rejects_invalid_inputs(
    first: torch.Tensor,
    second: torch.Tensor,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        compute_vicreg_loss(
            first,
            second,
        )


@pytest.mark.parametrize(
    "keyword_arguments",
    [
        {
            "invariance_weight": -1.0,
        },
        {
            "variance_weight": -1.0,
        },
        {
            "covariance_weight": -1.0,
        },
        {
            "invariance_weight": 0.0,
            "variance_weight": 0.0,
            "covariance_weight": 0.0,
        },
        {
            "target_standard_deviation": 0.0,
        },
        {
            "epsilon": 0.0,
        },
    ],
)
def test_invalid_vicreg_configuration_is_rejected(
    keyword_arguments: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        VICRegLossConfig(
            **keyword_arguments,
        )
