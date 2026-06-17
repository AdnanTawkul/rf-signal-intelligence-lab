from __future__ import annotations

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)


def test_extract_features_returns_expected_shape() -> None:
    model = BaselineIQCNN()
    inputs = torch.randn(3, 2, 256)

    embeddings = model.extract_features(inputs)

    assert embeddings.shape == (3, 128)


def test_forward_matches_classifier_applied_to_embeddings() -> None:
    torch.manual_seed(2026)

    model = BaselineIQCNN()
    model.eval()

    inputs = torch.randn(4, 2, 256)

    with torch.inference_mode():
        embeddings = model.extract_features(inputs)
        expected_logits = model.classifier(embeddings)
        actual_logits = model(inputs)

    torch.testing.assert_close(
        actual_logits,
        expected_logits,
    )


def test_extract_features_supports_groupnorm_configuration() -> None:
    configuration = BaselineCNNConfig(
        normalization="group",
        group_norm_groups=8,
    )
    model = BaselineIQCNN(configuration)
    inputs = torch.randn(2, 2, 128)

    embeddings = model.extract_features(inputs)

    assert embeddings.shape == (2, 128)
    assert torch.all(torch.isfinite(embeddings))


def test_extract_features_preserves_gradient_flow() -> None:
    model = BaselineIQCNN()
    inputs = torch.randn(
        2,
        2,
        128,
        requires_grad=True,
    )

    embeddings = model.extract_features(inputs)
    embeddings.mean().backward()

    assert inputs.grad is not None
    assert torch.all(torch.isfinite(inputs.grad))


@pytest.mark.parametrize(
    "inputs",
    [
        torch.randn(2, 128),
        torch.randn(2, 3, 128),
        torch.randn(2, 2, 4),
    ],
)
def test_extract_features_reuses_input_validation(
    inputs: torch.Tensor,
) -> None:
    model = BaselineIQCNN()

    with pytest.raises(ValueError):
        model.extract_features(inputs)
