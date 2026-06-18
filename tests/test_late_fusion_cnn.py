from __future__ import annotations

import math
from dataclasses import asdict

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineIQCNN,
    count_trainable_parameters,
)
from rfsil.models.late_fusion_cnn import (
    LateFusionCNNConfig,
    LateFusionIQDPhaseCNN,
)
from rfsil.models.model_factory import (
    create_model_from_checkpoint,
    create_model_from_mapping,
)


def test_late_fusion_parameter_count() -> None:
    model = LateFusionIQDPhaseCNN()

    assert (
        count_trainable_parameters(model)
        == 70_676
    )


def test_late_fusion_forward_shape() -> None:
    model = LateFusionIQDPhaseCNN()
    inputs = torch.randn(4, 2, 512)

    logits = model(inputs)

    assert logits.shape == (4, 4)
    assert torch.all(torch.isfinite(logits))


def test_late_fusion_embedding_shapes() -> None:
    model = LateFusionIQDPhaseCNN()
    inputs = torch.randn(3, 2, 256)

    iq_features, dphase_features = (
        model.extract_branch_features(inputs)
    )
    fused = model.extract_features(inputs)

    assert iq_features.shape == (3, 64)
    assert dphase_features.shape == (3, 64)
    assert fused.shape == (3, 256)


def test_dphase_branch_ignores_gain_and_global_phase(
) -> None:
    torch.manual_seed(2026)

    model = LateFusionIQDPhaseCNN()
    model.eval()

    inputs = torch.randn(3, 2, 256)
    angle = 0.81
    cosine = math.cos(angle)
    sine = math.sin(angle)

    transformed = torch.stack(
        (
            inputs[:, 0] * cosine
            - inputs[:, 1] * sine,
            inputs[:, 0] * sine
            + inputs[:, 1] * cosine,
        ),
        dim=1,
    ) * 2.7

    with torch.inference_mode():
        _, original_dphase = (
            model.extract_branch_features(
                inputs
            )
        )
        _, transformed_dphase = (
            model.extract_branch_features(
                transformed
            )
        )

    torch.testing.assert_close(
        original_dphase,
        transformed_dphase,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "inputs",
    [
        torch.randn(2, 3, 128),
        torch.randn(2, 2, 7),
        torch.randn(2, 128),
    ],
)
def test_late_fusion_rejects_invalid_inputs(
    inputs: torch.Tensor,
) -> None:
    model = LateFusionIQDPhaseCNN()

    with pytest.raises(ValueError):
        model(inputs)


@pytest.mark.parametrize(
    "content",
    [
        {
            "in_channels": 3,
        },
        {
            "kernel_size": 6,
        },
        {
            "branch_channels": (
                16,
                30,
                64,
            ),
        },
    ],
)
def test_invalid_late_fusion_config(
    content: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        LateFusionCNNConfig(
            **content,  # type: ignore[arg-type]
        )


def test_factory_preserves_legacy_baseline() -> None:
    model, configuration = (
        create_model_from_mapping(
            {
                "in_channels": 2,
                "num_classes": 4,
                "channels": [
                    32,
                    64,
                    128,
                ],
                "kernel_size": 7,
                "dropout": 0.2,
                "normalization": "group",
                "group_norm_groups": 8,
            }
        )
    )

    assert isinstance(
        model,
        BaselineIQCNN,
    )
    assert (
        configuration.in_channels
        == 2
    )


def test_factory_builds_late_fusion() -> None:
    model, configuration = (
        create_model_from_mapping(
            {
                "model_type": (
                    "late_fusion_iq_dphase"
                ),
                "branch_channels": [
                    16,
                    32,
                    64,
                ],
                "normalization": "group",
                "group_norm_groups": 8,
                "fusion_hidden": 256,
            }
        )
    )

    assert isinstance(
        model,
        LateFusionIQDPhaseCNN,
    )
    assert isinstance(
        configuration,
        LateFusionCNNConfig,
    )


def test_checkpoint_factory_reconstructs_late_fusion(
) -> None:
    configuration = LateFusionCNNConfig()

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
        LateFusionIQDPhaseCNN,
    )
    assert restored == configuration


def test_unknown_model_type_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported model_type",
    ):
        create_model_from_mapping(
            {
                "model_type": "unknown",
            }
        )
