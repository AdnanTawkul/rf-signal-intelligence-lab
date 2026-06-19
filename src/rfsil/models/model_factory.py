from __future__ import annotations

from collections.abc import Mapping

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)
from rfsil.models.late_fusion_cnn import (
    LateFusionCNNConfig,
    LateFusionIQDPhaseCNN,
)
from rfsil.models.residual_equalizer_cnn import (
    ResidualEqualizerCNNConfig,
    ResidualEqualizerIQCNN,
)

type ModelConfiguration = (
    BaselineCNNConfig
    | LateFusionCNNConfig
    | ResidualEqualizerCNNConfig
)

type ClassifierModel = (
    BaselineIQCNN
    | LateFusionIQDPhaseCNN
    | ResidualEqualizerIQCNN
)


def create_model_from_mapping(
    content: Mapping[str, object],
) -> tuple[
    ClassifierModel,
    ModelConfiguration,
]:
    """Build one supervised classifier from configuration values."""
    if not isinstance(content, Mapping):
        raise ValueError(
            "Model configuration must be "
            "a mapping."
        )

    model_type_value = content.get(
        "model_type",
        "baseline_iq_cnn",
    )

    if not isinstance(
        model_type_value,
        str,
    ):
        raise ValueError(
            "model_type must be a string."
        )

    model_type = (
        model_type_value.strip().lower()
    )

    if model_type in {
        "baseline",
        "baseline_iq_cnn",
        "iq_cnn",
    }:
        configuration = (
            BaselineCNNConfig.from_mapping(
                content
            )
        )

        return (
            BaselineIQCNN(configuration),
            configuration,
        )

    if model_type == (
        "late_fusion_iq_dphase"
    ):
        configuration = (
            LateFusionCNNConfig.from_mapping(
                content
            )
        )

        return (
            LateFusionIQDPhaseCNN(
                configuration
            ),
            configuration,
        )

    if model_type == (
        "residual_equalizer_iq_cnn"
    ):
        configuration = (
            ResidualEqualizerCNNConfig
            .from_mapping(content)
        )

        return (
            ResidualEqualizerIQCNN(
                configuration
            ),
            configuration,
        )

    raise ValueError(
        f"Unsupported model_type: "
        f"{model_type!r}."
    )


def create_model_from_checkpoint(
    checkpoint: Mapping[str, object],
) -> tuple[
    ClassifierModel,
    ModelConfiguration,
]:
    """Reconstruct a model from checkpoint metadata."""
    content = checkpoint.get(
        "model_configuration"
    )

    if not isinstance(content, Mapping):
        raise ValueError(
            "Checkpoint model_configuration "
            "must be a mapping."
        )

    return create_model_from_mapping(
        content
    )


__all__ = [
    "ClassifierModel",
    "ModelConfiguration",
    "create_model_from_checkpoint",
    "create_model_from_mapping",
]
