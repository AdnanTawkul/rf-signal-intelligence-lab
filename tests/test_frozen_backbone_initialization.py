from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
    count_trainable_parameters,
)
from rfsil.models.residual_equalizer_cnn import (
    ResidualEqualizerCNNConfig,
    ResidualEqualizerIQCNN,
)
from rfsil.training.backbone_initialization import (
    initialize_frozen_backbone_from_checkpoint,
)


def save_baseline_checkpoint(
    path: Path,
    configuration: BaselineCNNConfig,
) -> BaselineIQCNN:
    """Save one synthetic supervised baseline checkpoint."""
    torch.manual_seed(2026)

    model = BaselineIQCNN(configuration)

    torch.save(
        {
            "format_version": 1,
            "experiment_name": "test_baseline",
            "seed": 2026,
            "best_epoch": 17,
            "model_configuration": (
                asdict(configuration)
            ),
            "model_state_dict": (
                model.state_dict()
            ),
        },
        path,
    )

    return model


def create_equalizer() -> ResidualEqualizerIQCNN:
    """Create the matching residual equalizer."""
    return ResidualEqualizerIQCNN(
        ResidualEqualizerCNNConfig(
            normalization="group",
            group_norm_groups=8,
        )
    )


def test_frozen_backbone_matches_baseline(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path / "baseline.pt"
    )

    configuration = BaselineCNNConfig(
        normalization="group",
        group_norm_groups=8,
    )
    baseline = save_baseline_checkpoint(
        checkpoint_path,
        configuration,
    )
    equalizer = create_equalizer()

    metadata = (
        initialize_frozen_backbone_from_checkpoint(
            equalizer,
            checkpoint_path,
        )
    )

    baseline.eval()
    equalizer.eval()

    inputs = torch.randn(4, 2, 256)

    with torch.inference_mode():
        baseline_logits = baseline(inputs)
        equalizer_logits = equalizer(inputs)

    torch.testing.assert_close(
        equalizer_logits,
        baseline_logits,
        rtol=0.0,
        atol=0.0,
    )

    assert metadata.seed == 2026
    assert metadata.best_epoch == 17
    assert equalizer.backbone_frozen
    assert count_trainable_parameters(
        equalizer
    ) == 2_944


def test_frozen_backbone_remains_in_eval_mode(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path / "baseline.pt"
    )

    save_baseline_checkpoint(
        checkpoint_path,
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
        ),
    )

    equalizer = create_equalizer()

    initialize_frozen_backbone_from_checkpoint(
        equalizer,
        checkpoint_path,
    )

    equalizer.train()

    assert equalizer.training
    assert equalizer.equalizer.training
    assert not equalizer.backbone.training

    assert all(
        not parameter.requires_grad
        for parameter
        in equalizer.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter
        in equalizer.equalizer.parameters()
    )


def test_mismatched_backbone_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path / "baseline.pt"
    )

    save_baseline_checkpoint(
        checkpoint_path,
        BaselineCNNConfig(
            channels=(16, 32, 64),
            normalization="group",
            group_norm_groups=8,
        ),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        initialize_frozen_backbone_from_checkpoint(
            create_equalizer(),
            checkpoint_path,
        )


def test_wrong_target_model_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = (
        tmp_path / "baseline.pt"
    )

    baseline = save_baseline_checkpoint(
        checkpoint_path,
        BaselineCNNConfig(
            normalization="group",
            group_norm_groups=8,
        ),
    )

    with pytest.raises(
        TypeError,
        match="ResidualEqualizerIQCNN",
    ):
        initialize_frozen_backbone_from_checkpoint(
            baseline,  # type: ignore[arg-type]
            checkpoint_path,
        )
