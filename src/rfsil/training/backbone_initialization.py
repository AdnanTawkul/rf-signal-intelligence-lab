from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from rfsil.models.baseline_cnn import (
    BaselineIQCNN,
)
from rfsil.models.model_factory import (
    create_model_from_checkpoint,
)
from rfsil.models.residual_equalizer_cnn import (
    ResidualEqualizerIQCNN,
)


@dataclass(frozen=True, slots=True)
class FrozenBackboneInitialization:
    """Metadata imported from one supervised checkpoint."""

    experiment_name: str
    seed: int
    best_epoch: int
    checkpoint_path: Path


def initialize_frozen_backbone_from_checkpoint(
    model: ResidualEqualizerIQCNN,
    checkpoint_path: str | Path,
) -> FrozenBackboneInitialization:
    """Load and freeze a supervised baseline as the backbone."""
    if not isinstance(
        model,
        ResidualEqualizerIQCNN,
    ):
        raise TypeError(
            "model must be a "
            "ResidualEqualizerIQCNN."
        )

    resolved_path = Path(checkpoint_path)

    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)

    checkpoint = torch.load(
        resolved_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Checkpoint content must be a mapping."
        )

    baseline_model, baseline_configuration = (
        create_model_from_checkpoint(checkpoint)
    )

    if not isinstance(
        baseline_model,
        BaselineIQCNN,
    ):
        raise TypeError(
            "The initialization checkpoint must "
            "contain a BaselineIQCNN."
        )

    expected_configuration = (
        model.backbone.configuration
    )

    if baseline_configuration != expected_configuration:
        raise ValueError(
            "The supervised checkpoint configuration "
            "does not match the equalizer backbone."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(state_dict, Mapping):
        raise ValueError(
            "Checkpoint model_state_dict must "
            "be a mapping."
        )

    model.backbone.load_state_dict(
        state_dict,
        strict=True,
    )
    model.freeze_backbone()

    return FrozenBackboneInitialization(
        experiment_name=str(
            checkpoint.get(
                "experiment_name",
                "",
            )
        ),
        seed=int(checkpoint.get("seed", -1)),
        best_epoch=int(
            checkpoint.get("best_epoch", -1)
        ),
        checkpoint_path=resolved_path,
    )


__all__ = [
    "FrozenBackboneInitialization",
    "initialize_frozen_backbone_from_checkpoint",
]
