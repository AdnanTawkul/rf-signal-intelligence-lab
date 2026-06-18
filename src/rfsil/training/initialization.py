from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rfsil.models.baseline_cnn import BaselineIQCNN


@dataclass(frozen=True, slots=True)
class EncoderInitialization:
    """Metadata describing an imported SSL encoder."""

    method: str
    experiment_name: str | None
    seed: int | None
    best_epoch: int | None


def _optional_integer(
    value: object,
    name: str,
) -> int | None:
    """Validate an optional integer metadata value."""
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer or null."
        )

    return int(value)


def _load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Load and validate an SSL checkpoint mapping."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"SSL checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "SSL checkpoint must contain a mapping."
        )

    return checkpoint


def _validate_tensor_mapping(
    value: object,
) -> dict[str, Tensor]:
    """Validate an encoder state-dictionary mapping."""
    if not isinstance(value, dict):
        raise ValueError(
            "SSL checkpoint encoder_state_dict "
            "must be a mapping."
        )

    validated: dict[str, Tensor] = {}

    for key, tensor in value.items():
        if not isinstance(key, str):
            raise ValueError(
                "Encoder state-dictionary keys "
                "must be strings."
            )

        if not isinstance(tensor, Tensor):
            raise ValueError(
                "Encoder state-dictionary values "
                "must be tensors."
            )

        validated[key] = tensor

    return validated


def initialize_encoder_from_ssl_checkpoint(
    model: BaselineIQCNN,
    checkpoint_path: str | Path,
) -> EncoderInitialization:
    """Load only encoder parameters from an SSL checkpoint.

    The supervised classifier remains exactly as initialized by the
    caller. SimCLR and VICReg checkpoints are supported.
    """
    if not isinstance(model, BaselineIQCNN):
        raise TypeError(
            "model must be a BaselineIQCNN."
        )

    resolved_checkpoint_path = Path(
        checkpoint_path
    )
    checkpoint = _load_checkpoint(
        resolved_checkpoint_path
    )

    method = checkpoint.get("method")

    if method not in {"simclr", "vicreg"}:
        raise ValueError(
            "SSL checkpoint method must be "
            "'simclr' or 'vicreg'."
        )

    checkpoint_state = _validate_tensor_mapping(
        checkpoint.get("encoder_state_dict")
    )

    imported_encoder_state = {
        key: tensor
        for key, tensor in checkpoint_state.items()
        if not key.startswith("classifier.")
    }

    model_state = model.state_dict()

    expected_encoder_keys = {
        key
        for key in model_state
        if not key.startswith("classifier.")
    }
    imported_encoder_keys = set(
        imported_encoder_state
    )

    missing_keys = sorted(
        expected_encoder_keys
        - imported_encoder_keys
    )
    unexpected_keys = sorted(
        imported_encoder_keys
        - expected_encoder_keys
    )

    if missing_keys:
        raise ValueError(
            "SSL checkpoint is missing encoder keys: "
            + ", ".join(missing_keys)
        )

    if unexpected_keys:
        raise ValueError(
            "SSL checkpoint contains unexpected "
            "encoder keys: "
            + ", ".join(unexpected_keys)
        )

    for key in sorted(expected_encoder_keys):
        source_tensor = imported_encoder_state[key]
        target_tensor = model_state[key]

        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                "Encoder tensor shape mismatch for "
                f"{key}: checkpoint="
                f"{tuple(source_tensor.shape)}, model="
                f"{tuple(target_tensor.shape)}."
            )

        if not torch.all(torch.isfinite(source_tensor)):
            raise ValueError(
                "SSL checkpoint contains non-finite "
                f"values for encoder tensor {key}."
            )

    classifier_state_before = {
        key: tensor.detach().clone()
        for key, tensor in model_state.items()
        if key.startswith("classifier.")
    }

    updated_state = {
        key: tensor.detach().clone()
        for key, tensor in model_state.items()
    }

    for key, tensor in imported_encoder_state.items():
        updated_state[key] = tensor.detach().clone()

    model.load_state_dict(
        updated_state,
        strict=True,
    )

    loaded_state = model.state_dict()

    for key, expected_tensor in (
        classifier_state_before.items()
    ):
        if not torch.equal(
            loaded_state[key],
            expected_tensor,
        ):
            raise RuntimeError(
                "Supervised classifier changed while "
                "loading the SSL encoder."
            )

    experiment_name_value = checkpoint.get(
        "experiment_name"
    )

    if (
        experiment_name_value is not None
        and not isinstance(
            experiment_name_value,
            str,
        )
    ):
        raise ValueError(
            "experiment_name must be a string or null."
        )

    return EncoderInitialization(
        method=str(method),
        experiment_name=experiment_name_value,
        seed=_optional_integer(
            checkpoint.get("seed"),
            "seed",
        ),
        best_epoch=_optional_integer(
            checkpoint.get("best_epoch"),
            "best_epoch",
        ),
    )


__all__ = [
    "EncoderInitialization",
    "initialize_encoder_from_ssl_checkpoint",
]
