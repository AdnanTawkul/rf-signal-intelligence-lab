from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)
from rfsil.training.initialization import (
    initialize_encoder_from_ssl_checkpoint,
)


def create_model() -> BaselineIQCNN:
    return BaselineIQCNN(
        BaselineCNNConfig(
            channels=(8, 16, 32),
            normalization="group",
            group_norm_groups=8,
        )
    )


def create_checkpoint(
    path: Path,
    *,
    method: str = "simclr",
    state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    source_model = create_model()

    encoder_state = (
        {
            key: tensor.detach().clone()
            for key, tensor
            in source_model.state_dict().items()
        }
        if state is None
        else state
    )

    torch.save(
        {
            "format_version": 1,
            "method": method,
            "experiment_name": "ssl_test",
            "seed": 2026,
            "best_epoch": 12,
            "encoder_state_dict": encoder_state,
        },
        path,
    )

    return encoder_state


@pytest.mark.parametrize(
    "method",
    ["simclr", "vicreg"],
)
def test_supported_ssl_checkpoint_is_loaded(
    tmp_path: Path,
    method: str,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    source_state = create_checkpoint(
        checkpoint_path,
        method=method,
    )

    target_model = create_model()

    classifier_before = {
        key: tensor.detach().clone()
        for key, tensor
        in target_model.state_dict().items()
        if key.startswith("classifier.")
    }

    metadata = (
        initialize_encoder_from_ssl_checkpoint(
            target_model,
            checkpoint_path,
        )
    )

    loaded_state = target_model.state_dict()

    for key, tensor in source_state.items():
        if not key.startswith("classifier."):
            torch.testing.assert_close(
                loaded_state[key],
                tensor,
            )

    for key, tensor in classifier_before.items():
        torch.testing.assert_close(
            loaded_state[key],
            tensor,
        )

    assert metadata.method == method
    assert metadata.experiment_name == "ssl_test"
    assert metadata.seed == 2026
    assert metadata.best_epoch == 12


def test_ssl_classifier_is_not_imported(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    source_state = create_checkpoint(
        checkpoint_path
    )

    source_state["classifier.1.weight"].fill_(99.0)
    source_state["classifier.1.bias"].fill_(99.0)

    create_checkpoint(
        checkpoint_path,
        state=source_state,
    )

    target_model = create_model()

    classifier_before = (
        target_model.classifier[1]
        .weight.detach().clone()
    )

    initialize_encoder_from_ssl_checkpoint(
        target_model,
        checkpoint_path,
    )

    torch.testing.assert_close(
        target_model.classifier[1].weight,
        classifier_before,
    )


def test_missing_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            tmp_path / "missing.pt",
        )


def test_non_mapping_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    torch.save(["not", "a", "mapping"], checkpoint_path)

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )


def test_unsupported_method_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    create_checkpoint(
        checkpoint_path,
        method="unknown",
    )

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )


def test_missing_encoder_state_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"

    torch.save(
        {
            "method": "simclr",
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )


def test_missing_encoder_key_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    state = create_checkpoint(checkpoint_path)

    key = next(
        value
        for value in state
        if not value.startswith("classifier.")
    )
    del state[key]

    create_checkpoint(
        checkpoint_path,
        state=state,
    )

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )


def test_unexpected_encoder_key_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    state = create_checkpoint(checkpoint_path)
    state["features.unexpected.weight"] = (
        torch.ones(1)
    )

    create_checkpoint(
        checkpoint_path,
        state=state,
    )

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )


def test_shape_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    state = create_checkpoint(checkpoint_path)

    key = next(
        value
        for value, tensor in state.items()
        if (
            not value.startswith("classifier.")
            and tensor.ndim > 0
        )
    )
    state[key] = torch.zeros(1)

    create_checkpoint(
        checkpoint_path,
        state=state,
    )

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )


def test_non_finite_encoder_tensor_is_rejected(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    state = create_checkpoint(checkpoint_path)

    key = next(
        value
        for value, tensor in state.items()
        if (
            not value.startswith("classifier.")
            and torch.is_floating_point(tensor)
        )
    )

    state[key] = state[key].clone()
    state[key].reshape(-1)[0] = float("nan")

    create_checkpoint(
        checkpoint_path,
        state=state,
    )

    with pytest.raises(ValueError):
        initialize_encoder_from_ssl_checkpoint(
            create_model(),
            checkpoint_path,
        )
