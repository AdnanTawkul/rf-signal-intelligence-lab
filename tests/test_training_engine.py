from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from rfsil.models.baseline_cnn import BaselineIQCNN
from rfsil.training.engine import (
    run_evaluation_epoch,
    run_training_epoch,
    set_global_seed,
)


class TinyIQDataset(Dataset[dict[str, Tensor]]):
    """Small deterministic dataset for training-engine tests."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(42)
        self.iq = torch.randn(
            12,
            2,
            128,
            generator=generator,
        )
        self.labels = torch.tensor(
            [0, 1, 2, 3] * 3,
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "iq": self.iq[index],
            "label": self.labels[index],
        }


def create_loader() -> DataLoader:
    """Create a deterministic test DataLoader."""
    return DataLoader(
        TinyIQDataset(),
        batch_size=4,
        shuffle=False,
    )


def test_training_epoch_updates_model_parameters() -> None:
    set_global_seed(42)
    model = BaselineIQCNN()
    optimizer = Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    before = copy.deepcopy(model.state_dict())

    metrics = run_training_epoch(
        model=model,
        data_loader=create_loader(),
        optimizer=optimizer,
        loss_function=loss_function,
        device=torch.device("cpu"),
    )

    changed = any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )

    assert changed
    assert metrics.example_count == 12
    assert metrics.loss > 0.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_evaluation_epoch_does_not_update_parameters() -> None:
    model = BaselineIQCNN()
    loss_function = nn.CrossEntropyLoss()
    before = copy.deepcopy(model.state_dict())

    metrics = run_evaluation_epoch(
        model=model,
        data_loader=create_loader(),
        loss_function=loss_function,
        device=torch.device("cpu"),
    )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])

    assert metrics.example_count == 12
    assert metrics.loss > 0.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_seed_reproduces_model_initialization() -> None:
    set_global_seed(123)
    model_a = BaselineIQCNN()

    set_global_seed(123)
    model_b = BaselineIQCNN()

    for parameter_a, parameter_b in zip(
        model_a.parameters(),
        model_b.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            parameter_a,
            parameter_b,
        )


def test_training_epoch_rejects_empty_loader() -> None:
    empty_loader = DataLoader(
        [],
        batch_size=4,
    )

    with pytest.raises(ValueError):
        run_training_epoch(
            model=BaselineIQCNN(),
            data_loader=empty_loader,
            optimizer=Adam(
                BaselineIQCNN().parameters(),
                lr=0.001,
            ),
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
