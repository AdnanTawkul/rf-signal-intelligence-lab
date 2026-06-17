from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from rfsil.training.engine import run_training_epoch
from rfsil.training.losses import (
    ClassSNRWeightedCrossEntropyLoss,
)


def create_loss(
    weight: float = 2.0,
) -> ClassSNRWeightedCrossEntropyLoss:
    """Create the targeted 8PSK low-SNR loss."""
    return ClassSNRWeightedCrossEntropyLoss(
        target_class_index=2,
        target_snr_values_db=(-4.0, 0.0),
        target_weight=weight,
    )


def test_no_matching_examples_equals_standard_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0, 0.5],
            [0.0, 2.0, 0.5, -1.0],
        ],
    )
    labels = torch.tensor([0, 1])
    snr_db = torch.tensor([-4.0, 0.0])

    weighted = create_loss()(logits, labels, snr_db)
    standard = functional.cross_entropy(logits, labels)

    torch.testing.assert_close(weighted, standard)


def test_targeted_example_receives_extra_weight() -> None:
    logits = torch.tensor(
        [
            [3.0, 0.0, -2.0, 0.0],
            [0.0, 3.0, -2.0, 0.0],
        ],
    )
    labels = torch.tensor([2, 1])
    snr_db = torch.tensor([-4.0, -4.0])

    per_example = functional.cross_entropy(
        logits,
        labels,
        reduction="none",
    )
    expected = (
        per_example[0] * 2.0
        + per_example[1]
    ) / 3.0

    actual = create_loss()(logits, labels, snr_db)

    torch.testing.assert_close(actual, expected)


def test_non_target_snr_is_not_weighted() -> None:
    logits = torch.randn(3, 4)
    labels = torch.tensor([2, 2, 2])
    snr_db = torch.tensor([4.0, 8.0, 12.0])

    weighted = create_loss()(logits, labels, snr_db)
    standard = functional.cross_entropy(logits, labels)

    torch.testing.assert_close(weighted, standard)


def test_weighted_loss_produces_finite_gradients() -> None:
    logits = torch.randn(
        8,
        4,
        requires_grad=True,
    )
    labels = torch.tensor(
        [0, 1, 2, 3, 2, 1, 2, 0],
    )
    snr_db = torch.tensor(
        [-4.0, 0.0, -4.0, 4.0, 0.0, 8.0, 12.0, 20.0],
    )

    loss = create_loss()(logits, labels, snr_db)
    loss.backward()

    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_class_index", -1),
        ("target_snr_values_db", ()),
        ("target_snr_values_db", (float("nan"),)),
        ("target_weight", 0.0),
        ("target_weight", -1.0),
        ("snr_tolerance", 0.0),
    ],
)
def test_configuration_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "target_class_index": 2,
        "target_snr_values_db": (-4.0, 0.0),
        "target_weight": 2.0,
        "snr_tolerance": 1e-4,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        ClassSNRWeightedCrossEntropyLoss(
            **arguments,  # type: ignore[arg-type]
        )


def test_training_engine_accepts_metadata_aware_loss() -> None:
    torch.manual_seed(42)

    examples = [
        {
            "iq": torch.randn(2, 16),
            "label": torch.tensor(index % 4),
            "snr_db": torch.tensor(
                -4.0 if index % 2 == 0 else 0.0
            ),
        }
        for index in range(8)
    ]
    loader = DataLoader(
        examples,
        batch_size=4,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32, 4),
    )
    optimizer = SGD(
        model.parameters(),
        lr=0.01,
    )

    metrics = run_training_epoch(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        loss_function=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        metadata_loss_function=create_loss(),
    )

    assert metrics.example_count == 8
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss > 0.0


def test_training_engine_requires_snr_metadata() -> None:
    examples = [
        {
            "iq": torch.randn(2, 16),
            "label": torch.tensor(2),
        }
    ]
    loader = DataLoader(
        examples,
        batch_size=1,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32, 4),
    )
    optimizer = SGD(
        model.parameters(),
        lr=0.01,
    )

    with pytest.raises(KeyError):
        run_training_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            loss_function=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            metadata_loss_function=create_loss(),
        )
