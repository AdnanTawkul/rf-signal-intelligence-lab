from __future__ import annotations

import pytest

from rfsil.training.budget import (
    calculate_steps_per_epoch,
    derive_training_budget,
)


@pytest.mark.parametrize(
    (
        "example_count",
        "expected_steps_per_epoch",
        "expected_epochs",
    ),
    (
        (56, 1, 1320),
        (280, 3, 440),
        (560, 5, 264),
        (1400, 11, 120),
        (5600, 44, 30),
    ),
)
def test_derives_exact_ssl_fraction_budgets(
    example_count: int,
    expected_steps_per_epoch: int,
    expected_epochs: int,
) -> None:
    budget = derive_training_budget(
        example_count=example_count,
        batch_size=128,
        target_optimizer_steps=1320,
        drop_last=False,
    )

    assert (
        budget.steps_per_epoch
        == expected_steps_per_epoch
    )
    assert budget.epochs == expected_epochs
    assert (
        budget.actual_optimizer_steps
        == 1320
    )
    assert budget.exact_match is True


def test_steps_per_epoch_keeps_partial_batch() -> None:
    steps = calculate_steps_per_epoch(
        example_count=280,
        batch_size=128,
        drop_last=False,
    )

    assert steps == 3


def test_steps_per_epoch_drops_partial_batch() -> None:
    steps = calculate_steps_per_epoch(
        example_count=280,
        batch_size=128,
        drop_last=True,
    )

    assert steps == 2


def test_rejects_empty_drop_last_loader() -> None:
    with pytest.raises(
        ValueError,
        match="no training batches",
    ):
        calculate_steps_per_epoch(
            example_count=56,
            batch_size=128,
            drop_last=True,
        )


def test_rejects_nondivisible_exact_budget() -> None:
    with pytest.raises(
        ValueError,
        match="not exactly divisible",
    ):
        derive_training_budget(
            example_count=280,
            batch_size=128,
            target_optimizer_steps=1000,
            drop_last=False,
            require_exact=True,
        )


def test_can_round_up_nonexact_budget() -> None:
    budget = derive_training_budget(
        example_count=280,
        batch_size=128,
        target_optimizer_steps=1000,
        drop_last=False,
        require_exact=False,
    )

    assert budget.steps_per_epoch == 3
    assert budget.epochs == 334
    assert (
        budget.actual_optimizer_steps
        == 1002
    )
    assert budget.exact_match is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("example_count", 0),
        ("example_count", True),
        ("batch_size", 0),
        ("batch_size", 2.5),
        ("target_optimizer_steps", -1),
    ),
)
def test_rejects_invalid_positive_integer(
    field: str,
    value: object,
) -> None:
    arguments = {
        "example_count": 280,
        "batch_size": 128,
        "target_optimizer_steps": 1320,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        derive_training_budget(**arguments)


def test_rejects_nonboolean_drop_last() -> None:
    with pytest.raises(
        ValueError,
        match="drop_last must be a boolean",
    ):
        calculate_steps_per_epoch(
            example_count=280,
            batch_size=128,
            drop_last=0,
        )


def test_rejects_nonboolean_exact_flag() -> None:
    with pytest.raises(
        ValueError,
        match="require_exact must be a boolean",
    ):
        derive_training_budget(
            example_count=280,
            batch_size=128,
            target_optimizer_steps=1320,
            require_exact=1,
        )
