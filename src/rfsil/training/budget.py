from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    """Derived optimizer-step budget for one training run."""

    example_count: int
    batch_size: int
    drop_last: bool
    steps_per_epoch: int
    target_optimizer_steps: int
    epochs: int
    actual_optimizer_steps: int
    exact_match: bool


def _validate_positive_integer(
    value: object,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"{name} must be an integer."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return validated


def calculate_steps_per_epoch(
    *,
    example_count: int,
    batch_size: int,
    drop_last: bool = False,
) -> int:
    """Calculate optimizer updates produced by one epoch."""
    validated_example_count = (
        _validate_positive_integer(
            example_count,
            "example_count",
        )
    )
    validated_batch_size = (
        _validate_positive_integer(
            batch_size,
            "batch_size",
        )
    )

    if not isinstance(drop_last, bool):
        raise ValueError(
            "drop_last must be a boolean."
        )

    if drop_last:
        steps_per_epoch = (
            validated_example_count
            // validated_batch_size
        )
    else:
        steps_per_epoch = math.ceil(
            validated_example_count
            / validated_batch_size
        )

    if steps_per_epoch <= 0:
        raise ValueError(
            "The DataLoader would produce no "
            "training batches."
        )

    return steps_per_epoch


def derive_training_budget(
    *,
    example_count: int,
    batch_size: int,
    target_optimizer_steps: int,
    drop_last: bool = False,
    require_exact: bool = True,
) -> TrainingBudget:
    """Derive epochs needed for a target update budget."""
    validated_example_count = (
        _validate_positive_integer(
            example_count,
            "example_count",
        )
    )
    validated_batch_size = (
        _validate_positive_integer(
            batch_size,
            "batch_size",
        )
    )
    validated_target_steps = (
        _validate_positive_integer(
            target_optimizer_steps,
            "target_optimizer_steps",
        )
    )

    if not isinstance(require_exact, bool):
        raise ValueError(
            "require_exact must be a boolean."
        )

    steps_per_epoch = (
        calculate_steps_per_epoch(
            example_count=(
                validated_example_count
            ),
            batch_size=(
                validated_batch_size
            ),
            drop_last=drop_last,
        )
    )

    quotient, remainder = divmod(
        validated_target_steps,
        steps_per_epoch,
    )

    if require_exact and remainder != 0:
        raise ValueError(
            "target_optimizer_steps is not "
            "exactly divisible by "
            "steps_per_epoch."
        )

    if remainder == 0:
        epochs = quotient
    else:
        epochs = math.ceil(
            validated_target_steps
            / steps_per_epoch
        )

    actual_optimizer_steps = (
        epochs * steps_per_epoch
    )

    return TrainingBudget(
        example_count=(
            validated_example_count
        ),
        batch_size=validated_batch_size,
        drop_last=drop_last,
        steps_per_epoch=steps_per_epoch,
        target_optimizer_steps=(
            validated_target_steps
        ),
        epochs=epochs,
        actual_optimizer_steps=(
            actual_optimizer_steps
        ),
        exact_match=(
            actual_optimizer_steps
            == validated_target_steps
        ),
    )


__all__ = [
    "TrainingBudget",
    "calculate_steps_per_epoch",
    "derive_training_budget",
]
