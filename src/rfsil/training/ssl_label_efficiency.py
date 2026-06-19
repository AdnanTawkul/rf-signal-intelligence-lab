from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rfsil.training.budget import (
    TrainingBudget,
    derive_training_budget,
)

_IDENTIFIER_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_]*$"
)


@dataclass(frozen=True, slots=True)
class LabelFraction:
    """One class-SNR-stratified labeled-data level."""

    identifier: str
    display_name: str
    examples_per_class_snr: int


@dataclass(frozen=True, slots=True)
class SweepRunPlan:
    """One generated downstream-training run."""

    fraction: LabelFraction
    method: str
    seed: int
    selected_training_examples: int
    budget: TrainingBudget
    content: dict[str, Any]
    generated_config_path: Path
    output_directory: Path


def _validate_positive_integer(
    value: object,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(value)


def validate_seeds(
    value: object,
) -> tuple[int, ...]:
    """Validate unique downstream seeds."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            "seeds must be a non-empty list."
        )

    seeds = tuple(
        _validate_positive_integer(
            seed,
            "seed",
        )
        for seed in value
    )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "seeds must not contain duplicates."
        )

    return seeds


def validate_label_fractions(
    value: object,
) -> tuple[LabelFraction, ...]:
    """Validate configured labeled-data levels."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            "label_fractions must be a non-empty list."
        )

    fractions: list[LabelFraction] = []

    for index, raw_fraction in enumerate(value):
        if not isinstance(raw_fraction, dict):
            raise ValueError(
                f"label_fractions[{index}] must "
                "be a mapping."
            )

        identifier = str(
            raw_fraction.get("identifier", "")
        ).strip()
        display_name = str(
            raw_fraction.get("display_name", "")
        ).strip()

        if not _IDENTIFIER_PATTERN.fullmatch(
            identifier
        ):
            raise ValueError(
                "Fraction identifiers must contain "
                "only lowercase letters, digits, "
                "and underscores."
            )

        if not display_name:
            raise ValueError(
                "Fraction display_name must not "
                "be empty."
            )

        examples_per_class_snr = (
            _validate_positive_integer(
                raw_fraction.get(
                    "examples_per_class_snr"
                ),
                (
                    "examples_per_class_snr"
                ),
            )
        )

        fractions.append(
            LabelFraction(
                identifier=identifier,
                display_name=display_name,
                examples_per_class_snr=(
                    examples_per_class_snr
                ),
            )
        )

    identifiers = [
        fraction.identifier
        for fraction in fractions
    ]

    if len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError(
            "Fraction identifiers must "
            "not contain duplicates."
        )

    return tuple(fractions)


def serialize_path(
    path: Path,
    *,
    project_root: Path,
) -> str:
    """Serialize project paths relatively when possible."""
    resolved = path.resolve()
    root = project_root.resolve()

    try:
        return resolved.relative_to(
            root
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_run_plan(
    *,
    template: Mapping[str, Any],
    method: str,
    seed: int,
    fraction: LabelFraction,
    stratum_count: int,
    batch_size: int,
    target_optimizer_steps: int,
    drop_last: bool,
    require_exact: bool,
    experiment_prefix: str,
    output_root: Path,
    figure_root: Path,
    project_root: Path,
) -> SweepRunPlan:
    """Generate one exact-budget downstream config."""
    if not _IDENTIFIER_PATTERN.fullmatch(
        method
    ):
        raise ValueError(
            "method must be a lowercase identifier."
        )

    validated_seed = _validate_positive_integer(
        seed,
        "seed",
    )
    validated_stratum_count = (
        _validate_positive_integer(
            stratum_count,
            "stratum_count",
        )
    )

    content = deepcopy(dict(template))

    training = content.get("training")

    if not isinstance(training, dict):
        raise ValueError(
            "Template training configuration "
            "must be a mapping."
        )

    output = content.get("output")

    if not isinstance(output, dict):
        raise ValueError(
            "Template output configuration "
            "must be a mapping."
        )

    selected_training_examples = (
        validated_stratum_count
        * fraction.examples_per_class_snr
    )

    budget = derive_training_budget(
        example_count=(
            selected_training_examples
        ),
        batch_size=batch_size,
        target_optimizer_steps=(
            target_optimizer_steps
        ),
        drop_last=drop_last,
        require_exact=require_exact,
    )

    # train_baseline.py reads the top-level seed.
    content["seed"] = validated_seed

    # Remove the ineffective historical field.
    training.pop("seed", None)
    training.pop("epochs", None)

    training["batch_size"] = batch_size
    training["examples_per_class_snr"] = (
        fraction.examples_per_class_snr
    )
    training["subset_seed"] = validated_seed
    training["target_optimizer_steps"] = (
        target_optimizer_steps
    )
    training[
        "require_exact_optimizer_steps"
    ] = require_exact
    training["drop_last"] = drop_last

    run_name = (
        f"{experiment_prefix}_"
        f"{fraction.identifier}_"
        f"{method}_seed_{validated_seed}"
    )
    run_directory = (
        output_root
        / fraction.identifier
        / method
        / f"seed_{validated_seed}"
    )
    figure_path = (
        figure_root
        / (
            f"{experiment_prefix}_"
            f"{fraction.identifier}_"
            f"{method}_seed_{validated_seed}.png"
        )
    )
    generated_config_path = (
        output_root
        / "generated_configs"
        / fraction.identifier
        / f"{method}_seed_{validated_seed}.yaml"
    )

    content["experiment_name"] = run_name
    output["directory"] = serialize_path(
        run_directory,
        project_root=project_root,
    )
    output["figure_path"] = serialize_path(
        figure_path,
        project_root=project_root,
    )

    return SweepRunPlan(
        fraction=fraction,
        method=method,
        seed=validated_seed,
        selected_training_examples=(
            selected_training_examples
        ),
        budget=budget,
        content=content,
        generated_config_path=(
            generated_config_path
        ),
        output_directory=run_directory,
    )


__all__ = [
    "LabelFraction",
    "SweepRunPlan",
    "build_run_plan",
    "serialize_path",
    "validate_label_fractions",
    "validate_seeds",
]
