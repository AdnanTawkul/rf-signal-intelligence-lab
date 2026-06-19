from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class SweepRunExpectation:
    """Validated expectations for one planned run."""

    fraction_identifier: str
    method: str
    seed: int
    subset_seed: int
    examples_per_class_snr: int
    selected_training_examples: int
    experiment_name: str
    training_budget: dict[str, object]
    generated_config_path: Path
    output_directory: Path


@dataclass(frozen=True, slots=True)
class CompletedSweepRun:
    """Validated outputs from one completed run."""

    fraction_identifier: str
    method: str
    seed: int
    best_validation_accuracy: float
    best_epoch: int
    summary_path: Path
    checkpoint_path: Path
    history_path: Path


def _require_mapping(
    value: object,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping."
        )

    return value


def _require_positive_integer(
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


def _resolve_path(
    value: object,
    *,
    project_root: Path,
    name: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{name} must be a non-empty path."
        )

    path = Path(value)

    if path.is_absolute():
        return path

    return project_root / path


def load_run_expectation(
    record: object,
    *,
    project_root: Path,
) -> SweepRunExpectation:
    """Validate one dry-run manifest record."""
    mapping = _require_mapping(
        record,
        "manifest run record",
    )

    fraction_identifier = str(
        mapping.get(
            "fraction_identifier",
            "",
        )
    ).strip()
    method = str(
        mapping.get("method", "")
    ).strip()

    if not fraction_identifier:
        raise ValueError(
            "fraction_identifier must not "
            "be empty."
        )

    if not method:
        raise ValueError(
            "method must not be empty."
        )

    seed = _require_positive_integer(
        mapping.get("training_seed"),
        "training_seed",
    )
    subset_seed = _require_positive_integer(
        mapping.get("subset_seed"),
        "subset_seed",
    )
    examples_per_class_snr = (
        _require_positive_integer(
            mapping.get(
                "examples_per_class_snr"
            ),
            "examples_per_class_snr",
        )
    )
    selected_training_examples = (
        _require_positive_integer(
            mapping.get(
                "selected_training_examples"
            ),
            "selected_training_examples",
        )
    )

    generated_config_path = _resolve_path(
        mapping.get(
            "generated_config_path"
        ),
        project_root=project_root,
        name="generated_config_path",
    )
    output_directory = _resolve_path(
        mapping.get("output_directory"),
        project_root=project_root,
        name="output_directory",
    )

    if not generated_config_path.is_file():
        raise FileNotFoundError(
            generated_config_path
        )

    config = yaml.safe_load(
        generated_config_path.read_text(
            encoding="utf-8"
        )
    )
    config_mapping = _require_mapping(
        config,
        "generated training config",
    )

    experiment_name = str(
        config_mapping.get(
            "experiment_name",
            "",
        )
    ).strip()

    if not experiment_name:
        raise ValueError(
            "Generated experiment_name must "
            "not be empty."
        )

    if config_mapping.get("seed") != seed:
        raise ValueError(
            "Generated top-level seed does "
            "not match the manifest."
        )

    training = _require_mapping(
        config_mapping.get("training"),
        "generated training section",
    )

    if training.get("subset_seed") != subset_seed:
        raise ValueError(
            "Generated subset seed does not "
            "match the manifest."
        )

    if (
        training.get(
            "examples_per_class_snr"
        )
        != examples_per_class_snr
    ):
        raise ValueError(
            "Generated examples-per-stratum "
            "does not match the manifest."
        )

    raw_budget = _require_mapping(
        mapping.get("training_budget"),
        "training_budget",
    )
    training_budget = dict(raw_budget)

    if (
        training.get(
            "target_optimizer_steps"
        )
        != training_budget.get(
            "target_optimizer_steps"
        )
    ):
        raise ValueError(
            "Generated target optimizer steps "
            "do not match the manifest."
        )

    if (
        training_budget.get(
            "example_count"
        )
        != selected_training_examples
    ):
        raise ValueError(
            "Manifest example count does not "
            "match selected training examples."
        )

    return SweepRunExpectation(
        fraction_identifier=(
            fraction_identifier
        ),
        method=method,
        seed=seed,
        subset_seed=subset_seed,
        examples_per_class_snr=(
            examples_per_class_snr
        ),
        selected_training_examples=(
            selected_training_examples
        ),
        experiment_name=experiment_name,
        training_budget=training_budget,
        generated_config_path=(
            generated_config_path
        ),
        output_directory=output_directory,
    )


def load_completed_run(
    expectation: SweepRunExpectation,
    *,
    project_root: Path,
) -> CompletedSweepRun | None:
    """Load and validate a completed training run."""
    summary_path = (
        expectation.output_directory
        / "summary.json"
    )

    if not summary_path.is_file():
        return None

    content = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )
    summary = _require_mapping(
        content,
        "training summary",
    )

    if (
        summary.get("experiment_name")
        != expectation.experiment_name
    ):
        raise ValueError(
            "Completed experiment name does "
            "not match the planned run."
        )

    if summary.get("seed") != expectation.seed:
        raise ValueError(
            "Completed training seed does not "
            "match the planned run."
        )

    subset = _require_mapping(
        summary.get("labeled_subset"),
        "labeled_subset",
    )

    expected_subset = {
        "examples_per_class_snr": (
            expectation.examples_per_class_snr
        ),
        "subset_seed": (
            expectation.subset_seed
        ),
        "selected_training_examples": (
            expectation.selected_training_examples
        ),
    }

    for name, expected_value in (
        expected_subset.items()
    ):
        if subset.get(name) != expected_value:
            raise ValueError(
                "Completed labeled-subset "
                f"field {name!r} does not match."
            )

    if (
        summary.get("training_budget")
        != expectation.training_budget
    ):
        raise ValueError(
            "Completed training budget does "
            "not match the planned run."
        )

    best_epoch = _require_positive_integer(
        summary.get("best_epoch"),
        "best_epoch",
    )

    planned_epochs = _require_positive_integer(
        expectation.training_budget.get(
            "epochs"
        ),
        "planned epochs",
    )

    if best_epoch > planned_epochs:
        raise ValueError(
            "best_epoch exceeds the planned "
            "training duration."
        )

    raw_accuracy = summary.get(
        "best_validation_accuracy"
    )

    if isinstance(raw_accuracy, bool):
        raise ValueError(
            "best_validation_accuracy must "
            "be numeric."
        )

    try:
        best_validation_accuracy = float(
            raw_accuracy
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "best_validation_accuracy must "
            "be numeric."
        ) from error

    if (
        not math.isfinite(
            best_validation_accuracy
        )
        or best_validation_accuracy < 0.0
        or best_validation_accuracy > 1.0
    ):
        raise ValueError(
            "best_validation_accuracy must "
            "be between zero and one."
        )

    checkpoint_path = _resolve_path(
        summary.get("checkpoint_path"),
        project_root=project_root,
        name="checkpoint_path",
    )
    history_path = _resolve_path(
        summary.get("history_path"),
        project_root=project_root,
        name="history_path",
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    if not history_path.is_file():
        raise FileNotFoundError(
            history_path
        )

    return CompletedSweepRun(
        fraction_identifier=(
            expectation.fraction_identifier
        ),
        method=expectation.method,
        seed=expectation.seed,
        best_validation_accuracy=(
            best_validation_accuracy
        ),
        best_epoch=best_epoch,
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


__all__ = [
    "CompletedSweepRun",
    "SweepRunExpectation",
    "load_completed_run",
    "load_run_expectation",
]
