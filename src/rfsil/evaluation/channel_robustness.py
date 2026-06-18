from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SeedSweepMetrics:
    """Metrics loaded from one seed-sweep evaluation."""

    condition: str
    seeds: tuple[int, ...]
    class_names: tuple[str, ...]
    snr_values_db: tuple[float, ...]
    overall_accuracy: Float64Array
    class_accuracy: Float64Array
    accuracy_by_snr: Float64Array


def _validate_accuracy(
    value: object,
    name: str,
) -> float:
    """Validate an accuracy value in the interval [0, 1]."""
    result = float(value)

    if not np.isfinite(result):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= result <= 1.0:
        raise ValueError(
            f"{name} must be in the interval [0, 1]."
        )

    return result


def _validate_nonempty_string(
    value: object,
    name: str,
) -> str:
    """Validate a nonempty string."""
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must not be empty."
        )

    return result


def _parse_snr_mapping(
    value: object,
    expected_snr_values: tuple[float, ...],
    name: str,
) -> list[float]:
    """Parse a JSON SNR-to-accuracy mapping."""
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be a mapping."
        )

    parsed = {
        float(key): _validate_accuracy(
            item,
            f"{name}[{key}]",
        )
        for key, item in value.items()
    }

    if tuple(sorted(parsed)) != expected_snr_values:
        raise ValueError(
            f"{name} contains unexpected SNR values."
        )

    return [
        parsed[snr]
        for snr in expected_snr_values
    ]


def load_seed_sweep_metrics(
    aggregate_path: str | Path,
    condition: str,
) -> SeedSweepMetrics:
    """Load and validate one aggregate seed-sweep JSON file."""
    path = Path(aggregate_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Aggregate metrics file does not exist: {path}"
        )

    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Aggregate metrics must be a JSON mapping."
        )

    selected_condition = _validate_nonempty_string(
        condition,
        "condition",
    )

    class_names_value = content.get(
        "class_names"
    )

    if (
        not isinstance(class_names_value, list)
        or not class_names_value
    ):
        raise ValueError(
            "class_names must be a nonempty list."
        )

    class_names = tuple(
        _validate_nonempty_string(
            class_name,
            "class name",
        )
        for class_name in class_names_value
    )

    if len(set(class_names)) != len(class_names):
        raise ValueError(
            "class_names must not contain duplicates."
        )

    runs = content.get("runs")

    if not isinstance(runs, list) or not runs:
        raise ValueError(
            "runs must be a nonempty list."
        )

    first_run = runs[0]

    if not isinstance(first_run, dict):
        raise ValueError(
            "Every run must be a mapping."
        )

    first_snr_mapping = first_run.get(
        "accuracy_by_snr"
    )

    if (
        not isinstance(first_snr_mapping, dict)
        or not first_snr_mapping
    ):
        raise ValueError(
            "accuracy_by_snr must be a nonempty mapping."
        )

    snr_values_db = tuple(
        sorted(
            float(key)
            for key in first_snr_mapping
        )
    )

    seeds: list[int] = []
    overall_accuracy: list[float] = []
    class_accuracy: list[list[float]] = []
    accuracy_by_snr: list[list[float]] = []

    for run_index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(
                f"run {run_index} must be a mapping."
            )

        seed = run.get("seed")

        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError(
                f"run {run_index} seed must be an integer."
            )

        class_mapping = run.get(
            "class_accuracy"
        )

        if not isinstance(class_mapping, dict):
            raise ValueError(
                f"run {run_index} class_accuracy "
                "must be a mapping."
            )

        if set(class_mapping) != set(class_names):
            raise ValueError(
                f"run {run_index} has unexpected "
                "class names."
            )

        seeds.append(seed)
        overall_accuracy.append(
            _validate_accuracy(
                run.get("overall_accuracy"),
                f"run {run_index} overall_accuracy",
            )
        )
        class_accuracy.append(
            [
                _validate_accuracy(
                    class_mapping[class_name],
                    (
                        f"run {run_index} "
                        f"class_accuracy[{class_name}]"
                    ),
                )
                for class_name in class_names
            ]
        )
        accuracy_by_snr.append(
            _parse_snr_mapping(
                run.get("accuracy_by_snr"),
                snr_values_db,
                f"run {run_index} accuracy_by_snr",
            )
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "runs must not contain duplicate seeds."
        )

    return SeedSweepMetrics(
        condition=selected_condition,
        seeds=tuple(seeds),
        class_names=class_names,
        snr_values_db=snr_values_db,
        overall_accuracy=np.asarray(
            overall_accuracy,
            dtype=np.float64,
        ),
        class_accuracy=np.asarray(
            class_accuracy,
            dtype=np.float64,
        ),
        accuracy_by_snr=np.asarray(
            accuracy_by_snr,
            dtype=np.float64,
        ),
    )


def compare_seed_sweep_conditions(
    conditions: Mapping[str, SeedSweepMetrics],
    reference_condition: str = "clean",
) -> dict[str, Any]:
    """Compare paired seed-sweep results across channel conditions."""
    if not conditions:
        raise ValueError(
            "conditions must not be empty."
        )

    if reference_condition not in conditions:
        raise ValueError(
            "reference_condition must exist in conditions."
        )

    reference = conditions[reference_condition]

    for condition_name, metrics in conditions.items():
        if metrics.seeds != reference.seeds:
            raise ValueError(
                f"Condition {condition_name!r} does not "
                "use the same paired seeds."
            )

        if metrics.class_names != reference.class_names:
            raise ValueError(
                f"Condition {condition_name!r} does not "
                "use the same class names."
            )

        if (
            metrics.snr_values_db
            != reference.snr_values_db
        ):
            raise ValueError(
                f"Condition {condition_name!r} does not "
                "use the same SNR values."
            )

    result_conditions: dict[str, Any] = {}

    for condition_name, metrics in conditions.items():
        paired_drops = (
            reference.overall_accuracy
            - metrics.overall_accuracy
        )
        class_mean = np.mean(
            metrics.class_accuracy,
            axis=0,
        )
        class_std = np.std(
            metrics.class_accuracy,
            axis=0,
        )
        reference_class_mean = np.mean(
            reference.class_accuracy,
            axis=0,
        )
        snr_mean = np.mean(
            metrics.accuracy_by_snr,
            axis=0,
        )
        snr_std = np.std(
            metrics.accuracy_by_snr,
            axis=0,
        )
        reference_snr_mean = np.mean(
            reference.accuracy_by_snr,
            axis=0,
        )

        result_conditions[condition_name] = {
            "overall_mean": float(
                np.mean(metrics.overall_accuracy)
            ),
            "overall_standard_deviation": float(
                np.std(metrics.overall_accuracy)
            ),
            "overall_minimum": float(
                np.min(metrics.overall_accuracy)
            ),
            "overall_maximum": float(
                np.max(metrics.overall_accuracy)
            ),
            "paired_drop_from_reference": {
                "mean": float(
                    np.mean(paired_drops)
                ),
                "standard_deviation": float(
                    np.std(paired_drops)
                ),
                "minimum": float(
                    np.min(paired_drops)
                ),
                "maximum": float(
                    np.max(paired_drops)
                ),
                "per_seed": {
                    str(seed): float(drop)
                    for seed, drop in zip(
                        metrics.seeds,
                        paired_drops,
                        strict=True,
                    )
                },
            },
            "class_accuracy_mean": {
                class_name: float(value)
                for class_name, value in zip(
                    metrics.class_names,
                    class_mean,
                    strict=True,
                )
            },
            "class_accuracy_standard_deviation": {
                class_name: float(value)
                for class_name, value in zip(
                    metrics.class_names,
                    class_std,
                    strict=True,
                )
            },
            "class_drop_from_reference": {
                class_name: float(drop)
                for class_name, drop in zip(
                    metrics.class_names,
                    reference_class_mean - class_mean,
                    strict=True,
                )
            },
            "accuracy_by_snr_mean": {
                str(snr): float(value)
                for snr, value in zip(
                    metrics.snr_values_db,
                    snr_mean,
                    strict=True,
                )
            },
            "accuracy_by_snr_standard_deviation": {
                str(snr): float(value)
                for snr, value in zip(
                    metrics.snr_values_db,
                    snr_std,
                    strict=True,
                )
            },
            "snr_drop_from_reference": {
                str(snr): float(drop)
                for snr, drop in zip(
                    metrics.snr_values_db,
                    reference_snr_mean - snr_mean,
                    strict=True,
                )
            },
        }

    return {
        "format_version": 1,
        "reference_condition": reference_condition,
        "seeds": list(reference.seeds),
        "class_names": list(reference.class_names),
        "snr_values_db": list(
            reference.snr_values_db
        ),
        "conditions": result_conditions,
    }


__all__ = [
    "SeedSweepMetrics",
    "compare_seed_sweep_conditions",
    "load_seed_sweep_metrics",
]
