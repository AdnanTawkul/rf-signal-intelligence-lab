from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from rfsil.evaluation.channel_shift import (
    ShiftDetectionMetrics,
)

SHIFT_DETECTOR_SYSTEMS = (
    "lag8",
    "all_iq_linear",
    "output_energy",
    "iq_energy_fusion",
)

_METRIC_NAMES = (
    "auroc",
    "average_precision",
    "fpr_at_target_tpr",
)


def _validate_string(
    value: object,
    *,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _validate_unit_interval(
    value: object,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be finite and "
            "within [0, 1]."
        )

    number = float(value)

    if (
        not math.isfinite(number)
        or number < 0.0
        or number > 1.0
    ):
        raise ValueError(
            f"{name} must be finite and "
            "within [0, 1]."
        )

    return number


def _validate_optional_positive_float(
    value: object,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    number = float(value)

    if (
        not math.isfinite(number)
        or number <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    return number


@dataclass(frozen=True, slots=True)
class DetectorComparisonRecord:
    """One detector evaluated for one checkpoint and condition."""

    fraction_identifier: str
    method: str
    seed: int
    condition: str
    system_name: str
    metrics: ShiftDetectionMetrics
    development_auroc: float
    selected_l2_strength: float | None = None
    direction: str | None = None

    def __post_init__(self) -> None:
        fraction = _validate_string(
            self.fraction_identifier,
            name="fraction_identifier",
        )
        method = _validate_string(
            self.method,
            name="method",
        )
        condition = _validate_string(
            self.condition,
            name="condition",
        )
        system_name = _validate_string(
            self.system_name,
            name="system_name",
        )

        if system_name not in (
            SHIFT_DETECTOR_SYSTEMS
        ):
            raise ValueError(
                "system_name must be one of "
                f"{SHIFT_DETECTOR_SYSTEMS}."
            )

        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, Integral)
            or int(self.seed) < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer."
            )

        if not isinstance(
            self.metrics,
            ShiftDetectionMetrics,
        ):
            raise ValueError(
                "metrics must be a "
                "ShiftDetectionMetrics instance."
            )

        development_auroc = (
            _validate_unit_interval(
                self.development_auroc,
                name="development_auroc",
            )
        )
        selected_l2_strength = (
            _validate_optional_positive_float(
                self.selected_l2_strength,
                name="selected_l2_strength",
            )
        )

        direction = self.direction

        if direction is not None:
            direction = _validate_string(
                direction,
                name="direction",
            )

            if direction not in (
                "larger_is_shift_like",
                "smaller_is_shift_like",
            ):
                raise ValueError(
                    "direction must describe a "
                    "supported score direction."
                )

        object.__setattr__(
            self,
            "fraction_identifier",
            fraction,
        )
        object.__setattr__(
            self,
            "method",
            method,
        )
        object.__setattr__(
            self,
            "seed",
            int(self.seed),
        )
        object.__setattr__(
            self,
            "condition",
            condition,
        )
        object.__setattr__(
            self,
            "system_name",
            system_name,
        )
        object.__setattr__(
            self,
            "development_auroc",
            development_auroc,
        )
        object.__setattr__(
            self,
            "selected_l2_strength",
            selected_l2_strength,
        )
        object.__setattr__(
            self,
            "direction",
            direction,
        )

    @property
    def checkpoint_identifier(self) -> str:
        """Return the checkpoint identifier."""
        return (
            f"{self.fraction_identifier}/"
            f"{self.method}/"
            f"seed_{self.seed}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible record data."""
        return {
            "fraction_identifier": (
                self.fraction_identifier
            ),
            "method": self.method,
            "seed": self.seed,
            "checkpoint_identifier": (
                self.checkpoint_identifier
            ),
            "condition": self.condition,
            "system_name": self.system_name,
            "development_auroc": (
                self.development_auroc
            ),
            "selected_l2_strength": (
                self.selected_l2_strength
            ),
            "direction": self.direction,
            "metrics": self.metrics.to_dict(),
        }


def _numeric_summary(
    values: Sequence[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "Summary values must be a non-empty "
            "one-dimensional sequence."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Summary values must be finite."
        )

    return {
        "mean": float(np.mean(array)),
        "std": float(
            np.std(
                array,
                ddof=1,
            )
            if array.size > 1
            else 0.0
        ),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _aggregate_system_condition(
    records: Sequence[
        DetectorComparisonRecord
    ],
    *,
    system_name: str,
    condition: str,
) -> dict[str, Any]:
    metrics = {}

    for metric_name in _METRIC_NAMES:
        metrics[metric_name] = (
            _numeric_summary(
                [
                    float(
                        getattr(
                            record.metrics,
                            metric_name,
                        )
                    )
                    for record in records
                ]
            )
        )

    auroc = np.asarray(
        [
            record.metrics.auroc
            for record in records
        ],
        dtype=np.float64,
    )
    fpr95 = np.asarray(
        [
            record
            .metrics
            .fpr_at_target_tpr
            for record in records
        ],
        dtype=np.float64,
    )

    return {
        "system_name": system_name,
        "condition": condition,
        "run_count": len(records),
        "metrics": metrics,
        "threshold_counts": {
            "auroc_at_least_0_7": int(
                np.count_nonzero(
                    auroc >= 0.7
                )
            ),
            "auroc_at_least_0_8": int(
                np.count_nonzero(
                    auroc >= 0.8
                )
            ),
            "auroc_at_least_0_9": int(
                np.count_nonzero(
                    auroc >= 0.9
                )
            ),
            "fpr95_at_most_0_5": int(
                np.count_nonzero(
                    fpr95 <= 0.5
                )
            ),
            "fpr95_at_most_0_2": int(
                np.count_nonzero(
                    fpr95 <= 0.2
                )
            ),
        },
    }


def _paired_fusion_change(
    paired_records: Sequence[
        tuple[
            DetectorComparisonRecord,
            DetectorComparisonRecord,
        ]
    ],
    *,
    baseline_system: str,
    condition: str,
) -> dict[str, Any]:
    auroc_change = [
        fusion.metrics.auroc
        - baseline.metrics.auroc
        for baseline, fusion
        in paired_records
    ]
    average_precision_change = [
        fusion.metrics.average_precision
        - baseline.metrics.average_precision
        for baseline, fusion
        in paired_records
    ]
    fpr_change = [
        fusion.metrics.fpr_at_target_tpr
        - baseline.metrics.fpr_at_target_tpr
        for baseline, fusion
        in paired_records
    ]

    return {
        "baseline_system": baseline_system,
        "fusion_system": "iq_energy_fusion",
        "condition": condition,
        "run_count": len(paired_records),
        "fusion_minus_baseline": {
            "auroc": _numeric_summary(
                auroc_change
            ),
            "average_precision": (
                _numeric_summary(
                    average_precision_change
                )
            ),
            "fpr_at_target_tpr": (
                _numeric_summary(
                    fpr_change
                )
            ),
        },
        "improvement_counts": {
            "auroc_improved": sum(
                change > 0.0
                for change in auroc_change
            ),
            "average_precision_improved": sum(
                change > 0.0
                for change
                in average_precision_change
            ),
            "fpr95_improved": sum(
                change < 0.0
                for change in fpr_change
            ),
        },
    }


def aggregate_detector_comparison_records(
    records: object,
) -> dict[str, Any]:
    """Aggregate the predefined four-system comparison."""
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not records
    ):
        raise ValueError(
            "records must be a non-empty sequence."
        )

    validated = tuple(records)

    if not all(
        isinstance(
            record,
            DetectorComparisonRecord,
        )
        for record in validated
    ):
        raise ValueError(
            "Every record must be a "
            "DetectorComparisonRecord."
        )

    indexed: dict[
        tuple[str, str, int, str, str],
        DetectorComparisonRecord,
    ] = {}

    for record in validated:
        key = (
            record.fraction_identifier,
            record.method,
            record.seed,
            record.condition,
            record.system_name,
        )

        if key in indexed:
            raise ValueError(
                "Duplicate detector comparison "
                f"record: {key}."
            )

        indexed[key] = record

    checkpoint_conditions: dict[
        tuple[str, str, int, str],
        set[str],
    ] = defaultdict(set)

    for record in validated:
        key = (
            record.fraction_identifier,
            record.method,
            record.seed,
            record.condition,
        )
        checkpoint_conditions[key].add(
            record.system_name
        )

    expected_systems = set(
        SHIFT_DETECTOR_SYSTEMS
    )

    incomplete = {
        key: sorted(found)
        for key, found
        in checkpoint_conditions.items()
        if found != expected_systems
    }

    if incomplete:
        raise ValueError(
            "Every checkpoint/condition must "
            "contain all four systems."
        )

    systems = tuple(
        SHIFT_DETECTOR_SYSTEMS
    )
    conditions = tuple(
        sorted(
            {
                record.condition
                for record in validated
            }
        )
    )
    checkpoints = {
        (
            record.fraction_identifier,
            record.method,
            record.seed,
        )
        for record in validated
    }

    system_condition_groups = []

    for condition in conditions:
        for system_name in systems:
            group = [
                record
                for record in validated
                if (
                    record.condition
                    == condition
                    and record.system_name
                    == system_name
                )
            ]

            system_condition_groups.append(
                _aggregate_system_condition(
                    group,
                    system_name=system_name,
                    condition=condition,
                )
            )

    best_system_by_condition = []

    for condition in conditions:
        candidates = [
            group
            for group
            in system_condition_groups
            if group["condition"] == condition
        ]

        best = min(
            candidates,
            key=lambda group: (
                -group["metrics"]["auroc"][
                    "mean"
                ],
                group["metrics"][
                    "fpr_at_target_tpr"
                ]["mean"],
                group["system_name"],
            ),
        )

        best_system_by_condition.append(
            {
                "condition": condition,
                "system_name": (
                    best["system_name"]
                ),
                "mean_auroc": (
                    best["metrics"]["auroc"][
                        "mean"
                    ]
                ),
                "mean_average_precision": (
                    best["metrics"][
                        "average_precision"
                    ]["mean"]
                ),
                "mean_fpr_at_target_tpr": (
                    best["metrics"][
                        "fpr_at_target_tpr"
                    ]["mean"]
                ),
            }
        )

    fusion_paired_changes = []

    for condition in conditions:
        condition_keys = sorted(
            key
            for key
            in checkpoint_conditions
            if key[3] == condition
        )

        for baseline_system in (
            "lag8",
            "all_iq_linear",
            "output_energy",
        ):
            pairs = []

            for (
                fraction,
                method,
                seed,
                _,
            ) in condition_keys:
                baseline = indexed[
                    (
                        fraction,
                        method,
                        seed,
                        condition,
                        baseline_system,
                    )
                ]
                fusion = indexed[
                    (
                        fraction,
                        method,
                        seed,
                        condition,
                        "iq_energy_fusion",
                    )
                ]
                pairs.append(
                    (
                        baseline,
                        fusion,
                    )
                )

            fusion_paired_changes.append(
                _paired_fusion_change(
                    pairs,
                    baseline_system=(
                        baseline_system
                    ),
                    condition=condition,
                )
            )

    return {
        "record_count": len(validated),
        "checkpoint_count": len(checkpoints),
        "condition_count": len(conditions),
        "system_count": len(systems),
        "systems": list(systems),
        "conditions": list(conditions),
        "system_condition_groups": (
            system_condition_groups
        ),
        "best_system_by_condition": (
            best_system_by_condition
        ),
        "fusion_paired_changes": (
            fusion_paired_changes
        ),
    }


__all__ = [
    "SHIFT_DETECTOR_SYSTEMS",
    "DetectorComparisonRecord",
    "aggregate_detector_comparison_records",
]
