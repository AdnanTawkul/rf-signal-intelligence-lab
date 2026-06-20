from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

DETECTION_METRICS = (
    "auroc",
    "average_precision",
    "fpr_at_target_tpr",
)


def _numeric_summary(
    values: Sequence[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            "Summary values must be a "
            "non-empty one-dimensional sequence."
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


def _validate_record(
    value: object,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Every channel-shift record must "
            "be a mapping."
        )

    required_strings = (
        "fraction_identifier",
        "method",
        "condition",
        "score_name",
    )

    for key in required_strings:
        item = value.get(key)

        if (
            not isinstance(item, str)
            or not item
        ):
            raise ValueError(
                f"{key} must be a non-empty string."
            )

    seed = value.get("seed")

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError(
            "seed must be an integer."
        )

    metrics = value.get("metrics")

    if not isinstance(metrics, Mapping):
        raise ValueError(
            "metrics must be a mapping."
        )

    required_metrics = (
        *DETECTION_METRICS,
        "clean_mean",
        "shifted_mean",
        "clean_std",
        "shifted_std",
    )

    for key in required_metrics:
        raw = metrics.get(key)

        try:
            number = float(raw)
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"Metric {key!r} must be numeric."
            ) from error

        if not math.isfinite(number):
            raise ValueError(
                f"Metric {key!r} must be finite."
            )

    for key in DETECTION_METRICS:
        number = float(metrics[key])

        if number < 0.0 or number > 1.0:
            raise ValueError(
                f"Metric {key!r} must be "
                "within [0, 1]."
            )

    return value


def _aggregate_group(
    records: Sequence[Mapping[str, Any]],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    for metric_name in DETECTION_METRICS:
        values = [
            float(
                record["metrics"][
                    metric_name
                ]
            )
            for record in records
        ]
        metrics[metric_name] = (
            _numeric_summary(values)
        )

    auroc_values = np.asarray(
        [
            float(
                record["metrics"]["auroc"]
            )
            for record in records
        ],
        dtype=np.float64,
    )

    clean_means = [
        float(
            record["metrics"]["clean_mean"]
        )
        for record in records
    ]
    shifted_means = [
        float(
            record["metrics"]["shifted_mean"]
        )
        for record in records
    ]
    score_changes = [
        shifted - clean
        for clean, shifted in zip(
            clean_means,
            shifted_means,
            strict=True,
        )
    ]

    return {
        **dict(identity),
        "run_count": len(records),
        "seeds": sorted(
            {
                int(record["seed"])
                for record in records
            }
        ),
        "metrics": metrics,
        "auroc_direction_counts": {
            "above_chance": int(
                np.count_nonzero(
                    auroc_values > 0.5
                )
            ),
            "below_chance": int(
                np.count_nonzero(
                    auroc_values < 0.5
                )
            ),
            "at_chance": int(
                np.count_nonzero(
                    np.isclose(
                        auroc_values,
                        0.5,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            ),
            "at_least_0_7": int(
                np.count_nonzero(
                    auroc_values >= 0.7
                )
            ),
            "at_least_0_8": int(
                np.count_nonzero(
                    auroc_values >= 0.8
                )
            ),
        },
        "score_statistics": {
            "clean_mean": (
                _numeric_summary(
                    clean_means
                )
            ),
            "shifted_mean": (
                _numeric_summary(
                    shifted_means
                )
            ),
            "shifted_minus_clean": (
                _numeric_summary(
                    score_changes
                )
            ),
        },
    }


def _build_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    key_names: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[str, ...],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for record in records:
        key = tuple(
            str(record[name])
            for name in key_names
        )
        grouped[key].append(record)

    output = []

    for key, group_records in sorted(
        grouped.items()
    ):
        identity = {
            name: value
            for name, value in zip(
                key_names,
                key,
                strict=True,
            )
        }

        output.append(
            _aggregate_group(
                group_records,
                identity=identity,
            )
        )

    return output


def aggregate_channel_shift_records(
    records: object,
) -> dict[str, Any]:
    """Aggregate output-only shift detection records."""
    if (
        not isinstance(records, list)
        or not records
    ):
        raise ValueError(
            "records must be a non-empty list."
        )

    validated = [
        _validate_record(record)
        for record in records
    ]

    detailed_groups = _build_groups(
        validated,
        key_names=(
            "fraction_identifier",
            "method",
            "condition",
            "score_name",
        ),
    )
    condition_score_groups = _build_groups(
        validated,
        key_names=(
            "condition",
            "score_name",
        ),
    )
    fraction_condition_score_groups = (
        _build_groups(
            validated,
            key_names=(
                "fraction_identifier",
                "condition",
                "score_name",
            ),
        )
    )

    return {
        "record_count": len(validated),
        "detailed_group_count": len(
            detailed_groups
        ),
        "condition_score_group_count": len(
            condition_score_groups
        ),
        "fraction_condition_score_group_count": (
            len(
                fraction_condition_score_groups
            )
        ),
        "groups": detailed_groups,
        "condition_score_groups": (
            condition_score_groups
        ),
        "fraction_condition_score_groups": (
            fraction_condition_score_groups
        ),
    }


__all__ = [
    "DETECTION_METRICS",
    "aggregate_channel_shift_records",
]
