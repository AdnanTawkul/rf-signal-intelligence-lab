from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]


def _aggregate_groups(
    aggregate: object,
) -> list[Mapping[str, Any]]:
    if not isinstance(aggregate, Mapping):
        raise ValueError(
            "aggregate must be a mapping."
        )

    groups = aggregate.get("groups")

    if (
        not isinstance(groups, list)
        or not groups
    ):
        raise ValueError(
            "aggregate groups must be a "
            "non-empty list."
        )

    if any(
        not isinstance(group, Mapping)
        for group in groups
    ):
        raise ValueError(
            "Every aggregate group must "
            "be a mapping."
        )

    return groups


def _find_group(
    groups: Sequence[Mapping[str, Any]],
    *,
    fraction_identifier: str,
    method: str,
    condition: str,
) -> Mapping[str, Any]:
    matches = [
        group
        for group in groups
        if (
            group.get(
                "fraction_identifier"
            )
            == fraction_identifier
            and group.get("method") == method
            and group.get("condition")
            == condition
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one aggregate "
            "group for "
            f"{fraction_identifier}/"
            f"{method}/{condition}; "
            f"found {len(matches)}."
        )

    return matches[0]


def metric_change_matrix(
    aggregate: object,
    *,
    rows: Sequence[
        tuple[str, str]
    ],
    conditions: Sequence[str],
    metric: str,
) -> Float64Array:
    """Build a row-by-condition metric-change matrix."""
    if not rows:
        raise ValueError(
            "rows must not be empty."
        )

    if not conditions:
        raise ValueError(
            "conditions must not be empty."
        )

    groups = _aggregate_groups(aggregate)
    matrix = np.empty(
        (
            len(rows),
            len(conditions),
        ),
        dtype=np.float64,
    )

    for row_index, (
        fraction_identifier,
        method,
    ) in enumerate(rows):
        for column_index, condition in enumerate(
            conditions
        ):
            group = _find_group(
                groups,
                fraction_identifier=(
                    fraction_identifier
                ),
                method=method,
                condition=condition,
            )

            try:
                value = float(
                    group["metrics"][
                        metric
                    ]["change"]["mean"]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Aggregate group does not "
                    f"contain metric {metric!r}."
                ) from error

            if not np.isfinite(value):
                raise ValueError(
                    "Metric-change values must "
                    "be finite."
                )

            matrix[
                row_index,
                column_index,
            ] = value

    return matrix


def temperature_matrices(
    aggregate: object,
    *,
    fractions: Sequence[str],
    methods: Sequence[str],
    condition: str = "clean",
) -> tuple[
    Float64Array,
    Float64Array,
]:
    """Build mean and standard-deviation matrices."""
    if not fractions:
        raise ValueError(
            "fractions must not be empty."
        )

    if not methods:
        raise ValueError(
            "methods must not be empty."
        )

    groups = _aggregate_groups(aggregate)
    means = np.empty(
        (
            len(methods),
            len(fractions),
        ),
        dtype=np.float64,
    )
    standard_deviations = np.empty_like(
        means
    )

    for method_index, method in enumerate(
        methods
    ):
        for fraction_index, fraction in enumerate(
            fractions
        ):
            group = _find_group(
                groups,
                fraction_identifier=fraction,
                method=method,
                condition=condition,
            )

            try:
                mean = float(
                    group["temperature"]["mean"]
                )
                standard_deviation = float(
                    group["temperature"]["std"]
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                raise ValueError(
                    "Aggregate group has invalid "
                    "temperature statistics."
                ) from error

            if (
                not np.isfinite(mean)
                or mean <= 0.0
                or not np.isfinite(
                    standard_deviation
                )
                or standard_deviation < 0.0
            ):
                raise ValueError(
                    "Temperature statistics must "
                    "be finite and valid."
                )

            means[
                method_index,
                fraction_index,
            ] = mean
            standard_deviations[
                method_index,
                fraction_index,
            ] = standard_deviation

    return means, standard_deviations


def selective_accuracy_change_matrices(
    records: object,
    *,
    fractions: Sequence[str],
    conditions: Sequence[str],
    coverages: Sequence[float],
) -> dict[str, Float64Array]:
    """Average selective-accuracy changes."""
    if (
        not isinstance(records, list)
        or not records
    ):
        raise ValueError(
            "records must be a non-empty list."
        )

    if not fractions:
        raise ValueError(
            "fractions must not be empty."
        )

    if not conditions:
        raise ValueError(
            "conditions must not be empty."
        )

    if not coverages:
        raise ValueError(
            "coverages must not be empty."
        )

    grouped: dict[
        tuple[str, str],
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(
                "Every record must be "
                "a mapping."
            )

        key = (
            str(
                record[
                    "fraction_identifier"
                ]
            ),
            str(record["condition"]),
        )
        grouped[key].append(record)

    result: dict[str, Float64Array] = {}

    for condition in conditions:
        matrix = np.empty(
            (
                len(fractions),
                len(coverages),
            ),
            dtype=np.float64,
        )

        for fraction_index, fraction in enumerate(
            fractions
        ):
            matching_records = grouped.get(
                (
                    fraction,
                    condition,
                ),
                [],
            )

            if not matching_records:
                raise ValueError(
                    "No records found for "
                    f"{fraction}/{condition}."
                )

            for coverage_index, coverage in enumerate(
                coverages
            ):
                changes = []

                for record in matching_records:
                    baseline_points = record[
                        "selective_accuracy"
                    ]["baseline"]
                    calibrated_points = record[
                        "selective_accuracy"
                    ]["calibrated"]

                    baseline_matches = [
                        point
                        for point in baseline_points
                        if np.isclose(
                            float(
                                point[
                                    "requested_coverage"
                                ]
                            ),
                            coverage,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ]
                    calibrated_matches = [
                        point
                        for point
                        in calibrated_points
                        if np.isclose(
                            float(
                                point[
                                    "requested_coverage"
                                ]
                            ),
                            coverage,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    ]

                    if (
                        len(baseline_matches)
                        != 1
                        or len(
                            calibrated_matches
                        )
                        != 1
                    ):
                        raise ValueError(
                            "Selective-accuracy "
                            "coverage grid is "
                            "incomplete."
                        )

                    changes.append(
                        float(
                            calibrated_matches[
                                0
                            ]["accuracy"]
                        )
                        - float(
                            baseline_matches[
                                0
                            ]["accuracy"]
                        )
                    )

                matrix[
                    fraction_index,
                    coverage_index,
                ] = float(
                    np.mean(
                        np.asarray(
                            changes,
                            dtype=np.float64,
                        )
                    )
                )

        result[condition] = matrix

    return result


def reliability_curve(
    record: object,
    *,
    variant: str,
) -> dict[str, Float64Array]:
    """Extract non-empty reliability bins."""
    if not isinstance(record, Mapping):
        raise ValueError(
            "record must be a mapping."
        )

    if variant not in {
        "baseline",
        "calibrated",
    }:
        raise ValueError(
            "variant must be baseline "
            "or calibrated."
        )

    try:
        bins = record[variant]["bins"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Record does not contain "
            "reliability bins."
        ) from error

    if not isinstance(bins, list):
        raise ValueError(
            "Reliability bins must be a list."
        )

    centers = []
    accuracies = []
    confidences = []
    counts = []

    for bin_record in bins:
        count = int(
            bin_record["example_count"]
        )

        if count <= 0:
            continue

        lower = float(
            bin_record["lower_bound"]
        )
        upper = float(
            bin_record["upper_bound"]
        )
        accuracy = float(
            bin_record["accuracy"]
        )
        confidence = float(
            bin_record["mean_confidence"]
        )

        values = (
            lower,
            upper,
            accuracy,
            confidence,
        )

        if not all(
            np.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Reliability-bin values must "
                "be finite."
            )

        centers.append(
            0.5 * (lower + upper)
        )
        accuracies.append(accuracy)
        confidences.append(confidence)
        counts.append(float(count))

    if not centers:
        raise ValueError(
            "Record has no non-empty "
            "reliability bins."
        )

    return {
        "bin_centers": np.asarray(
            centers,
            dtype=np.float64,
        ),
        "accuracy": np.asarray(
            accuracies,
            dtype=np.float64,
        ),
        "mean_confidence": np.asarray(
            confidences,
            dtype=np.float64,
        ),
        "example_count": np.asarray(
            counts,
            dtype=np.float64,
        ),
    }


def find_record(
    records: object,
    *,
    fraction_identifier: str,
    method: str,
    seed: int,
    condition: str,
) -> Mapping[str, Any]:
    """Locate exactly one raw analysis record."""
    if not isinstance(records, list):
        raise ValueError(
            "records must be a list."
        )

    matches = [
        record
        for record in records
        if (
            record.get(
                "fraction_identifier"
            )
            == fraction_identifier
            and record.get("method") == method
            and int(record.get("seed"))
            == seed
            and record.get("condition")
            == condition
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected one raw record for "
            f"{fraction_identifier}/"
            f"{method}/{seed}/{condition}; "
            f"found {len(matches)}."
        )

    return matches[0]


def headline_summary(
    records: object,
) -> dict[str, Any]:
    """Create compact report-ready statistics."""
    if (
        not isinstance(records, list)
        or not records
    ):
        raise ValueError(
            "records must be a non-empty list."
        )

    condition_groups: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)
    fraction_groups: dict[
        str,
        list[Mapping[str, Any]],
    ] = defaultdict(list)

    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(
                "Every record must be "
                "a mapping."
            )

        condition_groups[
            str(record["condition"])
        ].append(record)
        fraction_groups[
            str(
                record[
                    "fraction_identifier"
                ]
            )
        ].append(record)

    def summarize_group(
        group: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "run_count": len(group),
        }

        for metric in (
            "negative_log_likelihood",
            "expected_calibration_error",
            "brier_score",
        ):
            changes = np.asarray(
                [
                    float(
                        record[
                            "metric_changes"
                        ][metric]
                    )
                    for record in group
                ],
                dtype=np.float64,
            )

            result[metric] = {
                "mean_change": float(
                    np.mean(changes)
                ),
                "improved_count": int(
                    np.count_nonzero(
                        changes < 0.0
                    )
                ),
                "worsened_count": int(
                    np.count_nonzero(
                        changes > 0.0
                    )
                ),
                "tied_count": int(
                    np.count_nonzero(
                        np.isclose(
                            changes,
                            0.0,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                ),
            }

        return result

    return {
        "record_count": len(records),
        "accuracy_preserved_count": sum(
            record.get(
                "accuracy_preserved"
            )
            is True
            for record in records
        ),
        "conditions": {
            name: summarize_group(group)
            for name, group
            in sorted(
                condition_groups.items()
            )
        },
        "fractions": {
            name: summarize_group(group)
            for name, group
            in sorted(
                fraction_groups.items()
            )
        },
    }


__all__ = [
    "find_record",
    "headline_summary",
    "metric_change_matrix",
    "reliability_curve",
    "selective_accuracy_change_matrices",
    "temperature_matrices",
]
