from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.calibration_visualization import (
    find_record,
    headline_summary,
    metric_change_matrix,
    reliability_curve,
    selective_accuracy_change_matrices,
    temperature_matrices,
)


def aggregate_fixture() -> dict:
    groups = []

    for fraction, temperature in (
        ("labels_001pct", 2.0),
        ("labels_100pct", 0.5),
    ):
        for method in (
            "random",
            "simclr",
        ):
            for condition, change in (
                ("clean", -0.1),
                ("severe", 0.5),
            ):
                groups.append(
                    {
                        "fraction_identifier": (
                            fraction
                        ),
                        "method": method,
                        "condition": condition,
                        "temperature": {
                            "mean": temperature,
                            "std": 0.1,
                        },
                        "metrics": {
                            "negative_log_likelihood": {
                                "change": {
                                    "mean": change,
                                }
                            }
                        },
                    }
                )

    return {"groups": groups}


def raw_record(
    *,
    fraction: str = "labels_001pct",
    method: str = "random",
    seed: int = 2026,
    condition: str = "clean",
    nll_change: float = -0.1,
) -> dict:
    return {
        "fraction_identifier": fraction,
        "method": method,
        "seed": seed,
        "condition": condition,
        "accuracy_preserved": True,
        "metric_changes": {
            "negative_log_likelihood": (
                nll_change
            ),
            "expected_calibration_error": (
                -0.02
            ),
            "brier_score": -0.01,
        },
        "baseline": {
            "bins": [
                {
                    "lower_bound": 0.0,
                    "upper_bound": 0.5,
                    "example_count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                },
                {
                    "lower_bound": 0.5,
                    "upper_bound": 1.0,
                    "example_count": 10,
                    "accuracy": 0.8,
                    "mean_confidence": 0.7,
                },
            ]
        },
        "calibrated": {
            "bins": [
                {
                    "lower_bound": 0.0,
                    "upper_bound": 0.5,
                    "example_count": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                },
                {
                    "lower_bound": 0.5,
                    "upper_bound": 1.0,
                    "example_count": 10,
                    "accuracy": 0.8,
                    "mean_confidence": 0.78,
                },
            ]
        },
        "selective_accuracy": {
            "baseline": [
                {
                    "requested_coverage": 1.0,
                    "accuracy": 0.8,
                },
                {
                    "requested_coverage": 0.5,
                    "accuracy": 0.9,
                },
            ],
            "calibrated": [
                {
                    "requested_coverage": 1.0,
                    "accuracy": 0.8,
                },
                {
                    "requested_coverage": 0.5,
                    "accuracy": 0.91,
                },
            ],
        },
    }


def test_builds_metric_change_matrix() -> None:
    matrix = metric_change_matrix(
        aggregate_fixture(),
        rows=[
            (
                "labels_001pct",
                "random",
            ),
            (
                "labels_100pct",
                "simclr",
            ),
        ],
        conditions=[
            "clean",
            "severe",
        ],
        metric=(
            "negative_log_likelihood"
        ),
    )

    np.testing.assert_allclose(
        matrix,
        [
            [-0.1, 0.5],
            [-0.1, 0.5],
        ],
    )


def test_metric_matrix_rejects_missing_group(
) -> None:
    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        metric_change_matrix(
            aggregate_fixture(),
            rows=[
                (
                    "labels_005pct",
                    "random",
                )
            ],
            conditions=["clean"],
            metric=(
                "negative_log_likelihood"
            ),
        )


def test_builds_temperature_matrices() -> None:
    means, standard_deviations = (
        temperature_matrices(
            aggregate_fixture(),
            fractions=[
                "labels_001pct",
                "labels_100pct",
            ],
            methods=[
                "random",
                "simclr",
            ],
        )
    )

    np.testing.assert_allclose(
        means,
        [
            [2.0, 0.5],
            [2.0, 0.5],
        ],
    )
    np.testing.assert_allclose(
        standard_deviations,
        0.1,
    )


def test_selective_change_matrix() -> None:
    records = [
        raw_record(
            method="random",
            seed=2026,
        ),
        raw_record(
            method="simclr",
            seed=2027,
        ),
    ]

    matrices = (
        selective_accuracy_change_matrices(
            records,
            fractions=[
                "labels_001pct"
            ],
            conditions=["clean"],
            coverages=[
                1.0,
                0.5,
            ],
        )
    )

    np.testing.assert_allclose(
        matrices["clean"],
        [[0.0, 0.01]],
    )


def test_selective_matrix_rejects_missing_coverage(
) -> None:
    with pytest.raises(
        ValueError,
        match="coverage grid",
    ):
        selective_accuracy_change_matrices(
            [raw_record()],
            fractions=[
                "labels_001pct"
            ],
            conditions=["clean"],
            coverages=[0.9],
        )


def test_reliability_filters_empty_bins() -> None:
    result = reliability_curve(
        raw_record(),
        variant="baseline",
    )

    np.testing.assert_allclose(
        result["bin_centers"],
        [0.75],
    )
    np.testing.assert_allclose(
        result["accuracy"],
        [0.8],
    )
    np.testing.assert_allclose(
        result["mean_confidence"],
        [0.7],
    )
    np.testing.assert_allclose(
        result["example_count"],
        [10.0],
    )


def test_reliability_rejects_invalid_variant(
) -> None:
    with pytest.raises(
        ValueError,
        match="variant",
    ):
        reliability_curve(
            raw_record(),
            variant="unknown",
        )


def test_finds_unique_record() -> None:
    record = find_record(
        [raw_record()],
        fraction_identifier=(
            "labels_001pct"
        ),
        method="random",
        seed=2026,
        condition="clean",
    )

    assert record["seed"] == 2026


def test_headline_summary_by_condition() -> None:
    summary = headline_summary(
        [
            raw_record(
                condition="clean",
                nll_change=-0.1,
            ),
            raw_record(
                condition="severe",
                seed=2027,
                nll_change=0.5,
            ),
        ]
    )

    clean = summary["conditions"][
        "clean"
    ]["negative_log_likelihood"]
    severe = summary["conditions"][
        "severe"
    ]["negative_log_likelihood"]

    assert clean["improved_count"] == 1
    assert severe["worsened_count"] == 1


def test_headline_summary_preservation_count(
) -> None:
    summary = headline_summary(
        [
            raw_record(seed=2026),
            raw_record(seed=2027),
        ]
    )

    assert summary["record_count"] == 2
    assert (
        summary[
            "accuracy_preserved_count"
        ]
        == 2
    )
