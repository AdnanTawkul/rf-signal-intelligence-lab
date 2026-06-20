from __future__ import annotations

import pytest

from rfsil.evaluation.calibration_analysis import (
    aggregate_temperature_transfer_records,
    evaluate_temperature_transfer,
    selective_accuracy_curve,
)
from rfsil.evaluation.calibration_artifacts import (
    build_calibration_artifact,
)


def example_artifact():
    return build_calibration_artifact(
        labels=[0, 0, 1, 1],
        logits=[
            [8.0, 0.0],
            [7.0, 0.0],
            [0.0, 8.0],
            [8.0, 0.0],
        ],
        class_names=(
            "BPSK",
            "QPSK",
        ),
    )


def test_full_coverage_matches_accuracy() -> None:
    points = selective_accuracy_curve(
        labels=[0, 1, 1, 0],
        probabilities=[
            [0.95, 0.05],
            [0.90, 0.10],
            [0.20, 0.80],
            [0.55, 0.45],
        ],
        coverages=[1.0],
    )

    assert points[0].actual_coverage == 1.0
    assert points[0].accuracy == pytest.approx(
        0.75
    )


def test_top_half_uses_highest_confidence() -> None:
    points = selective_accuracy_curve(
        labels=[0, 1, 1, 0],
        probabilities=[
            [0.95, 0.05],
            [0.90, 0.10],
            [0.20, 0.80],
            [0.55, 0.45],
        ],
        coverages=[0.5],
    )

    point = points[0]

    assert point.selected_count == 2
    assert point.actual_coverage == 0.5
    assert point.accuracy == pytest.approx(
        0.5
    )
    assert point.risk == pytest.approx(
        0.5
    )


def test_requested_coverage_uses_ceiling() -> None:
    points = selective_accuracy_curve(
        labels=[0, 0, 0, 0],
        probabilities=[
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ],
        coverages=[0.51],
    )

    assert points[0].selected_count == 3
    assert points[0].actual_coverage == (
        pytest.approx(0.75)
    )


@pytest.mark.parametrize(
    "coverages",
    (
        [],
        [0.0],
        [1.1],
        [float("nan")],
        [True],
    ),
)
def test_rejects_invalid_coverages(
    coverages: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="coverage",
    ):
        selective_accuracy_curve(
            labels=[0],
            probabilities=[[0.8, 0.2]],
            coverages=coverages,
        )


def test_transfer_preserves_accuracy() -> None:
    artifact = example_artifact()
    result = evaluate_temperature_transfer(
        artifact,
        temperature=2.0,
        bin_count=5,
    )

    assert result["accuracy_preserved"]
    assert (
        result["baseline"]["accuracy"]
        == result["calibrated"]["accuracy"]
    )


def test_transfer_records_metric_changes() -> None:
    result = evaluate_temperature_transfer(
        example_artifact(),
        temperature=2.0,
        bin_count=5,
    )

    assert (
        result["calibrated"][
            "negative_log_likelihood"
        ]
        < result["baseline"][
            "negative_log_likelihood"
        ]
    )
    assert (
        result["metric_changes"][
            "negative_log_likelihood"
        ]
        < 0.0
    )


def create_record(
    *,
    seed: int,
    temperature: float,
) -> dict:
    comparison = evaluate_temperature_transfer(
        example_artifact(),
        temperature=temperature,
        bin_count=5,
        coverages=[1.0, 0.5],
    )

    return {
        "fraction_identifier": (
            "labels_001pct"
        ),
        "method": "random",
        "seed": seed,
        "condition": "clean",
        **comparison,
    }


def test_aggregate_summarizes_seeds() -> None:
    result = (
        aggregate_temperature_transfer_records(
            [
                create_record(
                    seed=2026,
                    temperature=2.0,
                ),
                create_record(
                    seed=2027,
                    temperature=3.0,
                ),
            ]
        )
    )

    assert result["record_count"] == 2
    assert result["group_count"] == 1

    group = result["groups"][0]

    assert group["run_count"] == 2
    assert group["seeds"] == [
        2026,
        2027,
    ]
    assert group["temperature"][
        "mean"
    ] == pytest.approx(2.5)


def test_aggregate_counts_nll_improvements() -> None:
    result = (
        aggregate_temperature_transfer_records(
            [
                create_record(
                    seed=2026,
                    temperature=2.0,
                ),
                create_record(
                    seed=2027,
                    temperature=3.0,
                ),
            ]
        )
    )

    counts = result["groups"][0][
        "metrics"
    ]["negative_log_likelihood"][
        "comparison_counts"
    ]

    assert counts["improved"] == 2
    assert counts["worsened"] == 0
