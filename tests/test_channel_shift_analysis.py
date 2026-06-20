from __future__ import annotations

import pytest

from rfsil.evaluation.channel_shift_analysis import (
    aggregate_channel_shift_records,
)


def create_record(
    *,
    fraction: str = "labels_001pct",
    method: str = "random",
    seed: int = 2026,
    condition: str = "mild",
    score_name: str = "energy",
    auroc: float = 0.6,
    average_precision: float = 0.62,
    fpr95: float = 0.8,
    clean_mean: float = -6.0,
    shifted_mean: float = -5.0,
) -> dict:
    return {
        "fraction_identifier": fraction,
        "method": method,
        "seed": seed,
        "condition": condition,
        "score_name": score_name,
        "metrics": {
            "auroc": auroc,
            "average_precision": (
                average_precision
            ),
            "fpr_at_target_tpr": fpr95,
            "target_tpr": 0.95,
            "clean_count": 1400,
            "shifted_count": 1400,
            "clean_mean": clean_mean,
            "shifted_mean": shifted_mean,
            "clean_std": 1.0,
            "shifted_std": 1.2,
        },
    }


def test_rejects_empty_records() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        aggregate_channel_shift_records(
            []
        )


def test_rejects_non_mapping_record() -> None:
    with pytest.raises(
        ValueError,
        match="mapping",
    ):
        aggregate_channel_shift_records(
            ["invalid"]
        )


def test_creates_detailed_group() -> None:
    result = (
        aggregate_channel_shift_records(
            [
                create_record(
                    seed=2026,
                    auroc=0.6,
                ),
                create_record(
                    seed=2027,
                    auroc=0.8,
                ),
            ]
        )
    )

    assert result["record_count"] == 2
    assert (
        result[
            "detailed_group_count"
        ]
        == 1
    )

    group = result["groups"][0]

    assert group["run_count"] == 2
    assert group["seeds"] == [
        2026,
        2027,
    ]
    assert group["metrics"]["auroc"][
        "mean"
    ] == pytest.approx(0.7)


def test_counts_auroc_direction() -> None:
    result = (
        aggregate_channel_shift_records(
            [
                create_record(
                    seed=2026,
                    auroc=0.4,
                ),
                create_record(
                    seed=2027,
                    auroc=0.5,
                ),
                create_record(
                    seed=2028,
                    auroc=0.8,
                ),
            ]
        )
    )

    counts = result["groups"][0][
        "auroc_direction_counts"
    ]

    assert counts["below_chance"] == 1
    assert counts["at_chance"] == 1
    assert counts["above_chance"] == 1
    assert counts["at_least_0_7"] == 1
    assert counts["at_least_0_8"] == 1


def test_summarizes_score_change() -> None:
    result = (
        aggregate_channel_shift_records(
            [
                create_record(
                    seed=2026,
                    clean_mean=-6.0,
                    shifted_mean=-5.0,
                ),
                create_record(
                    seed=2027,
                    clean_mean=-4.0,
                    shifted_mean=-2.0,
                ),
            ]
        )
    )

    change = result["groups"][0][
        "score_statistics"
    ]["shifted_minus_clean"]

    assert change["mean"] == pytest.approx(
        1.5
    )
    assert change["minimum"] == (
        pytest.approx(1.0)
    )
    assert change["maximum"] == (
        pytest.approx(2.0)
    )


def test_condition_score_aggregation() -> None:
    records = [
        create_record(
            fraction="labels_001pct",
            method="random",
            seed=2026,
        ),
        create_record(
            fraction="labels_100pct",
            method="vicreg",
            seed=2027,
        ),
    ]

    result = (
        aggregate_channel_shift_records(
            records
        )
    )

    assert (
        result[
            "condition_score_group_count"
        ]
        == 1
    )
    group = result[
        "condition_score_groups"
    ][0]
    assert group["run_count"] == 2


def test_fraction_condition_groups() -> None:
    result = (
        aggregate_channel_shift_records(
            [
                create_record(
                    method="random",
                    seed=2026,
                ),
                create_record(
                    method="simclr",
                    seed=2027,
                ),
            ]
        )
    )

    assert (
        result[
            "fraction_condition_score_group_count"
        ]
        == 1
    )


def test_separates_scores_and_conditions() -> None:
    result = (
        aggregate_channel_shift_records(
            [
                create_record(
                    condition="mild",
                    score_name="energy",
                ),
                create_record(
                    condition="severe",
                    score_name="energy",
                ),
                create_record(
                    condition="mild",
                    score_name=(
                        "predictive_entropy"
                    ),
                ),
            ]
        )
    )

    assert (
        result[
            "condition_score_group_count"
        ]
        == 3
    )


@pytest.mark.parametrize(
    ("key", "value"),
    (
        (
            "fraction_identifier",
            "",
        ),
        (
            "method",
            None,
        ),
        (
            "condition",
            5,
        ),
        (
            "score_name",
            "",
        ),
        (
            "seed",
            True,
        ),
    ),
)
def test_rejects_invalid_identity(
    key: str,
    value: object,
) -> None:
    record = create_record()
    record[key] = value

    with pytest.raises(ValueError):
        aggregate_channel_shift_records(
            [record]
        )


@pytest.mark.parametrize(
    ("metric", "value"),
    (
        ("auroc", -0.1),
        ("auroc", 1.1),
        ("average_precision", 2.0),
        (
            "fpr_at_target_tpr",
            float("nan"),
        ),
        ("clean_mean", float("inf")),
    ),
)
def test_rejects_invalid_metrics(
    metric: str,
    value: float,
) -> None:
    record = create_record()
    record["metrics"][metric] = value

    with pytest.raises(ValueError):
        aggregate_channel_shift_records(
            [record]
        )
