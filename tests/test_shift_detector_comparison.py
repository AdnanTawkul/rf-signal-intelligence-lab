from __future__ import annotations

import pytest

from rfsil.evaluation.channel_shift import (
    ShiftDetectionMetrics,
)
from rfsil.evaluation.shift_detector_comparison import (
    SHIFT_DETECTOR_SYSTEMS,
    DetectorComparisonRecord,
    aggregate_detector_comparison_records,
)


def create_metrics(
    *,
    auroc: float,
    average_precision: float | None = None,
    fpr95: float = 0.5,
) -> ShiftDetectionMetrics:
    return ShiftDetectionMetrics(
        auroc=auroc,
        average_precision=(
            auroc
            if average_precision is None
            else average_precision
        ),
        fpr_at_target_tpr=fpr95,
        target_tpr=0.95,
        clean_count=700,
        shifted_count=700,
        clean_mean=0.0,
        shifted_mean=1.0,
        clean_std=1.0,
        shifted_std=1.0,
    )


def create_record(
    *,
    fraction: str = "labels_001pct",
    method: str = "random",
    seed: int = 2026,
    condition: str = "mild",
    system_name: str = "lag8",
    auroc: float = 0.7,
    fpr95: float = 0.5,
) -> DetectorComparisonRecord:
    return DetectorComparisonRecord(
        fraction_identifier=fraction,
        method=method,
        seed=seed,
        condition=condition,
        system_name=system_name,
        metrics=create_metrics(
            auroc=auroc,
            fpr95=fpr95,
        ),
        development_auroc=0.75,
        selected_l2_strength=(
            0.1
            if system_name in (
                "all_iq_linear",
                "iq_energy_fusion",
            )
            else None
        ),
        direction=(
            "larger_is_shift_like"
            if system_name
            in ("lag8", "output_energy")
            else None
        ),
    )


def complete_records() -> list[
    DetectorComparisonRecord
]:
    records = []

    offsets = {
        "lag8": 0.00,
        "all_iq_linear": 0.05,
        "output_energy": -0.10,
        "iq_energy_fusion": 0.10,
    }

    for seed in (2026, 2027):
        for condition in (
            "mild",
            "severe",
        ):
            for system_name in (
                SHIFT_DETECTOR_SYSTEMS
            ):
                records.append(
                    create_record(
                        seed=seed,
                        condition=condition,
                        system_name=(
                            system_name
                        ),
                        auroc=(
                            0.7
                            + offsets[
                                system_name
                            ]
                        ),
                        fpr95=(
                            0.6
                            - offsets[
                                system_name
                            ]
                        ),
                    )
                )

    return records


def test_record_to_dict() -> None:
    record = create_record()
    payload = record.to_dict()

    assert (
        payload["checkpoint_identifier"]
        == "labels_001pct/random/seed_2026"
    )
    assert payload["system_name"] == "lag8"
    assert payload["metrics"]["auroc"] == (
        pytest.approx(0.7)
    )


def test_aggregate_counts() -> None:
    result = (
        aggregate_detector_comparison_records(
            complete_records()
        )
    )

    assert result["record_count"] == 16
    assert result["checkpoint_count"] == 2
    assert result["condition_count"] == 2
    assert result["system_count"] == 4
    assert len(
        result["system_condition_groups"]
    ) == 8


def test_aggregate_group_mean() -> None:
    result = (
        aggregate_detector_comparison_records(
            complete_records()
        )
    )

    group = next(
        group
        for group
        in result["system_condition_groups"]
        if (
            group["condition"] == "mild"
            and group["system_name"]
            == "all_iq_linear"
        )
    )

    assert group["run_count"] == 2
    assert group["metrics"]["auroc"][
        "mean"
    ] == pytest.approx(0.75)


def test_selects_best_system() -> None:
    result = (
        aggregate_detector_comparison_records(
            complete_records()
        )
    )

    assert all(
        entry["system_name"]
        == "iq_energy_fusion"
        for entry
        in result["best_system_by_condition"]
    )


def test_paired_fusion_changes() -> None:
    result = (
        aggregate_detector_comparison_records(
            complete_records()
        )
    )

    change = next(
        item
        for item
        in result["fusion_paired_changes"]
        if (
            item["condition"] == "mild"
            and item["baseline_system"]
            == "all_iq_linear"
        )
    )

    assert change[
        "fusion_minus_baseline"
    ]["auroc"]["mean"] == pytest.approx(
        0.05
    )
    assert (
        change["improvement_counts"][
            "auroc_improved"
        ]
        == 2
    )
    assert (
        change["improvement_counts"][
            "fpr95_improved"
        ]
        == 2
    )


def test_rejects_duplicate_records() -> None:
    records = complete_records()
    records.append(records[0])

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        aggregate_detector_comparison_records(
            records
        )


def test_rejects_missing_system() -> None:
    records = complete_records()
    records.pop()

    with pytest.raises(
        ValueError,
        match="all four systems",
    ):
        aggregate_detector_comparison_records(
            records
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fraction_identifier", ""),
        ("method", None),
        ("condition", ""),
        ("system_name", "unknown"),
        ("seed", True),
    ),
)
def test_rejects_invalid_identity(
    field: str,
    value: object,
) -> None:
    arguments = {
        "fraction_identifier": (
            "labels_001pct"
        ),
        "method": "random",
        "seed": 2026,
        "condition": "mild",
        "system_name": "lag8",
        "metrics": create_metrics(
            auroc=0.7
        ),
        "development_auroc": 0.75,
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        DetectorComparisonRecord(
            **arguments,
        )


@pytest.mark.parametrize(
    "value",
    (
        -0.1,
        1.1,
        float("nan"),
    ),
)
def test_rejects_invalid_development_auroc(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="development_auroc",
    ):
        DetectorComparisonRecord(
            fraction_identifier=(
                "labels_001pct"
            ),
            method="random",
            seed=2026,
            condition="mild",
            system_name="lag8",
            metrics=create_metrics(
                auroc=0.7
            ),
            development_auroc=value,
        )


@pytest.mark.parametrize(
    "value",
    (
        0.0,
        -1.0,
        float("nan"),
        True,
    ),
)
def test_rejects_invalid_l2(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="selected_l2_strength",
    ):
        DetectorComparisonRecord(
            fraction_identifier=(
                "labels_001pct"
            ),
            method="random",
            seed=2026,
            condition="mild",
            system_name="all_iq_linear",
            metrics=create_metrics(
                auroc=0.7
            ),
            development_auroc=0.75,
            selected_l2_strength=value,
        )
