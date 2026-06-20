from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.iq_feature_detection import (
    IQFeatureDirectionSelection,
    analyze_iq_feature_detection,
    evaluate_iq_feature_directions,
    select_iq_feature_directions,
    summarize_iq_feature_results,
)
from rfsil.evaluation.paired_shift_split import (
    PairedShiftSplit,
)


def example_split() -> PairedShiftSplit:
    return PairedShiftSplit(
        development_indices=np.asarray(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
        test_indices=np.asarray(
            [4, 5, 6, 7],
            dtype=np.int64,
        ),
        development_fraction=0.5,
        split_seed=2026,
        snr_decimals=6,
        stratum_count=1,
    )


def example_values():
    clean = np.zeros(
        (8, 3),
        dtype=np.float64,
    )

    mild = np.zeros_like(clean)
    moderate = np.zeros_like(clean)
    severe = np.zeros_like(clean)

    for matrix, scale in (
        (mild, 1.0),
        (moderate, 2.0),
        (severe, 3.0),
    ):
        matrix[:, 0] = scale
        matrix[:, 1] = -scale

        # Development suggests larger means shift.
        matrix[:4, 2] = scale

        # Test deliberately reverses to prove
        # that direction is not selected on test.
        matrix[4:, 2] = -scale

    return {
        "clean": clean,
        "shifted": {
            "mild": mild,
            "moderate": moderate,
            "severe": severe,
        },
        "feature_names": (
            "increasing",
            "decreasing",
            "development_only",
        ),
    }


def test_selects_larger_direction() -> None:
    data = example_values()

    selections = select_iq_feature_directions(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    assert selections[0].multiplier == 1
    assert (
        selections[0].direction
        == "larger_is_shift_like"
    )
    assert (
        selections[0]
        .directed_development_metrics
        .auroc
        == pytest.approx(1.0)
    )


def test_selects_smaller_direction() -> None:
    data = example_values()

    selections = select_iq_feature_directions(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    assert selections[1].multiplier == -1
    assert (
        selections[1].direction
        == "smaller_is_shift_like"
    )
    assert (
        selections[1]
        .directed_development_metrics
        .auroc
        == pytest.approx(1.0)
    )


def test_same_direction_is_used_for_all_conditions(
) -> None:
    data = example_values()

    analysis = analyze_iq_feature_detection(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    multipliers = {
        result.multiplier
        for result
        in analysis.condition_results
        if result.feature_name
        == "decreasing"
    }

    assert multipliers == {-1}


def test_direction_uses_development_only() -> None:
    data = example_values()

    analysis = analyze_iq_feature_detection(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    reversed_results = [
        result
        for result
        in analysis.condition_results
        if result.feature_name
        == "development_only"
    ]

    assert all(
        result.multiplier == 1
        for result in reversed_results
    )
    assert all(
        result.directed_test_metrics.auroc
        == pytest.approx(0.0)
        for result in reversed_results
    )


def test_returns_every_condition_feature_pair(
) -> None:
    data = example_values()

    analysis = analyze_iq_feature_detection(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    assert len(analysis.selections) == 3
    assert len(
        analysis.condition_results
    ) == 9


def test_summary_ranks_strong_features_first(
) -> None:
    data = example_values()

    analysis = analyze_iq_feature_detection(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    names = [
        summary["feature_name"]
        for summary
        in analysis.feature_summaries
    ]

    assert names[-1] == "development_only"
    assert set(names[:2]) == {
        "increasing",
        "decreasing",
    }


def test_analysis_to_dict() -> None:
    data = example_values()

    analysis = analyze_iq_feature_detection(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )
    payload = analysis.to_dict()

    assert payload["feature_count"] == 3
    assert payload["condition_count"] == 3
    assert (
        payload["split"]["test_count"]
        == 4
    )
    assert len(payload["selections"]) == 3
    assert len(
        payload["condition_results"]
    ) == 9


@pytest.mark.parametrize(
    "clean_values",
    (
        np.ones(8),
        np.empty((0, 3)),
        np.empty((8, 0)),
        np.full(
            (8, 3),
            float("nan"),
        ),
        np.ones(
            (8, 3),
            dtype=np.bool_,
        ),
    ),
)
def test_rejects_invalid_clean_matrix(
    clean_values: object,
) -> None:
    data = example_values()

    with pytest.raises(ValueError):
        select_iq_feature_directions(
            clean_values,
            data["shifted"],
            data["feature_names"],
            example_split(),
        )


@pytest.mark.parametrize(
    "feature_names",
    (
        ("a", "b"),
        ("a", "a", "c"),
        ("a", "", "c"),
    ),
)
def test_rejects_invalid_feature_names(
    feature_names: object,
) -> None:
    data = example_values()

    with pytest.raises(ValueError):
        select_iq_feature_directions(
            data["clean"],
            data["shifted"],
            feature_names,
            example_split(),
        )


@pytest.mark.parametrize(
    "shifted_values",
    (
        {},
        [],
        {"": np.ones((8, 3))},
    ),
)
def test_rejects_invalid_shifted_mapping(
    shifted_values: object,
) -> None:
    data = example_values()

    with pytest.raises(ValueError):
        select_iq_feature_directions(
            data["clean"],
            shifted_values,
            data["feature_names"],
            example_split(),
        )


@pytest.mark.parametrize(
    "selection_change",
    (
        "missing",
        "duplicate",
        "invalid_multiplier",
    ),
)
def test_rejects_invalid_selections(
    selection_change: str,
) -> None:
    data = example_values()

    selections = list(
        select_iq_feature_directions(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            example_split(),
        )
    )

    if selection_change == "missing":
        selections.pop()
    elif selection_change == "duplicate":
        selections[1] = selections[0]
    else:
        original = selections[0]
        selections[0] = (
            IQFeatureDirectionSelection(
                feature_name=(
                    original.feature_name
                ),
                multiplier=2,
                direction=original.direction,
                raw_development_metrics=(
                    original
                    .raw_development_metrics
                ),
                directed_development_metrics=(
                    original
                    .directed_development_metrics
                ),
            )
        )

    with pytest.raises(ValueError):
        evaluate_iq_feature_directions(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            example_split(),
            selections,
        )


def test_summary_rejects_missing_results() -> None:
    data = example_values()

    selections = select_iq_feature_directions(
        data["clean"],
        data["shifted"],
        data["feature_names"],
        example_split(),
    )

    with pytest.raises(
        ValueError,
        match="condition results",
    ):
        summarize_iq_feature_results(
            selections,
            (),
        )
