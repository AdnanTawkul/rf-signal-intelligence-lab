from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.linear_shift_detector import (
    analyze_linear_iq_shift_detection,
    create_grouped_fold_assignments,
)
from rfsil.evaluation.paired_shift_split import (
    PairedShiftSplit,
)


def example_data():
    generator = np.random.default_rng(
        2026
    )
    example_count = 120
    feature_count = 5

    clean = generator.normal(
        size=(
            example_count,
            feature_count,
        )
    )

    shifted = {
        "mild": clean
        + np.asarray(
            [1.2, 0.4, 0.0, 0.0, 0.0]
        ),
        "moderate": clean
        + np.asarray(
            [1.8, 0.6, 0.0, 0.0, 0.0]
        ),
        "severe": clean
        + np.asarray(
            [2.4, 0.8, 0.0, 0.0, 0.0]
        ),
    }

    split = PairedShiftSplit(
        development_indices=np.arange(
            0,
            60,
            dtype=np.int64,
        ),
        test_indices=np.arange(
            60,
            120,
            dtype=np.int64,
        ),
        development_fraction=0.5,
        split_seed=2026,
        snr_decimals=6,
        stratum_count=1,
    )

    return {
        "clean": clean,
        "shifted": shifted,
        "feature_names": tuple(
            f"feature_{index}"
            for index in range(
                feature_count
            )
        ),
        "example_seed": np.arange(
            10_000,
            10_000 + example_count,
            dtype=np.uint64,
        ),
        "split": split,
    }


def test_grouped_folds_are_deterministic() -> None:
    groups = np.asarray(
        [1, 1, 2, 2, 3, 3, 4, 4],
        dtype=np.uint64,
    )

    first = (
        create_grouped_fold_assignments(
            groups,
            fold_count=2,
            cv_seed=2026,
        )
    )
    second = (
        create_grouped_fold_assignments(
            groups,
            fold_count=2,
            cv_seed=2026,
        )
    )

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_repeated_group_uses_one_fold() -> None:
    groups = np.asarray(
        [10, 10, 10, 20, 20, 30],
        dtype=np.uint64,
    )

    assignments = (
        create_grouped_fold_assignments(
            groups,
            fold_count=3,
        )
    )

    assert np.unique(
        assignments[groups == 10]
    ).size == 1
    assert np.unique(
        assignments[groups == 20]
    ).size == 1


def test_analysis_selects_candidate() -> None:
    data = example_data()

    analysis = (
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(
                1e-4,
                1e-2,
                1.0,
            ),
            fold_count=3,
        )
    )

    assert analysis.selected_l2_strength in {
        1e-4,
        1e-2,
        1.0,
    }
    assert len(analysis.candidates) == 3
    assert len(
        analysis.condition_results
    ) == 3


def test_detector_scores_shift_higher() -> None:
    data = example_data()

    analysis = (
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(1e-3, 1e-1),
            fold_count=3,
        )
    )

    assert all(
        result.metrics.auroc > 0.7
        for result
        in analysis.condition_results
    )


def test_detector_parameters_are_finite() -> None:
    data = example_data()

    analysis = (
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(1e-3,),
            fold_count=3,
        )
    )

    detector = analysis.detector

    assert np.all(
        np.isfinite(
            detector.feature_mean
        )
    )
    assert np.all(
        detector.feature_scale > 0.0
    )
    assert np.all(
        np.isfinite(
            detector.coefficients
        )
    )


def test_analysis_to_dict() -> None:
    data = example_data()

    analysis = (
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(1e-3,),
            fold_count=3,
        )
    )
    payload = analysis.to_dict()

    assert payload[
        "selected_l2_strength"
    ] == pytest.approx(1e-3)
    assert len(
        payload["condition_results"]
    ) == 3
    assert (
        payload["detector"][
            "feature_count"
        ]
        == 5
    )


@pytest.mark.parametrize(
    "clean",
    (
        np.ones(10),
        np.empty((0, 3)),
        np.empty((10, 0)),
        np.full(
            (10, 3),
            float("nan"),
        ),
        np.ones(
            (10, 3),
            dtype=np.bool_,
        ),
    ),
)
def test_rejects_invalid_clean_matrix(
    clean: object,
) -> None:
    data = example_data()

    with pytest.raises(ValueError):
        analyze_linear_iq_shift_detection(
            clean,
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(1.0,),
        )


@pytest.mark.parametrize(
    "strengths",
    (
        (),
        (0.0,),
        (-1.0,),
        (float("nan"),),
        (True,),
        (1.0, 1.0),
    ),
)
def test_rejects_invalid_strengths(
    strengths: object,
) -> None:
    data = example_data()

    with pytest.raises(ValueError):
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=strengths,
            fold_count=3,
        )


@pytest.mark.parametrize(
    ("fold_count", "cv_seed"),
    (
        (0, 2026),
        (-1, 2026),
        (True, 2026),
        (3, 0),
        (3, -1),
        (3, True),
    ),
)
def test_rejects_invalid_cv_options(
    fold_count: object,
    cv_seed: object,
) -> None:
    data = example_data()

    with pytest.raises(ValueError):
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(1.0,),
            fold_count=fold_count,
            cv_seed=cv_seed,
        )


def test_rejects_duplicate_example_seeds() -> None:
    data = example_data()
    seeds = data["example_seed"].copy()
    seeds[1] = seeds[0]

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            seeds,
            data["split"],
            l2_strengths=(1.0,),
            fold_count=3,
        )


def test_decision_function_rejects_wrong_width(
) -> None:
    data = example_data()

    analysis = (
        analyze_linear_iq_shift_detection(
            data["clean"],
            data["shifted"],
            data["feature_names"],
            data["example_seed"],
            data["split"],
            l2_strengths=(1.0,),
            fold_count=3,
        )
    )

    with pytest.raises(
        ValueError,
        match="feature count",
    ):
        analysis.detector.decision_function(
            np.ones((4, 2))
        )
