from __future__ import annotations

import math

import numpy as np
import pytest

from rfsil.evaluation.calibration import (
    evaluate_calibration,
    probabilities_from_logits,
)


def test_softmax_probabilities_sum_to_one() -> None:
    probabilities = probabilities_from_logits(
        [
            [1.0, 2.0, 3.0],
            [-2.0, 0.0, 4.0],
        ]
    )

    assert probabilities.shape == (2, 3)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(2),
        rtol=1e-12,
        atol=1e-12,
    )


def test_softmax_is_stable_for_large_logits() -> None:
    probabilities = probabilities_from_logits(
        [
            [10_000.0, 9_999.0],
            [-10_000.0, -10_001.0],
        ]
    )

    assert np.all(
        np.isfinite(probabilities)
    )
    assert probabilities[0, 0] > 0.5
    assert probabilities[1, 0] > 0.5


@pytest.mark.parametrize(
    "logits",
    (
        [1.0, 2.0],
        [[1.0]],
        [],
    ),
)
def test_rejects_invalid_logit_shapes(
    logits: object,
) -> None:
    with pytest.raises(ValueError):
        probabilities_from_logits(logits)


def test_rejects_nonfinite_logits() -> None:
    with pytest.raises(
        ValueError,
        match="finite",
    ):
        probabilities_from_logits(
            [[0.0, float("nan")]]
        )


def test_perfect_predictions_are_calibrated() -> None:
    result = evaluate_calibration(
        labels=[0, 1, 2],
        probabilities=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        bin_count=5,
    )

    assert result.accuracy == 1.0
    assert result.mean_confidence == 1.0
    assert (
        result.expected_calibration_error
        == pytest.approx(0.0)
    )
    assert (
        result.maximum_calibration_error
        == pytest.approx(0.0)
    )
    assert (
        result.negative_log_likelihood
        == pytest.approx(0.0)
    )
    assert result.brier_score == pytest.approx(
        0.0
    )


def test_computes_known_calibration_gap() -> None:
    result = evaluate_calibration(
        labels=[0, 1, 1, 0],
        probabilities=[
            [0.9, 0.1],
            [0.6, 0.4],
            [0.2, 0.8],
            [0.45, 0.55],
        ],
        bin_count=2,
    )

    assert result.accuracy == pytest.approx(
        0.5
    )
    assert result.mean_confidence == (
        pytest.approx(0.7125)
    )
    assert (
        result.expected_calibration_error
        == pytest.approx(0.2125)
    )
    assert (
        result.maximum_calibration_error
        == pytest.approx(0.2125)
    )


def test_computes_known_nll_and_brier() -> None:
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.25, 0.75],
        ]
    )

    result = evaluate_calibration(
        labels=[0, 1],
        probabilities=probabilities,
    )

    expected_nll = -0.5 * (
        math.log(0.8)
        + math.log(0.75)
    )

    assert (
        result.negative_log_likelihood
        == pytest.approx(expected_nll)
    )
    assert result.brier_score == pytest.approx(
        0.1025
    )


def test_confidence_on_boundary_uses_upper_bin() -> None:
    result = evaluate_calibration(
        labels=[0],
        probabilities=[[0.5, 0.5]],
        bin_count=2,
    )

    assert result.bins[0].example_count == 0
    assert result.bins[1].example_count == 1
    assert result.bins[1].lower_bound == 0.5
    assert result.bins[1].upper_bound == 1.0


def test_empty_bins_use_none_statistics() -> None:
    result = evaluate_calibration(
        labels=[0],
        probabilities=[[0.9, 0.1]],
        bin_count=4,
    )

    empty = result.bins[0]

    assert empty.example_count == 0
    assert empty.accuracy is None
    assert empty.mean_confidence is None
    assert empty.absolute_gap is None


def test_zero_true_probability_has_finite_nll() -> None:
    result = evaluate_calibration(
        labels=[1],
        probabilities=[[1.0, 0.0]],
    )

    assert math.isfinite(
        result.negative_log_likelihood
    )
    assert result.negative_log_likelihood > 0.0


def test_rejects_noninteger_labels() -> None:
    with pytest.raises(
        ValueError,
        match="integers",
    ):
        evaluate_calibration(
            labels=[0.0, 1.0],
            probabilities=[
                [0.8, 0.2],
                [0.2, 0.8],
            ],
        )


def test_rejects_out_of_range_labels() -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        evaluate_calibration(
            labels=[0, 2],
            probabilities=[
                [0.8, 0.2],
                [0.2, 0.8],
            ],
        )


def test_rejects_mismatched_example_counts() -> None:
    with pytest.raises(
        ValueError,
        match="same number",
    ):
        evaluate_calibration(
            labels=[0],
            probabilities=[
                [0.8, 0.2],
                [0.2, 0.8],
            ],
        )


def test_rejects_probability_rows_not_summing_to_one(
) -> None:
    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        evaluate_calibration(
            labels=[0],
            probabilities=[[0.4, 0.4]],
        )


@pytest.mark.parametrize(
    "probabilities",
    (
        [[-0.1, 1.1]],
        [[float("nan"), 0.0]],
        [[1.0]],
        [0.5, 0.5],
    ),
)
def test_rejects_invalid_probabilities(
    probabilities: object,
) -> None:
    with pytest.raises(ValueError):
        evaluate_calibration(
            labels=[0],
            probabilities=probabilities,
        )


@pytest.mark.parametrize(
    "bin_count",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_rejects_invalid_bin_count(
    bin_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="bin_count",
    ):
        evaluate_calibration(
            labels=[0],
            probabilities=[[0.8, 0.2]],
            bin_count=bin_count,
        )
