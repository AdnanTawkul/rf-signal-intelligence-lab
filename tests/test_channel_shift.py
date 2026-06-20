from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.channel_shift import (
    SHIFT_SCORE_NAMES,
    compute_shift_scores,
    evaluate_shift_detection,
    evaluate_shift_score_sets,
)


def example_logits() -> np.ndarray:
    return np.asarray(
        [
            [8.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_computes_all_shift_scores() -> None:
    scores = compute_shift_scores(
        example_logits()
    )

    assert tuple(scores) == SHIFT_SCORE_NAMES

    for values in scores.values():
        assert values.shape == (3,)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))


def test_uncertain_example_has_larger_scores() -> None:
    scores = compute_shift_scores(
        example_logits()
    )

    assert (
        scores["msp_uncertainty"][1]
        > scores["msp_uncertainty"][0]
    )
    assert (
        scores["predictive_entropy"][1]
        > scores["predictive_entropy"][0]
    )
    assert (
        scores["negative_logit_margin"][1]
        > scores["negative_logit_margin"][0]
    )


def test_provided_probabilities_match_derived(
) -> None:
    logits = example_logits()
    shifted = logits - np.max(
        logits,
        axis=1,
        keepdims=True,
    )
    exponentials = np.exp(shifted)
    probabilities = (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )

    derived = compute_shift_scores(logits)
    provided = compute_shift_scores(
        logits,
        probabilities=probabilities,
    )

    for name in SHIFT_SCORE_NAMES:
        np.testing.assert_allclose(
            provided[name],
            derived[name],
        )


def test_perfect_detection() -> None:
    result = evaluate_shift_detection(
        clean_scores=[0.0, 0.1, 0.2],
        shifted_scores=[0.8, 0.9, 1.0],
    )

    assert result.auroc == pytest.approx(
        1.0
    )
    assert (
        result.average_precision
        == pytest.approx(1.0)
    )
    assert (
        result.fpr_at_target_tpr
        == pytest.approx(0.0)
    )


def test_reversed_detection() -> None:
    result = evaluate_shift_detection(
        clean_scores=[0.8, 0.9, 1.0],
        shifted_scores=[0.0, 0.1, 0.2],
    )

    assert result.auroc == pytest.approx(
        0.0
    )
    assert result.fpr_at_target_tpr == (
        pytest.approx(1.0)
    )


def test_tied_scores_are_chance() -> None:
    result = evaluate_shift_detection(
        clean_scores=[0.0, 0.0],
        shifted_scores=[0.0, 0.0],
    )

    assert result.auroc == pytest.approx(
        0.5
    )
    assert (
        result.average_precision
        == pytest.approx(0.5)
    )
    assert result.fpr_at_target_tpr == (
        pytest.approx(1.0)
    )


def test_detection_metadata() -> None:
    result = evaluate_shift_detection(
        clean_scores=[0.0, 0.2],
        shifted_scores=[
            0.8,
            0.9,
            1.0,
        ],
        target_tpr=0.8,
    )

    assert result.clean_count == 2
    assert result.shifted_count == 3
    assert result.target_tpr == 0.8
    assert result.clean_mean == (
        pytest.approx(0.1)
    )
    assert result.shifted_mean == (
        pytest.approx(0.9)
    )
    assert result.to_dict()["auroc"] == (
        result.auroc
    )


def test_evaluates_matching_score_sets() -> None:
    result = evaluate_shift_score_sets(
        clean_scores={
            "entropy": [0.0, 0.1],
            "energy": [-5.0, -4.0],
        },
        shifted_scores={
            "entropy": [0.8, 0.9],
            "energy": [-2.0, -1.0],
        },
    )

    assert set(result) == {
        "entropy",
        "energy",
    }
    assert result["entropy"].auroc == (
        pytest.approx(1.0)
    )
    assert result["energy"].auroc == (
        pytest.approx(1.0)
    )


@pytest.mark.parametrize(
    "logits",
    (
        [1.0, 2.0],
        [],
        [[1.0]],
        [[1.0, float("nan")]],
        [[1.0 + 2.0j, 0.0]],
    ),
)
def test_rejects_invalid_logits(
    logits: object,
) -> None:
    with pytest.raises(ValueError):
        compute_shift_scores(logits)


@pytest.mark.parametrize(
    "probabilities",
    (
        [[0.6, 0.6, -0.2]] * 3,
        [[0.2, 0.2, 0.2]] * 3,
        [[0.5, 0.5]] * 3,
        [[0.5, 0.5, float("nan")]] * 3,
    ),
)
def test_rejects_invalid_probabilities(
    probabilities: object,
) -> None:
    with pytest.raises(ValueError):
        compute_shift_scores(
            example_logits(),
            probabilities=probabilities,
        )


@pytest.mark.parametrize(
    "scores",
    (
        [],
        [[0.1, 0.2]],
        [0.1, float("nan")],
        [True, False],
    ),
)
def test_rejects_invalid_score_vectors(
    scores: object,
) -> None:
    with pytest.raises(ValueError):
        evaluate_shift_detection(
            clean_scores=scores,
            shifted_scores=[0.5, 0.6],
        )


@pytest.mark.parametrize(
    "target_tpr",
    (
        0.0,
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_rejects_invalid_target_tpr(
    target_tpr: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="target_tpr",
    ):
        evaluate_shift_detection(
            clean_scores=[0.1],
            shifted_scores=[0.9],
            target_tpr=target_tpr,
        )


def test_rejects_mismatched_score_names() -> None:
    with pytest.raises(
        ValueError,
        match="identical names",
    ):
        evaluate_shift_score_sets(
            clean_scores={
                "entropy": [0.1],
            },
            shifted_scores={
                "energy": [0.9],
            },
        )
