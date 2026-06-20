from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.calibration import (
    evaluate_calibration,
    probabilities_from_logits,
)
from rfsil.evaluation.temperature_scaling import (
    apply_temperature,
    fit_temperature,
    negative_log_likelihood_from_logits,
    probabilities_with_temperature,
)


def example_logits() -> np.ndarray:
    return np.asarray(
        [
            [4.0, 1.0, -1.0],
            [0.0, 2.0, 1.0],
            [-1.0, 0.0, 3.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def example_labels() -> np.ndarray:
    return np.asarray(
        [0, 1, 2, 1],
        dtype=np.int64,
    )


def test_temperature_one_preserves_logits() -> None:
    result = apply_temperature(
        example_logits(),
        1.0,
    )

    np.testing.assert_allclose(
        result,
        example_logits(),
    )


def test_larger_temperature_softens_probabilities(
) -> None:
    baseline = probabilities_with_temperature(
        example_logits(),
        1.0,
    )
    softened = probabilities_with_temperature(
        example_logits(),
        2.0,
    )

    assert np.all(
        np.max(
            softened,
            axis=1,
        )
        < np.max(
            baseline,
            axis=1,
        )
    )


def test_smaller_temperature_sharpens_probabilities(
) -> None:
    baseline = probabilities_with_temperature(
        example_logits(),
        1.0,
    )
    sharpened = probabilities_with_temperature(
        example_logits(),
        0.5,
    )

    assert np.all(
        np.max(
            sharpened,
            axis=1,
        )
        > np.max(
            baseline,
            axis=1,
        )
    )


def test_temperature_preserves_argmax() -> None:
    original = np.argmax(
        example_logits(),
        axis=1,
    )

    for temperature in (
        0.1,
        0.5,
        1.0,
        2.0,
        10.0,
    ):
        scaled = apply_temperature(
            example_logits(),
            temperature,
        )

        np.testing.assert_array_equal(
            np.argmax(
                scaled,
                axis=1,
            ),
            original,
        )


def test_logit_nll_matches_probability_nll(
) -> None:
    logits = example_logits()
    labels = example_labels()
    probabilities = probabilities_from_logits(
        logits
    )
    probability_evaluation = (
        evaluate_calibration(
            labels,
            probabilities,
        )
    )

    logit_nll = (
        negative_log_likelihood_from_logits(
            labels,
            logits,
        )
    )

    assert logit_nll == pytest.approx(
        probability_evaluation
        .negative_log_likelihood
    )


def test_fit_improves_overconfident_logits() -> None:
    logits = np.asarray(
        [
            [8.0, 0.0],
            [7.0, 0.0],
            [0.0, 8.0],
            [8.0, 0.0],
        ],
        dtype=np.float64,
    )
    labels = np.asarray(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    result = fit_temperature(
        labels,
        logits,
    )

    assert result.temperature > 1.0
    assert (
        result.calibrated_nll
        < result.baseline_nll
    )
    assert result.nll_improvement > 0.0


def test_fit_result_contains_metadata() -> None:
    result = fit_temperature(
        example_labels(),
        example_logits(),
        temperature_bounds=(
            0.25,
            8.0,
        ),
        max_iterations=100,
    )

    assert result.example_count == 4
    assert result.class_count == 3
    assert result.lower_bound == 0.25
    assert result.upper_bound == 8.0
    assert result.converged
    assert result.function_evaluations > 0
    assert result.optimization_iterations > 0
    assert (
        result.calibrated_nll
        <= result.baseline_nll
    )
    assert result.to_dict()[
        "temperature"
    ] == result.temperature


@pytest.mark.parametrize(
    "temperature",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ),
)
def test_rejects_invalid_temperature(
    temperature: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="temperature",
    ):
        apply_temperature(
            example_logits(),
            temperature,
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
        negative_log_likelihood_from_logits(
            [0],
            logits,
        )


@pytest.mark.parametrize(
    ("labels", "logits"),
    (
        (
            [0.0, 1.0],
            [[2.0, 0.0], [0.0, 2.0]],
        ),
        (
            [0],
            [[2.0, 0.0], [0.0, 2.0]],
        ),
        (
            [0, 2],
            [[2.0, 0.0], [0.0, 2.0]],
        ),
    ),
)
def test_rejects_invalid_labels(
    labels: object,
    logits: object,
) -> None:
    with pytest.raises(ValueError):
        fit_temperature(
            labels,
            logits,
        )


@pytest.mark.parametrize(
    "bounds",
    (
        (1.0, 1.0),
        (2.0, 3.0),
        (0.0, 2.0),
        (float("nan"), 2.0),
    ),
)
def test_rejects_invalid_bounds(
    bounds: object,
) -> None:
    with pytest.raises(ValueError):
        fit_temperature(
            example_labels(),
            example_logits(),
            temperature_bounds=bounds,
        )


@pytest.mark.parametrize(
    "tolerance",
    (
        0.0,
        -1.0,
        float("nan"),
        True,
    ),
)
def test_rejects_invalid_tolerance(
    tolerance: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="optimization_tolerance",
    ):
        fit_temperature(
            example_labels(),
            example_logits(),
            optimization_tolerance=(
                tolerance
            ),
        )


@pytest.mark.parametrize(
    "max_iterations",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_rejects_invalid_max_iterations(
    max_iterations: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_iterations",
    ):
        fit_temperature(
            example_labels(),
            example_logits(),
            max_iterations=max_iterations,
        )
