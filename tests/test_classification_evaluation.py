from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.classification import evaluate_predictions


def test_perfect_predictions_produce_unit_accuracy() -> None:
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    predictions = labels.copy()
    snr_db = np.array([-4.0, 0.0, 4.0, 8.0], dtype=np.float32)

    evaluation = evaluate_predictions(
        labels,
        predictions,
        snr_db,
        num_classes=4,
    )

    assert evaluation.accuracy == pytest.approx(1.0)
    np.testing.assert_array_equal(
        evaluation.confusion_matrix,
        np.eye(4, dtype=np.int64),
    )
    np.testing.assert_allclose(
        evaluation.class_accuracy,
        np.ones(4, dtype=np.float32),
    )


def test_confusion_matrix_uses_true_rows_and_predicted_columns() -> None:
    labels = np.array(
        [0, 0, 1, 1, 2, 2, 3, 3],
        dtype=np.int64,
    )
    predictions = np.array(
        [0, 1, 1, 1, 2, 3, 0, 3],
        dtype=np.int64,
    )
    snr_db = np.zeros(8, dtype=np.float32)

    evaluation = evaluate_predictions(
        labels,
        predictions,
        snr_db,
        num_classes=4,
    )

    expected = np.array(
        [
            [1, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        evaluation.confusion_matrix,
        expected,
    )


def test_accuracy_is_computed_for_each_snr_level() -> None:
    labels = np.array(
        [0, 1, 0, 1, 0, 1],
        dtype=np.int64,
    )
    predictions = np.array(
        [0, 0, 0, 1, 1, 1],
        dtype=np.int64,
    )
    snr_db = np.array(
        [-4.0, -4.0, 0.0, 0.0, 8.0, 8.0],
        dtype=np.float32,
    )

    evaluation = evaluate_predictions(
        labels,
        predictions,
        snr_db,
        num_classes=2,
    )

    np.testing.assert_array_equal(
        evaluation.snr_values_db,
        np.array([-4.0, 0.0, 8.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        evaluation.snr_accuracy,
        np.array([0.5, 1.0, 0.5], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("labels", "predictions", "snr_db"),
    [
        (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        ),
        (
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
        (
            np.array([[0, 1]], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
    ],
)
def test_evaluation_rejects_invalid_array_shapes(
    labels: np.ndarray,
    predictions: np.ndarray,
    snr_db: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(
            labels,
            predictions,
            snr_db,
            num_classes=4,
        )


def test_evaluation_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(
            np.array([0, 4], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0.0, 0.0], dtype=np.float32),
            num_classes=4,
        )


def test_evaluation_rejects_nonfinite_snr() -> None:
    with pytest.raises(ValueError):
        evaluate_predictions(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            np.array([0.0, np.nan], dtype=np.float32),
            num_classes=4,
        )
