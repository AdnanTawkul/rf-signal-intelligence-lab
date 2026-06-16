from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.error_analysis import (
    compute_class_snr_analysis,
)


def test_class_snr_analysis_computes_expected_accuracy() -> None:
    labels = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=np.int64,
    )
    predictions = np.array(
        [0, 1, 0, 0, 1, 0, 1, 1],
        dtype=np.int64,
    )
    snr_db = np.array(
        [-4.0, -4.0, 8.0, 8.0, -4.0, -4.0, 8.0, 8.0],
        dtype=np.float32,
    )

    analysis = compute_class_snr_analysis(
        labels,
        predictions,
        snr_db,
        num_classes=2,
    )

    np.testing.assert_array_equal(
        analysis.snr_values_db,
        np.array([-4.0, 8.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        analysis.accuracy,
        np.array(
            [
                [0.5, 1.0],
                [0.5, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        analysis.counts,
        np.full((2, 2), 2, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        analysis.error_counts,
        np.array(
            [
                [1, 0],
                [1, 0],
            ],
            dtype=np.int64,
        ),
    )


def test_missing_class_snr_combination_has_nan_accuracy() -> None:
    analysis = compute_class_snr_analysis(
        labels=np.array([0, 1], dtype=np.int64),
        predictions=np.array([0, 1], dtype=np.int64),
        snr_db=np.array([-4.0, 8.0], dtype=np.float32),
        num_classes=2,
    )

    assert np.isnan(analysis.accuracy[0, 1])
    assert np.isnan(analysis.accuracy[1, 0])
    assert analysis.counts[0, 1] == 0
    assert analysis.counts[1, 0] == 0


def test_class_snr_analysis_sorts_snr_values() -> None:
    analysis = compute_class_snr_analysis(
        labels=np.array([0, 0, 1], dtype=np.int64),
        predictions=np.array([0, 0, 1], dtype=np.int64),
        snr_db=np.array([8.0, -4.0, 0.0], dtype=np.float32),
        num_classes=2,
    )

    np.testing.assert_array_equal(
        analysis.snr_values_db,
        np.array([-4.0, 0.0, 8.0], dtype=np.float32),
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
def test_class_snr_analysis_rejects_invalid_shapes(
    labels: np.ndarray,
    predictions: np.ndarray,
    snr_db: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        compute_class_snr_analysis(
            labels,
            predictions,
            snr_db,
            num_classes=4,
        )


def test_class_snr_analysis_rejects_invalid_class_indices() -> None:
    with pytest.raises(ValueError):
        compute_class_snr_analysis(
            labels=np.array([0, 4], dtype=np.int64),
            predictions=np.array([0, 1], dtype=np.int64),
            snr_db=np.array([0.0, 0.0], dtype=np.float32),
            num_classes=4,
        )


def test_class_snr_analysis_rejects_nonfinite_snr() -> None:
    with pytest.raises(ValueError):
        compute_class_snr_analysis(
            labels=np.array([0, 1], dtype=np.int64),
            predictions=np.array([0, 1], dtype=np.int64),
            snr_db=np.array([0.0, np.nan], dtype=np.float32),
            num_classes=4,
        )
