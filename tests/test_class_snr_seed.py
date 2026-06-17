from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.class_snr_seed import (
    ClassSNRSeedResult,
    aggregate_class_snr_seed_results,
)


def create_result(
    seed: int,
    accuracy: list[list[float]],
    snr_values_db: list[float] | None = None,
) -> ClassSNRSeedResult:
    """Create one class-by-SNR result fixture."""
    return ClassSNRSeedResult(
        seed=seed,
        snr_values_db=np.asarray(
            snr_values_db or [-4.0, 0.0, 4.0],
            dtype=np.float32,
        ),
        accuracy=np.asarray(
            accuracy,
            dtype=np.float32,
        ),
    )


def test_aggregate_computes_expected_mean() -> None:
    results = [
        create_result(
            1,
            [
                [0.6, 0.8, 1.0],
                [0.4, 0.7, 0.9],
            ],
        ),
        create_result(
            2,
            [
                [0.8, 0.9, 1.0],
                [0.6, 0.9, 1.0],
            ],
        ),
    ]

    aggregate = aggregate_class_snr_seed_results(results)

    np.testing.assert_allclose(
        aggregate.accuracy_mean,
        np.asarray(
            [
                [0.7, 0.85, 1.0],
                [0.5, 0.8, 0.95],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_aggregate_computes_population_standard_deviation() -> None:
    results = [
        create_result(
            1,
            [
                [0.6, 0.8, 1.0],
                [0.4, 0.7, 0.9],
            ],
        ),
        create_result(
            2,
            [
                [0.8, 0.9, 1.0],
                [0.6, 0.9, 1.0],
            ],
        ),
    ]

    aggregate = aggregate_class_snr_seed_results(results)

    expected = np.std(
        np.asarray(
            [
                results[0].accuracy,
                results[1].accuracy,
            ]
        ),
        axis=0,
    )

    np.testing.assert_allclose(
        aggregate.accuracy_std,
        expected,
        atol=1e-6,
    )


def test_aggregate_ignores_matching_nan_cells() -> None:
    results = [
        create_result(
            1,
            [
                [np.nan, 0.8, 1.0],
                [0.4, 0.7, 0.9],
            ],
        ),
        create_result(
            2,
            [
                [np.nan, 0.9, 1.0],
                [0.6, 0.9, 1.0],
            ],
        ),
    ]

    aggregate = aggregate_class_snr_seed_results(results)

    assert np.isnan(aggregate.accuracy_mean[0, 0])
    assert aggregate.accuracy_mean[0, 1] == pytest.approx(0.85)


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        aggregate_class_snr_seed_results([])


def test_aggregate_rejects_duplicate_seeds() -> None:
    results = [
        create_result(1, [[0.5, 0.7, 0.9]]),
        create_result(1, [[0.6, 0.8, 1.0]]),
    ]

    with pytest.raises(ValueError):
        aggregate_class_snr_seed_results(results)


def test_aggregate_rejects_mismatched_snr_values() -> None:
    results = [
        create_result(1, [[0.5, 0.7, 0.9]]),
        create_result(
            2,
            [[0.6, 0.8, 1.0]],
            snr_values_db=[-6.0, 0.0, 4.0],
        ),
    ]

    with pytest.raises(ValueError):
        aggregate_class_snr_seed_results(results)


def test_aggregate_rejects_mismatched_matrix_shapes() -> None:
    results = [
        create_result(1, [[0.5, 0.7, 0.9]]),
        create_result(
            2,
            [
                [0.6, 0.8, 1.0],
                [0.5, 0.7, 0.9],
            ],
        ),
    ]

    with pytest.raises(ValueError):
        aggregate_class_snr_seed_results(results)
