from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.seed_test import (
    SeedTestResult,
    aggregate_seed_test_results,
)


def create_result(
    seed: int,
    overall_accuracy: float,
    class_accuracy: list[float] | None = None,
    snr_values_db: list[float] | None = None,
    snr_accuracy: list[float] | None = None,
) -> SeedTestResult:
    """Create one seed-test result fixture."""
    return SeedTestResult(
        seed=seed,
        overall_accuracy=overall_accuracy,
        class_accuracy=np.asarray(
            class_accuracy or [0.9, 0.8, 0.7, 0.6],
            dtype=np.float32,
        ),
        snr_values_db=np.asarray(
            snr_values_db or [-4.0, 0.0, 4.0],
            dtype=np.float32,
        ),
        snr_accuracy=np.asarray(
            snr_accuracy or [0.6, 0.8, 0.9],
            dtype=np.float32,
        ),
    )


def test_aggregate_seed_test_results_computes_overall_statistics() -> None:
    results = [
        create_result(1, 0.90),
        create_result(2, 0.94),
        create_result(3, 0.92),
    ]

    aggregate = aggregate_seed_test_results(results)

    assert aggregate.overall_mean == pytest.approx(0.92)
    assert aggregate.overall_std == pytest.approx(
        float(np.std([0.90, 0.94, 0.92]))
    )
    assert aggregate.overall_minimum == pytest.approx(0.90)
    assert aggregate.overall_maximum == pytest.approx(0.94)


def test_aggregate_seed_test_results_computes_class_statistics() -> None:
    results = [
        create_result(
            1,
            0.90,
            class_accuracy=[1.0, 0.8, 0.9, 0.9],
        ),
        create_result(
            2,
            0.92,
            class_accuracy=[1.0, 0.9, 0.8, 1.0],
        ),
    ]

    aggregate = aggregate_seed_test_results(results)

    np.testing.assert_allclose(
        aggregate.class_accuracy_mean,
        np.array([1.0, 0.85, 0.85, 0.95]),
        atol=1e-6,
    )


def test_aggregate_seed_test_results_computes_snr_statistics() -> None:
    results = [
        create_result(
            1,
            0.90,
            snr_accuracy=[0.5, 0.8, 1.0],
        ),
        create_result(
            2,
            0.92,
            snr_accuracy=[0.7, 0.9, 1.0],
        ),
    ]

    aggregate = aggregate_seed_test_results(results)

    np.testing.assert_allclose(
        aggregate.snr_accuracy_mean,
        np.array([0.6, 0.85, 1.0]),
        atol=1e-6,
    )


def test_aggregate_seed_test_results_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        aggregate_seed_test_results([])


def test_aggregate_seed_test_results_rejects_duplicate_seeds() -> None:
    results = [
        create_result(42, 0.90),
        create_result(42, 0.91),
    ]

    with pytest.raises(ValueError):
        aggregate_seed_test_results(results)


def test_aggregate_seed_test_results_rejects_mismatched_snr_values() -> None:
    results = [
        create_result(1, 0.90),
        create_result(
            2,
            0.91,
            snr_values_db=[-6.0, 0.0, 4.0],
        ),
    ]

    with pytest.raises(ValueError):
        aggregate_seed_test_results(results)
