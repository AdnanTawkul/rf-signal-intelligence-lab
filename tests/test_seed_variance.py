from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.seed_variance import (
    SeedRunResult,
    aggregate_seed_results,
)


def create_result(
    seed: int,
    best_accuracy: float,
    final_accuracy: float,
) -> SeedRunResult:
    """Create one compact seed-result fixture."""
    return SeedRunResult(
        seed=seed,
        best_epoch=20,
        best_validation_accuracy=best_accuracy,
        final_train_loss=0.2,
        final_train_accuracy=0.9,
        final_validation_loss=0.3,
        final_validation_accuracy=final_accuracy,
    )


def test_aggregate_seed_results_computes_expected_statistics() -> None:
    results = [
        create_result(1, 0.90, 0.88),
        create_result(2, 0.94, 0.91),
        create_result(3, 0.92, 0.89),
    ]

    statistics = aggregate_seed_results(results)

    assert statistics.run_count == 3
    assert statistics.best_validation_mean == pytest.approx(0.92)
    assert statistics.best_validation_minimum == pytest.approx(0.90)
    assert statistics.best_validation_maximum == pytest.approx(0.94)
    assert statistics.final_validation_mean == pytest.approx(
        (0.88 + 0.91 + 0.89) / 3.0
    )
    assert statistics.best_seed == 2


def test_aggregate_seed_results_uses_population_standard_deviation() -> None:
    results = [
        create_result(1, 0.80, 0.75),
        create_result(2, 0.90, 0.85),
    ]

    statistics = aggregate_seed_results(results)

    assert statistics.best_validation_std == pytest.approx(
        float(np.std([0.80, 0.90]))
    )
    assert statistics.final_validation_std == pytest.approx(
        float(np.std([0.75, 0.85]))
    )


def test_aggregate_seed_results_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        aggregate_seed_results([])


def test_aggregate_seed_results_rejects_duplicate_seeds() -> None:
    results = [
        create_result(42, 0.90, 0.88),
        create_result(42, 0.91, 0.89),
    ]

    with pytest.raises(ValueError):
        aggregate_seed_results(results)
