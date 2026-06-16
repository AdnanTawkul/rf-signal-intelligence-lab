from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedRunResult:
    """Summary of one independently seeded training run."""

    seed: int
    best_epoch: int
    best_validation_accuracy: float
    final_train_loss: float
    final_train_accuracy: float
    final_validation_loss: float
    final_validation_accuracy: float


@dataclass(frozen=True, slots=True)
class SeedSweepStatistics:
    """Aggregate statistics across multiple training seeds."""

    run_count: int
    best_validation_mean: float
    best_validation_std: float
    best_validation_minimum: float
    best_validation_maximum: float
    final_validation_mean: float
    final_validation_std: float
    best_seed: int


def aggregate_seed_results(
    results: list[SeedRunResult],
) -> SeedSweepStatistics:
    """Aggregate validation performance across independent seeds."""
    if not results:
        raise ValueError("results must not be empty.")

    seeds = [result.seed for result in results]

    if len(seeds) != len(set(seeds)):
        raise ValueError("Each seed must appear exactly once.")

    best_validation = np.asarray(
        [
            result.best_validation_accuracy
            for result in results
        ],
        dtype=np.float64,
    )
    final_validation = np.asarray(
        [
            result.final_validation_accuracy
            for result in results
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(best_validation)):
        raise ValueError(
            "best_validation_accuracy values must be finite."
        )

    if not np.all(np.isfinite(final_validation)):
        raise ValueError(
            "final_validation_accuracy values must be finite."
        )

    best_index = int(np.argmax(best_validation))

    return SeedSweepStatistics(
        run_count=len(results),
        best_validation_mean=float(np.mean(best_validation)),
        best_validation_std=float(np.std(best_validation)),
        best_validation_minimum=float(np.min(best_validation)),
        best_validation_maximum=float(np.max(best_validation)),
        final_validation_mean=float(np.mean(final_validation)),
        final_validation_std=float(np.std(final_validation)),
        best_seed=results[best_index].seed,
    )


__all__ = [
    "SeedRunResult",
    "SeedSweepStatistics",
    "aggregate_seed_results",
]
