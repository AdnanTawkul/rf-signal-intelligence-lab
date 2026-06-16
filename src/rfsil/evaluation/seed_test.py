from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class SeedTestResult:
    """Held-out classification metrics for one training seed."""

    seed: int
    overall_accuracy: float
    class_accuracy: Float32Array
    snr_values_db: Float32Array
    snr_accuracy: Float32Array


@dataclass(frozen=True, slots=True)
class SeedTestAggregate:
    """Aggregate held-out metrics across independently trained models."""

    seeds: Int64Array
    overall_accuracy: Float32Array
    overall_mean: float
    overall_std: float
    overall_minimum: float
    overall_maximum: float
    class_accuracy_mean: Float32Array
    class_accuracy_std: Float32Array
    snr_values_db: Float32Array
    snr_accuracy_mean: Float32Array
    snr_accuracy_std: Float32Array


def aggregate_seed_test_results(
    results: list[SeedTestResult],
) -> SeedTestAggregate:
    """Aggregate held-out metrics across multiple training seeds."""
    if not results:
        raise ValueError("results must not be empty.")

    seeds = np.asarray(
        [result.seed for result in results],
        dtype=np.int64,
    )

    if len(seeds) != len(np.unique(seeds)):
        raise ValueError("Each seed must appear exactly once.")

    class_count = len(results[0].class_accuracy)
    reference_snr_values = np.asarray(
        results[0].snr_values_db,
        dtype=np.float32,
    )

    if class_count == 0:
        raise ValueError("class_accuracy must not be empty.")

    if reference_snr_values.size == 0:
        raise ValueError("snr_values_db must not be empty.")

    overall_accuracy = np.asarray(
        [result.overall_accuracy for result in results],
        dtype=np.float64,
    )

    class_accuracy_rows: list[Float32Array] = []
    snr_accuracy_rows: list[Float32Array] = []

    for result in results:
        class_accuracy = np.asarray(
            result.class_accuracy,
            dtype=np.float32,
        )
        snr_values = np.asarray(
            result.snr_values_db,
            dtype=np.float32,
        )
        snr_accuracy = np.asarray(
            result.snr_accuracy,
            dtype=np.float32,
        )

        if class_accuracy.shape != (class_count,):
            raise ValueError(
                "All class_accuracy arrays must have matching shapes."
            )

        if snr_values.shape != reference_snr_values.shape:
            raise ValueError(
                "All snr_values_db arrays must have matching shapes."
            )

        if snr_accuracy.shape != reference_snr_values.shape:
            raise ValueError(
                "All snr_accuracy arrays must match snr_values_db."
            )

        if not np.allclose(
            snr_values,
            reference_snr_values,
        ):
            raise ValueError(
                "All runs must use identical SNR values."
            )

        if not np.all(np.isfinite(class_accuracy)):
            raise ValueError(
                "class_accuracy must contain only finite values."
            )

        if not np.all(np.isfinite(snr_accuracy)):
            raise ValueError(
                "snr_accuracy must contain only finite values."
            )

        class_accuracy_rows.append(class_accuracy)
        snr_accuracy_rows.append(snr_accuracy)

    if not np.all(np.isfinite(overall_accuracy)):
        raise ValueError(
            "overall_accuracy must contain only finite values."
        )

    class_accuracy_matrix = np.stack(
        class_accuracy_rows,
        axis=0,
    )
    snr_accuracy_matrix = np.stack(
        snr_accuracy_rows,
        axis=0,
    )

    return SeedTestAggregate(
        seeds=seeds,
        overall_accuracy=overall_accuracy.astype(np.float32),
        overall_mean=float(np.mean(overall_accuracy)),
        overall_std=float(np.std(overall_accuracy)),
        overall_minimum=float(np.min(overall_accuracy)),
        overall_maximum=float(np.max(overall_accuracy)),
        class_accuracy_mean=np.mean(
            class_accuracy_matrix,
            axis=0,
        ).astype(np.float32),
        class_accuracy_std=np.std(
            class_accuracy_matrix,
            axis=0,
        ).astype(np.float32),
        snr_values_db=reference_snr_values,
        snr_accuracy_mean=np.mean(
            snr_accuracy_matrix,
            axis=0,
        ).astype(np.float32),
        snr_accuracy_std=np.std(
            snr_accuracy_matrix,
            axis=0,
        ).astype(np.float32),
    )


__all__ = [
    "SeedTestAggregate",
    "SeedTestResult",
    "aggregate_seed_test_results",
]
