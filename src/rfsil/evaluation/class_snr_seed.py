from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class ClassSNRSeedResult:
    """Class-by-SNR accuracy matrix for one training seed."""

    seed: int
    snr_values_db: Float32Array
    accuracy: Float32Array


@dataclass(frozen=True, slots=True)
class ClassSNRSeedAggregate:
    """Aggregate class-by-SNR statistics across training seeds."""

    seeds: Int64Array
    snr_values_db: Float32Array
    accuracy_mean: Float32Array
    accuracy_std: Float32Array


def aggregate_class_snr_seed_results(
    results: list[ClassSNRSeedResult],
) -> ClassSNRSeedAggregate:
    """Aggregate class-by-SNR accuracy matrices across seeds."""
    if not results:
        raise ValueError("results must not be empty.")

    seeds = np.asarray(
        [result.seed for result in results],
        dtype=np.int64,
    )

    if len(seeds) != len(np.unique(seeds)):
        raise ValueError("Each seed must appear exactly once.")

    reference_snr_values = np.asarray(
        results[0].snr_values_db,
        dtype=np.float32,
    )
    reference_shape = np.asarray(
        results[0].accuracy,
        dtype=np.float32,
    ).shape

    if reference_snr_values.ndim != 1:
        raise ValueError("snr_values_db must be one-dimensional.")

    if reference_snr_values.size == 0:
        raise ValueError("snr_values_db must not be empty.")

    if len(reference_shape) != 2:
        raise ValueError("accuracy must be two-dimensional.")

    if reference_shape[1] != len(reference_snr_values):
        raise ValueError(
            "accuracy columns must correspond to snr_values_db."
        )

    matrices: list[Float32Array] = []

    for result in results:
        snr_values = np.asarray(
            result.snr_values_db,
            dtype=np.float32,
        )
        accuracy = np.asarray(
            result.accuracy,
            dtype=np.float32,
        )

        if snr_values.shape != reference_snr_values.shape:
            raise ValueError(
                "All snr_values_db arrays must have matching shapes."
            )

        if not np.allclose(
            snr_values,
            reference_snr_values,
        ):
            raise ValueError(
                "All runs must use identical SNR values."
            )

        if accuracy.shape != reference_shape:
            raise ValueError(
                "All accuracy matrices must have matching shapes."
            )

        if np.any(np.isinf(accuracy)):
            raise ValueError(
                "accuracy matrices must not contain infinite values."
            )

        matrices.append(accuracy)

    accuracy_stack = np.stack(
        matrices,
        axis=0,
    )

    finite_mask = np.isfinite(accuracy_stack)
    finite_counts = np.sum(
        finite_mask,
        axis=0,
        dtype=np.int64,
    )
    valid_cells = finite_counts > 0

    finite_sums = np.sum(
        np.where(
            finite_mask,
            accuracy_stack,
            0.0,
        ),
        axis=0,
        dtype=np.float64,
    )

    accuracy_mean = np.full(
        reference_shape,
        np.nan,
        dtype=np.float32,
    )
    accuracy_mean[valid_cells] = (
        finite_sums[valid_cells]
        / finite_counts[valid_cells]
    ).astype(np.float32)

    centered = np.where(
        finite_mask,
        accuracy_stack - accuracy_mean[np.newaxis, ...],
        0.0,
    )
    squared_deviation_sums = np.sum(
        np.square(centered.astype(np.float64)),
        axis=0,
    )

    accuracy_std = np.full(
        reference_shape,
        np.nan,
        dtype=np.float32,
    )
    accuracy_std[valid_cells] = np.sqrt(
        squared_deviation_sums[valid_cells]
        / finite_counts[valid_cells]
    ).astype(np.float32)

    return ClassSNRSeedAggregate(
        seeds=seeds,
        snr_values_db=reference_snr_values,
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
    )


__all__ = [
    "ClassSNRSeedAggregate",
    "ClassSNRSeedResult",
    "aggregate_class_snr_seed_results",
]
