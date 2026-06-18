from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset, Subset

Int64Array = NDArray[np.int64]


def _validate_positive_integer(
    value: object,
    name: str,
) -> int:
    """Validate and return a positive integer."""
    if isinstance(value, bool) or not isinstance(
        value,
        Integral,
    ):
        raise ValueError(
            f"{name} must be an integer."
        )

    validated_value = int(value)

    if validated_value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return validated_value


def _validate_seed(seed: object) -> int:
    """Validate a deterministic random seed."""
    if isinstance(seed, bool) or not isinstance(
        seed,
        Integral,
    ):
        raise ValueError(
            "seed must be an integer."
        )

    return int(seed)


def _validate_labels(
    labels: object,
) -> Int64Array:
    """Validate class labels."""
    array = np.asarray(labels)

    if array.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if array.shape[0] == 0:
        raise ValueError(
            "labels must contain at least one example."
        )

    if not np.issubdtype(
        array.dtype,
        np.integer,
    ):
        raise TypeError(
            "labels must use an integer dtype."
        )

    converted = np.asarray(
        array,
        dtype=np.int64,
    )

    if np.any(converted < 0):
        raise ValueError(
            "labels must not contain negative values."
        )

    return converted


def _validate_snr_values(
    snr_db: object,
) -> NDArray[np.float64]:
    """Validate SNR metadata."""
    array = np.asarray(snr_db)

    if array.ndim != 1:
        raise ValueError(
            "snr_db must be one-dimensional."
        )

    if array.shape[0] == 0:
        raise ValueError(
            "snr_db must contain at least one example."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise TypeError(
            "snr_db must use a numeric dtype."
        )

    converted = np.asarray(
        array,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            "snr_db must contain only finite values."
        )

    return converted


def select_class_snr_stratified_indices(
    labels: object,
    snr_db: object,
    examples_per_stratum: int,
    seed: int,
) -> Int64Array:
    """Select an equal number of examples per class-SNR pair.

    Sampling is performed without replacement and is fully
    deterministic for a fixed seed.
    """
    validated_labels = _validate_labels(labels)
    validated_snr = _validate_snr_values(snr_db)

    if (
        validated_labels.shape[0]
        != validated_snr.shape[0]
    ):
        raise ValueError(
            "labels and snr_db must contain the "
            "same number of examples."
        )

    selected_examples_per_stratum = (
        _validate_positive_integer(
            examples_per_stratum,
            "examples_per_stratum",
        )
    )
    selected_seed = _validate_seed(seed)

    strata = sorted(
        {
            (
                int(label),
                float(snr),
            )
            for label, snr in zip(
                validated_labels,
                validated_snr,
                strict=True,
            )
        }
    )

    if not strata:
        raise ValueError(
            "No class-SNR strata were found."
        )

    generator = np.random.default_rng(
        selected_seed
    )
    selected_indices: list[int] = []

    for label, snr in strata:
        candidate_indices = np.flatnonzero(
            (validated_labels == label)
            & (validated_snr == snr)
        )

        if (
            candidate_indices.shape[0]
            < selected_examples_per_stratum
        ):
            raise ValueError(
                "Insufficient examples for "
                f"class={label}, SNR={snr:g} dB: "
                f"required="
                f"{selected_examples_per_stratum}, "
                f"available="
                f"{candidate_indices.shape[0]}."
            )

        sampled_indices = generator.choice(
            candidate_indices,
            size=selected_examples_per_stratum,
            replace=False,
        )

        selected_indices.extend(
            int(index)
            for index in sampled_indices
        )

    result = np.asarray(
        sorted(selected_indices),
        dtype=np.int64,
    )

    if np.unique(result).shape[0] != result.shape[0]:
        raise RuntimeError(
            "Stratified selection produced "
            "duplicate indices."
        )

    return result


def create_class_snr_stratified_subset(
    dataset: Dataset[Any],
    examples_per_stratum: int,
    seed: int,
) -> Subset[Any]:
    """Create a PyTorch subset balanced by class and SNR."""
    labels = getattr(dataset, "labels", None)
    snr_db = getattr(dataset, "snr_db", None)

    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            "dataset.labels must be a torch.Tensor."
        )

    if not isinstance(snr_db, torch.Tensor):
        raise TypeError(
            "dataset.snr_db must be a torch.Tensor."
        )

    if labels.shape[0] != len(dataset):
        raise ValueError(
            "dataset.labels length must match "
            "the dataset length."
        )

    if snr_db.shape[0] != len(dataset):
        raise ValueError(
            "dataset.snr_db length must match "
            "the dataset length."
        )

    indices = select_class_snr_stratified_indices(
        labels=labels.cpu().numpy(),
        snr_db=snr_db.cpu().numpy(),
        examples_per_stratum=(
            examples_per_stratum
        ),
        seed=seed,
    )

    return Subset(
        dataset,
        indices.tolist(),
    )


__all__ = [
    "create_class_snr_stratified_subset",
    "select_class_snr_stratified_indices",
]
