from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

Int64Array = NDArray[np.int64]
Float64Array = NDArray[np.float64]
UInt64Array = NDArray[np.uint64]


@dataclass(frozen=True, slots=True)
class PairedShiftSplit:
    """Development/test split shared by paired conditions."""

    development_indices: Int64Array
    test_indices: Int64Array
    development_fraction: float
    split_seed: int
    snr_decimals: int
    stratum_count: int

    @property
    def development_count(self) -> int:
        """Return the number of development examples."""
        return int(
            self.development_indices.size
        )

    @property
    def test_count(self) -> int:
        """Return the number of test examples."""
        return int(self.test_indices.size)

    @property
    def example_count(self) -> int:
        """Return the total number of examples."""
        return (
            self.development_count
            + self.test_count
        )

    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible split metadata."""
        output = asdict(self)
        output.pop(
            "development_indices"
        )
        output.pop("test_indices")
        output["development_count"] = (
            self.development_count
        )
        output["test_count"] = (
            self.test_count
        )
        output["example_count"] = (
            self.example_count
        )
        return output


def _validate_integer_vector(
    value: object,
    *,
    name: str,
) -> Int64Array:
    raw = np.asarray(value)

    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(
            f"{name} must be a non-empty "
            "one-dimensional array."
        )

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            f"{name} must use an integer dtype."
        )

    converted = np.asarray(
        raw,
        dtype=np.int64,
    )

    return np.ascontiguousarray(converted)


def _validate_snr(
    value: object,
    *,
    example_count: int,
) -> Float64Array:
    raw = np.asarray(value)

    if raw.shape != (example_count,):
        raise ValueError(
            "snr_db must match the example count."
        )

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "snr_db must contain real "
            "numeric values."
        )

    converted = np.asarray(
        raw,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            "snr_db must contain only "
            "finite values."
        )

    return np.ascontiguousarray(converted)


def _validate_seeds(
    value: object,
    *,
    example_count: int,
) -> UInt64Array:
    raw = np.asarray(value)

    if raw.shape != (example_count,):
        raise ValueError(
            "example_seed must match the "
            "example count."
        )

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "example_seed must use an "
            "integer dtype."
        )

    if (
        np.issubdtype(
            raw.dtype,
            np.signedinteger,
        )
        and np.any(raw < 0)
    ):
        raise ValueError(
            "example_seed must not contain "
            "negative values."
        )

    converted = np.asarray(
        raw,
        dtype=np.uint64,
    )

    if np.unique(converted).size != (
        converted.size
    ):
        raise ValueError(
            "example_seed values must be unique."
        )

    return np.ascontiguousarray(converted)


def _validate_fraction(
    value: object,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            "development_fraction must be "
            "within (0, 1)."
        )

    fraction = float(value)

    if (
        not math.isfinite(fraction)
        or fraction <= 0.0
        or fraction >= 1.0
    ):
        raise ValueError(
            "development_fraction must be "
            "within (0, 1)."
        )

    return fraction


def _validate_non_negative_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError(
            f"{name} must be a non-negative "
            "integer."
        )

    return int(value)


def _selection_key(
    example_seed: int,
    *,
    split_seed: int,
) -> int:
    payload = (
        f"{split_seed}:{example_seed}"
    ).encode("ascii")

    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"rfsil-split",
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    )


def create_paired_shift_split(
    labels: object,
    snr_db: object,
    example_seed: object,
    *,
    development_fraction: object = 0.50,
    split_seed: object = 2026,
    snr_decimals: object = 6,
) -> PairedShiftSplit:
    """Create a deterministic class/SNR-stratified split."""
    validated_labels = (
        _validate_integer_vector(
            labels,
            name="labels",
        )
    )

    if np.any(validated_labels < 0):
        raise ValueError(
            "labels must not contain "
            "negative values."
        )

    example_count = int(
        validated_labels.size
    )
    validated_snr = _validate_snr(
        snr_db,
        example_count=example_count,
    )
    validated_seeds = _validate_seeds(
        example_seed,
        example_count=example_count,
    )
    fraction = _validate_fraction(
        development_fraction
    )
    validated_split_seed = (
        _validate_non_negative_integer(
            split_seed,
            name="split_seed",
        )
    )
    validated_snr_decimals = (
        _validate_non_negative_integer(
            snr_decimals,
            name="snr_decimals",
        )
    )

    rounded_snr = np.round(
        validated_snr,
        decimals=validated_snr_decimals,
    )

    strata: dict[
        tuple[int, float],
        list[int],
    ] = {}

    for index, (
        label,
        snr,
    ) in enumerate(
        zip(
            validated_labels,
            rounded_snr,
            strict=True,
        )
    ):
        key = (
            int(label),
            float(snr),
        )
        strata.setdefault(
            key,
            [],
        ).append(index)

    development_indices: list[int] = []
    test_indices: list[int] = []

    for key in sorted(strata):
        indices = strata[key]

        if len(indices) < 2:
            raise ValueError(
                "Every class/SNR stratum must "
                "contain at least two examples."
            )

        ordered = sorted(
            indices,
            key=lambda index: (
                _selection_key(
                    int(
                        validated_seeds[
                            index
                        ]
                    ),
                    split_seed=(
                        validated_split_seed
                    ),
                ),
                int(
                    validated_seeds[index]
                ),
            ),
        )

        development_count = int(
            math.floor(
                len(ordered) * fraction
                + 0.5
            )
        )
        development_count = max(
            1,
            min(
                len(ordered) - 1,
                development_count,
            ),
        )

        development_indices.extend(
            ordered[:development_count]
        )
        test_indices.extend(
            ordered[development_count:]
        )

    development = np.asarray(
        sorted(development_indices),
        dtype=np.int64,
    )
    test = np.asarray(
        sorted(test_indices),
        dtype=np.int64,
    )

    if np.intersect1d(
        development,
        test,
    ).size:
        raise RuntimeError(
            "Development and test indices "
            "overlap."
        )

    combined = np.sort(
        np.concatenate(
            (
                development,
                test,
            )
        )
    )

    np.testing.assert_array_equal(
        combined,
        np.arange(
            example_count,
            dtype=np.int64,
        ),
    )

    return PairedShiftSplit(
        development_indices=development,
        test_indices=test,
        development_fraction=fraction,
        split_seed=validated_split_seed,
        snr_decimals=(
            validated_snr_decimals
        ),
        stratum_count=len(strata),
    )


__all__ = [
    "PairedShiftSplit",
    "create_paired_shift_split",
]
