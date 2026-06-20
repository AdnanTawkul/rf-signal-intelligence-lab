from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LoadedIQ:
    """Validated IQ windows loaded from one file."""

    source_path: Path
    array_key: str | None
    iq: Float32Array
    sample_indices: Int64Array
    labels: Int64Array | None = None
    snr_db: Float32Array | None = None

    @property
    def batch_size(self) -> int:
        """Return the number of loaded windows."""
        return int(self.iq.shape[0])

    @property
    def channel_count(self) -> int:
        """Return the number of real channels."""
        return int(self.iq.shape[1])

    @property
    def sample_count(self) -> int:
        """Return the number of samples per window."""
        return int(self.iq.shape[2])


def _validate_expected_sample_count(
    value: object,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            "expected_sample_count must be "
            "a positive integer or None."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            "expected_sample_count must be "
            "a positive integer or None."
        )

    return validated


def _validate_sample_index(
    value: object,
    *,
    batch_size: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            "sample_index must be an integer "
            "or None."
        )

    validated = int(value)

    if not 0 <= validated < batch_size:
        raise IndexError(
            "sample_index is out of range."
        )

    return validated


def _normalize_iq_array(
    value: object,
) -> Float32Array:
    """Convert supported IQ layouts to [batch, 2, samples]."""
    array = np.asarray(value)

    if array.size == 0:
        raise ValueError(
            "IQ input must not be empty."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise ValueError(
            "IQ input must contain numeric values."
        )

    if np.iscomplexobj(array):
        if array.ndim == 1:
            array = array[np.newaxis, :]
        elif array.ndim != 2:
            raise ValueError(
                "Complex IQ input must have shape "
                "[samples] or [batch, samples]."
            )

        normalized = np.stack(
            (
                np.real(array),
                np.imag(array),
            ),
            axis=1,
        )
    else:
        if array.ndim == 2:
            if array.shape[0] != 2:
                raise ValueError(
                    "Real IQ input must have shape "
                    "[2, samples] or "
                    "[batch, 2, samples]."
                )

            normalized = array[np.newaxis, :, :]
        elif array.ndim == 3:
            if array.shape[1] != 2:
                raise ValueError(
                    "Real IQ input must have shape "
                    "[2, samples] or "
                    "[batch, 2, samples]."
                )

            normalized = array
        else:
            raise ValueError(
                "Real IQ input must have shape "
                "[2, samples] or "
                "[batch, 2, samples]."
            )

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if normalized.shape[0] <= 0:
        raise ValueError(
            "IQ batch must not be empty."
        )

    if normalized.shape[2] <= 0:
        raise ValueError(
            "IQ windows must contain samples."
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "IQ input must contain only "
            "finite values."
        )

    return np.ascontiguousarray(normalized)


def _load_optional_vector(
    content: np.lib.npyio.NpzFile,
    *,
    key: str,
    batch_size: int,
    dtype: np.dtype,
) -> np.ndarray | None:
    if key not in content.files:
        return None

    raw = np.asarray(content[key])

    if raw.ndim != 1:
        raise ValueError(
            f"NPZ metadata {key!r} must be "
            "one-dimensional."
        )

    if raw.shape[0] != batch_size:
        raise ValueError(
            f"NPZ metadata {key!r} length does "
            "not match the IQ batch size."
        )

    if not np.issubdtype(
        raw.dtype,
        np.number,
    ):
        raise ValueError(
            f"NPZ metadata {key!r} must be "
            "numeric."
        )

    converted = np.asarray(
        raw,
        dtype=dtype,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"NPZ metadata {key!r} must contain "
            "only finite values."
        )

    return converted


def _select_sample(
    *,
    iq: Float32Array,
    labels: Int64Array | None,
    snr_db: Float32Array | None,
    sample_index: int | None,
) -> tuple[
    Float32Array,
    Int64Array,
    Int64Array | None,
    Float32Array | None,
]:
    batch_size = int(iq.shape[0])
    validated_index = _validate_sample_index(
        sample_index,
        batch_size=batch_size,
    )

    if validated_index is None:
        indices = np.arange(
            batch_size,
            dtype=np.int64,
        )

        return (
            iq,
            indices,
            labels,
            snr_db,
        )

    selection = slice(
        validated_index,
        validated_index + 1,
    )

    return (
        np.ascontiguousarray(iq[selection]),
        np.asarray(
            [validated_index],
            dtype=np.int64,
        ),
        (
            None
            if labels is None
            else np.ascontiguousarray(
                labels[selection]
            )
        ),
        (
            None
            if snr_db is None
            else np.ascontiguousarray(
                snr_db[selection]
            )
        ),
    )


def load_iq_file(
    input_path: str | Path,
    *,
    array_key: str = "iq",
    sample_index: int | None = None,
    expected_sample_count: int | None = None,
) -> LoadedIQ:
    """Load and validate IQ windows from NPY or NPZ."""
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"IQ input file does not exist: "
            f"{path}"
        )

    validated_sample_count = (
        _validate_expected_sample_count(
            expected_sample_count
        )
    )

    suffix = path.suffix.lower()

    if suffix == ".npy":
        raw_iq = np.load(
            path,
            allow_pickle=False,
        )
        iq = _normalize_iq_array(raw_iq)
        labels = None
        snr_db = None
        selected_key = None

    elif suffix == ".npz":
        if not isinstance(array_key, str):
            raise TypeError(
                "array_key must be a string."
            )

        normalized_key = array_key.strip()

        if not normalized_key:
            raise ValueError(
                "array_key must not be empty."
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as content:
            if normalized_key not in content.files:
                available = ", ".join(
                    sorted(content.files)
                )

                raise KeyError(
                    f"NPZ file does not contain "
                    f"{normalized_key!r}. "
                    f"Available keys: {available}"
                )

            iq = _normalize_iq_array(
                content[normalized_key]
            )

            labels = _load_optional_vector(
                content,
                key="labels",
                batch_size=int(iq.shape[0]),
                dtype=np.dtype(np.int64),
            )
            snr_db = _load_optional_vector(
                content,
                key="snr_db",
                batch_size=int(iq.shape[0]),
                dtype=np.dtype(np.float32),
            )

        selected_key = normalized_key

    else:
        raise ValueError(
            "IQ input file must use the "
            ".npy or .npz extension."
        )

    if (
        validated_sample_count is not None
        and iq.shape[2]
        != validated_sample_count
    ):
        raise ValueError(
            "IQ sample count does not match "
            f"the expected value of "
            f"{validated_sample_count}."
        )

    (
        selected_iq,
        selected_indices,
        selected_labels,
        selected_snr,
    ) = _select_sample(
        iq=iq,
        labels=labels,
        snr_db=snr_db,
        sample_index=sample_index,
    )

    return LoadedIQ(
        source_path=path,
        array_key=selected_key,
        iq=selected_iq,
        sample_indices=selected_indices,
        labels=selected_labels,
        snr_db=selected_snr,
    )


__all__ = [
    "LoadedIQ",
    "load_iq_file",
]
