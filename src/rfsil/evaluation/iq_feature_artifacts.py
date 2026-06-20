from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]
UInt64Array = NDArray[np.uint64]
BoolArray = NDArray[np.bool_]

IQ_FEATURE_ARTIFACT_FORMAT_VERSION = 1

_REQUIRED_KEYS = {
    "values",
    "feature_names",
    "labels",
    "snr_db",
    "frequency_offset_hz",
    "phase_offset_rad",
    "amplitude_scale",
    "time_shift_samples",
    "rayleigh_fading",
    "example_seed",
    "condition",
    "source_dataset",
    "format_version",
}


def _validate_non_empty_string(
    value: object,
    *,
    name: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    validated = value.strip()

    if not validated:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return validated


def _validate_float_vector(
    value: object,
    *,
    name: str,
    example_count: int,
) -> Float64Array:
    raw = np.asarray(value)

    if (
        raw.ndim != 1
        or raw.shape[0] != example_count
    ):
        raise ValueError(
            f"{name} must have shape "
            f"({example_count},)."
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
            f"{name} must contain real "
            "numeric values."
        )

    converted = np.asarray(
        raw,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    return np.ascontiguousarray(converted)


def _validate_integer_vector(
    value: object,
    *,
    name: str,
    example_count: int,
    allow_negative: bool,
) -> Int64Array:
    raw = np.asarray(value)

    if (
        raw.ndim != 1
        or raw.shape[0] != example_count
    ):
        raise ValueError(
            f"{name} must have shape "
            f"({example_count},)."
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

    if (
        not allow_negative
        and np.any(converted < 0)
    ):
        raise ValueError(
            f"{name} must not contain "
            "negative values."
        )

    return np.ascontiguousarray(converted)


def _validate_seed_vector(
    value: object,
    *,
    example_count: int,
) -> UInt64Array:
    raw = np.asarray(value)

    if (
        raw.ndim != 1
        or raw.shape[0] != example_count
    ):
        raise ValueError(
            "example_seed must have shape "
            f"({example_count},)."
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

    signed = np.asarray(
        raw,
        dtype=np.int64,
    )

    if np.any(signed < 0):
        raise ValueError(
            "example_seed must not contain "
            "negative values."
        )

    return np.ascontiguousarray(
        signed.astype(np.uint64)
    )


def _validate_boolean_vector(
    value: object,
    *,
    example_count: int,
) -> BoolArray:
    raw = np.asarray(value)

    if (
        raw.ndim != 1
        or raw.shape[0] != example_count
    ):
        raise ValueError(
            "rayleigh_fading must have shape "
            f"({example_count},)."
        )

    if not np.issubdtype(
        raw.dtype,
        np.bool_,
    ):
        raise ValueError(
            "rayleigh_fading must use a "
            "boolean dtype."
        )

    return np.ascontiguousarray(
        raw.astype(np.bool_, copy=False)
    )


@dataclass(frozen=True, slots=True)
class IQChannelFeatureArtifact:
    """Versioned IQ-feature artifact with paired metadata."""

    values: Float64Array
    feature_names: tuple[str, ...]
    labels: Int64Array
    snr_db: Float64Array
    frequency_offset_hz: Float64Array
    phase_offset_rad: Float64Array
    amplitude_scale: Float64Array
    time_shift_samples: Int64Array
    rayleigh_fading: BoolArray
    example_seed: UInt64Array
    condition: str
    source_dataset: str
    format_version: int = (
        IQ_FEATURE_ARTIFACT_FORMAT_VERSION
    )

    def __post_init__(self) -> None:
        values = np.asarray(
            self.values,
            dtype=np.float64,
        )

        if values.ndim != 2:
            raise ValueError(
                "values must have shape "
                "[examples, features]."
            )

        if values.shape[0] <= 0:
            raise ValueError(
                "values must contain at least "
                "one example."
            )

        if values.shape[1] <= 0:
            raise ValueError(
                "values must contain at least "
                "one feature."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "values must contain only "
                "finite values."
            )

        names = tuple(
            _validate_non_empty_string(
                name,
                name="feature name",
            )
            for name in self.feature_names
        )

        if len(names) != values.shape[1]:
            raise ValueError(
                "feature_names length must "
                "match the feature count."
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "feature_names must be unique."
            )

        example_count = int(
            values.shape[0]
        )

        labels = _validate_integer_vector(
            self.labels,
            name="labels",
            example_count=example_count,
            allow_negative=False,
        )
        snr_db = _validate_float_vector(
            self.snr_db,
            name="snr_db",
            example_count=example_count,
        )
        frequency_offset_hz = (
            _validate_float_vector(
                self.frequency_offset_hz,
                name="frequency_offset_hz",
                example_count=example_count,
            )
        )
        phase_offset_rad = (
            _validate_float_vector(
                self.phase_offset_rad,
                name="phase_offset_rad",
                example_count=example_count,
            )
        )
        amplitude_scale = (
            _validate_float_vector(
                self.amplitude_scale,
                name="amplitude_scale",
                example_count=example_count,
            )
        )
        time_shift_samples = (
            _validate_integer_vector(
                self.time_shift_samples,
                name="time_shift_samples",
                example_count=example_count,
                allow_negative=True,
            )
        )
        rayleigh_fading = (
            _validate_boolean_vector(
                self.rayleigh_fading,
                example_count=example_count,
            )
        )
        example_seed = _validate_seed_vector(
            self.example_seed,
            example_count=example_count,
        )

        condition = _validate_non_empty_string(
            self.condition,
            name="condition",
        )
        source_dataset = (
            _validate_non_empty_string(
                self.source_dataset,
                name="source_dataset",
            )
        )

        if (
            isinstance(
                self.format_version,
                bool,
            )
            or not isinstance(
                self.format_version,
                Integral,
            )
            or int(self.format_version) <= 0
        ):
            raise ValueError(
                "format_version must be a "
                "positive integer."
            )

        object.__setattr__(
            self,
            "values",
            np.ascontiguousarray(values),
        )
        object.__setattr__(
            self,
            "feature_names",
            names,
        )
        object.__setattr__(
            self,
            "labels",
            labels,
        )
        object.__setattr__(
            self,
            "snr_db",
            snr_db,
        )
        object.__setattr__(
            self,
            "frequency_offset_hz",
            frequency_offset_hz,
        )
        object.__setattr__(
            self,
            "phase_offset_rad",
            phase_offset_rad,
        )
        object.__setattr__(
            self,
            "amplitude_scale",
            amplitude_scale,
        )
        object.__setattr__(
            self,
            "time_shift_samples",
            time_shift_samples,
        )
        object.__setattr__(
            self,
            "rayleigh_fading",
            rayleigh_fading,
        )
        object.__setattr__(
            self,
            "example_seed",
            example_seed,
        )
        object.__setattr__(
            self,
            "condition",
            condition,
        )
        object.__setattr__(
            self,
            "source_dataset",
            source_dataset,
        )
        object.__setattr__(
            self,
            "format_version",
            int(self.format_version),
        )

    @property
    def example_count(self) -> int:
        """Return the number of feature rows."""
        return int(self.values.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of features."""
        return int(self.values.shape[1])

    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible artifact metadata."""
        return {
            "condition": self.condition,
            "source_dataset": (
                self.source_dataset
            ),
            "format_version": (
                self.format_version
            ),
            "example_count": (
                self.example_count
            ),
            "feature_count": (
                self.feature_count
            ),
            "feature_names": list(
                self.feature_names
            ),
        }


def save_iq_channel_feature_artifact(
    artifact: IQChannelFeatureArtifact,
    output_path: str | Path,
) -> Path:
    """Save one IQ-feature artifact without pickle."""
    if not isinstance(
        artifact,
        IQChannelFeatureArtifact,
    ):
        raise TypeError(
            "artifact must be an "
            "IQChannelFeatureArtifact."
        )

    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        values=artifact.values,
        feature_names=np.asarray(
            artifact.feature_names,
            dtype=np.str_,
        ),
        labels=artifact.labels,
        snr_db=artifact.snr_db,
        frequency_offset_hz=(
            artifact.frequency_offset_hz
        ),
        phase_offset_rad=(
            artifact.phase_offset_rad
        ),
        amplitude_scale=(
            artifact.amplitude_scale
        ),
        time_shift_samples=(
            artifact.time_shift_samples
        ),
        rayleigh_fading=(
            artifact.rayleigh_fading
        ),
        example_seed=artifact.example_seed,
        condition=np.asarray(
            artifact.condition,
            dtype=np.str_,
        ),
        source_dataset=np.asarray(
            artifact.source_dataset,
            dtype=np.str_,
        ),
        format_version=np.asarray(
            artifact.format_version,
            dtype=np.int64,
        ),
    )

    return path


def load_iq_channel_feature_artifact(
    input_path: str | Path,
) -> IQChannelFeatureArtifact:
    """Load and validate one IQ-feature artifact."""
    path = Path(input_path)

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        missing = (
            _REQUIRED_KEYS
            - set(archive.files)
        )

        if missing:
            raise ValueError(
                "IQ-feature artifact is missing "
                f"keys: {sorted(missing)}."
            )

        feature_names = tuple(
            str(value)
            for value in archive[
                "feature_names"
            ].tolist()
        )

        return IQChannelFeatureArtifact(
            values=archive["values"],
            feature_names=feature_names,
            labels=archive["labels"],
            snr_db=archive["snr_db"],
            frequency_offset_hz=archive[
                "frequency_offset_hz"
            ],
            phase_offset_rad=archive[
                "phase_offset_rad"
            ],
            amplitude_scale=archive[
                "amplitude_scale"
            ],
            time_shift_samples=archive[
                "time_shift_samples"
            ],
            rayleigh_fading=archive[
                "rayleigh_fading"
            ],
            example_seed=archive[
                "example_seed"
            ],
            condition=str(
                archive["condition"].item()
            ),
            source_dataset=str(
                archive[
                    "source_dataset"
                ].item()
            ),
            format_version=int(
                archive[
                    "format_version"
                ].item()
            ),
        )


__all__ = [
    "IQ_FEATURE_ARTIFACT_FORMAT_VERSION",
    "IQChannelFeatureArtifact",
    "load_iq_channel_feature_artifact",
    "save_iq_channel_feature_artifact",
]
