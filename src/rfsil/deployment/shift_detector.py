from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.iq_channel_features import (
    compute_iq_channel_features,
)
from rfsil.evaluation.linear_shift_detector import (
    StandardizedLinearShiftDetector,
)

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _nonempty_string(
    value: object,
    *,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(value)


def _finite_float(
    value: object,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite."
        )

    return number


def _positive_float(
    value: object,
    *,
    name: str,
) -> float:
    number = _finite_float(
        value,
        name=name,
    )

    if number <= 0.0:
        raise ValueError(
            f"{name} must be positive."
        )

    return number


def _unit_interval(
    value: object,
    *,
    name: str,
) -> float:
    number = _finite_float(
        value,
        name=name,
    )

    if not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} must be within [0, 1]."
        )

    return number


def _string_tuple(
    value: object,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name} must be a sequence."
        )

    try:
        raw_values = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a sequence."
        ) from error

    if not raw_values:
        raise ValueError(
            f"{name} must not be empty."
        )

    values = tuple(
        _nonempty_string(
            item,
            name=f"{name} entry",
        )
        for item in raw_values
    )

    if len(set(values)) != len(values):
        raise ValueError(
            f"{name} entries must be unique."
        )

    return values


def _positive_integer_tuple(
    value: object,
    *,
    name: str,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name} must be a sequence."
        )

    try:
        values = tuple(
            _positive_integer(
                item,
                name=f"{name} entry",
            )
            for item in value
        )
    except TypeError as error:
        raise ValueError(
            f"{name} must be a sequence."
        ) from error

    if not values:
        raise ValueError(
            f"{name} must not be empty."
        )

    if len(set(values)) != len(values):
        raise ValueError(
            f"{name} entries must be unique."
        )

    return values


def _float_vector(
    value: object,
    *,
    name: str,
    expected_size: int,
    positive: bool = False,
) -> Float64Array:
    raw = np.asarray(value)

    if raw.shape != (expected_size,):
        raise ValueError(
            f"{name} must contain exactly "
            f"{expected_size} values."
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

    array = np.asarray(
        raw,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    if positive and np.any(array <= 0.0):
        raise ValueError(
            f"{name} values must be positive."
        )

    return np.ascontiguousarray(array)


def _score_vector(
    value: object,
    *,
    name: str,
) -> Float64Array:
    raw = np.asarray(value)

    if raw.ndim != 1 or raw.size <= 0:
        raise ValueError(
            f"{name} must be a non-empty "
            "one-dimensional array."
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

    array = np.asarray(
        raw,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only "
            "finite values."
        )

    return np.ascontiguousarray(array)


@dataclass(frozen=True, slots=True)
class ShiftThresholdSelection:
    """Development-selected score threshold."""

    threshold: float
    target_tpr: float
    achieved_tpr: float
    achieved_fpr: float
    clean_count: int
    shifted_count: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible threshold data."""
        return {
            "threshold": self.threshold,
            "target_tpr": self.target_tpr,
            "achieved_tpr": self.achieved_tpr,
            "achieved_fpr": self.achieved_fpr,
            "clean_count": self.clean_count,
            "shifted_count": self.shifted_count,
            "decision_rule": (
                "score_greater_than_or_equal_to_"
                "threshold"
            ),
        }


def select_shift_threshold(
    clean_scores: object,
    shifted_scores: object,
    *,
    target_tpr: object = 0.95,
) -> ShiftThresholdSelection:
    """Select the highest threshold reaching target recall."""
    clean = _score_vector(
        clean_scores,
        name="clean_scores",
    )
    shifted = _score_vector(
        shifted_scores,
        name="shifted_scores",
    )
    target = _unit_interval(
        target_tpr,
        name="target_tpr",
    )

    if target <= 0.0:
        raise ValueError(
            "target_tpr must be greater than zero."
        )

    required_positive_count = int(
        math.ceil(
            target * shifted.size
        )
    )
    sorted_shifted = np.sort(shifted)

    threshold_index = (
        shifted.size
        - required_positive_count
    )
    threshold = float(
        sorted_shifted[threshold_index]
    )

    achieved_tpr = float(
        np.mean(
            shifted >= threshold
        )
    )
    achieved_fpr = float(
        np.mean(
            clean >= threshold
        )
    )

    return ShiftThresholdSelection(
        threshold=threshold,
        target_tpr=target,
        achieved_tpr=achieved_tpr,
        achieved_fpr=achieved_fpr,
        clean_count=int(clean.size),
        shifted_count=int(shifted.size),
    )


@dataclass(frozen=True, slots=True)
class IQShiftAssessment:
    """Batch channel-shift assessment."""

    feature_names: tuple[str, ...]
    feature_values: Float64Array
    scores: Float64Array
    threshold: float
    shift_like: BoolArray

    @property
    def batch_size(self) -> int:
        """Return the number of assessed windows."""
        return int(self.scores.size)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible assessment data."""
        records = []

        for index in range(self.batch_size):
            records.append(
                {
                    "position": index,
                    "score": float(
                        self.scores[index]
                    ),
                    "threshold": self.threshold,
                    "shift_like": bool(
                        self.shift_like[index]
                    ),
                    "features": {
                        name: float(
                            self.feature_values[
                                index,
                                feature_index,
                            ]
                        )
                        for feature_index, name
                        in enumerate(
                            self.feature_names
                        )
                    },
                }
            )

        return {
            "batch_size": self.batch_size,
            "feature_names": list(
                self.feature_names
            ),
            "threshold": self.threshold,
            "records": records,
        }


@dataclass(frozen=True, slots=True)
class IQShiftDetectorArtifact:
    """Portable all-IQ linear shift detector."""

    format_version: int
    artifact_name: str
    expected_sample_count: int
    feature_names: tuple[str, ...]
    feature_mean: Float64Array
    feature_scale: Float64Array
    coefficients: Float64Array
    intercept: float
    l2_strength: float
    threshold: float
    target_tpr: float
    development_auroc: float
    development_average_precision: float
    development_fpr_at_target_tpr: float
    development_clean_mean: float
    development_clean_std: float
    development_shifted_mean: float
    development_shifted_std: float
    autocorrelation_lags: tuple[int, ...]
    occupancy_fraction: float
    epsilon: float
    provenance: dict[str, object]

    def __post_init__(self) -> None:
        version = _positive_integer(
            self.format_version,
            name="format_version",
        )

        if version != 1:
            raise ValueError(
                "Only detector artifact format "
                "version 1 is supported."
            )

        name = _nonempty_string(
            self.artifact_name,
            name="artifact_name",
        )
        sample_count = _positive_integer(
            self.expected_sample_count,
            name="expected_sample_count",
        )
        feature_names = _string_tuple(
            self.feature_names,
            name="feature_names",
        )
        feature_count = len(feature_names)

        feature_mean = _float_vector(
            self.feature_mean,
            name="feature_mean",
            expected_size=feature_count,
        )
        feature_scale = _float_vector(
            self.feature_scale,
            name="feature_scale",
            expected_size=feature_count,
            positive=True,
        )
        coefficients = _float_vector(
            self.coefficients,
            name="coefficients",
            expected_size=feature_count,
        )

        intercept = _finite_float(
            self.intercept,
            name="intercept",
        )
        l2_strength = _positive_float(
            self.l2_strength,
            name="l2_strength",
        )
        threshold = _finite_float(
            self.threshold,
            name="threshold",
        )
        target_tpr = _unit_interval(
            self.target_tpr,
            name="target_tpr",
        )

        if target_tpr <= 0.0:
            raise ValueError(
                "target_tpr must be greater "
                "than zero."
            )

        development_auroc = _unit_interval(
            self.development_auroc,
            name="development_auroc",
        )
        development_average_precision = (
            _unit_interval(
                self.development_average_precision,
                name=(
                    "development_average_precision"
                ),
            )
        )
        development_fpr = _unit_interval(
            self.development_fpr_at_target_tpr,
            name=(
                "development_fpr_at_target_tpr"
            ),
        )
        clean_mean = _finite_float(
            self.development_clean_mean,
            name="development_clean_mean",
        )
        clean_std = _positive_float(
            self.development_clean_std,
            name="development_clean_std",
        )
        shifted_mean = _finite_float(
            self.development_shifted_mean,
            name="development_shifted_mean",
        )
        shifted_std = _positive_float(
            self.development_shifted_std,
            name="development_shifted_std",
        )
        lags = _positive_integer_tuple(
            self.autocorrelation_lags,
            name="autocorrelation_lags",
        )
        occupancy_fraction = (
            _unit_interval(
                self.occupancy_fraction,
                name="occupancy_fraction",
            )
        )

        if not 0.0 < occupancy_fraction < 1.0:
            raise ValueError(
                "occupancy_fraction must be "
                "strictly between zero and one."
            )

        epsilon = _positive_float(
            self.epsilon,
            name="epsilon",
        )

        if not isinstance(
            self.provenance,
            Mapping,
        ):
            raise ValueError(
                "provenance must be a mapping."
            )

        provenance = dict(
            self.provenance
        )

        object.__setattr__(
            self,
            "format_version",
            version,
        )
        object.__setattr__(
            self,
            "artifact_name",
            name,
        )
        object.__setattr__(
            self,
            "expected_sample_count",
            sample_count,
        )
        object.__setattr__(
            self,
            "feature_names",
            feature_names,
        )
        object.__setattr__(
            self,
            "feature_mean",
            feature_mean,
        )
        object.__setattr__(
            self,
            "feature_scale",
            feature_scale,
        )
        object.__setattr__(
            self,
            "coefficients",
            coefficients,
        )
        object.__setattr__(
            self,
            "intercept",
            intercept,
        )
        object.__setattr__(
            self,
            "l2_strength",
            l2_strength,
        )
        object.__setattr__(
            self,
            "threshold",
            threshold,
        )
        object.__setattr__(
            self,
            "target_tpr",
            target_tpr,
        )
        object.__setattr__(
            self,
            "development_auroc",
            development_auroc,
        )
        object.__setattr__(
            self,
            "development_average_precision",
            development_average_precision,
        )
        object.__setattr__(
            self,
            "development_fpr_at_target_tpr",
            development_fpr,
        )
        object.__setattr__(
            self,
            "development_clean_mean",
            clean_mean,
        )
        object.__setattr__(
            self,
            "development_clean_std",
            clean_std,
        )
        object.__setattr__(
            self,
            "development_shifted_mean",
            shifted_mean,
        )
        object.__setattr__(
            self,
            "development_shifted_std",
            shifted_std,
        )
        object.__setattr__(
            self,
            "autocorrelation_lags",
            lags,
        )
        object.__setattr__(
            self,
            "occupancy_fraction",
            occupancy_fraction,
        )
        object.__setattr__(
            self,
            "epsilon",
            epsilon,
        )
        object.__setattr__(
            self,
            "provenance",
            provenance,
        )

    @property
    def feature_count(self) -> int:
        """Return the detector feature count."""
        return len(self.feature_names)

    @property
    def detector(
        self,
    ) -> StandardizedLinearShiftDetector:
        """Construct the standardized linear model."""
        return StandardizedLinearShiftDetector(
            feature_names=self.feature_names,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            coefficients=self.coefficients,
            intercept=self.intercept,
            l2_strength=self.l2_strength,
        )

    def score_features(
        self,
        values: object,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> Float64Array:
        """Score a precomputed feature matrix."""
        if feature_names is not None:
            supplied_names = _string_tuple(
                feature_names,
                name="feature_names",
            )

            if supplied_names != self.feature_names:
                raise ValueError(
                    "Feature names or feature order "
                    "do not match the detector."
                )

        return self.detector.decision_function(
            values
        )

    def assess_iq(
        self,
        iq: object,
    ) -> IQShiftAssessment:
        """Extract IQ features and assess channel shift."""
        raw = np.asarray(iq)

        if raw.ndim == 2 or raw.ndim == 3:
            sample_count = int(
                raw.shape[-1]
            )
        else:
            raise ValueError(
                "iq must have shape [2, samples] "
                "or [batch, 2, samples]."
            )

        if sample_count != (
            self.expected_sample_count
        ):
            raise ValueError(
                "IQ sample count does not match "
                "the detector artifact."
            )

        features = compute_iq_channel_features(
            iq,
            autocorrelation_lags=(
                self.autocorrelation_lags
            ),
            occupancy_fraction=(
                self.occupancy_fraction
            ),
            epsilon=self.epsilon,
        )

        if (
            features.feature_names
            != self.feature_names
        ):
            raise RuntimeError(
                "Extracted feature names do not "
                "match the detector artifact."
            )

        scores = self.score_features(
            features.values,
            feature_names=(
                features.feature_names
            ),
        )
        shift_like = np.ascontiguousarray(
            scores >= self.threshold,
            dtype=np.bool_,
        )

        return IQShiftAssessment(
            feature_names=(
                features.feature_names
            ),
            feature_values=(
                np.ascontiguousarray(
                    features.values,
                    dtype=np.float64,
                )
            ),
            scores=np.ascontiguousarray(
                scores,
                dtype=np.float64,
            ),
            threshold=self.threshold,
            shift_like=shift_like,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the versioned JSON artifact."""
        return {
            "format_version": self.format_version,
            "artifact_name": self.artifact_name,
            "detector_type": (
                "standardized_l2_linear_iq_"
                "shift_detector"
            ),
            "input": {
                "expected_sample_count": (
                    self.expected_sample_count
                ),
                "layout": (
                    "[2, samples] or "
                    "[batch, 2, samples]"
                ),
            },
            "feature_extraction": {
                "feature_names": list(
                    self.feature_names
                ),
                "feature_count": (
                    self.feature_count
                ),
                "autocorrelation_lags": list(
                    self.autocorrelation_lags
                ),
                "occupancy_fraction": (
                    self.occupancy_fraction
                ),
                "epsilon": self.epsilon,
            },
            "model": {
                "feature_mean": (
                    self.feature_mean.tolist()
                ),
                "feature_scale": (
                    self.feature_scale.tolist()
                ),
                "coefficients": (
                    self.coefficients.tolist()
                ),
                "intercept": self.intercept,
                "l2_strength": (
                    self.l2_strength
                ),
            },
            "decision": {
                "score_direction": (
                    "larger_is_shift_like"
                ),
                "threshold": self.threshold,
                "target_tpr": self.target_tpr,
                "rule": (
                    "score >= threshold"
                ),
            },
            "development_reference": {
                "auroc": (
                    self.development_auroc
                ),
                "average_precision": (
                    self
                    .development_average_precision
                ),
                "fpr_at_target_tpr": (
                    self
                    .development_fpr_at_target_tpr
                ),
                "clean_score_mean": (
                    self.development_clean_mean
                ),
                "clean_score_std": (
                    self.development_clean_std
                ),
                "shifted_score_mean": (
                    self.development_shifted_mean
                ),
                "shifted_score_std": (
                    self.development_shifted_std
                ),
            },
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
    ) -> IQShiftDetectorArtifact:
        """Construct an artifact from parsed JSON."""
        if not isinstance(value, Mapping):
            raise ValueError(
                "Detector artifact must be a mapping."
            )

        input_data = value.get("input")
        feature_data = value.get(
            "feature_extraction"
        )
        model_data = value.get("model")
        decision_data = value.get(
            "decision"
        )
        development_data = value.get(
            "development_reference"
        )

        for name, section in (
            ("input", input_data),
            (
                "feature_extraction",
                feature_data,
            ),
            ("model", model_data),
            ("decision", decision_data),
            (
                "development_reference",
                development_data,
            ),
        ):
            if not isinstance(
                section,
                Mapping,
            ):
                raise ValueError(
                    f"{name} must be a mapping."
                )

        return cls(
            format_version=value.get(
                "format_version"
            ),
            artifact_name=value.get(
                "artifact_name"
            ),
            expected_sample_count=(
                input_data.get(
                    "expected_sample_count"
                )
            ),
            feature_names=feature_data.get(
                "feature_names"
            ),
            feature_mean=model_data.get(
                "feature_mean"
            ),
            feature_scale=model_data.get(
                "feature_scale"
            ),
            coefficients=model_data.get(
                "coefficients"
            ),
            intercept=model_data.get(
                "intercept"
            ),
            l2_strength=model_data.get(
                "l2_strength"
            ),
            threshold=decision_data.get(
                "threshold"
            ),
            target_tpr=decision_data.get(
                "target_tpr"
            ),
            development_auroc=(
                development_data.get(
                    "auroc"
                )
            ),
            development_average_precision=(
                development_data.get(
                    "average_precision"
                )
            ),
            development_fpr_at_target_tpr=(
                development_data.get(
                    "fpr_at_target_tpr"
                )
            ),
            development_clean_mean=(
                development_data.get(
                    "clean_score_mean"
                )
            ),
            development_clean_std=(
                development_data.get(
                    "clean_score_std"
                )
            ),
            development_shifted_mean=(
                development_data.get(
                    "shifted_score_mean"
                )
            ),
            development_shifted_std=(
                development_data.get(
                    "shifted_score_std"
                )
            ),
            autocorrelation_lags=(
                feature_data.get(
                    "autocorrelation_lags"
                )
            ),
            occupancy_fraction=(
                feature_data.get(
                    "occupancy_fraction"
                )
            ),
            epsilon=feature_data.get(
                "epsilon"
            ),
            provenance=dict(
                value.get(
                    "provenance",
                    {},
                )
            ),
        )


def load_iq_shift_detector(
    path: str | Path,
) -> IQShiftDetectorArtifact:
    """Load a versioned IQ shift detector."""
    artifact_path = Path(path)

    if not artifact_path.is_file():
        raise FileNotFoundError(
            "IQ shift detector artifact does "
            f"not exist: {artifact_path}"
        )

    payload = json.loads(
        artifact_path.read_text(
            encoding="utf-8"
        )
    )

    return IQShiftDetectorArtifact.from_dict(
        payload
    )


def save_iq_shift_detector(
    artifact: IQShiftDetectorArtifact,
    path: str | Path,
) -> Path:
    """Write a versioned IQ shift detector."""
    if not isinstance(
        artifact,
        IQShiftDetectorArtifact,
    ):
        raise TypeError(
            "artifact must be an "
            "IQShiftDetectorArtifact."
        )

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            artifact.to_dict(),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if output_path.stat().st_size <= 0:
        raise RuntimeError(
            "IQ shift detector artifact is empty."
        )

    return output_path


__all__ = [
    "IQShiftAssessment",
    "IQShiftDetectorArtifact",
    "ShiftThresholdSelection",
    "load_iq_shift_detector",
    "save_iq_shift_detector",
    "select_shift_threshold",
]
