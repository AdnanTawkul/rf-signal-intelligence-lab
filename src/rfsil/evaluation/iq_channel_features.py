from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

Float64Array = NDArray[np.float64]

DEFAULT_AUTOCORRELATION_LAGS = (
    1,
    2,
    4,
    8,
)


@dataclass(frozen=True, slots=True)
class IQChannelFeatureMatrix:
    """Per-window channel features extracted from IQ data."""

    feature_names: tuple[str, ...]
    values: Float64Array

    @property
    def example_count(self) -> int:
        """Return the number of IQ examples."""
        return int(self.values.shape[0])

    @property
    def feature_count(self) -> int:
        """Return the number of extracted features."""
        return int(self.values.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible feature metadata."""
        return {
            "feature_names": list(
                self.feature_names
            ),
            "example_count": (
                self.example_count
            ),
            "feature_count": (
                self.feature_count
            ),
        }


def _validate_iq(
    value: object,
) -> Float64Array:
    raw = np.asarray(value)

    if (
        np.issubdtype(
            raw.dtype,
            np.bool_,
        )
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "iq must contain real numeric "
            "I and Q channels."
        )

    iq = np.asarray(
        raw,
        dtype=np.float64,
    )

    if iq.ndim == 2:
        iq = iq[np.newaxis, ...]

    if iq.ndim != 3:
        raise ValueError(
            "iq must have shape "
            "[examples, 2, samples] or "
            "[2, samples]."
        )

    if iq.shape[0] <= 0:
        raise ValueError(
            "iq must contain at least "
            "one example."
        )

    if iq.shape[1] != 2:
        raise ValueError(
            "iq must contain exactly two "
            "channels."
        )

    if iq.shape[2] <= 1:
        raise ValueError(
            "iq must contain at least "
            "two samples."
        )

    if not np.all(np.isfinite(iq)):
        raise ValueError(
            "iq must contain only finite "
            "values."
        )

    return np.ascontiguousarray(iq)


def _validate_positive_number(
    value: object,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be positive "
            "and finite."
        )

    validated = float(value)

    if (
        not math.isfinite(validated)
        or validated <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive "
            "and finite."
        )

    return validated


def _validate_fraction(
    value: object,
    *,
    name: str,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(
            f"{name} must be finite and "
            "within (0, 1)."
        )

    validated = float(value)

    if (
        not math.isfinite(validated)
        or validated <= 0.0
        or validated >= 1.0
    ):
        raise ValueError(
            f"{name} must be finite and "
            "within (0, 1)."
        )

    return validated


def _validate_lags(
    value: object,
    *,
    sample_count: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(
            "autocorrelation_lags must be "
            "a non-empty sequence."
        )

    try:
        raw_lags = tuple(value)
    except TypeError as error:
        raise ValueError(
            "autocorrelation_lags must be "
            "a non-empty sequence."
        ) from error

    if not raw_lags:
        raise ValueError(
            "autocorrelation_lags must be "
            "a non-empty sequence."
        )

    lags: list[int] = []

    for raw_lag in raw_lags:
        if (
            isinstance(
                raw_lag,
                (bool, np.bool_),
            )
            or not isinstance(
                raw_lag,
                Integral,
            )
        ):
            raise ValueError(
                "Every autocorrelation lag "
                "must be a positive integer."
            )

        lag = int(raw_lag)

        if lag <= 0:
            raise ValueError(
                "Every autocorrelation lag "
                "must be a positive integer."
            )

        if lag >= sample_count:
            raise ValueError(
                "Every autocorrelation lag "
                "must be smaller than the "
                "IQ sample count."
            )

        lags.append(lag)

    if len(set(lags)) != len(lags):
        raise ValueError(
            "Autocorrelation lags must "
            "be unique."
        )

    return tuple(lags)


def _normalized_complex_iq(
    iq: Float64Array,
    *,
    epsilon: float,
) -> NDArray[np.complex128]:
    complex_iq = (
        iq[:, 0].astype(
            np.complex128
        )
        + 1j
        * iq[:, 1].astype(
            np.complex128
        )
    )

    centered = (
        complex_iq
        - np.mean(
            complex_iq,
            axis=1,
            keepdims=True,
        )
    )

    average_power = np.mean(
        np.abs(centered) ** 2,
        axis=1,
        keepdims=True,
    )

    if np.any(
        average_power <= epsilon
    ):
        raise ValueError(
            "Every IQ example must have "
            "positive centered power."
        )

    return centered / np.sqrt(
        average_power
    )


def _amplitude_features(
    complex_iq: NDArray[np.complex128],
    *,
    epsilon: float,
) -> tuple[
    tuple[str, ...],
    list[Float64Array],
]:
    amplitude = np.abs(complex_iq)
    mean = np.mean(
        amplitude,
        axis=1,
    )
    centered = (
        amplitude
        - mean[:, np.newaxis]
    )
    variance = np.mean(
        centered**2,
        axis=1,
    )
    standard_deviation = np.sqrt(
        variance
    )

    rms = np.sqrt(
        np.mean(
            amplitude**2,
            axis=1,
        )
    )

    values = [
        mean,
        standard_deviation
        / np.maximum(mean, epsilon),
        np.mean(
            centered**3,
            axis=1,
        )
        / np.maximum(
            standard_deviation**3,
            epsilon,
        ),
        np.mean(
            centered**4,
            axis=1,
        )
        / np.maximum(
            variance**2,
            epsilon,
        )
        - 3.0,
        np.max(
            amplitude,
            axis=1,
        )
        / np.maximum(rms, epsilon),
    ]

    names = (
        "amplitude_mean",
        "amplitude_coefficient_of_variation",
        "amplitude_skewness",
        "amplitude_excess_kurtosis",
        "peak_to_rms_ratio",
    )

    return names, values


def _iq_geometry_features(
    complex_iq: NDArray[np.complex128],
    *,
    epsilon: float,
) -> tuple[
    tuple[str, ...],
    list[Float64Array],
]:
    in_phase = complex_iq.real
    quadrature = complex_iq.imag

    variance_i = np.mean(
        in_phase**2,
        axis=1,
    )
    variance_q = np.mean(
        quadrature**2,
        axis=1,
    )
    covariance = np.mean(
        in_phase * quadrature,
        axis=1,
    )

    correlation = covariance / np.sqrt(
        np.maximum(
            variance_i * variance_q,
            epsilon,
        )
    )

    trace = variance_i + variance_q
    discriminant = np.sqrt(
        np.maximum(
            (
                variance_i
                - variance_q
            )
            ** 2
            + 4.0 * covariance**2,
            0.0,
        )
    )
    maximum_eigenvalue = 0.5 * (
        trace + discriminant
    )
    minimum_eigenvalue = 0.5 * (
        trace - discriminant
    )

    values = [
        np.abs(
            np.log(
                (
                    variance_i
                    + epsilon
                )
                / (
                    variance_q
                    + epsilon
                )
            )
        ),
        np.abs(correlation),
        np.log(
            (
                maximum_eigenvalue
                + epsilon
            )
            / (
                minimum_eigenvalue
                + epsilon
            )
        ),
    ]

    names = (
        "iq_variance_log_ratio_abs",
        "iq_correlation_abs",
        "iq_covariance_log_condition",
    )

    return names, values


def _phase_features(
    complex_iq: NDArray[np.complex128],
    *,
    epsilon: float,
) -> tuple[
    tuple[str, ...],
    list[Float64Array],
]:
    differential_phase = np.angle(
        complex_iq[:, 1:]
        * np.conjugate(
            complex_iq[:, :-1]
        )
    )

    circular_mean = np.mean(
        np.exp(
            1j * differential_phase
        ),
        axis=1,
    )
    concentration = np.clip(
        np.abs(circular_mean),
        epsilon,
        1.0,
    )

    values = [
        np.abs(
            np.angle(circular_mean)
        )
        / np.pi,
        1.0 - concentration,
        np.sqrt(
            np.maximum(
                -2.0
                * np.log(
                    concentration
                ),
                0.0,
            )
        ),
    ]

    names = (
        "dphase_mean_abs_normalized",
        "dphase_circular_dispersion",
        "dphase_circular_std",
    )

    return names, values


def _autocorrelation_features(
    complex_iq: NDArray[np.complex128],
    *,
    lags: tuple[int, ...],
    epsilon: float,
) -> tuple[
    tuple[str, ...],
    list[Float64Array],
]:
    power = np.mean(
        np.abs(complex_iq) ** 2,
        axis=1,
    )

    names = []
    values = []

    for lag in lags:
        correlation = np.mean(
            complex_iq[:, lag:]
            * np.conjugate(
                complex_iq[:, :-lag]
            ),
            axis=1,
        )

        names.append(
            f"autocorrelation_abs_lag_{lag}"
        )
        values.append(
            np.abs(correlation)
            / np.maximum(
                power,
                epsilon,
            )
        )

    return tuple(names), values


def _spectral_features(
    complex_iq: NDArray[np.complex128],
    *,
    occupancy_fraction: float,
    epsilon: float,
) -> tuple[
    tuple[str, ...],
    list[Float64Array],
]:
    sample_count = complex_iq.shape[1]

    window = np.hanning(
        sample_count
    ).astype(np.float64)

    spectrum = np.fft.fftshift(
        np.fft.fft(
            complex_iq
            * window[np.newaxis, :],
            axis=1,
        ),
        axes=1,
    )
    power = np.abs(spectrum) ** 2

    total_power = np.sum(
        power,
        axis=1,
        keepdims=True,
    )
    probabilities = (
        power
        / np.maximum(
            total_power,
            epsilon,
        )
    )

    entropy = -np.sum(
        probabilities
        * np.log(
            np.maximum(
                probabilities,
                epsilon,
            )
        ),
        axis=1,
    ) / math.log(sample_count)

    flatness = np.exp(
        np.mean(
            np.log(
                np.maximum(
                    power,
                    epsilon,
                )
            ),
            axis=1,
        )
    ) / np.maximum(
        np.mean(
            power,
            axis=1,
        ),
        epsilon,
    )

    peak_fraction = np.max(
        probabilities,
        axis=1,
    )

    descending = np.sort(
        probabilities,
        axis=1,
    )[:, ::-1]
    cumulative = np.cumsum(
        descending,
        axis=1,
    )
    occupancy_count = (
        np.argmax(
            cumulative
            >= occupancy_fraction,
            axis=1,
        )
        + 1
    )
    occupancy = (
        occupancy_count
        / sample_count
    )

    frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            sample_count,
            d=1.0,
        )
    )
    centroid = np.sum(
        probabilities
        * frequencies[np.newaxis, :],
        axis=1,
    )
    spread = np.sqrt(
        np.maximum(
            np.sum(
                probabilities
                * (
                    frequencies[
                        np.newaxis,
                        :
                    ]
                    - centroid[
                        :,
                        np.newaxis,
                    ]
                )
                ** 2,
                axis=1,
            ),
            0.0,
        )
    )

    values = [
        entropy,
        flatness,
        peak_fraction,
        occupancy,
        np.abs(centroid),
        spread,
    ]

    names = (
        "spectral_entropy",
        "spectral_flatness",
        "spectral_peak_fraction",
        "spectral_occupancy_fraction",
        "spectral_centroid_abs",
        "spectral_spread",
    )

    return names, values


def compute_iq_channel_features(
    iq: object,
    *,
    autocorrelation_lags: object = (
        DEFAULT_AUTOCORRELATION_LAGS
    ),
    occupancy_fraction: object = 0.90,
    epsilon: object = 1e-12,
) -> IQChannelFeatureMatrix:
    """Extract deterministic gain-invariant IQ features."""
    validated_iq = _validate_iq(iq)
    validated_epsilon = (
        _validate_positive_number(
            epsilon,
            name="epsilon",
        )
    )
    validated_occupancy = (
        _validate_fraction(
            occupancy_fraction,
            name="occupancy_fraction",
        )
    )
    validated_lags = _validate_lags(
        autocorrelation_lags,
        sample_count=int(
            validated_iq.shape[2]
        ),
    )

    complex_iq = _normalized_complex_iq(
        validated_iq,
        epsilon=validated_epsilon,
    )

    feature_names: list[str] = []
    feature_values: list[
        Float64Array
    ] = []

    builders = (
        _amplitude_features(
            complex_iq,
            epsilon=validated_epsilon,
        ),
        _iq_geometry_features(
            complex_iq,
            epsilon=validated_epsilon,
        ),
        _phase_features(
            complex_iq,
            epsilon=validated_epsilon,
        ),
        _autocorrelation_features(
            complex_iq,
            lags=validated_lags,
            epsilon=validated_epsilon,
        ),
        _spectral_features(
            complex_iq,
            occupancy_fraction=(
                validated_occupancy
            ),
            epsilon=validated_epsilon,
        ),
    )

    for names, values in builders:
        feature_names.extend(names)
        feature_values.extend(values)

    matrix = np.column_stack(
        feature_values
    ).astype(
        np.float64,
        copy=False,
    )

    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            "IQ feature extraction produced "
            "non-finite values."
        )

    return IQChannelFeatureMatrix(
        feature_names=tuple(
            feature_names
        ),
        values=np.ascontiguousarray(
            matrix
        ),
    )


__all__ = [
    "DEFAULT_AUTOCORRELATION_LAGS",
    "IQChannelFeatureMatrix",
    "compute_iq_channel_features",
]
