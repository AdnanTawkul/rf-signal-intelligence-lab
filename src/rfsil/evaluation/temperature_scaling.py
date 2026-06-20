from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from rfsil.evaluation.calibration import (
    probabilities_from_logits,
)

Float64Array = NDArray[np.float64]
Int64Array = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TemperatureScalingResult:
    """Result of validation-fitted scalar calibration."""

    temperature: float
    baseline_nll: float
    calibrated_nll: float
    nll_improvement: float
    example_count: int
    class_count: int
    lower_bound: float
    upper_bound: float
    optimization_iterations: int
    function_evaluations: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to JSON-compatible data."""
        return asdict(self)


def _validate_logits(
    logits: object,
) -> Float64Array:
    raw = np.asarray(logits)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or np.iscomplexobj(raw)
        or not np.issubdtype(
            raw.dtype,
            np.number,
        )
    ):
        raise ValueError(
            "logits must contain real numeric values."
        )

    validated = np.asarray(
        raw,
        dtype=np.float64,
    )

    if validated.ndim != 2:
        raise ValueError(
            "logits must have shape "
            "[examples, classes]."
        )

    if validated.shape[0] <= 0:
        raise ValueError(
            "logits must not be empty."
        )

    if validated.shape[1] < 2:
        raise ValueError(
            "logits must contain at least "
            "two classes."
        )

    if not np.all(np.isfinite(validated)):
        raise ValueError(
            "logits must contain only "
            "finite values."
        )

    return np.ascontiguousarray(validated)


def _validate_labels(
    labels: object,
    *,
    example_count: int,
    class_count: int,
) -> Int64Array:
    raw = np.asarray(labels)

    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(
            raw.dtype,
            np.integer,
        )
    ):
        raise ValueError(
            "labels must contain integers."
        )

    validated = np.asarray(
        raw,
        dtype=np.int64,
    )

    if validated.ndim != 1:
        raise ValueError(
            "labels must be one-dimensional."
        )

    if validated.shape[0] != example_count:
        raise ValueError(
            "labels and logits must contain "
            "the same number of examples."
        )

    if np.any(validated < 0) or np.any(
        validated >= class_count
    ):
        raise ValueError(
            "labels contain an out-of-range "
            "class index."
        )

    return np.ascontiguousarray(validated)


def _validate_positive_float(
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(
        value,
        (bool, np.bool_),
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must be positive and finite."
        ) from error

    if (
        not math.isfinite(validated)
        or validated <= 0.0
    ):
        raise ValueError(
            f"{name} must be positive and finite."
        )

    return validated


def _validate_max_iterations(
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            "max_iterations must be a "
            "positive integer."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            "max_iterations must be a "
            "positive integer."
        )

    return validated


def _validate_bounds(
    bounds: object,
) -> tuple[float, float]:
    if (
        isinstance(bounds, (str, bytes))
        or not isinstance(
            bounds,
            (tuple, list),
        )
        or len(bounds) != 2
    ):
        raise ValueError(
            "temperature_bounds must contain "
            "two values."
        )

    lower = _validate_positive_float(
        bounds[0],
        name="temperature lower bound",
    )
    upper = _validate_positive_float(
        bounds[1],
        name="temperature upper bound",
    )

    if lower >= upper:
        raise ValueError(
            "Temperature lower bound must be "
            "less than the upper bound."
        )

    if not lower <= 1.0 <= upper:
        raise ValueError(
            "Temperature bounds must include 1.0."
        )

    return lower, upper


def apply_temperature(
    logits: object,
    temperature: object,
) -> Float64Array:
    """Divide logits by a positive scalar temperature."""
    validated_logits = _validate_logits(
        logits
    )
    validated_temperature = (
        _validate_positive_float(
            temperature,
            name="temperature",
        )
    )

    return np.ascontiguousarray(
        validated_logits
        / validated_temperature
    )


def probabilities_with_temperature(
    logits: object,
    temperature: object,
) -> Float64Array:
    """Convert temperature-scaled logits to probabilities."""
    return probabilities_from_logits(
        apply_temperature(
            logits,
            temperature,
        )
    )


def _negative_log_likelihood_validated(
    logits: Float64Array,
    labels: Int64Array,
    temperature: float,
) -> float:
    scaled = logits / temperature
    row_maximum = np.max(
        scaled,
        axis=1,
        keepdims=True,
    )
    shifted = scaled - row_maximum
    log_partition = (
        row_maximum[:, 0]
        + np.log(
            np.sum(
                np.exp(shifted),
                axis=1,
            )
        )
    )
    true_logits = scaled[
        np.arange(
            logits.shape[0]
        ),
        labels,
    ]

    return float(
        np.mean(
            log_partition
            - true_logits
        )
    )


def negative_log_likelihood_from_logits(
    labels: object,
    logits: object,
    *,
    temperature: object = 1.0,
) -> float:
    """Compute stable multiclass NLL from logits."""
    validated_logits = _validate_logits(
        logits
    )
    validated_labels = _validate_labels(
        labels,
        example_count=int(
            validated_logits.shape[0]
        ),
        class_count=int(
            validated_logits.shape[1]
        ),
    )
    validated_temperature = (
        _validate_positive_float(
            temperature,
            name="temperature",
        )
    )

    return _negative_log_likelihood_validated(
        validated_logits,
        validated_labels,
        validated_temperature,
    )


def fit_temperature(
    labels: object,
    logits: object,
    *,
    temperature_bounds: (
        tuple[float, float]
    ) = (0.05, 20.0),
    optimization_tolerance: float = 1e-6,
    max_iterations: int = 200,
) -> TemperatureScalingResult:
    """Fit one scalar temperature by minimizing NLL."""
    validated_logits = _validate_logits(
        logits
    )
    validated_labels = _validate_labels(
        labels,
        example_count=int(
            validated_logits.shape[0]
        ),
        class_count=int(
            validated_logits.shape[1]
        ),
    )
    lower, upper = _validate_bounds(
        temperature_bounds
    )
    tolerance = _validate_positive_float(
        optimization_tolerance,
        name="optimization_tolerance",
    )
    validated_max_iterations = (
        _validate_max_iterations(
            max_iterations
        )
    )

    baseline_nll = (
        _negative_log_likelihood_validated(
            validated_logits,
            validated_labels,
            1.0,
        )
    )

    log_lower = math.log(lower)
    log_upper = math.log(upper)

    def objective(
        log_temperature: float,
    ) -> float:
        return (
            _negative_log_likelihood_validated(
                validated_logits,
                validated_labels,
                math.exp(log_temperature),
            )
        )

    optimization = minimize_scalar(
        objective,
        bounds=(
            log_lower,
            log_upper,
        ),
        method="bounded",
        options={
            "xatol": tolerance,
            "maxiter": (
                validated_max_iterations
            ),
        },
    )

    if (
        not optimization.success
        or not math.isfinite(
            float(optimization.x)
        )
        or not math.isfinite(
            float(optimization.fun)
        )
    ):
        raise RuntimeError(
            "Temperature optimization failed: "
            f"{optimization.message}"
        )

    fitted_temperature = math.exp(
        float(optimization.x)
    )
    fitted_nll = float(
        optimization.fun
    )

    if baseline_nll <= fitted_nll:
        selected_temperature = 1.0
        calibrated_nll = baseline_nll
    else:
        selected_temperature = (
            fitted_temperature
        )
        calibrated_nll = fitted_nll

    return TemperatureScalingResult(
        temperature=float(
            selected_temperature
        ),
        baseline_nll=float(
            baseline_nll
        ),
        calibrated_nll=float(
            calibrated_nll
        ),
        nll_improvement=float(
            baseline_nll
            - calibrated_nll
        ),
        example_count=int(
            validated_logits.shape[0]
        ),
        class_count=int(
            validated_logits.shape[1]
        ),
        lower_bound=lower,
        upper_bound=upper,
        optimization_iterations=int(
            getattr(
                optimization,
                "nit",
                0,
            )
        ),
        function_evaluations=int(
            optimization.nfev
        ),
        converged=bool(
            optimization.success
        ),
    )


__all__ = [
    "TemperatureScalingResult",
    "apply_temperature",
    "fit_temperature",
    "negative_log_likelihood_from_logits",
    "probabilities_with_temperature",
]
