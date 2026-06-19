from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from rfsil.deployment.inference import (
    BatchPrediction,
    IQInferenceEngine,
)

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]

RemainderPolicy = Literal[
    "drop",
    "pad",
    "error",
]


@dataclass(frozen=True, slots=True)
class WindowedIQ:
    """Sliding windows extracted from one IQ signal."""

    windows: Float32Array
    start_indices: Int64Array
    valid_sample_counts: Int64Array
    original_sample_count: int
    window_size: int
    stride: int
    remainder_policy: str
    pad_value: float

    @property
    def window_count(self) -> int:
        """Return the number of generated windows."""
        return int(self.windows.shape[0])


@dataclass(frozen=True, slots=True)
class SignalPrediction:
    """Signal-level aggregate of window predictions."""

    class_names: tuple[str, ...]
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: tuple[float, ...]
    window_count: int


def _validate_positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    validated = int(value)

    if validated <= 0:
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return validated


def _validate_pad_value(
    value: object,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "pad_value must be finite."
        )

    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "pad_value must be finite."
        ) from error

    if not math.isfinite(validated):
        raise ValueError(
            "pad_value must be finite."
        )

    return validated


def _validate_remainder_policy(
    value: object,
) -> RemainderPolicy:
    if not isinstance(value, str):
        raise ValueError(
            "remainder_policy must be one of: "
            "drop, pad, error."
        )

    normalized = value.strip().lower()

    if normalized not in {
        "drop",
        "pad",
        "error",
    }:
        raise ValueError(
            "remainder_policy must be one of: "
            "drop, pad, error."
        )

    return normalized  # type: ignore[return-value]


def _normalize_single_signal(
    value: object,
) -> Float32Array:
    """Normalize one signal to [2, samples]."""
    array = np.asarray(value)

    if array.size == 0:
        raise ValueError(
            "IQ signal must not be empty."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise ValueError(
            "IQ signal must contain numeric values."
        )

    if np.iscomplexobj(array):
        if array.ndim != 1:
            raise ValueError(
                "Complex long-signal input must "
                "have shape [samples]."
            )

        normalized = np.stack(
            (
                np.real(array),
                np.imag(array),
            ),
            axis=0,
        )
    else:
        if (
            array.ndim == 3
            and array.shape[0] == 1
            and array.shape[1] == 2
        ):
            normalized = array[0]
        elif (
            array.ndim == 2
            and array.shape[0] == 2
        ):
            normalized = array
        else:
            raise ValueError(
                "Real long-signal input must have "
                "shape [2, samples] or "
                "[1, 2, samples]."
            )

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if normalized.shape[1] <= 0:
        raise ValueError(
            "IQ signal must contain samples."
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "IQ signal must contain only "
            "finite values."
        )

    return np.ascontiguousarray(normalized)


def window_iq_signal(
    signal: object,
    *,
    window_size: int,
    stride: int | None = None,
    remainder_policy: str = "drop",
    pad_value: float = 0.0,
) -> WindowedIQ:
    """Extract overlapping windows from one IQ signal."""
    normalized = _normalize_single_signal(
        signal
    )
    validated_window_size = (
        _validate_positive_integer(
            window_size,
            name="window_size",
        )
    )
    validated_stride = (
        validated_window_size
        if stride is None
        else _validate_positive_integer(
            stride,
            name="stride",
        )
    )

    if validated_stride > validated_window_size:
        raise ValueError(
            "stride must not exceed window_size."
        )

    validated_policy = (
        _validate_remainder_policy(
            remainder_policy
        )
    )
    validated_pad_value = (
        _validate_pad_value(pad_value)
    )

    sample_count = int(
        normalized.shape[1]
    )

    if sample_count >= validated_window_size:
        full_starts = list(
            range(
                0,
                sample_count
                - validated_window_size
                + 1,
                validated_stride,
            )
        )
    else:
        full_starts = []

    covered_end = (
        full_starts[-1]
        + validated_window_size
        if full_starts
        else 0
    )
    uncovered_sample_count = (
        sample_count - covered_end
    )

    starts = list(full_starts)

    if validated_policy == "error":
        if uncovered_sample_count > 0:
            raise ValueError(
                "The IQ signal length does not "
                "fit the requested windowing "
                "configuration exactly."
            )

    elif validated_policy == "pad":
        if uncovered_sample_count > 0:
            next_start = (
                0
                if not full_starts
                else (
                    full_starts[-1]
                    + validated_stride
                )
            )
            starts.append(next_start)

    elif (
        validated_policy == "drop"
        and not starts
    ):
        raise ValueError(
            "The IQ signal is shorter than one "
            "complete window. Use "
            "remainder_policy='pad' to keep it."
        )

    if not starts:
        raise ValueError(
            "No IQ windows were generated."
        )

    windows = np.full(
        (
            len(starts),
            2,
            validated_window_size,
        ),
        validated_pad_value,
        dtype=np.float32,
    )
    valid_sample_counts = np.empty(
        len(starts),
        dtype=np.int64,
    )

    for index, start in enumerate(starts):
        stop = min(
            start + validated_window_size,
            sample_count,
        )
        valid_count = stop - start

        if valid_count <= 0:
            raise RuntimeError(
                "Generated an empty IQ window."
            )

        windows[
            index,
            :,
            :valid_count,
        ] = normalized[:, start:stop]

        valid_sample_counts[index] = (
            valid_count
        )

    return WindowedIQ(
        windows=np.ascontiguousarray(
            windows
        ),
        start_indices=np.asarray(
            starts,
            dtype=np.int64,
        ),
        valid_sample_counts=(
            valid_sample_counts
        ),
        original_sample_count=sample_count,
        window_size=validated_window_size,
        stride=validated_stride,
        remainder_policy=validated_policy,
        pad_value=validated_pad_value,
    )


def predict_window_batches(
    engine: IQInferenceEngine,
    windows: np.ndarray,
    *,
    batch_size: int = 64,
) -> BatchPrediction:
    """Run window inference in bounded batches."""
    validated_batch_size = (
        _validate_positive_integer(
            batch_size,
            name="batch_size",
        )
    )
    array = np.asarray(windows)

    if (
        array.ndim != 3
        or array.shape[0] <= 0
        or array.shape[1] != engine.in_channels
    ):
        raise ValueError(
            "windows must have shape "
            "[window_count, channels, samples]."
        )

    logits = []
    probabilities = []
    predicted_indices = []
    confidences = []
    predicted_labels: list[str] = []

    for start in range(
        0,
        int(array.shape[0]),
        validated_batch_size,
    ):
        stop = min(
            start + validated_batch_size,
            int(array.shape[0]),
        )
        result = engine.predict_batch(
            array[start:stop]
        )

        logits.append(result.logits)
        probabilities.append(
            result.probabilities
        )
        predicted_indices.append(
            result.predicted_indices
        )
        confidences.append(
            result.confidences
        )
        predicted_labels.extend(
            result.predicted_labels
        )

    return BatchPrediction(
        class_names=engine.class_names,
        logits=np.concatenate(
            logits,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        probabilities=np.concatenate(
            probabilities,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
        predicted_indices=np.concatenate(
            predicted_indices,
            axis=0,
        ).astype(
            np.int64,
            copy=False,
        ),
        predicted_labels=tuple(
            predicted_labels
        ),
        confidences=np.concatenate(
            confidences,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        ),
    )


def aggregate_window_predictions(
    prediction: BatchPrediction,
    *,
    weights: np.ndarray | None = None,
) -> SignalPrediction:
    """Aggregate window probabilities to one signal prediction."""
    probabilities = np.asarray(
        prediction.probabilities,
        dtype=np.float64,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[0] <= 0
        or probabilities.shape[1]
        != len(prediction.class_names)
    ):
        raise ValueError(
            "Window probabilities have an "
            "invalid shape."
        )

    if not np.all(
        np.isfinite(probabilities)
    ):
        raise ValueError(
            "Window probabilities must be finite."
        )

    if np.any(probabilities < 0.0):
        raise ValueError(
            "Window probabilities must not "
            "be negative."
        )

    row_sums = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        row_sums,
        1.0,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError(
            "Every probability row must "
            "sum to one."
        )

    if weights is None:
        validated_weights = np.ones(
            probabilities.shape[0],
            dtype=np.float64,
        )
    else:
        validated_weights = np.asarray(
            weights,
            dtype=np.float64,
        )

        if (
            validated_weights.ndim != 1
            or validated_weights.shape[0]
            != probabilities.shape[0]
        ):
            raise ValueError(
                "Aggregation weights must match "
                "the window count."
            )

        if not np.all(
            np.isfinite(validated_weights)
        ):
            raise ValueError(
                "Aggregation weights must "
                "be finite."
            )

        if np.any(validated_weights < 0.0):
            raise ValueError(
                "Aggregation weights must not "
                "be negative."
            )

        if float(
            validated_weights.sum()
        ) <= 0.0:
            raise ValueError(
                "At least one aggregation weight "
                "must be positive."
            )

    aggregated = np.average(
        probabilities,
        axis=0,
        weights=validated_weights,
    )
    aggregated = aggregated / aggregated.sum()

    predicted_index = int(
        np.argmax(aggregated)
    )

    return SignalPrediction(
        class_names=tuple(
            prediction.class_names
        ),
        predicted_index=predicted_index,
        predicted_label=(
            prediction.class_names[
                predicted_index
            ]
        ),
        confidence=float(
            aggregated[predicted_index]
        ),
        probabilities=tuple(
            float(value)
            for value in aggregated
        ),
        window_count=int(
            probabilities.shape[0]
        ),
    )


__all__ = [
    "RemainderPolicy",
    "SignalPrediction",
    "WindowedIQ",
    "aggregate_window_predictions",
    "predict_window_batches",
    "window_iq_signal",
]
