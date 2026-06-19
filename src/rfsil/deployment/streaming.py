from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.deployment.inference import (
    IQInferenceEngine,
    WindowPrediction,
)
from rfsil.deployment.windowing import (
    predict_window_batches,
)

Float32Array = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StreamWindow:
    """One complete window emitted by a stream buffer."""

    sequence_index: int
    start_sample: int
    stop_sample_exclusive: int
    iq: Float32Array


@dataclass(frozen=True, slots=True)
class StreamingPrediction:
    """Timestamped prediction for one streaming window."""

    sequence_index: int
    start_sample: int
    stop_sample_exclusive: int
    start_time_seconds: float | None
    end_time_seconds: float | None
    center_time_seconds: float | None
    predicted_index: int
    predicted_label: str
    confidence: float
    logits: tuple[float, ...]
    probabilities: tuple[float, ...]


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


def _validate_sample_rate(
    value: object,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "sample_rate_hz must be a positive "
            "finite number or None."
        )

    try:
        validated = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "sample_rate_hz must be a positive "
            "finite number or None."
        ) from error

    if (
        not math.isfinite(validated)
        or validated <= 0.0
    ):
        raise ValueError(
            "sample_rate_hz must be a positive "
            "finite number or None."
        )

    return validated


def _normalize_stream_chunk(
    value: Any,
) -> Float32Array:
    """Normalize one stream chunk to [2, samples]."""
    array = np.asarray(value)

    if array.size == 0:
        raise ValueError(
            "IQ stream chunk must not be empty."
        )

    if not np.issubdtype(
        array.dtype,
        np.number,
    ):
        raise ValueError(
            "IQ stream chunk must contain "
            "numeric values."
        )

    if np.iscomplexobj(array):
        if array.ndim != 1:
            raise ValueError(
                "Complex stream chunks must have "
                "shape [samples]."
            )

        normalized = np.stack(
            (
                np.real(array),
                np.imag(array),
            ),
            axis=0,
        )
    elif (
        array.ndim == 2
        and array.shape[0] == 2
    ):
        normalized = array
    elif (
        array.ndim == 3
        and array.shape[0] == 1
        and array.shape[1] == 2
    ):
        normalized = array[0]
    else:
        raise ValueError(
            "Real stream chunks must have shape "
            "[2, samples] or [1, 2, samples]."
        )

    normalized = np.asarray(
        normalized,
        dtype=np.float32,
    )

    if normalized.shape[1] <= 0:
        raise ValueError(
            "IQ stream chunk must contain samples."
        )

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "IQ stream chunk must contain only "
            "finite values."
        )

    return np.ascontiguousarray(normalized)


class IQStreamBuffer:
    """Fixed-memory overlapping IQ window buffer."""

    def __init__(
        self,
        *,
        window_size: int,
        hop_size: int | None = None,
    ) -> None:
        """Create an empty stream buffer."""
        self.window_size = (
            _validate_positive_integer(
                window_size,
                name="window_size",
            )
        )
        self.hop_size = (
            self.window_size
            if hop_size is None
            else _validate_positive_integer(
                hop_size,
                name="hop_size",
            )
        )

        if self.hop_size > self.window_size:
            raise ValueError(
                "hop_size must not exceed "
                "window_size."
            )

        self._buffer = np.empty(
            (2, self.window_size),
            dtype=np.float32,
        )

        self.reset()

    @property
    def total_samples_received(self) -> int:
        """Return the number of received samples."""
        return self._total_samples_received

    @property
    def buffered_sample_count(self) -> int:
        """Return valid samples in the ring buffer."""
        return self._buffered_sample_count

    @property
    def emitted_window_count(self) -> int:
        """Return the number of emitted windows."""
        return self._sequence_index

    @property
    def next_window_start(self) -> int:
        """Return the next absolute window start."""
        return self._next_window_start

    def reset(self) -> None:
        """Reset all stream state."""
        self._buffered_sample_count = 0
        self._total_samples_received = 0
        self._next_window_start = 0
        self._sequence_index = 0

    def push(
        self,
        chunk: Any,
    ) -> tuple[StreamWindow, ...]:
        """Append a chunk and emit complete windows."""
        normalized = _normalize_stream_chunk(
            chunk
        )
        emitted: list[StreamWindow] = []
        source_offset = 0
        source_sample_count = int(
            normalized.shape[1]
        )

        while source_offset < source_sample_count:
            required = (
                self.window_size
                - self._buffered_sample_count
            )
            available = (
                source_sample_count
                - source_offset
            )
            copy_count = min(
                required,
                available,
            )

            destination_start = (
                self._buffered_sample_count
            )
            destination_stop = (
                destination_start + copy_count
            )
            source_stop = (
                source_offset + copy_count
            )

            self._buffer[
                :,
                destination_start:destination_stop,
            ] = normalized[
                :,
                source_offset:source_stop,
            ]

            self._buffered_sample_count += (
                copy_count
            )
            self._total_samples_received += (
                copy_count
            )
            source_offset = source_stop

            if (
                self._buffered_sample_count
                != self.window_size
            ):
                continue

            start_sample = (
                self._next_window_start
            )
            stop_sample = (
                start_sample + self.window_size
            )

            emitted.append(
                StreamWindow(
                    sequence_index=(
                        self._sequence_index
                    ),
                    start_sample=start_sample,
                    stop_sample_exclusive=(
                        stop_sample
                    ),
                    iq=np.ascontiguousarray(
                        self._buffer.copy()
                    ),
                )
            )

            self._sequence_index += 1
            self._next_window_start += (
                self.hop_size
            )

            retained_sample_count = (
                self.window_size
                - self.hop_size
            )

            if retained_sample_count > 0:
                self._buffer[
                    :,
                    :retained_sample_count,
                ] = self._buffer[
                    :,
                    self.hop_size:,
                ]

            self._buffered_sample_count = (
                retained_sample_count
            )

        return tuple(emitted)


class StreamingIQClassifier:
    """Online IQ classifier backed by a stream buffer."""

    def __init__(
        self,
        *,
        engine: IQInferenceEngine,
        window_size: int | None = None,
        hop_size: int | None = None,
        sample_rate_hz: float | None = None,
        inference_batch_size: int = 64,
    ) -> None:
        """Create a streaming classifier."""
        resolved_window_size = (
            engine.expected_sample_count
            if window_size is None
            else window_size
        )

        if resolved_window_size is None:
            raise ValueError(
                "window_size is required when "
                "the inference engine has no "
                "expected sample count."
            )

        validated_window_size = (
            _validate_positive_integer(
                resolved_window_size,
                name="window_size",
            )
        )

        if (
            engine.expected_sample_count
            is not None
            and engine.expected_sample_count
            != validated_window_size
        ):
            raise ValueError(
                "Streaming window_size must match "
                "the inference engine expected "
                "sample count."
            )

        self.engine = engine
        self.sample_rate_hz = (
            _validate_sample_rate(
                sample_rate_hz
            )
        )
        self.inference_batch_size = (
            _validate_positive_integer(
                inference_batch_size,
                name="inference_batch_size",
            )
        )
        self.buffer = IQStreamBuffer(
            window_size=validated_window_size,
            hop_size=hop_size,
        )

    @property
    def window_size(self) -> int:
        """Return the streaming window size."""
        return self.buffer.window_size

    @property
    def hop_size(self) -> int:
        """Return the streaming hop size."""
        return self.buffer.hop_size

    def reset(self) -> None:
        """Reset the stream state."""
        self.buffer.reset()

    def _sample_time(
        self,
        sample_index: float,
    ) -> float | None:
        if self.sample_rate_hz is None:
            return None

        return (
            sample_index
            / self.sample_rate_hz
        )

    def push(
        self,
        chunk: Any,
    ) -> tuple[
        StreamingPrediction,
        ...,
    ]:
        """Append IQ samples and return new predictions."""
        windows = self.buffer.push(chunk)

        if not windows:
            return ()

        stacked = np.stack(
            [
                window.iq
                for window in windows
            ],
            axis=0,
        )

        batch_prediction = (
            predict_window_batches(
                self.engine,
                stacked,
                batch_size=(
                    self.inference_batch_size
                ),
            )
        )

        events: list[
            StreamingPrediction
        ] = []

        for index, window in enumerate(
            windows
        ):
            prediction: WindowPrediction = (
                batch_prediction.item(index)
            )
            center_sample = (
                window.start_sample
                + window.stop_sample_exclusive
            ) / 2.0

            events.append(
                StreamingPrediction(
                    sequence_index=(
                        window.sequence_index
                    ),
                    start_sample=(
                        window.start_sample
                    ),
                    stop_sample_exclusive=(
                        window
                        .stop_sample_exclusive
                    ),
                    start_time_seconds=(
                        self._sample_time(
                            window.start_sample
                        )
                    ),
                    end_time_seconds=(
                        self._sample_time(
                            window
                            .stop_sample_exclusive
                        )
                    ),
                    center_time_seconds=(
                        self._sample_time(
                            center_sample
                        )
                    ),
                    predicted_index=(
                        prediction
                        .predicted_index
                    ),
                    predicted_label=(
                        prediction
                        .predicted_label
                    ),
                    confidence=(
                        prediction.confidence
                    ),
                    logits=prediction.logits,
                    probabilities=(
                        prediction.probabilities
                    ),
                )
            )

        return tuple(events)


__all__ = [
    "IQStreamBuffer",
    "StreamWindow",
    "StreamingIQClassifier",
    "StreamingPrediction",
]
