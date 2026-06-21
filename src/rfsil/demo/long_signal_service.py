from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO
from numbers import Integral
from pathlib import Path

import numpy as np

from rfsil.deployment import (
    IQInferenceEngine,
    LoadedIQ,
    aggregate_window_predictions,
    predict_window_batches,
    rank_probabilities,
    window_iq_signal,
)
from rfsil.deployment.shift_detector import (
    IQShiftDetectorArtifact,
)


@dataclass(frozen=True, slots=True)
class LongSignalWindowRecord:
    """Prediction and shift result for one window."""

    window_index: int
    source_sample_index: int
    start_sample: int | None
    stop_sample_exclusive: int | None
    valid_sample_count: int
    predicted_index: int
    predicted_label: str
    confidence: float
    probabilities: tuple[float, ...]
    shift_score: float
    shift_threshold: float
    shift_like: bool
    true_index: int | None = None
    snr_db: float | None = None

    def to_dict(
        self,
        *,
        class_names: tuple[str, ...],
        top_k: int,
    ) -> dict[str, object]:
        """Return JSON-compatible window data."""
        return {
            "window_index": self.window_index,
            "source_sample_index": (
                self.source_sample_index
            ),
            "start_sample": self.start_sample,
            "stop_sample_exclusive": (
                self.stop_sample_exclusive
            ),
            "valid_sample_count": (
                self.valid_sample_count
            ),
            "predicted_index": (
                self.predicted_index
            ),
            "predicted_label": (
                self.predicted_label
            ),
            "confidence": self.confidence,
            "probabilities": list(
                self.probabilities
            ),
            "top_k": rank_probabilities(
                self.probabilities,
                class_names=class_names,
                top_k=top_k,
            ),
            "shift_score": self.shift_score,
            "shift_threshold": (
                self.shift_threshold
            ),
            "shift_like": self.shift_like,
            "true_index": self.true_index,
            "snr_db": self.snr_db,
        }


@dataclass(frozen=True, slots=True)
class LongSignalAnalysis:
    """GUI-ready batch or long-signal result."""

    source_mode: str
    class_names: tuple[str, ...]
    original_sample_count: int
    source_item_count: int
    analyzed_window_count: int
    truncated: bool
    window_size: int
    stride: int
    remainder_policy: str
    aggregate_predicted_index: int
    aggregate_predicted_label: str
    aggregate_confidence: float
    aggregate_probabilities: tuple[
        float,
        ...,
    ]
    shift_threshold: float
    window_records: tuple[
        LongSignalWindowRecord,
        ...,
    ]

    @property
    def shift_like_count(self) -> int:
        """Return the number of flagged windows."""
        return sum(
            record.shift_like
            for record in self.window_records
        )

    @property
    def shift_like_fraction(self) -> float:
        """Return the fraction of flagged windows."""
        return (
            self.shift_like_count
            / self.analyzed_window_count
        )

    @property
    def mean_confidence(self) -> float:
        """Return mean per-window confidence."""
        return float(
            np.mean(
                [
                    record.confidence
                    for record
                    in self.window_records
                ]
            )
        )

    def to_document(
        self,
        *,
        source_name: str,
        checkpoint_reference: str,
        detector_name: str,
        top_k: int,
    ) -> dict[str, object]:
        """Create a privacy-safe JSON document."""
        safe_source_name = Path(
            source_name
        ).name

        if not safe_source_name:
            raise ValueError(
                "source_name must not be empty."
            )

        if not checkpoint_reference.strip():
            raise ValueError(
                "checkpoint_reference must not "
                "be empty."
            )

        return {
            "format_version": 1,
            "analysis_type": (
                "long_iq_modulation_and_shift"
            ),
            "input": {
                "source_name": safe_source_name,
                "source_mode": self.source_mode,
                "original_sample_count": (
                    self.original_sample_count
                ),
                "source_item_count": (
                    self.source_item_count
                ),
            },
            "model": {
                "checkpoint_reference": (
                    checkpoint_reference
                ),
                "class_names": list(
                    self.class_names
                ),
            },
            "windowing": {
                "window_size": self.window_size,
                "stride": self.stride,
                "remainder_policy": (
                    self.remainder_policy
                ),
                "analyzed_window_count": (
                    self.analyzed_window_count
                ),
                "truncated": self.truncated,
            },
            "aggregate_prediction": {
                "predicted_index": (
                    self
                    .aggregate_predicted_index
                ),
                "predicted_label": (
                    self
                    .aggregate_predicted_label
                ),
                "confidence": (
                    self.aggregate_confidence
                ),
                "probabilities": list(
                    self
                    .aggregate_probabilities
                ),
                "top_k": rank_probabilities(
                    self.aggregate_probabilities,
                    class_names=(
                        self.class_names
                    ),
                    top_k=top_k,
                ),
            },
            "shift_summary": {
                "detector_name": detector_name,
                "threshold": (
                    self.shift_threshold
                ),
                "shift_like_count": (
                    self.shift_like_count
                ),
                "shift_like_fraction": (
                    self.shift_like_fraction
                ),
                "mean_window_confidence": (
                    self.mean_confidence
                ),
            },
            "window_records": [
                record.to_dict(
                    class_names=self.class_names,
                    top_k=top_k,
                )
                for record
                in self.window_records
            ],
        }

    def to_json(
        self,
        *,
        source_name: str,
        checkpoint_reference: str,
        detector_name: str,
        top_k: int,
    ) -> str:
        """Serialize the long-analysis document."""
        return json.dumps(
            self.to_document(
                source_name=source_name,
                checkpoint_reference=(
                    checkpoint_reference
                ),
                detector_name=detector_name,
                top_k=top_k,
            ),
            indent=2,
        ) + "\n"

    def to_csv(self) -> str:
        """Serialize per-window rows as CSV."""
        output = StringIO(
            newline=""
        )

        fieldnames = [
            "window_index",
            "source_sample_index",
            "start_sample",
            "stop_sample_exclusive",
            "valid_sample_count",
            "predicted_index",
            "predicted_label",
            "confidence",
            "shift_score",
            "shift_threshold",
            "shift_like",
            "true_index",
            "snr_db",
            *[
                f"probability_{name}"
                for name in self.class_names
            ],
        ]

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for record in self.window_records:
            row: dict[str, object] = {
                "window_index": (
                    record.window_index
                ),
                "source_sample_index": (
                    record.source_sample_index
                ),
                "start_sample": (
                    record.start_sample
                ),
                "stop_sample_exclusive": (
                    record.stop_sample_exclusive
                ),
                "valid_sample_count": (
                    record.valid_sample_count
                ),
                "predicted_index": (
                    record.predicted_index
                ),
                "predicted_label": (
                    record.predicted_label
                ),
                "confidence": record.confidence,
                "shift_score": (
                    record.shift_score
                ),
                "shift_threshold": (
                    record.shift_threshold
                ),
                "shift_like": (
                    record.shift_like
                ),
                "true_index": record.true_index,
                "snr_db": record.snr_db,
            }

            for class_name, probability in zip(
                self.class_names,
                record.probabilities,
                strict=True,
            ):
                row[
                    f"probability_{class_name}"
                ] = probability

            writer.writerow(row)

        return output.getvalue()


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


def analyze_long_iq(
    *,
    loaded: LoadedIQ,
    engine: IQInferenceEngine,
    shift_detector: IQShiftDetectorArtifact,
    window_size: object,
    stride: object,
    remainder_policy: str,
    batch_size: object,
    maximum_windows: object,
    pad_value: float = 0.0,
) -> LongSignalAnalysis:
    """Analyze a long signal or pre-windowed batch."""
    if not isinstance(loaded, LoadedIQ):
        raise TypeError(
            "loaded must be a LoadedIQ instance."
        )

    if not isinstance(
        shift_detector,
        IQShiftDetectorArtifact,
    ):
        raise TypeError(
            "shift_detector must be an "
            "IQShiftDetectorArtifact."
        )

    validated_window_size = (
        _positive_integer(
            window_size,
            name="window_size",
        )
    )
    validated_stride = _positive_integer(
        stride,
        name="stride",
    )
    validated_batch_size = (
        _positive_integer(
            batch_size,
            name="batch_size",
        )
    )
    validated_maximum_windows = (
        _positive_integer(
            maximum_windows,
            name="maximum_windows",
        )
    )
    normalized_remainder = str(
        remainder_policy
    ).strip().lower()

    if normalized_remainder not in {
        "drop",
        "pad",
        "error",
    }:
        raise ValueError(
            "remainder_policy must be drop, "
            "pad, or error."
        )

    if validated_window_size != (
        shift_detector.expected_sample_count
    ):
        raise ValueError(
            "window_size must match the shift "
            "detector sample count."
        )

    source_item_count = (
        loaded.batch_size
    )

    if loaded.batch_size > 1:
        if loaded.sample_count != (
            validated_window_size
        ):
            raise ValueError(
                "A pre-windowed batch must use "
                "the configured window size."
            )

        source_mode = (
            "prewindowed_batch"
        )
        available_window_count = (
            loaded.batch_size
        )
        analyzed_window_count = min(
            available_window_count,
            validated_maximum_windows,
        )
        windows = np.ascontiguousarray(
            loaded.iq[
                :analyzed_window_count
            ]
        )
        valid_sample_counts = np.full(
            analyzed_window_count,
            validated_window_size,
            dtype=np.int64,
        )
        start_indices = None
        original_sample_count = (
            loaded.sample_count
        )
    else:
        source_mode = (
            "continuous_long_signal"
        )
        windowed = window_iq_signal(
            loaded.iq,
            window_size=(
                validated_window_size
            ),
            stride=validated_stride,
            remainder_policy=(
                normalized_remainder
            ),
            pad_value=float(pad_value),
        )

        available_window_count = (
            windowed.window_count
        )

        if available_window_count <= 0:
            raise ValueError(
                "No analyzable windows were "
                "created from the signal."
            )

        analyzed_window_count = min(
            available_window_count,
            validated_maximum_windows,
        )
        windows = np.ascontiguousarray(
            windowed.windows[
                :analyzed_window_count
            ]
        )
        valid_sample_counts = (
            np.ascontiguousarray(
                windowed
                .valid_sample_counts[
                    :analyzed_window_count
                ]
            )
        )
        start_indices = (
            np.ascontiguousarray(
                windowed.start_indices[
                    :analyzed_window_count
                ]
            )
        )
        original_sample_count = (
            windowed.original_sample_count
        )

    predictions = predict_window_batches(
        engine,
        windows,
        batch_size=validated_batch_size,
    )
    aggregate = (
        aggregate_window_predictions(
            predictions,
            weights=valid_sample_counts,
        )
    )
    shift_assessment = (
        shift_detector.assess_iq(
            windows
        )
    )

    records = []

    for index in range(
        analyzed_window_count
    ):
        if start_indices is None:
            start_sample = None
            stop_sample = None
            source_sample_index = int(
                loaded.sample_indices[index]
            )
            metadata_index = index
        else:
            start_sample = int(
                start_indices[index]
            )
            stop_sample = (
                start_sample
                + int(
                    valid_sample_counts[index]
                )
            )
            source_sample_index = int(
                loaded.sample_indices[0]
            )
            metadata_index = 0

        true_index = (
            None
            if loaded.labels is None
            else int(
                loaded.labels[
                    metadata_index
                ]
            )
        )
        snr_db = (
            None
            if loaded.snr_db is None
            else float(
                loaded.snr_db[
                    metadata_index
                ]
            )
        )

        records.append(
            LongSignalWindowRecord(
                window_index=index,
                source_sample_index=(
                    source_sample_index
                ),
                start_sample=start_sample,
                stop_sample_exclusive=(
                    stop_sample
                ),
                valid_sample_count=int(
                    valid_sample_counts[index]
                ),
                predicted_index=int(
                    predictions
                    .predicted_indices[index]
                ),
                predicted_label=(
                    predictions
                    .predicted_labels[index]
                ),
                confidence=float(
                    predictions
                    .confidences[index]
                ),
                probabilities=tuple(
                    float(value)
                    for value in (
                        predictions
                        .probabilities[index]
                    )
                ),
                shift_score=float(
                    shift_assessment
                    .scores[index]
                ),
                shift_threshold=(
                    shift_detector.threshold
                ),
                shift_like=bool(
                    shift_assessment
                    .shift_like[index]
                ),
                true_index=true_index,
                snr_db=snr_db,
            )
        )

    return LongSignalAnalysis(
        source_mode=source_mode,
        class_names=tuple(
            engine.class_names
        ),
        original_sample_count=(
            original_sample_count
        ),
        source_item_count=(
            source_item_count
        ),
        analyzed_window_count=(
            analyzed_window_count
        ),
        truncated=(
            analyzed_window_count
            < available_window_count
        ),
        window_size=validated_window_size,
        stride=validated_stride,
        remainder_policy=(
            normalized_remainder
        ),
        aggregate_predicted_index=(
            aggregate.predicted_index
        ),
        aggregate_predicted_label=(
            aggregate.predicted_label
        ),
        aggregate_confidence=(
            aggregate.confidence
        ),
        aggregate_probabilities=tuple(
            aggregate.probabilities
        ),
        shift_threshold=(
            shift_detector.threshold
        ),
        window_records=tuple(records),
    )


__all__ = [
    "LongSignalAnalysis",
    "LongSignalWindowRecord",
    "analyze_long_iq",
]
