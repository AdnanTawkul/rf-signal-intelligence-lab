from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfsil.demo.long_signal_service import (
    analyze_long_iq,
)
from rfsil.deployment import (
    BatchPrediction,
    LoadedIQ,
)
from rfsil.deployment.shift_detector import (
    IQShiftDetectorArtifact,
)
from rfsil.evaluation.iq_channel_features import (
    compute_iq_channel_features,
)

CLASS_NAMES = (
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
)


class FakeEngine:
    class_names = CLASS_NAMES
    in_channels = 2
    expected_sample_count = 64

    def predict_batch(
        self,
        inputs: np.ndarray,
    ) -> BatchPrediction:
        batch_size = inputs.shape[0]
        probabilities = np.tile(
            np.asarray(
                [[0.1, 0.2, 0.6, 0.1]],
                dtype=np.float32,
            ),
            (batch_size, 1),
        )

        return BatchPrediction(
            class_names=self.class_names,
            logits=np.log(
                probabilities
            ),
            probabilities=probabilities,
            predicted_indices=np.full(
                batch_size,
                2,
                dtype=np.int64,
            ),
            predicted_labels=tuple(
                "8PSK"
                for _ in range(batch_size)
            ),
            confidences=np.full(
                batch_size,
                0.6,
                dtype=np.float32,
            ),
        )


def example_windows(
    *,
    batch_size: int,
    sample_count: int = 64,
) -> np.ndarray:
    generator = np.random.default_rng(
        2026
    )

    return generator.normal(
        size=(
            batch_size,
            2,
            sample_count,
        )
    ).astype(np.float32)


def example_detector() -> (
    IQShiftDetectorArtifact
):
    features = compute_iq_channel_features(
        example_windows(batch_size=2)
    )
    feature_count = len(
        features.feature_names
    )

    return IQShiftDetectorArtifact(
        format_version=1,
        artifact_name="test_detector",
        expected_sample_count=64,
        feature_names=(
            features.feature_names
        ),
        feature_mean=np.zeros(
            feature_count,
            dtype=np.float64,
        ),
        feature_scale=np.ones(
            feature_count,
            dtype=np.float64,
        ),
        coefficients=np.linspace(
            -0.1,
            0.1,
            num=feature_count,
            dtype=np.float64,
        ),
        intercept=0.0,
        l2_strength=0.1,
        threshold=0.0,
        target_tpr=0.95,
        development_auroc=0.9,
        development_average_precision=0.9,
        development_fpr_at_target_tpr=0.5,
        development_clean_mean=-0.2,
        development_clean_std=0.2,
        development_shifted_mean=0.2,
        development_shifted_std=0.4,
        autocorrelation_lags=(
            1,
            2,
            4,
            8,
        ),
        occupancy_fraction=0.9,
        epsilon=1e-12,
        provenance={},
    )


def test_analyzes_prewindowed_batch() -> None:
    loaded = LoadedIQ(
        source_path=Path("batch.npz"),
        array_key="iq",
        iq=example_windows(
            batch_size=5
        ),
        sample_indices=np.arange(
            100,
            105,
            dtype=np.int64,
        ),
        labels=np.full(
            5,
            2,
            dtype=np.int64,
        ),
        snr_db=np.full(
            5,
            4.0,
            dtype=np.float32,
        ),
    )

    result = analyze_long_iq(
        loaded=loaded,
        engine=FakeEngine(),
        shift_detector=(
            example_detector()
        ),
        window_size=64,
        stride=32,
        remainder_policy="drop",
        batch_size=2,
        maximum_windows=10,
    )

    assert (
        result.source_mode
        == "prewindowed_batch"
    )
    assert (
        result.analyzed_window_count
        == 5
    )
    assert (
        result.aggregate_predicted_label
        == "8PSK"
    )
    assert result.window_records[0]\
        .source_sample_index == 100
    assert (
        result.window_records[0]
        .start_sample
        is None
    )


def test_analyzes_continuous_signal() -> None:
    signal = example_windows(
        batch_size=1,
        sample_count=160,
    )

    loaded = LoadedIQ(
        source_path=Path("long.npy"),
        array_key=None,
        iq=signal,
        sample_indices=np.asarray(
            [7],
            dtype=np.int64,
        ),
    )

    result = analyze_long_iq(
        loaded=loaded,
        engine=FakeEngine(),
        shift_detector=(
            example_detector()
        ),
        window_size=64,
        stride=32,
        remainder_policy="drop",
        batch_size=8,
        maximum_windows=20,
    )

    assert (
        result.source_mode
        == "continuous_long_signal"
    )
    assert (
        result.analyzed_window_count
        == 4
    )
    assert (
        result.window_records[1]
        .start_sample
        == 32
    )
    assert (
        result.window_records[1]
        .stop_sample_exclusive
        == 96
    )


def test_limits_window_count() -> None:
    loaded = LoadedIQ(
        source_path=Path("batch.npy"),
        array_key=None,
        iq=example_windows(
            batch_size=12
        ),
        sample_indices=np.arange(
            12,
            dtype=np.int64,
        ),
    )

    result = analyze_long_iq(
        loaded=loaded,
        engine=FakeEngine(),
        shift_detector=(
            example_detector()
        ),
        window_size=64,
        stride=64,
        remainder_policy="drop",
        batch_size=4,
        maximum_windows=5,
    )

    assert (
        result.analyzed_window_count
        == 5
    )
    assert result.truncated is True


def test_json_export_is_path_safe() -> None:
    loaded = LoadedIQ(
        source_path=Path("batch.npy"),
        array_key=None,
        iq=example_windows(
            batch_size=2
        ),
        sample_indices=np.arange(
            2,
            dtype=np.int64,
        ),
    )

    result = analyze_long_iq(
        loaded=loaded,
        engine=FakeEngine(),
        shift_detector=(
            example_detector()
        ),
        window_size=64,
        stride=64,
        remainder_policy="drop",
        batch_size=2,
        maximum_windows=2,
    )

    document = result.to_document(
        source_name=(
            "../private/batch.npy"
        ),
        checkpoint_reference=(
            "experiment/best_model.pt"
        ),
        detector_name="detector_v1",
        top_k=4,
    )

    assert document["input"][
        "source_name"
    ] == "batch.npy"
    assert "private" not in str(document)


def test_csv_contains_every_window() -> None:
    loaded = LoadedIQ(
        source_path=Path("batch.npy"),
        array_key=None,
        iq=example_windows(
            batch_size=3
        ),
        sample_indices=np.arange(
            3,
            dtype=np.int64,
        ),
    )

    result = analyze_long_iq(
        loaded=loaded,
        engine=FakeEngine(),
        shift_detector=(
            example_detector()
        ),
        window_size=64,
        stride=64,
        remainder_policy="drop",
        batch_size=2,
        maximum_windows=3,
    )

    csv_text = result.to_csv()

    assert (
        "probability_8PSK"
        in csv_text
    )
    assert (
        len(csv_text.strip().splitlines())
        == 4
    )


def test_rejects_wrong_batch_window_size(
) -> None:
    loaded = LoadedIQ(
        source_path=Path("batch.npy"),
        array_key=None,
        iq=example_windows(
            batch_size=2,
            sample_count=32,
        ),
        sample_indices=np.arange(
            2,
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="window size",
    ):
        analyze_long_iq(
            loaded=loaded,
            engine=FakeEngine(),
            shift_detector=(
                example_detector()
            ),
            window_size=64,
            stride=32,
            remainder_policy="drop",
            batch_size=2,
            maximum_windows=2,
        )


@pytest.mark.parametrize(
    "remainder_policy",
    (
        "",
        "unknown",
    ),
)
def test_rejects_invalid_remainder_policy(
    remainder_policy: str,
) -> None:
    loaded = LoadedIQ(
        source_path=Path("signal.npy"),
        array_key=None,
        iq=example_windows(
            batch_size=1,
            sample_count=128,
        ),
        sample_indices=np.asarray(
            [0],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="remainder_policy",
    ):
        analyze_long_iq(
            loaded=loaded,
            engine=FakeEngine(),
            shift_detector=(
                example_detector()
            ),
            window_size=64,
            stride=32,
            remainder_policy=(
                remainder_policy
            ),
            batch_size=2,
            maximum_windows=10,
        )
