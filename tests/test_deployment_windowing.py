from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from rfsil.deployment.inference import (
    BatchPrediction,
    IQInferenceEngine,
)
from rfsil.deployment.windowing import (
    aggregate_window_predictions,
    predict_window_batches,
    window_iq_signal,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "predict_long_iq.py"
)


def write_checkpoint(
    path: Path,
) -> None:
    torch.manual_seed(2026)

    configuration = BaselineCNNConfig(
        in_channels=2,
        num_classes=4,
        channels=(8, 16),
        kernel_size=3,
        dropout=0.0,
        normalize_input_rms=False,
        normalization="group",
        group_norm_groups=4,
    )
    model = BaselineIQCNN(configuration)

    torch.save(
        {
            "format_version": 1,
            "experiment_name": (
                "long_inference_test"
            ),
            "seed": 2026,
            "model_configuration": (
                asdict(configuration)
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "class_names": [
                "bpsk",
                "qpsk",
                "8psk",
                "16qam",
            ],
        },
        path,
    )


def run_cli(
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()

    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_nonoverlapping_exact_windows() -> None:
    signal = np.arange(
        2 * 16,
        dtype=np.float32,
    ).reshape(2, 16)

    result = window_iq_signal(
        signal,
        window_size=8,
    )

    assert result.windows.shape == (
        2,
        2,
        8,
    )
    assert result.start_indices.tolist() == [
        0,
        8,
    ]
    assert (
        result.valid_sample_counts.tolist()
        == [8, 8]
    )


def test_overlapping_windows() -> None:
    signal = np.zeros(
        (2, 16),
        dtype=np.float32,
    )

    result = window_iq_signal(
        signal,
        window_size=8,
        stride=4,
    )

    assert result.start_indices.tolist() == [
        0,
        4,
        8,
    ]


def test_drops_trailing_samples() -> None:
    signal = np.zeros(
        (2, 18),
        dtype=np.float32,
    )

    result = window_iq_signal(
        signal,
        window_size=8,
        stride=8,
        remainder_policy="drop",
    )

    assert result.window_count == 2
    assert result.start_indices.tolist() == [
        0,
        8,
    ]


def test_pads_one_trailing_window() -> None:
    signal = np.ones(
        (2, 18),
        dtype=np.float32,
    )

    result = window_iq_signal(
        signal,
        window_size=8,
        stride=8,
        remainder_policy="pad",
        pad_value=-2.0,
    )

    assert result.window_count == 3
    assert result.start_indices.tolist() == [
        0,
        8,
        16,
    ]
    assert (
        result.valid_sample_counts.tolist()
        == [8, 8, 2]
    )
    np.testing.assert_array_equal(
        result.windows[2, :, :2],
        np.ones((2, 2)),
    )
    np.testing.assert_array_equal(
        result.windows[2, :, 2:],
        np.full((2, 6), -2.0),
    )


def test_pads_signal_shorter_than_window() -> None:
    signal = np.ones(
        (2, 3),
        dtype=np.float32,
    )

    result = window_iq_signal(
        signal,
        window_size=8,
        remainder_policy="pad",
    )

    assert result.windows.shape == (
        1,
        2,
        8,
    )
    assert result.valid_sample_counts.tolist() == [
        3
    ]


def test_drop_rejects_short_signal() -> None:
    with pytest.raises(
        ValueError,
        match="shorter than one",
    ):
        window_iq_signal(
            np.zeros(
                (2, 3),
                dtype=np.float32,
            ),
            window_size=8,
            remainder_policy="drop",
        )


def test_error_policy_rejects_remainder() -> None:
    with pytest.raises(
        ValueError,
        match="does not fit",
    ):
        window_iq_signal(
            np.zeros(
                (2, 18),
                dtype=np.float32,
            ),
            window_size=8,
            remainder_policy="error",
        )


def test_accepts_complex_signal() -> None:
    signal = np.asarray(
        [
            1.0 + 2.0j,
            3.0 + 4.0j,
            5.0 + 6.0j,
            7.0 + 8.0j,
        ],
        dtype=np.complex64,
    )

    result = window_iq_signal(
        signal,
        window_size=4,
    )

    np.testing.assert_array_equal(
        result.windows[0, 0],
        [1.0, 3.0, 5.0, 7.0],
    )
    np.testing.assert_array_equal(
        result.windows[0, 1],
        [2.0, 4.0, 6.0, 8.0],
    )


def test_rejects_stride_larger_than_window() -> None:
    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        window_iq_signal(
            np.zeros(
                (2, 16),
                dtype=np.float32,
            ),
            window_size=8,
            stride=9,
        )


def make_prediction() -> BatchPrediction:
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=np.float32,
    )

    return BatchPrediction(
        class_names=("a", "b"),
        logits=np.log(probabilities),
        probabilities=probabilities,
        predicted_indices=np.asarray(
            [0, 1],
            dtype=np.int64,
        ),
        predicted_labels=("a", "b"),
        confidences=np.asarray(
            [0.8, 0.8],
            dtype=np.float32,
        ),
    )


def test_aggregates_mean_probabilities() -> None:
    result = aggregate_window_predictions(
        make_prediction()
    )

    assert result.probabilities == pytest.approx(
        (0.5, 0.5)
    )
    assert result.predicted_index == 0
    assert result.window_count == 2


def test_aggregates_weighted_probabilities() -> None:
    result = aggregate_window_predictions(
        make_prediction(),
        weights=np.asarray(
            [3.0, 1.0],
            dtype=np.float32,
        ),
    )

    assert result.probabilities == pytest.approx(
        (0.65, 0.35)
    )
    assert result.predicted_label == "a"


def test_rejects_weight_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="window count",
    ):
        aggregate_window_predictions(
            make_prediction(),
            weights=np.asarray([1.0]),
        )


def test_predicts_windows_in_batches(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    engine = IQInferenceEngine.from_checkpoint(
        checkpoint,
        device="cpu",
        expected_sample_count=64,
    )

    result = predict_window_batches(
        engine,
        np.zeros(
            (5, 2, 64),
            dtype=np.float32,
        ),
        batch_size=2,
    )

    assert len(result) == 5
    assert result.probabilities.shape == (
        5,
        4,
    )


def test_cli_predicts_long_signal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "long.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 150),
            dtype=np.float32,
        ),
    )

    result = run_cli(
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--device",
        "cpu",
        "--window-size",
        "64",
        "--stride",
        "32",
        "--remainder",
        "pad",
        "--batch-size",
        "2",
        "--top-k",
        "2",
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )

    document = json.loads(result.stdout)

    assert document["input"][
        "original_sample_count"
    ] == 150
    assert document["windowing"][
        "window_count"
    ] == 4
    assert len(
        document["window_predictions"]
    ) == 4
    assert len(
        document[
            "aggregate_prediction"
        ]["top_k"]
    ) == 2


def test_cli_requires_one_signal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "batch.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 2, 150),
            dtype=np.float32,
        ),
    )

    result = run_cli(
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--device",
        "cpu",
        "--window-size",
        "64",
    )

    assert result.returncode == 2
    assert "exactly one signal" in (
        result.stderr
    )


def test_cli_error_policy_rejects_tail(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "long.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 150),
            dtype=np.float32,
        ),
    )

    result = run_cli(
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--device",
        "cpu",
        "--window-size",
        "64",
        "--stride",
        "64",
        "--remainder",
        "error",
    )

    assert result.returncode == 2
    assert "does not fit" in result.stderr
