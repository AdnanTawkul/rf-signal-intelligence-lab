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
    IQInferenceEngine,
)
from rfsil.deployment.streaming import (
    IQStreamBuffer,
    StreamingIQClassifier,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "stream_iq.py"
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
                "streaming_test"
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


def test_emits_after_multiple_chunks() -> None:
    buffer = IQStreamBuffer(
        window_size=8,
        hop_size=8,
    )

    assert buffer.push(
        np.zeros(
            (2, 3),
            dtype=np.float32,
        )
    ) == ()

    windows = buffer.push(
        np.ones(
            (2, 5),
            dtype=np.float32,
        )
    )

    assert len(windows) == 1
    assert windows[0].start_sample == 0
    assert (
        windows[0].stop_sample_exclusive
        == 8
    )
    assert buffer.total_samples_received == 8


def test_emits_multiple_windows_from_one_chunk() -> None:
    buffer = IQStreamBuffer(
        window_size=8,
        hop_size=8,
    )

    windows = buffer.push(
        np.zeros(
            (2, 24),
            dtype=np.float32,
        )
    )

    assert len(windows) == 3
    assert [
        window.start_sample
        for window in windows
    ] == [0, 8, 16]


def test_overlapping_stream_windows() -> None:
    signal = np.arange(
        2 * 12,
        dtype=np.float32,
    ).reshape(2, 12)

    buffer = IQStreamBuffer(
        window_size=8,
        hop_size=4,
    )

    windows = buffer.push(signal)

    assert len(windows) == 2
    assert [
        window.start_sample
        for window in windows
    ] == [0, 4]

    np.testing.assert_array_equal(
        windows[0].iq,
        signal[:, 0:8],
    )
    np.testing.assert_array_equal(
        windows[1].iq,
        signal[:, 4:12],
    )


def test_chunk_boundaries_do_not_change_windows() -> None:
    signal = np.arange(
        2 * 20,
        dtype=np.float32,
    ).reshape(2, 20)

    whole = IQStreamBuffer(
        window_size=8,
        hop_size=4,
    )
    whole_windows = whole.push(signal)

    chunked = IQStreamBuffer(
        window_size=8,
        hop_size=4,
    )

    chunked_windows = (
        chunked.push(signal[:, :3])
        + chunked.push(signal[:, 3:11])
        + chunked.push(signal[:, 11:])
    )

    assert len(chunked_windows) == len(
        whole_windows
    )

    for first, second in zip(
        whole_windows,
        chunked_windows,
        strict=True,
    ):
        assert (
            first.start_sample
            == second.start_sample
        )
        np.testing.assert_array_equal(
            first.iq,
            second.iq,
        )


def test_accepts_complex_chunks() -> None:
    buffer = IQStreamBuffer(
        window_size=4,
    )

    windows = buffer.push(
        np.asarray(
            [
                1.0 + 2.0j,
                3.0 + 4.0j,
                5.0 + 6.0j,
                7.0 + 8.0j,
            ],
            dtype=np.complex64,
        )
    )

    np.testing.assert_array_equal(
        windows[0].iq[0],
        [1.0, 3.0, 5.0, 7.0],
    )
    np.testing.assert_array_equal(
        windows[0].iq[1],
        [2.0, 4.0, 6.0, 8.0],
    )


def test_reset_clears_state() -> None:
    buffer = IQStreamBuffer(
        window_size=8,
        hop_size=4,
    )

    buffer.push(
        np.zeros(
            (2, 10),
            dtype=np.float32,
        )
    )
    buffer.reset()

    assert buffer.total_samples_received == 0
    assert buffer.buffered_sample_count == 0
    assert buffer.emitted_window_count == 0
    assert buffer.next_window_start == 0


def test_rejects_invalid_hop_size() -> None:
    with pytest.raises(
        ValueError,
        match="must not exceed",
    ):
        IQStreamBuffer(
            window_size=8,
            hop_size=9,
        )


def test_rejects_invalid_chunk_shape() -> None:
    buffer = IQStreamBuffer(
        window_size=8,
    )

    with pytest.raises(
        ValueError,
        match="stream chunks",
    ):
        buffer.push(
            np.zeros(
                (8, 2),
                dtype=np.float32,
            )
        )


def test_streaming_classifier_adds_timestamps(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    engine = IQInferenceEngine.from_checkpoint(
        checkpoint,
        device="cpu",
        expected_sample_count=64,
    )
    classifier = StreamingIQClassifier(
        engine=engine,
        hop_size=32,
        sample_rate_hz=1000.0,
        inference_batch_size=2,
    )

    events = classifier.push(
        np.zeros(
            (2, 96),
            dtype=np.float32,
        )
    )

    assert len(events) == 2
    assert events[0].start_time_seconds == (
        pytest.approx(0.0)
    )
    assert events[0].end_time_seconds == (
        pytest.approx(0.064)
    )
    assert events[0].center_time_seconds == (
        pytest.approx(0.032)
    )
    assert events[1].start_time_seconds == (
        pytest.approx(0.032)
    )


def test_streaming_classifier_without_rate(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    engine = IQInferenceEngine.from_checkpoint(
        checkpoint,
        device="cpu",
        expected_sample_count=64,
    )
    classifier = StreamingIQClassifier(
        engine=engine,
    )

    events = classifier.push(
        np.zeros(
            (2, 64),
            dtype=np.float32,
        )
    )

    assert len(events) == 1
    assert events[0].start_time_seconds is None
    assert events[0].end_time_seconds is None
    assert events[0].center_time_seconds is None


def test_streaming_window_must_match_engine(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    engine = IQInferenceEngine.from_checkpoint(
        checkpoint,
        device="cpu",
        expected_sample_count=64,
    )

    with pytest.raises(
        ValueError,
        match="must match",
    ):
        StreamingIQClassifier(
            engine=engine,
            window_size=32,
        )


@pytest.mark.parametrize(
    "sample_rate_hz",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ),
)
def test_rejects_invalid_sample_rate(
    tmp_path: Path,
    sample_rate_hz: object,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    engine = IQInferenceEngine.from_checkpoint(
        checkpoint,
        device="cpu",
        expected_sample_count=64,
    )

    with pytest.raises(
        ValueError,
        match="sample_rate_hz",
    ):
        StreamingIQClassifier(
            engine=engine,
            sample_rate_hz=sample_rate_hz,
        )


def test_cli_streams_long_signal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "signal.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 160),
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
        "--hop-size",
        "32",
        "--chunk-size",
        "17",
        "--sample-rate-hz",
        "1000",
        "--top-k",
        "2",
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )

    document = json.loads(result.stdout)

    assert document["streaming"][
        "prediction_count"
    ] == 4
    assert len(document["predictions"]) == 4

    assert document["predictions"][0][
        "start_time_seconds"
    ] == pytest.approx(0.0)

    assert document["predictions"][1][
        "start_time_seconds"
    ] == pytest.approx(0.032)

    assert len(
        document["predictions"][0]["top_k"]
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
            (2, 2, 128),
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


def test_cli_rejects_short_signal(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "short.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 32),
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
    assert "No complete streaming windows" in (
        result.stderr
    )


def test_cli_writes_output(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "signal.npy"
    output_path = tmp_path / "stream.json"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 128),
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
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
    assert output_path.is_file()

    assert json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    ) == json.loads(result.stdout)
