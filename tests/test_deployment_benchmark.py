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

from rfsil.deployment.benchmark import (
    benchmark_checkpoint,
    build_benchmark_batch,
    summarize_latencies,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "benchmark_inference.py"
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
                "benchmark_test"
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


def test_summarizes_latencies() -> None:
    summary = summarize_latencies(
        [0.001, 0.002, 0.003]
    )

    assert summary.iteration_count == 3
    assert summary.mean_ms == pytest.approx(
        2.0
    )
    assert summary.minimum_ms == pytest.approx(
        1.0
    )
    assert summary.median_ms == pytest.approx(
        2.0
    )
    assert summary.maximum_ms == pytest.approx(
        3.0
    )


def test_rejects_empty_latencies() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        summarize_latencies([])


def test_builds_repeated_batch() -> None:
    source = np.arange(
        2 * 2 * 4,
        dtype=np.float32,
    ).reshape(2, 2, 4)

    result = build_benchmark_batch(
        source,
        batch_size=5,
    )

    assert result.shape == (5, 2, 4)

    np.testing.assert_array_equal(
        result[0],
        source[0],
    )
    np.testing.assert_array_equal(
        result[1],
        source[1],
    )
    np.testing.assert_array_equal(
        result[2],
        source[0],
    )


@pytest.mark.parametrize(
    "batch_size",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_rejects_invalid_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="batch_size",
    ):
        build_benchmark_batch(
            np.zeros(
                (1, 2, 8),
                dtype=np.float32,
            ),
            batch_size=batch_size,
        )


def test_benchmarks_cpu_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    result = benchmark_checkpoint(
        checkpoint_path=checkpoint,
        inputs=np.zeros(
            (2, 2, 64),
            dtype=np.float32,
        ),
        device="cpu",
        expected_sample_count=64,
        warmup_iterations=1,
        measurement_iterations=3,
    )

    assert result.device == "cpu"
    assert result.batch_size == 2
    assert result.sample_count == 64
    assert (
        result.steady_state_latency
        .iteration_count
        == 3
    )
    assert result.model_load_ms > 0.0
    assert result.first_inference_ms > 0.0
    assert (
        result.throughput_windows_per_second
        > 0.0
    )
    assert result.cuda_peak_memory_bytes is None


@pytest.mark.parametrize(
    (
        "warmup_iterations",
        "measurement_iterations",
    ),
    (
        (-1, 1),
        (True, 1),
        (0, 0),
        (0, -1),
    ),
)
def test_rejects_invalid_iteration_counts(
    tmp_path: Path,
    warmup_iterations: object,
    measurement_iterations: object,
) -> None:
    checkpoint = tmp_path / "model.pt"
    write_checkpoint(checkpoint)

    with pytest.raises(ValueError):
        benchmark_checkpoint(
            checkpoint_path=checkpoint,
            inputs=np.zeros(
                (1, 2, 64),
                dtype=np.float32,
            ),
            device="cpu",
            expected_sample_count=64,
            warmup_iterations=(
                warmup_iterations
            ),
            measurement_iterations=(
                measurement_iterations
            ),
        )


def test_cli_writes_cpu_benchmark(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "input.npy"
    output_path = tmp_path / "benchmark.json"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 64),
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
        "--batch-size",
        "1",
        "--batch-size",
        "4",
        "--expected-samples",
        "64",
        "--warmup-iterations",
        "1",
        "--measurement-iterations",
        "2",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
    assert output_path.is_file()

    document = json.loads(result.stdout)

    assert document["configuration"][
        "devices"
    ] == ["cpu"]

    assert document["configuration"][
        "batch_sizes"
    ] == [1, 4]

    assert len(document["results"]) == 2

    for benchmark in document["results"]:
        assert benchmark["device"] == "cpu"
        assert (
            benchmark[
                "throughput_windows_per_second"
            ]
            > 0.0
        )


def test_cli_rejects_invalid_batch_size(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "input.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (2, 64),
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
        "--batch-size",
        "0",
        "--expected-samples",
        "64",
    )

    assert result.returncode == 2
    assert "batch size" in result.stderr
