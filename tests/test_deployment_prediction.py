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
)
from rfsil.deployment.iq_io import LoadedIQ
from rfsil.deployment.prediction import (
    build_prediction_document,
    rank_probabilities,
    validate_top_k,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "predict_iq.py"
)


def write_checkpoint(
    path: Path,
) -> None:
    """Write one small deterministic checkpoint."""
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
                "prediction_cli_test"
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


def make_loaded(
    tmp_path: Path,
    *,
    labels: np.ndarray | None = None,
    snr_db: np.ndarray | None = None,
) -> LoadedIQ:
    return LoadedIQ(
        source_path=tmp_path / "input.npz",
        array_key="iq",
        iq=np.zeros(
            (2, 2, 8),
            dtype=np.float32,
        ),
        sample_indices=np.asarray(
            [4, 7],
            dtype=np.int64,
        ),
        labels=labels,
        snr_db=snr_db,
    )


def make_prediction() -> BatchPrediction:
    probabilities = np.asarray(
        [
            [0.1, 0.6, 0.2, 0.1],
            [0.7, 0.1, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    return BatchPrediction(
        class_names=(
            "bpsk",
            "qpsk",
            "8psk",
            "16qam",
        ),
        logits=np.log(probabilities),
        probabilities=probabilities,
        predicted_indices=np.asarray(
            [1, 0],
            dtype=np.int64,
        ),
        predicted_labels=(
            "qpsk",
            "bpsk",
        ),
        confidences=np.asarray(
            [0.6, 0.7],
            dtype=np.float32,
        ),
    )


def test_ranks_probabilities() -> None:
    ranked = rank_probabilities(
        [0.1, 0.5, 0.3, 0.1],
        class_names=(
            "a",
            "b",
            "c",
            "d",
        ),
        top_k=2,
    )

    assert ranked == [
        {
            "rank": 1,
            "class_index": 1,
            "label": "b",
            "probability": 0.5,
        },
        {
            "rank": 2,
            "class_index": 2,
            "label": "c",
            "probability": 0.3,
        },
    ]


def test_default_top_k_includes_all() -> None:
    ranked = rank_probabilities(
        [0.25, 0.75],
        class_names=("a", "b"),
    )

    assert len(ranked) == 2


@pytest.mark.parametrize(
    ("top_k", "class_count"),
    (
        (0, 4),
        (-1, 4),
        (True, 4),
        (1.5, 4),
        (5, 4),
    ),
)
def test_rejects_invalid_top_k(
    top_k: object,
    class_count: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="top_k",
    ):
        validate_top_k(
            top_k,
            class_count=class_count,
        )


def test_builds_document_with_metadata(
    tmp_path: Path,
) -> None:
    loaded = make_loaded(
        tmp_path,
        labels=np.asarray(
            [1, 3],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [-2.0, 4.0],
            dtype=np.float32,
        ),
    )

    document = build_prediction_document(
        loaded=loaded,
        prediction=make_prediction(),
        checkpoint_path=(
            tmp_path / "model.pt"
        ),
        device="cpu",
        input_scale=1.0,
        top_k=2,
        checkpoint_metadata={
            "seed": 2026,
        },
    )

    assert document["input"][
        "batch_size"
    ] == 2

    first = document["predictions"][0]

    assert first["sample_index"] == 4
    assert first["true_index"] == 1
    assert first["true_label"] == "qpsk"
    assert first["correct"] is True
    assert first["snr_db"] == pytest.approx(
        -2.0
    )
    assert len(first["top_k"]) == 2


def test_builds_document_without_metadata(
    tmp_path: Path,
) -> None:
    document = build_prediction_document(
        loaded=make_loaded(tmp_path),
        prediction=make_prediction(),
        checkpoint_path=(
            tmp_path / "model.pt"
        ),
        device="cpu",
        input_scale=1.0,
    )

    first = document["predictions"][0]

    assert "true_index" not in first
    assert "true_label" not in first
    assert "correct" not in first
    assert "snr_db" not in first


def test_rejects_out_of_range_label(
    tmp_path: Path,
) -> None:
    loaded = make_loaded(
        tmp_path,
        labels=np.asarray(
            [1, 9],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Ground-truth",
    ):
        build_prediction_document(
            loaded=loaded,
            prediction=make_prediction(),
            checkpoint_path=(
                tmp_path / "model.pt"
            ),
            device="cpu",
            input_scale=1.0,
        )


def test_rejects_batch_mismatch(
    tmp_path: Path,
) -> None:
    loaded = LoadedIQ(
        source_path=tmp_path / "input.npy",
        array_key=None,
        iq=np.zeros(
            (1, 2, 8),
            dtype=np.float32,
        ),
        sample_indices=np.asarray(
            [0],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Prediction count",
    ):
        build_prediction_document(
            loaded=loaded,
            prediction=make_prediction(),
            checkpoint_path=(
                tmp_path / "model.pt"
            ),
            device="cpu",
            input_scale=1.0,
        )


def test_cli_predicts_selected_npz_sample(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "dataset.npz"

    write_checkpoint(checkpoint)

    np.savez_compressed(
        input_path,
        iq=np.zeros(
            (2, 2, 64),
            dtype=np.float32,
        ),
        labels=np.asarray(
            [1, 2],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [-4.0, 6.0],
            dtype=np.float32,
        ),
    )

    result = run_cli(
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--sample-index",
        "1",
        "--device",
        "cpu",
        "--expected-samples",
        "64",
        "--top-k",
        "2",
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )

    document = json.loads(result.stdout)

    assert document["input"][
        "batch_size"
    ] == 1
    assert document["input"][
        "sample_indices"
    ] == [1]

    prediction = document[
        "predictions"
    ][0]

    assert prediction["sample_index"] == 1
    assert prediction["true_index"] == 2
    assert prediction["true_label"] == "8psk"
    assert prediction["snr_db"] == pytest.approx(
        6.0
    )
    assert len(prediction["top_k"]) == 2


def test_cli_writes_output_file(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "signal.npy"
    output_path = (
        tmp_path / "output" / "prediction.json"
    )

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
        "--expected-samples",
        "64",
        "--output",
        str(output_path),
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )
    assert output_path.is_file()

    stdout_document = json.loads(
        result.stdout
    )
    file_document = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert file_document == stdout_document


def test_cli_predicts_npy_batch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "batch.npy"

    write_checkpoint(checkpoint)

    np.save(
        input_path,
        np.zeros(
            (3, 2, 64),
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
        "--expected-samples",
        "64",
    )

    assert result.returncode == 0, (
        result.stdout + result.stderr
    )

    document = json.loads(result.stdout)

    assert document["input"][
        "batch_size"
    ] == 3
    assert len(
        document["predictions"]
    ) == 3

    for prediction in document[
        "predictions"
    ]:
        assert "true_index" not in prediction
        assert len(
            prediction["top_k"]
        ) == 4


def test_cli_rejects_invalid_top_k(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    input_path = tmp_path / "signal.npy"

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
        "--expected-samples",
        "64",
        "--top-k",
        "0",
    )

    assert result.returncode == 2
    assert "top_k" in result.stderr
