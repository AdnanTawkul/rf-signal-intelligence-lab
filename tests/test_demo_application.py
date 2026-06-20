from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from rfsil.demo.application import (
    build_signal_view_data,
    discover_checkpoints,
    load_demo_config,
    load_uploaded_iq,
    run_single_window_prediction,
    select_loaded_window,
)
from rfsil.deployment import (
    BatchPrediction,
    LoadedIQ,
)


class FakeEngine:
    class_names = (
        "BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
    )
    device = "cpu"
    input_scale = 1.0
    checkpoint_metadata = {
        "seed": 2026,
    }

    def predict_batch(
        self,
        inputs: np.ndarray,
    ) -> BatchPrediction:
        assert inputs.shape == (
            1,
            2,
            16,
        )

        return BatchPrediction(
            class_names=self.class_names,
            logits=np.asarray(
                [[0.0, 2.0, 0.0, -1.0]],
                dtype=np.float32,
            ),
            probabilities=np.asarray(
                [[0.1, 0.7, 0.15, 0.05]],
                dtype=np.float32,
            ),
            predicted_indices=np.asarray(
                [1],
                dtype=np.int64,
            ),
            predicted_labels=(
                "QPSK",
            ),
            confidences=np.asarray(
                [0.7],
                dtype=np.float32,
            ),
        )


def test_load_demo_config(
    tmp_path: Path,
) -> None:
    config_path = (
        tmp_path / "demo.yaml"
    )
    config_path.write_text(
        """
experiment_name: test_demo

checkpoint:
  search_root: results
  preferred_path: results/example/best_model.pt

inference:
  expected_sample_count: 2048
  input_scale: 1.0
  default_device: auto
  top_k: 4

visualization:
  default_sample_rate_hz: 1000000.0
  maximum_waveform_points: 256
  maximum_constellation_points: 512
  spectrum_fft_size: 1024
""".strip(),
        encoding="utf-8",
    )

    config = load_demo_config(
        config_path,
        project_root=tmp_path,
    )

    assert (
        config.checkpoint_search_root
        == tmp_path / "results"
    )
    assert (
        config.preferred_checkpoint
        == (
            tmp_path
            / "results"
            / "example"
            / "best_model.pt"
        )
    )
    assert (
        config.expected_sample_count
        == 2048
    )


def test_discovers_checkpoints(
    tmp_path: Path,
) -> None:
    first = (
        tmp_path
        / "experiment_a"
        / "seed_2026"
        / "best_model.pt"
    )
    second = (
        tmp_path
        / "experiment_b"
        / "best_model.pt"
    )

    first.parent.mkdir(
        parents=True
    )
    second.parent.mkdir(
        parents=True
    )
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    options = discover_checkpoints(
        tmp_path
    )

    assert len(options) == 2
    assert all(
        option.size_bytes > 0
        for option in options
    )
    assert {
        option.path
        for option in options
    } == {
        first.resolve(),
        second.resolve(),
    }


def test_loads_uploaded_npy() -> None:
    buffer = BytesIO()
    iq = np.ones(
        (2, 16),
        dtype=np.float32,
    )
    np.save(buffer, iq)

    loaded = load_uploaded_iq(
        filename="example.npy",
        content=buffer.getvalue(),
        expected_sample_count=16,
    )

    assert loaded.batch_size == 1
    assert loaded.sample_count == 16
    np.testing.assert_array_equal(
        loaded.iq[0],
        iq,
    )


def test_loads_uploaded_npz_metadata() -> None:
    buffer = BytesIO()
    iq = np.ones(
        (2, 2, 16),
        dtype=np.float32,
    )

    np.savez_compressed(
        buffer,
        iq=iq,
        labels=np.asarray(
            [1, 2],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [0.0, 8.0],
            dtype=np.float32,
        ),
    )

    loaded = load_uploaded_iq(
        filename="example.npz",
        content=buffer.getvalue(),
        expected_sample_count=16,
    )

    assert loaded.batch_size == 2
    np.testing.assert_array_equal(
        loaded.labels,
        [1, 2],
    )
    np.testing.assert_allclose(
        loaded.snr_db,
        [0.0, 8.0],
    )


def test_selects_loaded_window() -> None:
    loaded = LoadedIQ(
        source_path=Path("input.npz"),
        array_key="iq",
        iq=np.ones(
            (3, 2, 16),
            dtype=np.float32,
        ),
        sample_indices=np.asarray(
            [10, 11, 12],
            dtype=np.int64,
        ),
    )

    selected = select_loaded_window(
        loaded,
        1,
    )

    assert selected.batch_size == 1
    assert int(
        selected.sample_indices[0]
    ) == 11


def test_builds_signal_view_data() -> None:
    sample_rate_hz = 1024.0
    sample_count = 1024
    tone_frequency_hz = 128.0
    time_s = (
        np.arange(sample_count)
        / sample_rate_hz
    )
    samples = np.exp(
        1j
        * 2.0
        * np.pi
        * tone_frequency_hz
        * time_s
    )

    iq = np.stack(
        (
            samples.real,
            samples.imag,
        ),
        axis=0,
    ).astype(np.float32)

    view = build_signal_view_data(
        iq,
        sample_rate_hz=sample_rate_hz,
        maximum_waveform_points=128,
        maximum_constellation_points=256,
        spectrum_fft_size=1024,
    )

    assert (
        view.waveform_i.size
        <= 128
    )
    assert (
        view.constellation_i.size
        <= 256
    )

    peak_frequency = (
        view.spectrum_frequency_hz[
            int(
                np.argmax(
                    view.spectrum_power_db
                )
            )
        ]
    )

    assert peak_frequency == (
        pytest.approx(
            tone_frequency_hz,
            abs=1e-6,
        )
    )


def test_rejects_invalid_uploaded_extension(
) -> None:
    with pytest.raises(
        ValueError,
        match="npy or .npz",
    ):
        load_uploaded_iq(
            filename="example.txt",
            content=b"content",
        )


def test_runs_single_window_prediction(
    tmp_path: Path,
) -> None:
    loaded = LoadedIQ(
        source_path=Path("input.npy"),
        array_key=None,
        iq=np.ones(
            (1, 2, 16),
            dtype=np.float32,
        ),
        sample_indices=np.asarray(
            [0],
            dtype=np.int64,
        ),
    )

    result = (
        run_single_window_prediction(
            engine=FakeEngine(),
            loaded=loaded,
            position=0,
            checkpoint_path=(
                tmp_path
                / "best_model.pt"
            ),
            top_k=2,
        )
    )

    assert (
        result.predicted_record[
            "predicted_label"
        ]
        == "QPSK"
    )
    assert len(
        result.predicted_record[
            "top_k"
        ]
    ) == 2
    assert (
        '"predicted_label": "QPSK"'
        in result.to_json()
    )

def test_builds_public_prediction_document(
) -> None:
    from rfsil.demo.application import (
        build_public_prediction_document,
    )

    original = {
        "model": {
            "checkpoint_path": (
                "G:/private/results/best_model.pt"
            ),
            "device": "cuda",
        },
        "input": {
            "source_path": (
                "G:/private/data/validation.npz"
            ),
            "sample_count": 2048,
        },
    }

    exported = (
        build_public_prediction_document(
            original,
            source_name=(
                "../uploads/validation.npz"
            ),
            checkpoint_reference=(
                "experiment/seed_2026/"
                "best_model.pt"
            ),
        )
    )

    assert exported["model"][
        "checkpoint_path"
    ] == (
        "experiment/seed_2026/"
        "best_model.pt"
    )
    assert exported["input"][
        "source_path"
    ] == "validation.npz"

    assert original["model"][
        "checkpoint_path"
    ] == (
        "G:/private/results/best_model.pt"
    )
