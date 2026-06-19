from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from rfsil.deployment.inference import (
    IQInferenceEngine,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)


def write_checkpoint(
    path: Path,
    *,
    include_class_names: bool = True,
) -> BaselineIQCNN:
    """Create one deterministic test checkpoint."""
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
    model.eval()

    checkpoint = {
        "format_version": 1,
        "experiment_name": (
            "deployment_inference_test"
        ),
        "seed": 2026,
        "model_configuration": asdict(
            configuration
        ),
        "model_state_dict": (
            model.state_dict()
        ),
    }

    if include_class_names:
        checkpoint["class_names"] = [
            "bpsk",
            "qpsk",
            "8psk",
            "16qam",
        ]

    torch.save(checkpoint, path)

    return model


def test_loads_checkpoint_and_predicts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    original = write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    generator = torch.Generator()
    generator.manual_seed(2027)
    inputs = torch.randn(
        3,
        2,
        64,
        generator=generator,
    )

    with torch.inference_mode():
        expected_logits = original(inputs)

    result = engine.predict_batch(inputs)

    assert len(result) == 3
    assert result.class_names == (
        "bpsk",
        "qpsk",
        "8psk",
        "16qam",
    )
    assert result.logits.shape == (3, 4)
    assert result.probabilities.shape == (
        3,
        4,
    )
    assert result.predicted_indices.shape == (
        3,
    )
    assert result.confidences.shape == (3,)

    np.testing.assert_allclose(
        result.logits,
        expected_logits.numpy(),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.probabilities.sum(axis=1),
        np.ones(3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_predicts_single_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    prediction = engine.predict_window(
        np.zeros(
            (2, 64),
            dtype=np.float32,
        )
    )

    assert 0 <= prediction.predicted_index < 4
    assert prediction.predicted_label in {
        "bpsk",
        "qpsk",
        "8psk",
        "16qam",
    }
    assert 0.0 <= prediction.confidence <= 1.0
    assert len(prediction.logits) == 4
    assert len(prediction.probabilities) == 4
    assert sum(
        prediction.probabilities
    ) == pytest.approx(1.0)


def test_accepts_singleton_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    result = engine.predict_window(
        torch.zeros(1, 2, 64)
    )

    assert result.predicted_label in (
        engine.class_names
    )


def test_rejects_wrong_channel_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    with pytest.raises(
        ValueError,
        match="channel count",
    ):
        engine.predict_batch(
            np.zeros(
                (3, 64),
                dtype=np.float32,
            )
        )


def test_rejects_wrong_sample_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    with pytest.raises(
        ValueError,
        match="sample count",
    ):
        engine.predict_batch(
            np.zeros(
                (2, 63),
                dtype=np.float32,
            )
        )


def test_rejects_nonfinite_input(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    inputs = np.zeros(
        (2, 64),
        dtype=np.float32,
    )
    inputs[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        engine.predict_window(inputs)


@pytest.mark.parametrize(
    "input_scale",
    (
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ),
)
def test_rejects_invalid_input_scale(
    tmp_path: Path,
    input_scale: object,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    with pytest.raises(
        ValueError,
        match="input_scale",
    ):
        IQInferenceEngine.from_checkpoint(
            path,
            device="cpu",
            input_scale=input_scale,
        )


def test_rejects_missing_class_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(
        path,
        include_class_names=False,
    )

    with pytest.raises(
        ValueError,
        match="class_names",
    ):
        IQInferenceEngine.from_checkpoint(
            path,
            device="cpu",
        )


def test_rejects_multiwindow_single_call(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.pt"
    write_checkpoint(path)

    engine = IQInferenceEngine.from_checkpoint(
        path,
        device="cpu",
        expected_sample_count=64,
    )

    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        engine.predict_window(
            np.zeros(
                (2, 2, 64),
                dtype=np.float32,
            )
        )


def test_missing_checkpoint_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        IQInferenceEngine.from_checkpoint(
            tmp_path / "missing.pt",
            device="cpu",
        )
