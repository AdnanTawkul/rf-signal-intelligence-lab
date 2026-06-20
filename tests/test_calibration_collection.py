from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)

from rfsil.evaluation.classification import (
    collect_calibration_predictions,
    collect_predictions,
)


class CalibrationDataset(Dataset):
    """Small deterministic classification dataset."""

    amplitudes = (-1.0, 0.5, 2.0)
    labels = (0, 1, 1)
    snr_values = (-4.0, 0.0, 8.0)

    def __len__(self) -> int:
        return len(self.amplitudes)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        return {
            "iq": torch.full(
                (2, 8),
                self.amplitudes[index],
                dtype=torch.float32,
            ),
            "label": torch.tensor(
                self.labels[index],
                dtype=torch.int64,
            ),
            "snr_db": torch.tensor(
                self.snr_values[index],
                dtype=torch.float32,
            ),
        }


class LinearLogitModel(nn.Module):
    """Generate deterministic three-class logits."""

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        score = inputs[
            :,
            0,
            :,
        ].mean(dim=1)

        return torch.stack(
            (
                -score,
                score,
                torch.zeros_like(score),
            ),
            dim=1,
        )


class InvalidShapeModel(nn.Module):
    """Return invalid one-dimensional logits."""

    def forward(
        self,
        inputs: Tensor,
    ) -> Tensor:
        return inputs[
            :,
            0,
            :,
        ].mean(dim=1)


class EmptyDataset(Dataset):
    """Dataset containing no examples."""

    def __len__(self) -> int:
        return 0

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        raise IndexError(index)


def create_loader(
    *,
    batch_size: int = 2,
) -> DataLoader:
    return DataLoader(
        CalibrationDataset(),
        batch_size=batch_size,
        shuffle=False,
    )


def test_collects_logits_and_probabilities() -> None:
    result = (
        collect_calibration_predictions(
            model=LinearLogitModel(),
            data_loader=create_loader(),
            device=torch.device("cpu"),
            class_names=(
                "bpsk",
                "qpsk",
                "8psk",
            ),
        )
    )

    assert result.example_count == 3
    assert result.class_count == 3
    assert result.class_names == (
        "bpsk",
        "qpsk",
        "8psk",
    )

    np.testing.assert_array_equal(
        result.labels,
        np.asarray(
            [0, 1, 1],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        result.predictions,
        np.asarray(
            [0, 1, 1],
            dtype=np.int64,
        ),
    )
    np.testing.assert_allclose(
        result.probabilities.sum(axis=1),
        np.ones(3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_collects_snr_values() -> None:
    result = (
        collect_calibration_predictions(
            model=LinearLogitModel(),
            data_loader=create_loader(),
            device=torch.device("cpu"),
        )
    )

    assert result.snr_db is not None

    np.testing.assert_allclose(
        result.snr_db,
        [-4.0, 0.0, 8.0],
    )


def test_input_scale_changes_logits() -> None:
    unscaled = (
        collect_calibration_predictions(
            model=LinearLogitModel(),
            data_loader=create_loader(),
            device=torch.device("cpu"),
            input_scale=1.0,
        )
    )
    scaled = (
        collect_calibration_predictions(
            model=LinearLogitModel(),
            data_loader=create_loader(),
            device=torch.device("cpu"),
            input_scale=2.0,
        )
    )

    np.testing.assert_allclose(
        scaled.logits[:, :2],
        2.0 * unscaled.logits[:, :2],
    )


def test_legacy_collection_remains_compatible() -> None:
    result = collect_predictions(
        model=LinearLogitModel(),
        data_loader=create_loader(),
        device=torch.device("cpu"),
    )

    assert result.labels.dtype == np.int64
    assert result.predictions.dtype == np.int64
    assert result.snr_db.dtype == np.float32

    np.testing.assert_array_equal(
        result.predictions,
        [0, 1, 1],
    )


def test_rejects_invalid_model_output_shape() -> None:
    with pytest.raises(
        ValueError,
        match="Model logits must have shape",
    ):
        collect_calibration_predictions(
            model=InvalidShapeModel(),
            data_loader=create_loader(),
            device=torch.device("cpu"),
        )


def test_rejects_empty_loader() -> None:
    loader = DataLoader(
        EmptyDataset(),
        batch_size=2,
        shuffle=False,
    )

    with pytest.raises(
        ValueError,
        match="produced no examples",
    ):
        collect_calibration_predictions(
            model=LinearLogitModel(),
            data_loader=loader,
            device=torch.device("cpu"),
        )
