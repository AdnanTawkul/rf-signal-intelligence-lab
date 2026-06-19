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
    collect_predictions,
)


class ConstantIQDataset(Dataset):
    """One deterministic IQ example."""

    def __len__(self) -> int:
        return 1

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor]:
        if index != 0:
            raise IndexError(index)

        return {
            "iq": torch.full(
                (2, 8),
                0.4,
                dtype=torch.float32,
            ),
            "label": torch.tensor(
                1,
                dtype=torch.int64,
            ),
            "snr_db": torch.tensor(
                0.0,
                dtype=torch.float32,
            ),
        }


class AmplitudeThresholdModel(nn.Module):
    """Switch class when scaled amplitude crosses 0.5."""

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
                0.5 - score,
                score - 0.5,
            ),
            dim=1,
        )


def create_loader() -> DataLoader:
    return DataLoader(
        ConstantIQDataset(),
        batch_size=1,
        shuffle=False,
    )


def test_collect_predictions_applies_input_scale() -> None:
    model = AmplitudeThresholdModel()
    device = torch.device("cpu")

    unscaled = collect_predictions(
        model=model,
        data_loader=create_loader(),
        device=device,
    )
    explicitly_unscaled = collect_predictions(
        model=model,
        data_loader=create_loader(),
        device=device,
        input_scale=1.0,
    )
    scaled = collect_predictions(
        model=model,
        data_loader=create_loader(),
        device=device,
        input_scale=2.0,
    )

    np.testing.assert_array_equal(
        unscaled.predictions,
        np.asarray([0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        explicitly_unscaled.predictions,
        unscaled.predictions,
    )
    np.testing.assert_array_equal(
        scaled.predictions,
        np.asarray([1], dtype=np.int64),
    )


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
def test_collect_predictions_rejects_invalid_scale(
    input_scale: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="input_scale must be",
    ):
        collect_predictions(
            model=AmplitudeThresholdModel(),
            data_loader=create_loader(),
            device=torch.device("cpu"),
            input_scale=input_scale,
        )
