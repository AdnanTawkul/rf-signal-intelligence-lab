from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rfsil.data.dataset import (
    DatasetGenerationConfig,
    build_dataset_split,
    save_dataset_split,
)
from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)


def create_split_file(tmp_path: Path) -> Path:
    """Create a small generated split for PyTorch loader tests."""
    configuration = DatasetGenerationConfig(
        dataset_name="torch_loader_test",
        sample_count=128,
        sample_rate_hz=1_000_000.0,
        samples_per_symbol=8,
        rolloff=0.35,
        span_symbols=8,
        snr_values_db=(-4.0, 8.0),
        frequency_offset_range_hz=(-2_000.0, 2_000.0),
        phase_offset_range_rad=(-0.5, 0.5),
        amplitude_scale_range=(0.8, 1.2),
        time_shift_range_samples=(-2, 2),
        rayleigh_probability=0.25,
    )

    dataset_split = build_dataset_split(
        configuration=configuration,
        examples_per_class_per_snr=1,
        seed=42,
    )

    output_path = tmp_path / "split.npz"
    save_dataset_split(dataset_split, output_path)

    return output_path


def test_npz_iq_dataset_length_and_item_types(tmp_path: Path) -> None:
    dataset = NPZIQDataset(create_split_file(tmp_path))

    assert len(dataset) == 8

    item = dataset[0]

    assert item["iq"].shape == (2, 128)
    assert item["iq"].dtype == torch.float32
    assert item["label"].dtype == torch.int64
    assert item["snr_db"].dtype == torch.float32
    assert item["frequency_offset_hz"].dtype == torch.float32
    assert item["phase_offset_rad"].dtype == torch.float32
    assert item["amplitude_scale"].dtype == torch.float32
    assert item["time_shift_samples"].dtype == torch.int32
    assert item["rayleigh_fading"].dtype == torch.bool
    assert item["example_seed"].dtype == torch.int64


def test_data_loader_returns_expected_batch_shapes(tmp_path: Path) -> None:
    dataset = NPZIQDataset(create_split_file(tmp_path))

    loader = create_data_loader(
        dataset,
        DataLoaderConfig(
            batch_size=4,
            shuffle=False,
            pin_memory=False,
        ),
    )

    batch = next(iter(loader))

    assert batch["iq"].shape == (4, 2, 128)
    assert batch["label"].shape == (4,)
    assert batch["snr_db"].shape == (4,)
    assert batch["iq"].dtype == torch.float32
    assert batch["label"].dtype == torch.int64


def test_seeded_shuffle_is_reproducible(tmp_path: Path) -> None:
    dataset = NPZIQDataset(create_split_file(tmp_path))
    configuration = DataLoaderConfig(
        batch_size=8,
        shuffle=True,
        pin_memory=False,
        seed=123,
    )

    batch_a = next(
        iter(create_data_loader(dataset, configuration))
    )
    batch_b = next(
        iter(create_data_loader(dataset, configuration))
    )

    torch.testing.assert_close(
        batch_a["example_seed"],
        batch_b["example_seed"],
    )


def test_different_loader_seeds_change_shuffle_order(
    tmp_path: Path,
) -> None:
    dataset = NPZIQDataset(create_split_file(tmp_path))

    batch_a = next(
        iter(
            create_data_loader(
                dataset,
                DataLoaderConfig(
                    batch_size=8,
                    shuffle=True,
                    pin_memory=False,
                    seed=1,
                ),
            )
        )
    )
    batch_b = next(
        iter(
            create_data_loader(
                dataset,
                DataLoaderConfig(
                    batch_size=8,
                    shuffle=True,
                    pin_memory=False,
                    seed=2,
                ),
            )
        )
    )

    assert not torch.equal(
        batch_a["example_seed"],
        batch_b["example_seed"],
    )


def test_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        NPZIQDataset(tmp_path / "missing.npz")


@pytest.mark.parametrize(
    "invalid_configuration",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"num_workers": -1},
        {"shuffle": 1},
        {"pin_memory": 1},
        {"drop_last": 1},
        {"seed": True},
    ],
)
def test_data_loader_config_rejects_invalid_values(
    invalid_configuration: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(
            **invalid_configuration,  # type: ignore[arg-type]
        )


def test_iq_tensor_matches_saved_numpy_values(tmp_path: Path) -> None:
    split_path = create_split_file(tmp_path)

    with np.load(split_path, allow_pickle=False) as data:
        expected_iq = data["iq"][0].copy()

    dataset = NPZIQDataset(split_path)

    np.testing.assert_array_equal(
        dataset[0]["iq"].numpy(),
        expected_iq,
    )
