from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from rfsil.data.dataset import SyntheticDatasetSplit, load_dataset_split


class IQDatasetItem(TypedDict):
    """One labeled RF IQ example returned by the PyTorch dataset."""

    iq: Tensor
    label: Tensor
    snr_db: Tensor
    frequency_offset_hz: Tensor
    phase_offset_rad: Tensor
    amplitude_scale: Tensor
    time_shift_samples: Tensor
    rayleigh_fading: Tensor
    example_seed: Tensor


@dataclass(frozen=True, slots=True)
class DataLoaderConfig:
    """Configuration for a reproducible PyTorch DataLoader."""

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate DataLoader settings."""
        _validate_positive_integer(self.batch_size, "batch_size")
        _validate_nonnegative_integer(self.num_workers, "num_workers")

        if not isinstance(self.shuffle, bool):
            raise ValueError("shuffle must be a boolean.")

        if not isinstance(self.pin_memory, bool):
            raise ValueError("pin_memory must be a boolean.")

        if not isinstance(self.drop_last, bool):
            raise ValueError("drop_last must be a boolean.")

        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer.")


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated


def _validate_nonnegative_integer(value: object, name: str) -> int:
    """Validate and return a nonnegative integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated < 0:
        raise ValueError(f"{name} must be nonnegative.")

    return validated


class NPZIQDataset(Dataset[IQDatasetItem]):
    """Eagerly loaded PyTorch dataset backed by one generated NPZ split.

    This implementation is appropriate for smoke tests and moderate datasets.
    Before scaling to multi-gigabyte datasets, storage will be changed to
    sharded files so training does not require loading one huge split at once.
    """

    def __init__(self, input_path: str | Path) -> None:
        """Load and validate one synthetic dataset split."""
        self.input_path = Path(input_path)

        if not self.input_path.is_file():
            raise FileNotFoundError(
                f"Dataset split does not exist: {self.input_path}"
            )

        dataset_split = load_dataset_split(self.input_path)
        self._validate_split(dataset_split)

        self.iq = torch.from_numpy(
            np.ascontiguousarray(dataset_split.iq)
        )
        self.labels = torch.from_numpy(
            np.ascontiguousarray(dataset_split.labels)
        )
        self.snr_db = torch.from_numpy(
            np.ascontiguousarray(dataset_split.snr_db)
        )
        self.frequency_offset_hz = torch.from_numpy(
            np.ascontiguousarray(dataset_split.frequency_offset_hz)
        )
        self.phase_offset_rad = torch.from_numpy(
            np.ascontiguousarray(dataset_split.phase_offset_rad)
        )
        self.amplitude_scale = torch.from_numpy(
            np.ascontiguousarray(dataset_split.amplitude_scale)
        )
        self.time_shift_samples = torch.from_numpy(
            np.ascontiguousarray(dataset_split.time_shift_samples)
        )
        self.rayleigh_fading = torch.from_numpy(
            np.ascontiguousarray(dataset_split.rayleigh_fading)
        )
        self.example_seed = torch.from_numpy(
            np.ascontiguousarray(
                dataset_split.example_seed.astype(np.int64)
            )
        )

    @staticmethod
    def _validate_split(dataset_split: SyntheticDatasetSplit) -> None:
        """Validate tensor shapes before exposing the dataset."""
        if dataset_split.iq.ndim != 3:
            raise ValueError(
                "IQ data must have shape [examples, channels, samples]."
            )

        if dataset_split.iq.shape[1] != 2:
            raise ValueError(
                "IQ data must have exactly two channels: I and Q."
            )

        example_count = dataset_split.iq.shape[0]

        metadata_arrays = {
            "labels": dataset_split.labels,
            "snr_db": dataset_split.snr_db,
            "frequency_offset_hz": dataset_split.frequency_offset_hz,
            "phase_offset_rad": dataset_split.phase_offset_rad,
            "amplitude_scale": dataset_split.amplitude_scale,
            "time_shift_samples": dataset_split.time_shift_samples,
            "rayleigh_fading": dataset_split.rayleigh_fading,
            "example_seed": dataset_split.example_seed,
        }

        for name, values in metadata_arrays.items():
            if values.shape != (example_count,):
                raise ValueError(
                    f"{name} must have shape ({example_count},)."
                )

        if not np.all(np.isfinite(dataset_split.iq)):
            raise ValueError("IQ data must contain only finite values.")

        if not np.all(np.isfinite(dataset_split.snr_db)):
            raise ValueError("SNR values must contain only finite values.")

    def __len__(self) -> int:
        """Return the number of examples."""
        return int(self.iq.shape[0])

    def __getitem__(self, index: int) -> IQDatasetItem:
        """Return one example and its metadata."""
        return {
            "iq": self.iq[index],
            "label": self.labels[index],
            "snr_db": self.snr_db[index],
            "frequency_offset_hz": self.frequency_offset_hz[index],
            "phase_offset_rad": self.phase_offset_rad[index],
            "amplitude_scale": self.amplitude_scale[index],
            "time_shift_samples": self.time_shift_samples[index],
            "rayleigh_fading": self.rayleigh_fading[index],
            "example_seed": self.example_seed[index],
        }


def create_data_loader(
    dataset: Dataset[IQDatasetItem],
    configuration: DataLoaderConfig,
) -> DataLoader[IQDatasetItem]:
    """Create a reproducibly seeded PyTorch DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(int(configuration.seed))

    return DataLoader(
        dataset,
        batch_size=int(configuration.batch_size),
        shuffle=configuration.shuffle,
        num_workers=int(configuration.num_workers),
        pin_memory=configuration.pin_memory,
        drop_last=configuration.drop_last,
        generator=generator,
    )


__all__ = [
    "DataLoaderConfig",
    "IQDatasetItem",
    "NPZIQDataset",
    "create_data_loader",
]
