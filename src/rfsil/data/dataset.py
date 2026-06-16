from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rfsil.data.synthetic import (
    MODULATION_CLASSES,
    MODULATION_TO_LABEL,
    SyntheticExampleConfig,
    generate_synthetic_example,
)

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
Int32Array = NDArray[np.int32]
UInt32Array = NDArray[np.uint32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class DatasetGenerationConfig:
    """Configuration shared by all examples in a synthetic dataset."""

    dataset_name: str
    sample_count: int
    sample_rate_hz: float
    samples_per_symbol: int
    rolloff: float
    span_symbols: int
    snr_values_db: tuple[float, ...]
    frequency_offset_range_hz: tuple[float, float]
    phase_offset_range_rad: tuple[float, float]
    amplitude_scale_range: tuple[float, float]
    time_shift_range_samples: tuple[int, int]
    rayleigh_probability: float

    def __post_init__(self) -> None:
        """Validate the complete dataset configuration."""
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty.")

        _validate_positive_integer(self.sample_count, "sample_count")
        _validate_positive_integer(
            self.samples_per_symbol,
            "samples_per_symbol",
        )
        _validate_positive_integer(
            self.span_symbols,
            "span_symbols",
        )

        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive and finite.")

        if not np.isfinite(self.rolloff) or not 0.0 < self.rolloff <= 1.0:
            raise ValueError("rolloff must be finite and in the interval (0, 1].")

        if not self.snr_values_db:
            raise ValueError("snr_values_db must not be empty.")

        if not np.all(np.isfinite(self.snr_values_db)):
            raise ValueError("snr_values_db must contain only finite values.")

        _validate_float_range(
            self.frequency_offset_range_hz,
            "frequency_offset_range_hz",
        )
        _validate_float_range(
            self.phase_offset_range_rad,
            "phase_offset_range_rad",
        )

        amplitude_minimum, amplitude_maximum = _validate_float_range(
            self.amplitude_scale_range,
            "amplitude_scale_range",
        )

        if amplitude_minimum <= 0.0:
            raise ValueError("amplitude_scale_range must remain positive.")

        _validate_integer_range(
            self.time_shift_range_samples,
            "time_shift_range_samples",
        )

        if (
            not np.isfinite(self.rayleigh_probability)
            or not 0.0 <= self.rayleigh_probability <= 1.0
        ):
            raise ValueError(
                "rayleigh_probability must be finite and in the interval [0, 1]."
            )


@dataclass(frozen=True, slots=True)
class SyntheticDatasetSplit:
    """One balanced synthetic dataset split."""

    iq: Float32Array
    labels: Int64Array
    snr_db: Float32Array
    frequency_offset_hz: Float32Array
    phase_offset_rad: Float32Array
    amplitude_scale: Float32Array
    time_shift_samples: Int32Array
    rayleigh_fading: BoolArray
    example_seed: UInt32Array


def _validate_positive_integer(value: object, name: str) -> int:
    """Validate and return a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")

    validated = int(value)

    if validated <= 0:
        raise ValueError(f"{name} must be positive.")

    return validated


def _validate_float_range(
    value_range: tuple[float, float],
    name: str,
) -> tuple[float, float]:
    """Validate an ordered finite floating-point range."""
    if len(value_range) != 2:
        raise ValueError(f"{name} must contain exactly two values.")

    minimum = float(value_range[0])
    maximum = float(value_range[1])

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"{name} must contain only finite values.")

    if minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed its maximum.")

    return minimum, maximum


def _validate_integer_range(
    value_range: tuple[int, int],
    name: str,
) -> tuple[int, int]:
    """Validate an ordered integer range."""
    if len(value_range) != 2:
        raise ValueError(f"{name} must contain exactly two values.")

    minimum, maximum = value_range

    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, Integral)
        or isinstance(maximum, bool)
        or not isinstance(maximum, Integral)
    ):
        raise ValueError(f"{name} must contain integers.")

    if minimum > maximum:
        raise ValueError(f"{name} minimum must not exceed its maximum.")

    return int(minimum), int(maximum)


def _draw_child_seed(rng: np.random.Generator) -> int:
    """Draw one deterministic unsigned 32-bit seed."""
    return int(
        rng.integers(
            low=0,
            high=np.iinfo(np.uint32).max,
            dtype=np.uint32,
        )
    )


def build_dataset_split(
    configuration: DatasetGenerationConfig,
    examples_per_class_per_snr: int,
    seed: int,
) -> SyntheticDatasetSplit:
    """Build one balanced and deterministically shuffled dataset split."""
    examples_per_group = _validate_positive_integer(
        examples_per_class_per_snr,
        "examples_per_class_per_snr",
    )

    total_examples = (
        len(MODULATION_CLASSES)
        * len(configuration.snr_values_db)
        * examples_per_group
    )

    iq = np.empty(
        (
            total_examples,
            2,
            configuration.sample_count,
        ),
        dtype=np.float32,
    )
    labels = np.empty(total_examples, dtype=np.int64)
    snr_db = np.empty(total_examples, dtype=np.float32)
    frequency_offset_hz = np.empty(total_examples, dtype=np.float32)
    phase_offset_rad = np.empty(total_examples, dtype=np.float32)
    amplitude_scale = np.empty(total_examples, dtype=np.float32)
    time_shift_samples = np.empty(total_examples, dtype=np.int32)
    rayleigh_fading = np.empty(total_examples, dtype=np.bool_)
    example_seed = np.empty(total_examples, dtype=np.uint32)

    frequency_minimum, frequency_maximum = (
        configuration.frequency_offset_range_hz
    )
    phase_minimum, phase_maximum = configuration.phase_offset_range_rad
    amplitude_minimum, amplitude_maximum = configuration.amplitude_scale_range
    time_shift_minimum, time_shift_maximum = (
        configuration.time_shift_range_samples
    )

    rng = np.random.default_rng(seed)
    example_index = 0

    for modulation in MODULATION_CLASSES:
        for requested_snr_db in configuration.snr_values_db:
            for _ in range(examples_per_group):
                selected_seed = _draw_child_seed(rng)
                selected_frequency_offset = float(
                    rng.uniform(
                        frequency_minimum,
                        frequency_maximum,
                    )
                )
                selected_phase_offset = float(
                    rng.uniform(
                        phase_minimum,
                        phase_maximum,
                    )
                )
                selected_amplitude_scale = float(
                    rng.uniform(
                        amplitude_minimum,
                        amplitude_maximum,
                    )
                )
                selected_time_shift = int(
                    rng.integers(
                        low=time_shift_minimum,
                        high=time_shift_maximum + 1,
                    )
                )
                selected_rayleigh_fading = bool(
                    rng.random() < configuration.rayleigh_probability
                )

                example_configuration = SyntheticExampleConfig(
                    sample_count=configuration.sample_count,
                    sample_rate_hz=configuration.sample_rate_hz,
                    samples_per_symbol=configuration.samples_per_symbol,
                    rolloff=configuration.rolloff,
                    span_symbols=configuration.span_symbols,
                    snr_db=float(requested_snr_db),
                    frequency_offset_hz=selected_frequency_offset,
                    phase_offset_rad=selected_phase_offset,
                    amplitude_scale=selected_amplitude_scale,
                    time_shift_samples=selected_time_shift,
                    apply_rayleigh_fading=selected_rayleigh_fading,
                )

                example = generate_synthetic_example(
                    modulation=modulation,
                    configuration=example_configuration,
                    seed=selected_seed,
                )

                iq[example_index, 0] = example.samples.real
                iq[example_index, 1] = example.samples.imag
                labels[example_index] = example.label
                snr_db[example_index] = requested_snr_db
                frequency_offset_hz[example_index] = selected_frequency_offset
                phase_offset_rad[example_index] = selected_phase_offset
                amplitude_scale[example_index] = selected_amplitude_scale
                time_shift_samples[example_index] = selected_time_shift
                rayleigh_fading[example_index] = selected_rayleigh_fading
                example_seed[example_index] = selected_seed

                example_index += 1

    permutation = rng.permutation(total_examples)

    return SyntheticDatasetSplit(
        iq=iq[permutation],
        labels=labels[permutation],
        snr_db=snr_db[permutation],
        frequency_offset_hz=frequency_offset_hz[permutation],
        phase_offset_rad=phase_offset_rad[permutation],
        amplitude_scale=amplitude_scale[permutation],
        time_shift_samples=time_shift_samples[permutation],
        rayleigh_fading=rayleigh_fading[permutation],
        example_seed=example_seed[permutation],
    )


def save_dataset_split(
    dataset_split: SyntheticDatasetSplit,
    output_path: Path,
) -> None:
    """Save one dataset split in compressed NumPy format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        iq=dataset_split.iq,
        labels=dataset_split.labels,
        snr_db=dataset_split.snr_db,
        frequency_offset_hz=dataset_split.frequency_offset_hz,
        phase_offset_rad=dataset_split.phase_offset_rad,
        amplitude_scale=dataset_split.amplitude_scale,
        time_shift_samples=dataset_split.time_shift_samples,
        rayleigh_fading=dataset_split.rayleigh_fading,
        example_seed=dataset_split.example_seed,
    )


def load_dataset_split(
    input_path: Path,
) -> SyntheticDatasetSplit:
    """Load one dataset split without enabling pickle deserialization."""
    with np.load(input_path, allow_pickle=False) as data:
        return SyntheticDatasetSplit(
            iq=data["iq"].astype(np.float32),
            labels=data["labels"].astype(np.int64),
            snr_db=data["snr_db"].astype(np.float32),
            frequency_offset_hz=data[
                "frequency_offset_hz"
            ].astype(np.float32),
            phase_offset_rad=data["phase_offset_rad"].astype(np.float32),
            amplitude_scale=data["amplitude_scale"].astype(np.float32),
            time_shift_samples=data[
                "time_shift_samples"
            ].astype(np.int32),
            rayleigh_fading=data["rayleigh_fading"].astype(np.bool_),
            example_seed=data["example_seed"].astype(np.uint32),
        )


def write_dataset_manifest(
    configuration: DatasetGenerationConfig,
    split_files: dict[str, Path],
    split_sizes: dict[str, int],
    output_path: Path,
) -> None:
    """Write a human-readable dataset manifest."""
    manifest: dict[str, Any] = {
        "format_version": 1,
        "dataset_name": configuration.dataset_name,
        "classes": [
            modulation.value
            for modulation in MODULATION_CLASSES
        ],
        "label_map": {
            modulation.value: label
            for modulation, label in MODULATION_TO_LABEL.items()
        },
        "configuration": asdict(configuration),
        "splits": {
            split_name: {
                "file": str(split_files[split_name]),
                "example_count": split_sizes[split_name],
            }
            for split_name in split_files
        },
        "tensor_format": {
            "iq_shape": "[examples, 2, samples]",
            "channel_0": "in_phase",
            "channel_1": "quadrature",
            "iq_dtype": "float32",
            "label_dtype": "int64",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "DatasetGenerationConfig",
    "SyntheticDatasetSplit",
    "build_dataset_split",
    "load_dataset_split",
    "save_dataset_split",
    "write_dataset_manifest",
]
