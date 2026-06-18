from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfsil.data.dataset import (
    DatasetGenerationConfig,
    build_dataset_split,
    save_dataset_split,
    write_dataset_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a reproducible synthetic RF IQ dataset.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML dataset configuration.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load and validate a YAML mapping."""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError("Dataset configuration must be a YAML mapping.")

    return content


def build_configuration(
    content: dict[str, Any],
) -> DatasetGenerationConfig:
    """Convert YAML values into a validated dataset configuration."""
    return DatasetGenerationConfig(
        dataset_name=str(content["dataset_name"]),
        sample_count=int(content["sample_count"]),
        sample_rate_hz=float(content["sample_rate_hz"]),
        samples_per_symbol=int(content["samples_per_symbol"]),
        rolloff=float(content["rolloff"]),
        span_symbols=int(content["span_symbols"]),
        snr_values_db=tuple(
            float(value)
            for value in content["snr_values_db"]
        ),
        frequency_offset_range_hz=tuple(
            float(value)
            for value in content["frequency_offset_range_hz"]
        ),
        phase_offset_range_rad=tuple(
            float(value)
            for value in content["phase_offset_range_rad"]
        ),
        amplitude_scale_range=tuple(
            float(value)
            for value in content["amplitude_scale_range"]
        ),
        time_shift_range_samples=tuple(
            int(value)
            for value in content["time_shift_range_samples"]
        ),
        rayleigh_probability=float(content["rayleigh_probability"]),

        multipath_profile=(

            None

            if content.get("multipath_profile") is None

            else str(content["multipath_profile"])

        ),
    )


def main() -> None:
    arguments = parse_arguments()

    config_path = arguments.config

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    content = load_yaml(config_path)
    configuration = build_configuration(content)

    output_directory = PROJECT_ROOT / str(
        content["output_directory"]
    )
    split_plan = {
        str(name): int(count)
        for name, count in content["splits"].items()
    }

    root_seed = int(content["seed"])
    seed_sequence = np.random.SeedSequence(root_seed)
    split_seed_sequences = seed_sequence.spawn(len(split_plan))

    split_files: dict[str, Path] = {}
    split_sizes: dict[str, int] = {}

    print(f"Dataset: {configuration.dataset_name}")
    print(f"Output directory: {output_directory}")
    print("Classes: 4")
    print(f"SNR levels: {len(configuration.snr_values_db)}")

    for (
        split_name,
        examples_per_group,
    ), split_seed_sequence in zip(
        split_plan.items(),
        split_seed_sequences,
        strict=True,
    ):
        split_seed = int(
            split_seed_sequence.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )

        dataset_split = build_dataset_split(
            configuration=configuration,
            examples_per_class_per_snr=examples_per_group,
            seed=split_seed,
        )

        output_path = output_directory / f"{split_name}.npz"
        save_dataset_split(dataset_split, output_path)

        split_files[split_name] = output_path.relative_to(
            PROJECT_ROOT
        )
        split_sizes[split_name] = len(dataset_split.labels)

        print(
            f"{split_name}: "
            f"examples={len(dataset_split.labels)}, "
            f"iq_shape={dataset_split.iq.shape}, "
            f"file={output_path}"
        )

    manifest_path = output_directory / "manifest.json"

    write_dataset_manifest(
        configuration=configuration,
        split_files=split_files,
        split_sizes=split_sizes,
        output_path=manifest_path,
    )

    total_examples = sum(split_sizes.values())

    print(f"Total examples: {total_examples}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
