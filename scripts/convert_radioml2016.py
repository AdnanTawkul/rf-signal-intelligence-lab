from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfsil.data.dataset import save_dataset_split
from rfsil.data.radioml2016 import (
    RADIOML_TO_PROJECT,
    build_radioml2016_four_class_splits,
    load_radioml2016_dictionary,
)
from rfsil.data.synthetic import MODULATION_TO_LABEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert RadioML 2016.10A into the "
            "project's canonical NPZ format."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the conversion YAML configuration.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def serialize_project_path(path: Path) -> str:
    """Serialize a path relative to the repository when possible."""
    resolved = path.resolve()

    try:
        return resolved.relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_configuration(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(
            "Conversion configuration must be a mapping."
        )

    return content


def hash_file(
    path: Path,
    algorithm: str,
) -> str:
    """Calculate a file hash without loading the whole file."""
    digest = hashlib.new(algorithm)

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def validate_source_content(
    content: object,
) -> Mapping[str, object]:
    """Validate the source configuration mapping."""
    if not isinstance(content, Mapping):
        raise ValueError(
            "source must be a YAML mapping."
        )

    required = {
        "dataset_name",
        "archive_path",
        "pickle_path",
        "expected_archive_md5",
        "license",
    }

    missing = required - set(content)

    if missing:
        raise ValueError(
            "Missing source configuration values: "
            f"{sorted(missing)}"
        )

    return content


def validate_split_content(
    content: object,
) -> dict[str, int]:
    """Validate split counts from YAML."""
    if not isinstance(content, Mapping):
        raise ValueError(
            "splits must be a YAML mapping."
        )

    result: dict[str, int] = {}

    for name in (
        "train",
        "validation",
        "test",
    ):
        value = content.get(name)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"splits.{name} must be "
                "a positive integer."
            )

        result[name] = value

    if set(content) != set(result):
        raise ValueError(
            "splits must contain exactly train, "
            "validation, and test."
        )

    return result


def build_group_counts(
    labels: np.ndarray,
    snr_db: np.ndarray,
    snr_values: tuple[int, ...],
) -> dict[str, dict[str, int]]:
    """Count examples for every class and SNR pair."""
    result: dict[str, dict[str, int]] = {}

    for source_name, modulation in RADIOML_TO_PROJECT:
        label = MODULATION_TO_LABEL[modulation]

        result[source_name] = {
            str(snr): int(
                np.sum(
                    (labels == label)
                    & (snr_db == float(snr))
                )
            )
            for snr in snr_values
        }

    return result


def main() -> None:
    """Convert and save the selected RadioML subset."""
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_configuration(config_path)

    dataset_name = str(
        content["dataset_name"]
    ).strip()

    if not dataset_name:
        raise ValueError(
            "dataset_name must not be empty."
        )

    seed = int(content["seed"])
    source = validate_source_content(
        content["source"]
    )
    split_counts = validate_split_content(
        content["splits"]
    )

    archive_path = resolve_project_path(
        str(source["archive_path"])
    )
    pickle_path = resolve_project_path(
        str(source["pickle_path"])
    )
    output_directory = resolve_project_path(
        str(content["output_directory"])
    )

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    if not pickle_path.is_file():
        raise FileNotFoundError(pickle_path)

    expected_md5 = str(
        source["expected_archive_md5"]
    ).strip().lower()
    actual_md5 = hash_file(
        archive_path,
        "md5",
    )

    if actual_md5 != expected_md5:
        raise ValueError(
            "Archive MD5 mismatch: "
            f"expected {expected_md5}, "
            f"found {actual_md5}."
        )

    print("Archive MD5 verification: OK")
    print(f"Loading: {pickle_path}")

    groups = load_radioml2016_dictionary(
        pickle_path
    )

    splits, snr_values = (
        build_radioml2016_four_class_splits(
            groups,
            split_counts=split_counts,
            seed=seed,
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_files: dict[str, Path] = {}

    for split_name, dataset_split in splits.items():
        output_path = (
            output_directory
            / f"{split_name}.npz"
        )

        save_dataset_split(
            dataset_split,
            output_path,
        )
        split_files[split_name] = output_path

        print(
            f"Saved {split_name}: "
            f"{dataset_split.iq.shape}"
        )

    class_mapping = {
        source_name: {
            "project_name": modulation.value,
            "display_name": modulation.value.upper(),
            "label": int(
                MODULATION_TO_LABEL[modulation]
            ),
        }
        for source_name, modulation
        in RADIOML_TO_PROJECT
    }

    manifest = {
        "format_version": 1,
        "dataset_name": dataset_name,
        "source": {
            "dataset_name": str(
                source["dataset_name"]
            ),
            "archive_path": serialize_project_path(
                archive_path
            ),
            "pickle_path": serialize_project_path(
                pickle_path
            ),
            "download_url": source.get(
                "download_url"
            ),
            "license": str(source["license"]),
            "archive_hashes": {
                "md5": actual_md5,
                "sha256": hash_file(
                    archive_path,
                    "sha256",
                ),
            },
            "pickle_hashes": {
                "md5": hash_file(
                    pickle_path,
                    "md5",
                ),
                "sha256": hash_file(
                    pickle_path,
                    "sha256",
                ),
            },
            "total_group_count": len(groups),
            "total_example_count": int(
                sum(
                    group.shape[0]
                    for group in groups.values()
                )
            ),
        },
        "selection": {
            "source_to_project_mapping": (
                class_mapping
            ),
            "selected_group_count": (
                len(RADIOML_TO_PROJECT)
                * len(snr_values)
            ),
            "selected_example_count": int(
                sum(
                    split.iq.shape[0]
                    for split in splits.values()
                )
            ),
            "snr_values_db": list(snr_values),
            "samples_per_example": int(
                next(iter(splits.values()))
                .iq.shape[2]
            ),
        },
        "split_policy": {
            "type": (
                "deterministic_stratified_"
                "within_class_snr_group"
            ),
            "seed": seed,
            "examples_per_class_snr": (
                split_counts
            ),
        },
        "splits": {
            split_name: {
                "file": serialize_project_path(
                    split_files[split_name]
                ),
                "example_count": int(
                    dataset_split.iq.shape[0]
                ),
                "iq_shape": list(
                    dataset_split.iq.shape
                ),
                "group_counts": build_group_counts(
                    dataset_split.labels,
                    dataset_split.snr_db,
                    snr_values,
                ),
            }
            for split_name, dataset_split
            in splits.items()
        },
        "tensor_format": {
            "iq_shape": (
                "[examples, 2, samples]"
            ),
            "channel_0": "in_phase",
            "channel_1": "quadrature",
            "iq_dtype": "float32",
            "label_dtype": "int64",
        },
        "metadata_availability": {
            "snr_db": (
                "provided by the source dataset"
            ),
            "frequency_offset_hz": (
                "unavailable; compatibility "
                "value set to 0.0"
            ),
            "phase_offset_rad": (
                "unavailable; compatibility "
                "value set to 0.0"
            ),
            "amplitude_scale": (
                "unavailable; compatibility "
                "value set to 1.0"
            ),
            "time_shift_samples": (
                "unavailable; compatibility "
                "value set to 0"
            ),
            "rayleigh_fading": (
                "unavailable; compatibility "
                "value set to false"
            ),
            "example_seed": (
                "deterministic compatibility "
                "identifier; not a source "
                "generation seed"
            ),
        },
    }

    manifest_path = (
        output_directory / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Manifest: {manifest_path}")
    print(
        "RadioML 2016 four-class conversion: OK"
    )


if __name__ == "__main__":
    main()
