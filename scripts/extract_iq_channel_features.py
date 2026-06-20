from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfsil.data.dataset import (
    SyntheticDatasetSplit,
    load_dataset_split,
)
from rfsil.evaluation.iq_channel_features import (
    compute_iq_channel_features,
)
from rfsil.evaluation.iq_feature_artifacts import (
    IQChannelFeatureArtifact,
    save_iq_channel_feature_artifact,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract versioned IQ channel "
            "features from held-out datasets."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml_mapping(
    path: Path,
) -> dict[str, Any]:
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Configuration must be a mapping."
        )

    return content


def verify_paired_metadata(
    reference: SyntheticDatasetSplit,
    candidate: SyntheticDatasetSplit,
    *,
    condition: str,
) -> None:
    arrays = {
        "labels": (
            reference.labels,
            candidate.labels,
        ),
        "snr_db": (
            reference.snr_db,
            candidate.snr_db,
        ),
        "frequency_offset_hz": (
            reference.frequency_offset_hz,
            candidate.frequency_offset_hz,
        ),
        "phase_offset_rad": (
            reference.phase_offset_rad,
            candidate.phase_offset_rad,
        ),
        "amplitude_scale": (
            reference.amplitude_scale,
            candidate.amplitude_scale,
        ),
        "time_shift_samples": (
            reference.time_shift_samples,
            candidate.time_shift_samples,
        ),
        "rayleigh_fading": (
            reference.rayleigh_fading,
            candidate.rayleigh_fading,
        ),
        "example_seed": (
            reference.example_seed,
            candidate.example_seed,
        ),
    }

    for name, (
        reference_values,
        candidate_values,
    ) in arrays.items():
        if not np.array_equal(
            reference_values,
            candidate_values,
        ):
            raise ValueError(
                "Dataset metadata is not "
                f"paired for condition "
                f"{condition!r}: {name}."
            )


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml_mapping(
        config_path
    )

    conditions = content["conditions"]
    feature_config = content["features"]
    output_config = content["output"]

    if (
        not isinstance(conditions, dict)
        or not conditions
    ):
        raise ValueError(
            "conditions must be a "
            "non-empty mapping."
        )

    output_directory = (
        resolve_project_path(
            output_config["directory"]
        )
    )
    artifact_filename = str(
        output_config[
            "artifact_filename"
        ]
    )
    manifest_path = (
        output_directory
        / str(
            output_config[
                "manifest_filename"
            ]
        )
    )

    autocorrelation_lags = tuple(
        int(value)
        for value in feature_config[
            "autocorrelation_lags"
        ]
    )
    occupancy_fraction = float(
        feature_config[
            "occupancy_fraction"
        ]
    )
    epsilon = float(
        feature_config["epsilon"]
    )

    reference_dataset = None
    reference_feature_names = None
    manifest_records = []

    for condition, condition_config in (
        conditions.items()
    ):
        dataset_path = resolve_project_path(
            condition_config["test_path"]
        )
        dataset = load_dataset_split(
            dataset_path
        )

        if reference_dataset is None:
            reference_dataset = dataset
        else:
            verify_paired_metadata(
                reference_dataset,
                dataset,
                condition=str(condition),
            )

        features = (
            compute_iq_channel_features(
                dataset.iq,
                autocorrelation_lags=(
                    autocorrelation_lags
                ),
                occupancy_fraction=(
                    occupancy_fraction
                ),
                epsilon=epsilon,
            )
        )

        if reference_feature_names is None:
            reference_feature_names = (
                features.feature_names
            )
        elif (
            features.feature_names
            != reference_feature_names
        ):
            raise RuntimeError(
                "Feature names differ "
                "between conditions."
            )

        artifact = IQChannelFeatureArtifact(
            values=features.values,
            feature_names=(
                features.feature_names
            ),
            labels=dataset.labels,
            snr_db=dataset.snr_db,
            frequency_offset_hz=(
                dataset.frequency_offset_hz
            ),
            phase_offset_rad=(
                dataset.phase_offset_rad
            ),
            amplitude_scale=(
                dataset.amplitude_scale
            ),
            time_shift_samples=(
                dataset.time_shift_samples
            ),
            rayleigh_fading=(
                dataset.rayleigh_fading
            ),
            example_seed=(
                dataset.example_seed
            ),
            condition=str(condition),
            source_dataset=(
                dataset_path
                .resolve()
                .as_posix()
            ),
        )

        artifact_path = (
            output_directory
            / str(condition)
            / artifact_filename
        )
        save_iq_channel_feature_artifact(
            artifact,
            artifact_path,
        )

        record = artifact.summary()
        record["artifact_path"] = (
            artifact_path
            .resolve()
            .as_posix()
        )
        manifest_records.append(record)

        print(
            f"{condition}: "
            f"{artifact.example_count} ? "
            f"{artifact.feature_count} -> "
            f"{artifact_path}"
        )

    manifest = {
        "format_version": 1,
        "experiment_name": str(
            content["experiment_name"]
        ),
        "config_path": (
            config_path
            .resolve()
            .as_posix()
        ),
        "condition_count": len(
            manifest_records
        ),
        "feature_count": len(
            reference_feature_names or ()
        ),
        "feature_names": list(
            reference_feature_names or ()
        ),
        "conditions": manifest_records,
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
