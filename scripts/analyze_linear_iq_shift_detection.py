from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfsil.evaluation.iq_feature_artifacts import (
    IQChannelFeatureArtifact,
    load_iq_channel_feature_artifact,
)
from rfsil.evaluation.linear_shift_detector import (
    analyze_linear_iq_shift_detection,
)
from rfsil.evaluation.paired_shift_split import (
    create_paired_shift_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and evaluate a standardized "
            "linear IQ shift detector."
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


def verify_paired_artifacts(
    reference: IQChannelFeatureArtifact,
    candidate: IQChannelFeatureArtifact,
) -> None:
    if (
        reference.feature_names
        != candidate.feature_names
    ):
        raise ValueError(
            "Feature names differ between "
            "conditions."
        )

    for name in (
        "labels",
        "snr_db",
        "frequency_offset_hz",
        "phase_offset_rad",
        "amplitude_scale",
        "time_shift_samples",
        "rayleigh_fading",
        "example_seed",
    ):
        if not np.array_equal(
            getattr(reference, name),
            getattr(candidate, name),
        ):
            raise ValueError(
                "Feature artifact metadata "
                f"is not paired: {name}."
            )


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml_mapping(
        config_path
    )

    input_config = content["input"]
    split_config = content["split"]
    detector_config = content["detector"]
    analysis_config = content["analysis"]
    output_config = content["output"]

    root = resolve_project_path(
        input_config["directory"]
    )
    filename = str(
        input_config["artifact_filename"]
    )
    clean_condition = str(
        input_config["clean_condition"]
    )
    shifted_conditions = tuple(
        str(value)
        for value in input_config[
            "shifted_conditions"
        ]
    )

    clean_path = (
        root
        / clean_condition
        / filename
    )
    clean = (
        load_iq_channel_feature_artifact(
            clean_path
        )
    )

    shifted = {}

    for condition in shifted_conditions:
        artifact = (
            load_iq_channel_feature_artifact(
                root
                / condition
                / filename
            )
        )
        verify_paired_artifacts(
            clean,
            artifact,
        )
        shifted[condition] = artifact

    split = create_paired_shift_split(
        clean.labels,
        clean.snr_db,
        clean.example_seed,
        development_fraction=float(
            split_config[
                "development_fraction"
            ]
        ),
        split_seed=int(
            split_config["split_seed"]
        ),
        snr_decimals=int(
            split_config["snr_decimals"]
        ),
    )

    analysis = (
        analyze_linear_iq_shift_detection(
            clean.values,
            {
                condition: artifact.values
                for condition, artifact
                in shifted.items()
            },
            clean.feature_names,
            clean.example_seed,
            split,
            l2_strengths=tuple(
                detector_config[
                    "l2_strengths"
                ]
            ),
            fold_count=int(
                detector_config[
                    "fold_count"
                ]
            ),
            cv_seed=int(
                detector_config["cv_seed"]
            ),
            target_tpr=float(
                analysis_config["target_tpr"]
            ),
        )
    )

    payload = {
        "format_version": 1,
        "experiment_name": str(
            content["experiment_name"]
        ),
        "config_path": (
            config_path
            .resolve()
            .as_posix()
        ),
        "analysis": analysis.to_dict(),
    }

    output_directory = resolve_project_path(
        output_config["directory"]
    )
    output_path = (
        output_directory
        / str(
            output_config[
                "summary_filename"
            ]
        )
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Cross-validation candidates")
    print("=" * 88)
    print(
        "L2 strength | Mean AUROC | "
        "Worst AUROC | Mean AP | Mean FPR95"
    )
    print("-" * 88)

    for candidate in (
        analysis.candidates
    ):
        print(
            f"{candidate.l2_strength:11.6g} | "
            f"{candidate.mean_auroc:.4f}     | "
            f"{candidate.minimum_auroc:.4f}      | "
            f"{candidate.mean_average_precision:.4f}  | "
            f"{candidate.mean_fpr_at_target_tpr:.4f}"
        )

    print()
    print(
        "Selected L2 strength: "
        f"{analysis.selected_l2_strength}"
    )
    print(
        "Pooled development AUROC: "
        f"{analysis.development_metrics.auroc:.4f}"
    )

    print()
    print("Untouched test results")
    print("=" * 80)
    print(
        "Condition  | AUROC  | "
        "Average precision | FPR@95TPR"
    )
    print("-" * 80)

    for result in (
        analysis.condition_results
    ):
        metrics = result.metrics

        print(
            f"{result.condition:10s} | "
            f"{metrics.auroc:.4f} | "
            f"{metrics.average_precision:.4f}            | "
            f"{metrics.fpr_at_target_tpr:.4f}"
        )

    coefficients = sorted(
        zip(
            analysis.detector.feature_names,
            analysis.detector.coefficients,
            strict=True,
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    print()
    print("Largest standardized coefficients")
    print("=" * 72)

    for name, coefficient in (
        coefficients[:10]
    ):
        print(
            f"{name:40s} | "
            f"{coefficient:+.6f}"
        )

    print()
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
