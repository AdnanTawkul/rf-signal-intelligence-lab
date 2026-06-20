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
from rfsil.evaluation.iq_feature_detection import (
    analyze_iq_feature_detection,
)
from rfsil.evaluation.paired_shift_split import (
    create_paired_shift_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate individual IQ channel "
            "features for shift detection."
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
        candidate.feature_names
        != reference.feature_names
    ):
        raise ValueError(
            "Feature names differ between "
            "conditions."
        )

    arrays = (
        "labels",
        "snr_db",
        "frequency_offset_hz",
        "phase_offset_rad",
        "amplitude_scale",
        "time_shift_samples",
        "rayleigh_fading",
        "example_seed",
    )

    for name in arrays:
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
    analysis_config = content["analysis"]
    output_config = content["output"]

    artifact_root = resolve_project_path(
        input_config["directory"]
    )
    artifact_filename = str(
        input_config["artifact_filename"]
    )
    clean_condition = str(
        input_config["clean_condition"]
    )
    shifted_conditions = tuple(
        str(condition)
        for condition in input_config[
            "shifted_conditions"
        ]
    )

    clean_path = (
        artifact_root
        / clean_condition
        / artifact_filename
    )
    clean_artifact = (
        load_iq_channel_feature_artifact(
            clean_path
        )
    )

    shifted_artifacts = {}

    for condition in shifted_conditions:
        artifact = (
            load_iq_channel_feature_artifact(
                artifact_root
                / condition
                / artifact_filename
            )
        )
        verify_paired_artifacts(
            clean_artifact,
            artifact,
        )
        shifted_artifacts[
            condition
        ] = artifact

    split = create_paired_shift_split(
        clean_artifact.labels,
        clean_artifact.snr_db,
        clean_artifact.example_seed,
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

    analysis = analyze_iq_feature_detection(
        clean_artifact.values,
        {
            condition: artifact.values
            for condition, artifact
            in shifted_artifacts.items()
        },
        clean_artifact.feature_names,
        split,
        target_tpr=float(
            analysis_config["target_tpr"]
        ),
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
        "clean_artifact_path": (
            clean_path
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

    print("Split")
    print("=" * 72)
    print(
        f"Development examples: "
        f"{split.development_count}"
    )
    print(
        f"Test examples: "
        f"{split.test_count}"
    )
    print(
        f"Class/SNR strata: "
        f"{split.stratum_count}"
    )

    print()
    print("Overall individual-feature ranking")
    print("=" * 112)
    print(
        "Rank | Feature                                  | "
        "Direction | Mean AUROC | Worst AUROC | "
        "Mean FPR95"
    )
    print("-" * 112)

    for rank, summary in enumerate(
        analysis.feature_summaries,
        start=1,
    ):
        print(
            f"{rank:4d} | "
            f"{summary['feature_name']:40s} | "
            f"{summary['direction']:21s} | "
            f"{summary['mean_test_auroc']:.4f}     | "
            f"{summary['minimum_test_auroc']:.4f}      | "
            f"{summary[
                'mean_test_fpr_at_target_tpr'
            ]:.4f}"
        )

    print()
    print("Best feature by shifted condition")
    print("=" * 92)

    for condition in analysis.conditions:
        candidates = [
            result
            for result
            in analysis.condition_results
            if result.condition == condition
        ]
        best = max(
            candidates,
            key=lambda result: (
                result
                .directed_test_metrics
                .auroc
            ),
        )

        print(
            f"{condition:10s} | "
            f"{best.feature_name:40s} | "
            f"AUROC="
            f"{best.directed_test_metrics.auroc:.4f} | "
            f"AP="
            f"{best.directed_test_metrics.average_precision:.4f} | "
            f"FPR95="
            f"{best.directed_test_metrics.fpr_at_target_tpr:.4f}"
        )

    print()
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
