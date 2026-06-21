from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfsil.deployment.shift_detector import (
    IQShiftDetectorArtifact,
    save_iq_shift_detector,
    select_shift_threshold,
)
from rfsil.evaluation.channel_shift import (
    evaluate_shift_detection,
)
from rfsil.evaluation.iq_feature_artifacts import (
    IQChannelFeatureArtifact,
    load_iq_channel_feature_artifact,
)
from rfsil.evaluation.linear_shift_detector import (
    StandardizedLinearShiftDetector,
)
from rfsil.evaluation.paired_shift_split import (
    create_paired_shift_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the development-selected "
            "all-IQ channel-shift detector."
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
            "IQ feature names differ between "
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
                "IQ feature metadata is not "
                f"paired: {name}."
            )


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    config = load_yaml_mapping(
        config_path
    )

    input_config = config["input"]
    split_config = config["split"]
    feature_config = config[
        "feature_extraction"
    ]
    deployment_config = config[
        "deployment"
    ]
    output_config = config["output"]

    analysis_path = resolve_project_path(
        input_config["analysis_summary"]
    )
    analysis_payload = json.loads(
        analysis_path.read_text(
            encoding="utf-8"
        )
    )
    analysis = analysis_payload[
        "analysis"
    ]
    detector_data = analysis["detector"]
    development_data = analysis[
        "development_metrics"
    ]

    feature_root = resolve_project_path(
        input_config[
            "iq_feature_directory"
        ]
    )
    feature_filename = str(
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
        feature_root
        / clean_condition
        / feature_filename
    )
    clean = load_iq_channel_feature_artifact(
        clean_path
    )
    shifted = {}

    for condition in shifted_conditions:
        path = (
            feature_root
            / condition
            / feature_filename
        )
        artifact = (
            load_iq_channel_feature_artifact(
                path
            )
        )
        verify_paired_artifacts(
            clean,
            artifact,
        )
        shifted[condition] = artifact

    stored_feature_names = tuple(
        detector_data["feature_names"]
    )

    if clean.feature_names != (
        stored_feature_names
    ):
        raise ValueError(
            "Analysis detector feature names "
            "do not match the IQ artifacts."
        )

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

    detector = StandardizedLinearShiftDetector(
        feature_names=(
            stored_feature_names
        ),
        feature_mean=np.asarray(
            detector_data["feature_mean"],
            dtype=np.float64,
        ),
        feature_scale=np.asarray(
            detector_data["feature_scale"],
            dtype=np.float64,
        ),
        coefficients=np.asarray(
            detector_data["coefficients"],
            dtype=np.float64,
        ),
        intercept=float(
            detector_data["intercept"]
        ),
        l2_strength=float(
            detector_data["l2_strength"]
        ),
    )

    development_indices = (
        split.development_indices
    )
    clean_scores = (
        detector.decision_function(
            clean.values[
                development_indices
            ]
        )
    )
    shifted_scores = np.concatenate(
        [
            detector.decision_function(
                artifact.values[
                    development_indices
                ]
            )
            for artifact
            in shifted.values()
        ]
    )

    target_tpr = float(
        analysis["target_tpr"]
    )
    threshold = select_shift_threshold(
        clean_scores,
        shifted_scores,
        target_tpr=target_tpr,
    )
    recalculated = evaluate_shift_detection(
        clean_scores,
        shifted_scores,
        target_tpr=target_tpr,
    )

    expected_fpr = float(
        development_data[
            "fpr_at_target_tpr"
        ]
    )
    tolerance = (
        1.0 / clean_scores.size
        + 1e-12
    )

    if abs(
        threshold.achieved_fpr
        - expected_fpr
    ) > tolerance:
        raise RuntimeError(
            "Exported decision threshold does "
            "not reproduce the stored "
            "development FPR95."
        )

    if not np.isclose(
        recalculated.auroc,
        float(
            development_data["auroc"]
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Recalculated development AUROC "
            "does not match the analysis."
        )

    artifact = IQShiftDetectorArtifact(
        format_version=1,
        artifact_name=str(
            deployment_config[
                "artifact_name"
            ]
        ),
        expected_sample_count=int(
            deployment_config[
                "expected_sample_count"
            ]
        ),
        feature_names=(
            detector.feature_names
        ),
        feature_mean=(
            detector.feature_mean
        ),
        feature_scale=(
            detector.feature_scale
        ),
        coefficients=(
            detector.coefficients
        ),
        intercept=detector.intercept,
        l2_strength=(
            detector.l2_strength
        ),
        threshold=threshold.threshold,
        target_tpr=target_tpr,
        development_auroc=float(
            development_data["auroc"]
        ),
        development_average_precision=(
            float(
                development_data[
                    "average_precision"
                ]
            )
        ),
        development_fpr_at_target_tpr=(
            expected_fpr
        ),
        development_clean_mean=float(
            development_data[
                "clean_mean"
            ]
        ),
        development_clean_std=float(
            development_data[
                "clean_std"
            ]
        ),
        development_shifted_mean=float(
            development_data[
                "shifted_mean"
            ]
        ),
        development_shifted_std=float(
            development_data[
                "shifted_std"
            ]
        ),
        autocorrelation_lags=tuple(
            int(value)
            for value in feature_config[
                "autocorrelation_lags"
            ]
        ),
        occupancy_fraction=float(
            feature_config[
                "occupancy_fraction"
            ]
        ),
        epsilon=float(
            feature_config["epsilon"]
        ),
        provenance={
            "experiment_name": str(
                config["experiment_name"]
            ),
            "analysis_summary": (
                analysis_path
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "clean_feature_artifact": (
                clean_path
                .relative_to(PROJECT_ROOT)
                .as_posix()
            ),
            "shifted_conditions": list(
                shifted_conditions
            ),
            "development_count": (
                split.development_count
            ),
            "test_count": (
                split.test_count
            ),
            "split_seed": int(
                split_config["split_seed"]
            ),
            "development_fraction": (
                float(
                    split_config[
                        "development_fraction"
                    ]
                )
            ),
            "threshold_selection": (
                threshold.to_dict()
            ),
        },
    )

    output_path = resolve_project_path(
        output_config["artifact_path"]
    )
    save_iq_shift_detector(
        artifact,
        output_path,
    )

    print("Deployable IQ shift detector")
    print("=" * 80)
    print(
        f"Artifact: {output_path}"
    )
    print(
        f"Features: "
        f"{artifact.feature_count}"
    )
    print(
        f"Expected samples: "
        f"{artifact.expected_sample_count}"
    )
    print(
        f"Selected L2: "
        f"{artifact.l2_strength}"
    )
    print(
        f"Threshold: "
        f"{artifact.threshold:.8f}"
    )
    print(
        f"Target TPR: "
        f"{artifact.target_tpr:.4f}"
    )
    print(
        f"Achieved development TPR: "
        f"{threshold.achieved_tpr:.4f}"
    )
    print(
        f"Achieved development FPR: "
        f"{threshold.achieved_fpr:.4f}"
    )
    print(
        f"Development AUROC: "
        f"{artifact.development_auroc:.4f}"
    )
    print("Integrity result: OK")


if __name__ == "__main__":
    main()
