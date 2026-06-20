from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rfsil.evaluation.calibration_artifacts import (
    load_calibration_artifact,
)
from rfsil.evaluation.channel_shift import (
    compute_shift_scores,
)
from rfsil.evaluation.iq_feature_artifacts import (
    IQChannelFeatureArtifact,
    load_iq_channel_feature_artifact,
)
from rfsil.evaluation.iq_feature_detection import (
    analyze_iq_feature_detection,
)
from rfsil.evaluation.linear_shift_detector import (
    analyze_linear_iq_shift_detection,
)
from rfsil.evaluation.paired_shift_split import (
    create_paired_shift_split,
)
from rfsil.evaluation.shift_detector_comparison import (
    DetectorComparisonRecord,
    aggregate_detector_comparison_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare IQ and output-based "
            "channel-shift detectors."
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


def verify_paired_iq_artifacts(
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


def verify_calibration_alignment(
    calibration,
    iq_artifact: IQChannelFeatureArtifact,
    *,
    path: Path,
) -> None:
    if (
        calibration.example_count
        != iq_artifact.example_count
    ):
        raise ValueError(
            "Calibration/IQ example-count "
            f"mismatch: {path}."
        )

    if not np.array_equal(
        calibration.labels,
        iq_artifact.labels,
    ):
        raise ValueError(
            "Calibration/IQ label-order "
            f"mismatch: {path}."
        )

    if not np.allclose(
        calibration.snr_db,
        iq_artifact.snr_db,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Calibration/IQ SNR-order "
            f"mismatch: {path}."
        )


def discover_checkpoints(
    clean_root: Path,
    *,
    artifact_filename: str,
) -> tuple[
    tuple[str, str, int, str],
    ...,
]:
    paths = sorted(
        clean_root.rglob(
            artifact_filename
        )
    )
    checkpoints = []

    for path in paths:
        relative = path.relative_to(
            clean_root
        )

        if len(relative.parts) != 4:
            raise ValueError(
                "Unexpected clean calibration "
                f"path: {relative}."
            )

        (
            fraction,
            method,
            seed_directory,
            filename,
        ) = relative.parts

        if filename != artifact_filename:
            raise ValueError(
                f"Unexpected filename: {filename}."
            )

        if not seed_directory.startswith(
            "seed_"
        ):
            raise ValueError(
                "Unexpected seed directory: "
                f"{seed_directory}."
            )

        seed = int(
            seed_directory.removeprefix(
                "seed_"
            )
        )

        checkpoints.append(
            (
                fraction,
                method,
                seed,
                seed_directory,
            )
        )

    return tuple(checkpoints)


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

    conditions = (
        str(input_config["clean_condition"]),
        *tuple(
            str(value)
            for value in input_config[
                "shifted_conditions"
            ]
        ),
    )
    clean_condition = conditions[0]
    shifted_conditions = conditions[1:]

    iq_root = resolve_project_path(
        input_config[
            "iq_feature_directory"
        ]
    )
    iq_filename = str(
        input_config[
            "iq_feature_filename"
        ]
    )
    calibration_root = (
        resolve_project_path(
            input_config[
                "calibration_directory"
            ]
        )
    )
    calibration_filename = str(
        input_config[
            "calibration_filename"
        ]
    )

    iq_artifacts = {
        condition: (
            load_iq_channel_feature_artifact(
                iq_root
                / condition
                / iq_filename
            )
        )
        for condition in conditions
    }

    clean_iq = iq_artifacts[
        clean_condition
    ]

    for condition in shifted_conditions:
        verify_paired_iq_artifacts(
            clean_iq,
            iq_artifacts[condition],
        )

    split = create_paired_shift_split(
        clean_iq.labels,
        clean_iq.snr_db,
        clean_iq.example_seed,
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

    target_tpr = float(
        analysis_config["target_tpr"]
    )
    lag8_feature_name = str(
        analysis_config[
            "lag8_feature_name"
        ]
    )

    if lag8_feature_name not in (
        clean_iq.feature_names
    ):
        raise ValueError(
            "Configured lag-8 feature was "
            "not found."
        )

    lag8_index = (
        clean_iq.feature_names.index(
            lag8_feature_name
        )
    )

    lag8_analysis = (
        analyze_iq_feature_detection(
            clean_iq.values[
                :,
                [lag8_index],
            ],
            {
                condition: (
                    iq_artifacts[
                        condition
                    ].values[
                        :,
                        [lag8_index],
                    ]
                )
                for condition
                in shifted_conditions
            },
            (lag8_feature_name,),
            split,
            target_tpr=target_tpr,
        )
    )
    lag8_selection = (
        lag8_analysis.selections[0]
    )
    lag8_results = {
        result.condition: (
            result.directed_test_metrics
        )
        for result
        in lag8_analysis.condition_results
    }

    l2_strengths = tuple(
        detector_config["l2_strengths"]
    )
    fold_count = int(
        detector_config["fold_count"]
    )
    cv_seed = int(
        detector_config["cv_seed"]
    )

    all_iq_analysis = (
        analyze_linear_iq_shift_detection(
            clean_iq.values,
            {
                condition: (
                    iq_artifacts[
                        condition
                    ].values
                )
                for condition
                in shifted_conditions
            },
            clean_iq.feature_names,
            clean_iq.example_seed,
            split,
            l2_strengths=l2_strengths,
            fold_count=fold_count,
            cv_seed=cv_seed,
            target_tpr=target_tpr,
        )
    )
    all_iq_results = {
        result.condition: result.metrics
        for result
        in all_iq_analysis.condition_results
    }

    checkpoints = discover_checkpoints(
        calibration_root
        / clean_condition,
        artifact_filename=(
            calibration_filename
        ),
    )

    expected_checkpoint_count = int(
        analysis_config[
            "expected_checkpoint_count"
        ]
    )

    if len(checkpoints) != (
        expected_checkpoint_count
    ):
        raise ValueError(
            "Unexpected checkpoint count: "
            f"{len(checkpoints)}."
        )

    records = []
    checkpoint_summaries = []

    for checkpoint_index, (
        fraction,
        method,
        seed,
        seed_directory,
    ) in enumerate(
        checkpoints,
        start=1,
    ):
        calibrations = {}
        energies = {}

        for condition in conditions:
            path = (
                calibration_root
                / condition
                / fraction
                / method
                / seed_directory
                / calibration_filename
            )
            calibration = (
                load_calibration_artifact(
                    path
                )
            )
            verify_calibration_alignment(
                calibration,
                iq_artifacts[condition],
                path=path,
            )
            calibrations[condition] = (
                calibration
            )
            energies[condition] = (
                compute_shift_scores(
                    calibration.logits,
                    probabilities=(
                        calibration
                        .probabilities
                    ),
                )["energy"]
            )

        energy_analysis = (
            analyze_iq_feature_detection(
                energies[clean_condition][
                    :,
                    np.newaxis,
                ],
                {
                    condition: (
                        energies[condition][
                            :,
                            np.newaxis,
                        ]
                    )
                    for condition
                    in shifted_conditions
                },
                ("output_energy",),
                split,
                target_tpr=target_tpr,
            )
        )
        energy_selection = (
            energy_analysis.selections[0]
        )
        energy_results = {
            result.condition: (
                result.directed_test_metrics
            )
            for result
            in energy_analysis.condition_results
        }

        fusion_names = (
            *clean_iq.feature_names,
            "output_energy",
        )
        fusion_clean = np.column_stack(
            (
                clean_iq.values,
                energies[clean_condition],
            )
        )
        fusion_shifted = {
            condition: np.column_stack(
                (
                    iq_artifacts[
                        condition
                    ].values,
                    energies[condition],
                )
            )
            for condition
            in shifted_conditions
        }

        fusion_analysis = (
            analyze_linear_iq_shift_detection(
                fusion_clean,
                fusion_shifted,
                fusion_names,
                clean_iq.example_seed,
                split,
                l2_strengths=l2_strengths,
                fold_count=fold_count,
                cv_seed=cv_seed,
                target_tpr=target_tpr,
            )
        )
        fusion_results = {
            result.condition: result.metrics
            for result
            in fusion_analysis.condition_results
        }

        for condition in shifted_conditions:
            records.extend(
                (
                    DetectorComparisonRecord(
                        fraction_identifier=(
                            fraction
                        ),
                        method=method,
                        seed=seed,
                        condition=condition,
                        system_name="lag8",
                        metrics=(
                            lag8_results[
                                condition
                            ]
                        ),
                        development_auroc=(
                            lag8_selection
                            .directed_development_metrics
                            .auroc
                        ),
                        direction=(
                            lag8_selection
                            .direction
                        ),
                    ),
                    DetectorComparisonRecord(
                        fraction_identifier=(
                            fraction
                        ),
                        method=method,
                        seed=seed,
                        condition=condition,
                        system_name=(
                            "all_iq_linear"
                        ),
                        metrics=(
                            all_iq_results[
                                condition
                            ]
                        ),
                        development_auroc=(
                            all_iq_analysis
                            .development_metrics
                            .auroc
                        ),
                        selected_l2_strength=(
                            all_iq_analysis
                            .selected_l2_strength
                        ),
                    ),
                    DetectorComparisonRecord(
                        fraction_identifier=(
                            fraction
                        ),
                        method=method,
                        seed=seed,
                        condition=condition,
                        system_name=(
                            "output_energy"
                        ),
                        metrics=(
                            energy_results[
                                condition
                            ]
                        ),
                        development_auroc=(
                            energy_selection
                            .directed_development_metrics
                            .auroc
                        ),
                        direction=(
                            energy_selection
                            .direction
                        ),
                    ),
                    DetectorComparisonRecord(
                        fraction_identifier=(
                            fraction
                        ),
                        method=method,
                        seed=seed,
                        condition=condition,
                        system_name=(
                            "iq_energy_fusion"
                        ),
                        metrics=(
                            fusion_results[
                                condition
                            ]
                        ),
                        development_auroc=(
                            fusion_analysis
                            .development_metrics
                            .auroc
                        ),
                        selected_l2_strength=(
                            fusion_analysis
                            .selected_l2_strength
                        ),
                    ),
                )
            )

        checkpoint_summaries.append(
            {
                "fraction_identifier": (
                    fraction
                ),
                "method": method,
                "seed": seed,
                "energy_direction": (
                    energy_selection.direction
                ),
                "energy_development_auroc": (
                    energy_selection
                    .directed_development_metrics
                    .auroc
                ),
                "fusion_l2_strength": (
                    fusion_analysis
                    .selected_l2_strength
                ),
                "fusion_development_auroc": (
                    fusion_analysis
                    .development_metrics
                    .auroc
                ),
                "fusion_energy_coefficient": (
                    float(
                        fusion_analysis
                        .detector
                        .coefficients[-1]
                    )
                ),
            }
        )

        if (
            checkpoint_index == 1
            or checkpoint_index % 10 == 0
            or checkpoint_index
            == len(checkpoints)
        ):
            print(
                "Processed checkpoints: "
                f"{checkpoint_index}/"
                f"{len(checkpoints)}"
            )

    aggregate = (
        aggregate_detector_comparison_records(
            records
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
        "record_count": len(records),
        "checkpoint_count": len(
            checkpoints
        ),
        "static_baselines": {
            "lag8": (
                lag8_analysis.to_dict()
            ),
            "all_iq_linear": (
                all_iq_analysis.to_dict()
            ),
        },
        "checkpoint_summaries": (
            checkpoint_summaries
        ),
        "records": [
            record.to_dict()
            for record in records
        ],
        "aggregate": aggregate,
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

    print()
    print("Aggregate held-out comparison")
    print("=" * 104)
    print(
        "Condition | System             | "
        "Mean AUROC | Mean AP | Mean FPR95 | "
        "AUROC >= 0.8"
    )
    print("-" * 104)

    for group in aggregate[
        "system_condition_groups"
    ]:
        print(
            f"{group['condition']:9s} | "
            f"{group['system_name']:18s} | "
            f"{group['metrics']['auroc']['mean']:.4f}     | "
            f"{group['metrics']['average_precision']['mean']:.4f}  | "
            f"{group['metrics']['fpr_at_target_tpr']['mean']:.4f}     | "
            f"{group['threshold_counts']['auroc_at_least_0_8']:2d}/"
            f"{group['run_count']}"
        )

    print()
    print("Fusion paired changes")
    print("=" * 104)
    print(
        "Condition | Baseline           | "
        "AUROC change | FPR95 change | "
        "AUROC improved | FPR95 improved"
    )
    print("-" * 104)

    for change in aggregate[
        "fusion_paired_changes"
    ]:
        print(
            f"{change['condition']:9s} | "
            f"{change['baseline_system']:18s} | "
            f"{change['fusion_minus_baseline']['auroc']['mean']:+.4f}       | "
            f"{change['fusion_minus_baseline']['fpr_at_target_tpr']['mean']:+.4f}       | "
            f"{change['improvement_counts']['auroc_improved']:2d}/"
            f"{change['run_count']}          | "
            f"{change['improvement_counts']['fpr95_improved']:2d}/"
            f"{change['run_count']}"
        )

    print()
    print(f"Records: {len(records)}")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
