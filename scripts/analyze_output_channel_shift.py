from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from rfsil.evaluation.calibration_artifacts import (
    load_calibration_artifact,
)
from rfsil.evaluation.channel_shift import (
    SHIFT_SCORE_NAMES,
    compute_shift_scores,
    evaluate_shift_score_sets,
)
from rfsil.evaluation.channel_shift_analysis import (
    aggregate_channel_shift_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate output-based channel-shift "
            "scores across held-out artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fraction",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
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
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Configuration must be a mapping."
        )

    return content


def validate_requested_values(
    requested: set[Any],
    available: set[Any],
    *,
    name: str,
) -> None:
    unknown = requested - available

    if unknown:
        raise ValueError(
            f"Unknown {name}: "
            f"{sorted(unknown)}."
        )


def main() -> None:
    arguments = parse_arguments()

    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml_mapping(
        config_path
    )

    input_content = content["input"]
    layout = content["layout"]
    analysis = content["analysis"]
    output = content["output"]

    artifact_root = resolve_project_path(
        input_content["artifact_root"]
    )
    artifact_filename = str(
        input_content["artifact_filename"]
    )

    clean_condition = str(
        layout["clean_condition"]
    )
    all_conditions = tuple(
        str(value)
        for value in layout[
            "shifted_conditions"
        ]
    )
    all_fractions = tuple(
        str(value)
        for value in layout["fractions"]
    )
    all_methods = tuple(
        str(value)
        for value in layout["methods"]
    )
    all_seeds = tuple(
        int(value)
        for value in layout["seeds"]
    )

    requested_fractions = set(
        arguments.fraction
    )
    requested_methods = set(
        arguments.method
    )
    requested_seeds = set(
        arguments.seed
    )
    requested_conditions = set(
        arguments.condition
    )
    requested_scores = set(
        arguments.score
    )

    validate_requested_values(
        requested_fractions,
        set(all_fractions),
        name="fractions",
    )
    validate_requested_values(
        requested_methods,
        set(all_methods),
        name="methods",
    )
    validate_requested_values(
        requested_seeds,
        set(all_seeds),
        name="seeds",
    )
    validate_requested_values(
        requested_conditions,
        set(all_conditions),
        name="conditions",
    )
    validate_requested_values(
        requested_scores,
        set(SHIFT_SCORE_NAMES),
        name="scores",
    )

    fractions = tuple(
        value
        for value in all_fractions
        if (
            not requested_fractions
            or value in requested_fractions
        )
    )
    methods = tuple(
        value
        for value in all_methods
        if (
            not requested_methods
            or value in requested_methods
        )
    )
    seeds = tuple(
        value
        for value in all_seeds
        if (
            not requested_seeds
            or value in requested_seeds
        )
    )
    conditions = tuple(
        value
        for value in all_conditions
        if (
            not requested_conditions
            or value in requested_conditions
        )
    )
    score_names = tuple(
        value
        for value in SHIFT_SCORE_NAMES
        if (
            not requested_scores
            or value in requested_scores
        )
    )

    target_tpr = float(
        analysis["target_tpr"]
    )

    records: list[dict[str, Any]] = []

    checkpoint_count = (
        len(fractions)
        * len(methods)
        * len(seeds)
    )
    comparison_count = (
        checkpoint_count
        * len(conditions)
        * len(score_names)
    )

    print(
        f"Selected checkpoints: "
        f"{checkpoint_count}"
    )
    print(
        f"Selected conditions: "
        f"{len(conditions)}"
    )
    print(
        f"Selected scores: "
        f"{len(score_names)}"
    )
    print(
        f"Expected comparisons: "
        f"{comparison_count}"
    )

    for fraction in fractions:
        for method in methods:
            for seed in seeds:
                seed_directory = (
                    f"seed_{seed}"
                )
                clean_path = (
                    artifact_root
                    / clean_condition
                    / fraction
                    / method
                    / seed_directory
                    / artifact_filename
                )

                clean_artifact = (
                    load_calibration_artifact(
                        clean_path
                    )
                )
                clean_scores = (
                    compute_shift_scores(
                        clean_artifact.logits,
                        probabilities=(
                            clean_artifact
                            .probabilities
                        ),
                    )
                )

                clean_scores = {
                    name: clean_scores[name]
                    for name in score_names
                }

                for condition in conditions:
                    shifted_path = (
                        artifact_root
                        / condition
                        / fraction
                        / method
                        / seed_directory
                        / artifact_filename
                    )
                    shifted_artifact = (
                        load_calibration_artifact(
                            shifted_path
                        )
                    )

                    if (
                        shifted_artifact
                        .example_count
                        != clean_artifact
                        .example_count
                    ):
                        raise ValueError(
                            "Clean and shifted "
                            "artifacts have different "
                            "example counts."
                        )

                    if (
                        shifted_artifact
                        .class_names
                        != clean_artifact
                        .class_names
                    ):
                        raise ValueError(
                            "Clean and shifted "
                            "artifacts have different "
                            "class orders."
                        )

                    shifted_scores = (
                        compute_shift_scores(
                            shifted_artifact.logits,
                            probabilities=(
                                shifted_artifact
                                .probabilities
                            ),
                        )
                    )
                    shifted_scores = {
                        name: shifted_scores[name]
                        for name in score_names
                    }

                    evaluations = (
                        evaluate_shift_score_sets(
                            clean_scores,
                            shifted_scores,
                            target_tpr=target_tpr,
                        )
                    )

                    for score_name in score_names:
                        records.append(
                            {
                                "fraction_identifier": (
                                    fraction
                                ),
                                "method": method,
                                "seed": seed,
                                "condition": condition,
                                "score_name": (
                                    score_name
                                ),
                                "clean_artifact_path": (
                                    clean_path
                                    .resolve()
                                    .as_posix()
                                ),
                                "shifted_artifact_path": (
                                    shifted_path
                                    .resolve()
                                    .as_posix()
                                ),
                                "score_direction": (
                                    "larger_is_shift_like"
                                ),
                                "metrics": (
                                    evaluations[
                                        score_name
                                    ].to_dict()
                                ),
                            }
                        )

    aggregate = (
        aggregate_channel_shift_records(
            records
        )
    )

    payload = {
        "format_version": 1,
        "config_path": (
            config_path
            .resolve()
            .as_posix()
        ),
        "record_count": len(records),
        "records": records,
        "aggregate": aggregate,
    }

    output_directory = resolve_project_path(
        output["directory"]
    )
    output_path = (
        output_directory
        / str(output["summary_filename"])
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
    print(
        "Condition | Score                    | "
        "AUROC mean | AP mean | FPR@95 mean | "
        "AUROC >= 0.7"
    )
    print("=" * 106)

    for group in aggregate[
        "condition_score_groups"
    ]:
        auroc = group["metrics"]["auroc"]
        average_precision = group[
            "metrics"
        ]["average_precision"]
        fpr95 = group["metrics"][
            "fpr_at_target_tpr"
        ]
        counts = group[
            "auroc_direction_counts"
        ]

        print(
            f"{group['condition']:9s} | "
            f"{group['score_name']:24s} | "
            f"{auroc['mean']:.4f}     | "
            f"{average_precision['mean']:.4f}  | "
            f"{fpr95['mean']:.4f}      | "
            f"{counts['at_least_0_7']:2d}/"
            f"{group['run_count']}"
        )

    print()
    print(f"Records: {len(records)}")
    print(
        "Detailed groups: "
        f"{aggregate[
            'detailed_group_count'
        ]}"
    )
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
