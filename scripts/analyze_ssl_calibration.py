from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from rfsil.evaluation.calibration_analysis import (
    aggregate_temperature_transfer_records,
    evaluate_temperature_transfer,
)
from rfsil.evaluation.calibration_artifacts import (
    load_calibration_artifact,
)
from rfsil.evaluation.ssl_label_efficiency import (
    build_metrics_path,
    validate_test_conditions,
)
from rfsil.evaluation.temperature_workflow import (
    load_temperature_summary,
)
from rfsil.training.ssl_sweep_execution import (
    load_run_expectation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply validation-fitted temperatures "
            "to held-out SSL artifacts."
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

    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_mapping(
    path: Path,
) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        content = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    else:
        content = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

    if not isinstance(content, dict):
        raise ValueError(
            f"Expected a mapping: {path}"
        )

    return content


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_mapping(config_path)

    training_manifest_path = (
        resolve_project_path(
            content["training_manifest"]
        )
    )
    held_out_config_path = (
        resolve_project_path(
            content[
                "held_out_evaluation_config"
            ]
        )
    )

    training_manifest = load_mapping(
        training_manifest_path
    )
    held_out_config = load_mapping(
        held_out_config_path
    )

    raw_runs = training_manifest.get(
        "runs"
    )

    if not isinstance(raw_runs, list):
        raise ValueError(
            "Training manifest runs must "
            "be a list."
        )

    expectations = [
        load_run_expectation(
            record,
            project_root=PROJECT_ROOT,
        )
        for record in raw_runs
    ]

    fractions = set(arguments.fraction)
    methods = set(arguments.method)
    seeds = set(arguments.seed)

    expectations = [
        expectation
        for expectation in expectations
        if (
            (
                not fractions
                or expectation
                .fraction_identifier
                in fractions
            )
            and (
                not methods
                or expectation.method
                in methods
            )
            and (
                not seeds
                or expectation.seed
                in seeds
            )
        )
    ]

    conditions = validate_test_conditions(
        held_out_config["conditions"],
        project_root=PROJECT_ROOT,
    )
    requested_conditions = set(
        arguments.condition
    )
    conditions = tuple(
        condition
        for condition in conditions
        if (
            not requested_conditions
            or condition.identifier
            in requested_conditions
        )
    )

    if not expectations:
        raise ValueError(
            "No checkpoints match filters."
        )

    if not conditions:
        raise ValueError(
            "No conditions match filters."
        )

    analysis = content["analysis"]
    output = content["output"]

    bin_count = int(
        analysis["bin_count"]
    )
    coverages = tuple(
        float(value)
        for value in analysis[
            "coverages"
        ]
    )
    artifact_filename = str(
        analysis["artifact_filename"]
    )
    temperature_filename = str(
        analysis[
            "temperature_summary_filename"
        ]
    )

    held_out_root = resolve_project_path(
        held_out_config[
            "output"
        ]["directory"]
    )
    output_directory = (
        resolve_project_path(
            output["directory"]
        )
    )
    output_path = (
        output_directory
        / output["summary_filename"]
    )

    records: list[dict[str, Any]] = []

    print(
        f"Selected checkpoints: "
        f"{len(expectations)}"
    )
    print(
        f"Selected conditions: "
        f"{len(conditions)}"
    )

    for expectation in expectations:
        temperature_path = (
            expectation.output_directory
            / temperature_filename
        )
        temperature_summary = (
            load_temperature_summary(
                temperature_path
            )
        )
        temperature = float(
            temperature_summary[
                "temperature_scaling"
            ]["temperature"]
        )

        for condition in conditions:
            metrics_path = build_metrics_path(
                output_root=held_out_root,
                expectation=expectation,
                condition=condition,
            )
            artifact_path = (
                metrics_path.parent
                / artifact_filename
            )
            artifact = (
                load_calibration_artifact(
                    artifact_path
                )
            )

            comparison = (
                evaluate_temperature_transfer(
                    artifact,
                    temperature=temperature,
                    bin_count=bin_count,
                    coverages=coverages,
                )
            )

            records.append(
                {
                    "fraction_identifier": (
                        expectation
                        .fraction_identifier
                    ),
                    "method": (
                        expectation.method
                    ),
                    "seed": expectation.seed,
                    "condition": (
                        condition.identifier
                    ),
                    "artifact_path": (
                        artifact_path
                        .resolve()
                        .as_posix()
                    ),
                    "temperature_summary_path": (
                        temperature_path
                        .resolve()
                        .as_posix()
                    ),
                    **comparison,
                }
            )

    aggregate = (
        aggregate_temperature_transfer_records(
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
        "group_count": (
            aggregate["group_count"]
        ),
        "records": records,
        "aggregate": aggregate,
    }

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
        "Fraction | Method | Condition | "
        "NLL change | ECE change | "
        "NLL improved"
    )
    print("=" * 96)

    for group in aggregate["groups"]:
        nll = group["metrics"][
            "negative_log_likelihood"
        ]
        ece = group["metrics"][
            "expected_calibration_error"
        ]
        counts = nll[
            "comparison_counts"
        ]

        print(
            f"{group[
                'fraction_identifier'
            ]:14s} | "
            f"{group['method']:7s} | "
            f"{group['condition']:8s} | "
            f"{nll['change']['mean']:+.5f} | "
            f"{ece['change']['mean']:+.5f} | "
            f"{counts['improved']}/"
            f"{group['run_count']}"
        )

    print()
    print(f"Records: {len(records)}")
    print(
        f"Aggregate groups: "
        f"{aggregate['group_count']}"
    )
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
