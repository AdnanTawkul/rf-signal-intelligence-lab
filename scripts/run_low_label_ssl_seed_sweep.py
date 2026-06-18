from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run paired random, SimCLR, and VICReg "
            "low-label supervised seed sweeps."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration must be a mapping: {path}"
        )

    return content


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON mapping."""
    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            f"JSON content must be a mapping: {path}"
        )

    return content


def validate_seeds(value: object) -> list[int]:
    """Validate the configured seed list."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            "seeds must be a nonempty list."
        )

    seeds: list[int] = []

    for seed in value:
        if isinstance(seed, bool) or not isinstance(
            seed,
            int,
        ):
            raise ValueError(
                "Every seed must be an integer."
            )

        seeds.append(int(seed))

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Seeds must not contain duplicates."
        )

    return seeds


def prepare_run_config(
    template: dict[str, Any],
    method: str,
    seed: int,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    """Create one paired per-method, per-seed config."""
    content = deepcopy(template)

    training = content.get("training")

    if not isinstance(training, dict):
        raise ValueError(
            "Template training configuration "
            "must be a mapping."
        )

    training["seed"] = seed
    training["subset_seed"] = seed

    expected_examples = training.get(
        "examples_per_class_snr"
    )

    if expected_examples != 10:
        raise ValueError(
            "The paired sweep expects exactly "
            "10 examples per class-SNR stratum."
        )

    if int(training.get("epochs", 0)) != 330:
        raise ValueError(
            "The paired sweep expects the "
            "330-epoch step-matched protocol."
        )

    run_name = (
        f"baseline_cnn_groupnorm_{method}_"
        f"low_label_10_steps_matched_seed_{seed}"
    )

    if method == "random":
        run_name = (
            "baseline_cnn_groupnorm_"
            f"low_label_10_steps_matched_seed_{seed}"
        )

    run_directory = output_root / method / f"seed_{seed}"

    content["experiment_name"] = run_name
    content["output"]["directory"] = (
        run_directory.relative_to(
            PROJECT_ROOT
        ).as_posix()
    )
    content["output"]["figure_path"] = (
        Path("reports/figures")
        / (
            "low_label_ssl_step_matched_"
            f"{method}_seed_{seed}.png"
        )
    ).as_posix()

    config_directory = output_root / "generated_configs"
    config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_config_path = (
        config_directory
        / f"{method}_seed_{seed}.yaml"
    )

    generated_config_path.write_text(
        yaml.safe_dump(
            content,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return content, generated_config_path


def run_training(config_path: Path) -> None:
    """Run one supervised training process."""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train_baseline.py"),
        "--config",
        str(config_path),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def summarize_method(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one initialization method."""
    accuracies = [
        float(record["best_validation_accuracy"])
        for record in records
    ]
    epochs = [
        int(record["best_epoch"])
        for record in records
    ]

    return {
        "seed_count": len(records),
        "mean_validation_accuracy": mean(
            accuracies
        ),
        "validation_accuracy_standard_deviation": (
            pstdev(accuracies)
        ),
        "minimum_validation_accuracy": min(
            accuracies
        ),
        "maximum_validation_accuracy": max(
            accuracies
        ),
        "mean_best_epoch": mean(epochs),
        "records": records,
    }


def main() -> None:
    """Run the paired sweep."""
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml(config_path)

    experiment_name = str(
        content["experiment_name"]
    )
    seeds = validate_seeds(content["seeds"])

    methods_content = content.get("methods")

    if not isinstance(methods_content, dict):
        raise ValueError(
            "methods must be a mapping."
        )

    expected_methods = (
        "random",
        "simclr",
        "vicreg",
    )

    if set(methods_content) != set(
        expected_methods
    ):
        raise ValueError(
            "methods must contain exactly random, "
            "simclr, and vicreg."
        )

    output_content = content.get("output")

    if not isinstance(output_content, dict):
        raise ValueError(
            "output must be a mapping."
        )

    output_root = resolve_project_path(
        output_content["directory"]
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_configs: dict[
        str,
        dict[str, Any],
    ] = {}

    for method in expected_methods:
        method_content = methods_content[method]

        if not isinstance(method_content, dict):
            raise ValueError(
                f"Method {method} must be a mapping."
            )

        template_path = resolve_project_path(
            method_content["template_config"]
        )
        template_configs[method] = load_yaml(
            template_path
        )

    print(f"Experiment: {experiment_name}")
    print(f"Seeds: {seeds}")
    print(
        "Methods: random, simclr, vicreg"
    )
    print(
        "Pairing rule: training seed = subset seed"
    )

    method_records: dict[
        str,
        list[dict[str, Any]],
    ] = {
        method: []
        for method in expected_methods
    }

    for seed in seeds:
        print("")
        print("=" * 72)
        print(f"Paired seed {seed}")
        print("=" * 72)

        seed_summaries: dict[
            str,
            dict[str, Any],
        ] = {}

        for method in expected_methods:
            print("")
            print(
                f"Running {method} | seed={seed}"
            )

            _, generated_config_path = (
                prepare_run_config(
                    template=(
                        template_configs[method]
                    ),
                    method=method,
                    seed=seed,
                    output_root=output_root,
                )
            )

            run_training(
                generated_config_path
            )

            run_directory = (
                output_root
                / method
                / f"seed_{seed}"
            )
            summary_path = (
                run_directory / "summary.json"
            )
            summary = load_json(summary_path)

            subset = summary.get(
                "labeled_subset"
            )

            if not isinstance(subset, dict):
                raise ValueError(
                    "Training summary is missing "
                    "labeled_subset metadata."
                )

            if (
                int(
                    subset[
                        "selected_training_examples"
                    ]
                )
                != 280
            ):
                raise ValueError(
                    "Unexpected labeled subset size."
                )

            if (
                int(
                    subset[
                        "examples_per_class_snr"
                    ]
                )
                != 10
            ):
                raise ValueError(
                    "Unexpected examples-per-stratum."
                )

            if int(subset["subset_seed"]) != seed:
                raise ValueError(
                    "Subset seed does not match "
                    "the paired training seed."
                )

            record = {
                "seed": seed,
                "best_validation_accuracy": float(
                    summary[
                        "best_validation_accuracy"
                    ]
                ),
                "best_epoch": int(
                    summary["best_epoch"]
                ),
                "summary_path": (
                    summary_path.relative_to(
                        PROJECT_ROOT
                    ).as_posix()
                ),
            }

            method_records[method].append(
                record
            )
            seed_summaries[method] = record

        random_accuracy = seed_summaries[
            "random"
        ]["best_validation_accuracy"]

        print("")
        print(f"Paired seed {seed} result")

        for method in expected_methods:
            accuracy = seed_summaries[method][
                "best_validation_accuracy"
            ]

            print(
                f"{method:7s} | "
                f"validation={accuracy:.4f} | "
                f"change vs random="
                f"{accuracy - random_accuracy:+.4f}"
            )

    aggregate_methods = {
        method: summarize_method(
            method_records[method]
        )
        for method in expected_methods
    }

    paired_differences = {
        method: [
            float(method_record[
                "best_validation_accuracy"
            ])
            - float(random_record[
                "best_validation_accuracy"
            ])
            for method_record, random_record in zip(
                method_records[method],
                method_records["random"],
                strict=True,
            )
        ]
        for method in ("simclr", "vicreg")
    }

    aggregate = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "seeds": seeds,
        "paired_protocol": {
            "training_seed_equals_subset_seed": True,
            "selected_training_examples": 280,
            "examples_per_class_snr": 10,
            "epochs": 330,
        },
        "methods": aggregate_methods,
        "paired_changes_vs_random": {
            method: {
                "mean": mean(values),
                "standard_deviation": (
                    pstdev(values)
                ),
                "minimum": min(values),
                "maximum": max(values),
                "seeds_improved": sum(
                    value > 0.0
                    for value in values
                ),
                "seed_count": len(values),
                "per_seed": values,
            }
            for method, values
            in paired_differences.items()
        },
    }

    aggregate_path = (
        output_root
        / str(
            output_content.get(
                "aggregate_name",
                "aggregate.json",
            )
        )
    )
    aggregate_path.write_text(
        json.dumps(
            aggregate,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("=" * 72)
    print("Five-seed paired aggregate")
    print("=" * 72)

    for method in expected_methods:
        method_summary = aggregate_methods[
            method
        ]

        print(
            f"{method:7s} | "
            f"validation="
            f"{method_summary['mean_validation_accuracy']:.4f} "
            f"? "
            f"{method_summary['validation_accuracy_standard_deviation']:.4f}"
        )

    print("")

    for method in ("simclr", "vicreg"):
        change = aggregate[
            "paired_changes_vs_random"
        ][method]

        print(
            f"{method:7s} vs random | "
            f"mean change={change['mean']:+.4f} "
            f"? {change['standard_deviation']:.4f} | "
            f"improved seeds="
            f"{change['seeds_improved']}/"
            f"{change['seed_count']}"
        )

    print(f"Aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
