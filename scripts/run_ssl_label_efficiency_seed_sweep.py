from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from rfsil.training.ssl_label_efficiency import (
    build_run_plan,
    validate_label_fractions,
    validate_seeds,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the exact-budget SSL "
            "label-efficiency seed sweep."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate and validate all configs "
            "without launching training."
        ),
    )
    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    """Load one YAML mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration must be a mapping: {path}"
        )

    return content


def validate_methods(
    value: object,
) -> dict[str, Path]:
    """Validate the three required method templates."""
    if not isinstance(value, dict):
        raise ValueError(
            "methods must be a mapping."
        )

    expected = (
        "random",
        "simclr",
        "vicreg",
    )

    if set(value) != set(expected):
        raise ValueError(
            "methods must contain exactly random, "
            "simclr, and vicreg."
        )

    result: dict[str, Path] = {}

    for method in expected:
        method_content = value[method]

        if not isinstance(
            method_content,
            dict,
        ):
            raise ValueError(
                f"methods.{method} must "
                "be a mapping."
            )

        result[method] = (
            resolve_project_path(
                method_content[
                    "template_config"
                ]
            )
        )

    return result


def main() -> None:
    """Generate and validate the v2 run matrix."""
    arguments = parse_arguments()

    if not arguments.dry_run:
        raise ValueError(
            "Execution mode is not enabled yet. "
            "Use --dry-run to validate the matrix."
        )

    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml(config_path)

    experiment_name = str(
        content["experiment_name"]
    ).strip()

    if not experiment_name:
        raise ValueError(
            "experiment_name must not be empty."
        )

    experiment_prefix = str(
        content["experiment_prefix"]
    ).strip()

    if not experiment_prefix:
        raise ValueError(
            "experiment_prefix must not be empty."
        )

    seeds = validate_seeds(
        content["seeds"]
    )
    fractions = validate_label_fractions(
        content["label_fractions"]
    )
    method_paths = validate_methods(
        content["methods"]
    )

    budget_content = content.get(
        "training_budget"
    )

    if not isinstance(
        budget_content,
        dict,
    ):
        raise ValueError(
            "training_budget must be a mapping."
        )

    stratum_count = int(
        budget_content["stratum_count"]
    )
    batch_size = int(
        budget_content["batch_size"]
    )
    target_optimizer_steps = int(
        budget_content[
            "target_optimizer_steps"
        ]
    )
    drop_last = budget_content.get(
        "drop_last",
        False,
    )
    require_exact = budget_content.get(
        "require_exact",
        True,
    )

    if not isinstance(drop_last, bool):
        raise ValueError(
            "training_budget.drop_last must "
            "be a boolean."
        )

    if not isinstance(require_exact, bool):
        raise ValueError(
            "training_budget.require_exact "
            "must be a boolean."
        )

    output_content = content.get("output")

    if not isinstance(
        output_content,
        dict,
    ):
        raise ValueError(
            "output must be a mapping."
        )

    output_root = resolve_project_path(
        output_content["directory"]
    )
    figure_root = resolve_project_path(
        output_content.get(
            "figure_directory",
            "reports/figures",
        )
    )
    manifest_name = str(
        output_content.get(
            "manifest_name",
            "dry_run_manifest.json",
        )
    )

    templates = {
        method: load_yaml(path)
        for method, path
        in method_paths.items()
    }

    run_records: list[
        dict[str, object]
    ] = []

    for fraction in fractions:
        for method in (
            "random",
            "simclr",
            "vicreg",
        ):
            for seed in seeds:
                plan = build_run_plan(
                    template=templates[method],
                    method=method,
                    seed=seed,
                    fraction=fraction,
                    stratum_count=(
                        stratum_count
                    ),
                    batch_size=batch_size,
                    target_optimizer_steps=(
                        target_optimizer_steps
                    ),
                    drop_last=drop_last,
                    require_exact=(
                        require_exact
                    ),
                    experiment_prefix=(
                        experiment_prefix
                    ),
                    output_root=output_root,
                    figure_root=figure_root,
                    project_root=PROJECT_ROOT,
                )

                plan.generated_config_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                plan.generated_config_path.write_text(
                    yaml.safe_dump(
                        plan.content,
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )

                run_records.append(
                    {
                        "fraction_identifier": (
                            fraction.identifier
                        ),
                        "fraction_display_name": (
                            fraction.display_name
                        ),
                        "examples_per_class_snr": (
                            fraction.examples_per_class_snr
                        ),
                        "selected_training_examples": (
                            plan.selected_training_examples
                        ),
                        "method": method,
                        "training_seed": seed,
                        "subset_seed": seed,
                        "generated_config_path": (
                            plan.generated_config_path.resolve().as_posix()
                        ),
                        "output_directory": (
                            plan.output_directory.resolve().as_posix()
                        ),
                        "training_budget": asdict(
                            plan.budget
                        ),
                    }
                )

    expected_run_count = (
        len(fractions)
        * 3
        * len(seeds)
    )

    if len(run_records) != expected_run_count:
        raise RuntimeError(
            "Generated run count does not "
            "match the configured matrix."
        )

    if not all(
        record["training_budget"][
            "actual_optimizer_steps"
        ]
        == target_optimizer_steps
        for record in run_records
    ):
        raise RuntimeError(
            "At least one run does not match "
            "the target optimizer-step budget."
        )

    manifest = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "dry_run": True,
        "run_count": len(run_records),
        "method_count": 3,
        "seed_count": len(seeds),
        "fraction_count": len(fractions),
        "training_seed_equals_subset_seed": True,
        "target_optimizer_steps": (
            target_optimizer_steps
        ),
        "runs": run_records,
    }

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path = (
        output_root / manifest_name
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Experiment: {experiment_name}")
    print(f"Fractions: {len(fractions)}")
    print("Methods: 3")
    print(f"Seeds: {len(seeds)}")
    print(f"Generated runs: {len(run_records)}")
    print(
        "Target optimizer steps per run: "
        f"{target_optimizer_steps}"
    )

    print()
    print(
        "Fraction | Examples | Steps/epoch | "
        "Epochs | Updates"
    )
    print("-" * 64)

    for fraction in fractions:
        record = next(
            item
            for item in run_records
            if (
                item["fraction_identifier"]
                == fraction.identifier
            )
        )
        budget = record["training_budget"]

        print(
            f"{fraction.display_name:8s} | "
            f"{record['selected_training_examples']:8d} | "
            f"{budget['steps_per_epoch']:11d} | "
            f"{budget['epochs']:6d} | "
            f"{budget['actual_optimizer_steps']:7d}"
        )

    print()
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
