from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Refit frozen classifier heads for multiple CNN checkpoints."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/refit_groupnorm_head_seed_sweep_v1.yaml"
        ),
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return path if path.is_absolute() else PROJECT_ROOT / path


def serialize_project_path(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    resolved_path = path.resolve()
    resolved_root = PROJECT_ROOT.resolve()

    try:
        return resolved_path.relative_to(
            resolved_root
        ).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Seed-sweep configuration must be a YAML mapping."
        )

    return content


def validate_seeds(value: object) -> tuple[int, ...]:
    """Validate the configured training seeds."""
    if not isinstance(value, list) or not value:
        raise ValueError(
            "seeds must be a non-empty list."
        )

    seeds = tuple(int(seed) for seed in value)

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Every seed must appear exactly once."
        )

    return seeds


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml(config_path)

    experiment_name = str(
        content["experiment_name"]
    )
    seeds = validate_seeds(
        content["seeds"]
    )

    source_directory = resolve_project_path(
        content["source_checkpoint_directory"]
    )

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_config_directory = (
        output_directory / "generated_configs"
    )
    generated_config_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_content = content["dataset"]
    refit_content = content["refit"]
    evaluation_content = content["evaluation"]

    records: list[dict[str, Any]] = []

    print(f"Experiment: {experiment_name}")
    print(f"Seeds: {list(seeds)}")

    for seed in seeds:
        source_checkpoint = (
            source_directory
            / f"seed_{seed}"
            / "best_model.pt"
        )

        if not source_checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint was not found: {source_checkpoint}"
            )

        seed_output_directory = (
            output_directory / f"seed_{seed}"
        )
        seed_config_path = (
            generated_config_directory
            / f"refit_seed_{seed}.yaml"
        )

        seed_configuration = {
            "experiment_name": (
                f"{experiment_name}_seed_{seed}"
            ),
            "source_checkpoint_path": (
                serialize_project_path(
                    source_checkpoint
                )
            ),
            "dataset": dataset_content,
            "refit": {
                **refit_content,
                "random_state": seed,
            },
            "evaluation": evaluation_content,
            "output": {
                "directory": serialize_project_path(
                    seed_output_directory
                ),
                "checkpoint_name": "best_model.pt",
                "summary_name": "summary.json",
            },
        }

        seed_config_path.write_text(
            yaml.safe_dump(
                seed_configuration,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        print("")
        print("=" * 72)
        print(f"Refitting seed {seed}")
        print("=" * 72)

        subprocess.run(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "refit_frozen_head.py"
                ),
                "--config",
                str(seed_config_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        summary_path = (
            seed_output_directory / "summary.json"
        )
        summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        records.append(
            {
                "seed": seed,
                "selected_regularization_c": float(
                    summary[
                        "selected_regularization_c"
                    ]
                ),
                "original_validation_accuracy": float(
                    summary[
                        "original_validation_accuracy"
                    ]
                ),
                "refit_validation_accuracy": float(
                    summary[
                        "refit_validation_accuracy"
                    ]
                ),
                "validation_accuracy_change": float(
                    summary[
                        "validation_accuracy_change"
                    ]
                ),
                "checkpoint_path": serialize_project_path(
                    seed_output_directory
                    / "best_model.pt"
                ),
                "summary_path": serialize_project_path(
                    summary_path
                ),
            }
        )

    original_values = np.asarray(
        [
            record["original_validation_accuracy"]
            for record in records
        ],
        dtype=np.float64,
    )
    refit_values = np.asarray(
        [
            record["refit_validation_accuracy"]
            for record in records
        ],
        dtype=np.float64,
    )
    changes = refit_values - original_values

    aggregate = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "seeds": list(seeds),
        "runs": records,
        "validation": {
            "original_mean_accuracy": float(
                original_values.mean()
            ),
            "original_standard_deviation": float(
                original_values.std()
            ),
            "refit_mean_accuracy": float(
                refit_values.mean()
            ),
            "refit_standard_deviation": float(
                refit_values.std()
            ),
            "mean_accuracy_change": float(
                changes.mean()
            ),
            "change_standard_deviation": float(
                changes.std()
            ),
            "improved_seed_count": int(
                np.count_nonzero(changes > 0.0)
            ),
        },
    }

    aggregate_path = (
        output_directory
        / str(
            output_content.get(
                "aggregate_summary",
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
    print("Five-seed validation aggregate")
    print(
        "Original accuracy: "
        f"{original_values.mean():.4f} ? "
        f"{original_values.std():.4f}"
    )
    print(
        "Refitted accuracy: "
        f"{refit_values.mean():.4f} ? "
        f"{refit_values.std():.4f}"
    )
    print(
        "Mean change: "
        f"{changes.mean():+.4f} ? "
        f"{changes.std():.4f}"
    )
    print(
        "Seeds improved: "
        f"{np.count_nonzero(changes > 0)}/{len(seeds)}"
    )
    print(f"Aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
