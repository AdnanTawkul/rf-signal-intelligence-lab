from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.evaluation.seed_variance import (
    SeedRunResult,
    aggregate_seed_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the baseline CNN across multiple random seeds.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/baseline_seed_sweep_v1.yaml"
        ),
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError("Seed-sweep configuration must be a mapping.")

    return content


def load_seed_result(summary_path: Path) -> SeedRunResult:
    """Load one training summary as a typed seed result."""
    content = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    return SeedRunResult(
        seed=int(content["seed"]),
        best_epoch=int(content["best_epoch"]),
        best_validation_accuracy=float(
            content["best_validation_accuracy"]
        ),
        final_train_loss=float(
            content["final_train_loss"]
        ),
        final_train_accuracy=float(
            content["final_train_accuracy"]
        ),
        final_validation_loss=float(
            content["final_validation_loss"]
        ),
        final_validation_accuracy=float(
            content["final_validation_accuracy"]
        ),
    )


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    content = load_yaml(config_path)

    experiment_name = str(content["experiment_name"])
    base_training_config = resolve_project_path(
        content["base_training_config"]
    )
    seeds = [int(seed) for seed in content["seeds"]]

    if not seeds:
        raise ValueError("At least one seed is required.")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed values must be unique.")

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    figure_path = resolve_project_path(
        output_content["figure_path"]
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[SeedRunResult] = []

    print(f"Seed sweep: {experiment_name}")
    print(f"Runs: {len(seeds)}")
    print(f"Seeds: {seeds}")

    for run_index, seed in enumerate(seeds, start=1):
        run_name = f"baseline_cnn_v1_seed_{seed}"
        run_output_directory = (
            output_directory
            / f"seed_{seed}"
        )
        run_figure_path = (
            figure_path.parent
            / f"baseline_cnn_v1_seed_{seed}_training.png"
        )

        print("")
        print(
            f"Starting run {run_index}/{len(seeds)} "
            f"with seed {seed}"
        )

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_baseline.py"),
            "--config",
            str(base_training_config),
            "--seed",
            str(seed),
            "--experiment-name",
            run_name,
            "--output-directory",
            str(run_output_directory),
            "--figure-path",
            str(run_figure_path),
        ]

        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
        )

        summary_path = run_output_directory / "summary.json"

        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Training summary was not created: {summary_path}"
            )

        result = load_seed_result(summary_path)
        results.append(result)

        print(
            f"Completed seed {seed}: "
            f"best val acc="
            f"{result.best_validation_accuracy:.4f}, "
            f"final val acc="
            f"{result.final_validation_accuracy:.4f}"
        )

    statistics = aggregate_seed_results(results)

    aggregate_path = output_directory / "aggregate.json"
    aggregate_content = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "base_training_config": str(
            base_training_config.relative_to(PROJECT_ROOT)
        ),
        "seeds": seeds,
        "runs": [
            asdict(result)
            for result in results
        ],
        "statistics": asdict(statistics),
    }

    aggregate_path.write_text(
        json.dumps(aggregate_content, indent=2),
        encoding="utf-8",
    )

    seed_labels = [
        str(result.seed)
        for result in results
    ]
    best_accuracies = np.asarray(
        [
            result.best_validation_accuracy
            for result in results
        ],
        dtype=np.float64,
    )
    final_accuracies = np.asarray(
        [
            result.final_validation_accuracy
            for result in results
        ],
        dtype=np.float64,
    )

    positions = np.arange(len(results))
    bar_width = 0.38

    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        positions - bar_width / 2.0,
        best_accuracies,
        width=bar_width,
        label="Best validation accuracy",
    )
    axis.bar(
        positions + bar_width / 2.0,
        final_accuracies,
        width=bar_width,
        label="Final validation accuracy",
    )
    axis.axhline(
        statistics.best_validation_mean,
        linestyle="--",
        label=(
            "Mean best validation accuracy "
            f"{statistics.best_validation_mean:.3f}"
        ),
    )
    axis.set_title("Baseline CNN Validation Accuracy Across Seeds")
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(
        positions,
        labels=seed_labels,
    )
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print("")
    print("Seed sweep complete")
    print(f"Runs: {statistics.run_count}")
    print(
        "Mean best validation accuracy: "
        f"{statistics.best_validation_mean:.4f}"
    )
    print(
        "Best-validation standard deviation: "
        f"{statistics.best_validation_std:.4f}"
    )
    print(
        "Best-validation range: "
        f"{statistics.best_validation_minimum:.4f} to "
        f"{statistics.best_validation_maximum:.4f}"
    )
    print(
        "Mean final validation accuracy: "
        f"{statistics.final_validation_mean:.4f}"
    )
    print(
        "Final-validation standard deviation: "
        f"{statistics.final_validation_std:.4f}"
    )
    print(f"Best seed: {statistics.best_seed}")
    print(f"Aggregate results: {aggregate_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
