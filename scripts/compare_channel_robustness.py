from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.evaluation.channel_robustness import (
    SeedSweepMetrics,
    compare_seed_sweep_conditions,
    load_seed_sweep_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired channel-robustness "
            "seed-sweep evaluations."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def load_configuration(
    path: Path,
) -> dict[str, Any]:
    """Load one YAML configuration mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Configuration must be a mapping."
        )

    return content


def create_figure(
    conditions: dict[str, SeedSweepMetrics],
    output_path: Path,
) -> None:
    """Create a consolidated robustness figure."""
    condition_names = list(conditions)
    display_names = [
        name.replace("_", " ").title()
        for name in condition_names
    ]

    overall_means = [
        float(
            np.mean(
                conditions[name].overall_accuracy
            )
        )
        for name in condition_names
    ]
    overall_stds = [
        float(
            np.std(
                conditions[name].overall_accuracy
            )
        )
        for name in condition_names
    ]

    reference = conditions[condition_names[0]]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5),
    )

    axes[0].errorbar(
        display_names,
        overall_means,
        yerr=overall_stds,
        marker="o",
        capsize=4,
    )
    axes[0].set_title(
        "Overall Accuracy by Channel Condition"
    )
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.3)

    condition_indices = np.arange(
        len(condition_names)
    )

    for class_index, class_name in enumerate(
        reference.class_names
    ):
        values = [
            float(
                np.mean(
                    conditions[name]
                    .class_accuracy[:, class_index]
                )
            )
            for name in condition_names
        ]

        axes[1].plot(
            condition_indices,
            values,
            marker="o",
            label=class_name,
        )

    axes[1].set_xticks(
        condition_indices,
        display_names,
    )
    axes[1].set_title(
        "Per-Class Accuracy Degradation"
    )
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    for condition_name, display_name in zip(
        condition_names,
        display_names,
        strict=True,
    ):
        metrics = conditions[condition_name]
        mean_accuracy = np.mean(
            metrics.accuracy_by_snr,
            axis=0,
        )
        standard_deviation = np.std(
            metrics.accuracy_by_snr,
            axis=0,
        )

        axes[2].plot(
            metrics.snr_values_db,
            mean_accuracy,
            marker="o",
            label=display_name,
        )
        axes[2].fill_between(
            metrics.snr_values_db,
            mean_accuracy - standard_deviation,
            mean_accuracy + standard_deviation,
            alpha=0.15,
        )

    axes[2].set_title(
        "Accuracy by SNR and Channel Condition"
    )
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    figure.suptitle(
        "Baseline CNN Multipath Robustness"
    )
    figure.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def main() -> None:
    """Run the robustness comparison."""
    arguments = parse_arguments()
    config_path = resolve_path(
        arguments.config
    )
    content = load_configuration(
        config_path
    )

    conditions_content = content.get(
        "conditions"
    )

    if (
        not isinstance(conditions_content, dict)
        or not conditions_content
    ):
        raise ValueError(
            "conditions must be a nonempty mapping."
        )

    conditions = {
        str(condition_name): (
            load_seed_sweep_metrics(
                resolve_path(path_value),
                str(condition_name),
            )
        )
        for condition_name, path_value
        in conditions_content.items()
    }

    reference_condition = str(
        content.get(
            "reference_condition",
            "clean",
        )
    )

    summary = compare_seed_sweep_conditions(
        conditions,
        reference_condition=reference_condition,
    )
    summary["experiment_name"] = str(
        content["experiment_name"]
    )

    output_content = content.get("output")

    if not isinstance(output_content, dict):
        raise ValueError(
            "output must be a mapping."
        )

    summary_path = resolve_path(
        output_content["summary_json"]
    )
    figure_path = resolve_path(
        output_content["figure_path"]
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    create_figure(
        conditions,
        figure_path,
    )

    print("Channel robustness comparison")
    print(
        f"Reference condition: "
        f"{reference_condition}"
    )
    print("")

    for condition_name in conditions:
        condition = summary["conditions"][
            condition_name
        ]
        drop = condition[
            "paired_drop_from_reference"
        ]

        print(
            f"{condition_name:8s} | "
            f"accuracy="
            f"{condition['overall_mean']:.4f} "
            f"? "
            f"{condition['overall_standard_deviation']:.4f} "
            f"| paired drop="
            f"{drop['mean']:+.4f} "
            f"? {drop['standard_deviation']:.4f}"
        )

    print("")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
