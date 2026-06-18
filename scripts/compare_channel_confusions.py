from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.evaluation.confusion_robustness import (
    ConditionConfusionSummary,
    compare_condition_confusions,
    load_condition_confusions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare confusion matrices across "
            "paired channel conditions."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    """Load a YAML mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Configuration must be a mapping."
        )

    return content


def create_confusion_figure(
    conditions: dict[
        str,
        ConditionConfusionSummary,
    ],
    comparison: dict[str, Any],
    output_path: Path,
) -> None:
    """Plot pooled normalized confusion matrices."""
    condition_names = list(conditions)
    class_names = next(
        iter(conditions.values())
    ).class_names

    figure, axes = plt.subplots(
        1,
        len(condition_names),
        figsize=(
            4.8 * len(condition_names),
            4.8,
        ),
        constrained_layout=True,
        squeeze=False,
    )

    image = None

    for axis, condition_name in zip(
        axes[0],
        condition_names,
        strict=True,
    ):
        matrix = np.asarray(
            comparison["conditions"][
                condition_name
            ][
                "pooled_normalized_confusion_matrix"
            ],
            dtype=np.float64,
        )

        image = axis.imshow(
            matrix,
            vmin=0.0,
            vmax=1.0,
        )

        axis.set_title(
            condition_name.replace(
                "_",
                " ",
            ).title()
        )
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_xticks(
            np.arange(len(class_names)),
            class_names,
            rotation=45,
            ha="right",
        )
        axis.set_yticks(
            np.arange(len(class_names)),
            class_names,
        )

        for row in range(len(class_names)):
            for column in range(
                len(class_names)
            ):
                value = float(
                    matrix[row, column]
                )

                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    if image is None:
        raise RuntimeError(
            "No confusion matrices were plotted."
        )

    figure.colorbar(
        image,
        ax=list(axes[0]),
        fraction=0.025,
        pad=0.02,
        label="Row-normalized frequency",
    )
    figure.suptitle(
        "Multipath Robustness: "
        "Five-Seed Pooled Confusion Matrices"
    )

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
    """Run the confusion comparison."""
    arguments = parse_arguments()
    configuration = load_yaml(
        resolve_path(arguments.config)
    )

    seeds = [
        int(seed)
        for seed in configuration["seeds"]
    ]
    class_names = [
        str(name)
        for name in configuration[
            "class_names"
        ]
    ]

    conditions_content = configuration[
        "conditions"
    ]

    if not isinstance(
        conditions_content,
        dict,
    ):
        raise ValueError(
            "conditions must be a mapping."
        )

    conditions = {
        str(condition_name): (
            load_condition_confusions(
                condition=str(condition_name),
                predictions_directory=resolve_path(
                    directory
                ),
                seeds=seeds,
                class_names=class_names,
            )
        )
        for condition_name, directory
        in conditions_content.items()
    }

    reference_condition = str(
        configuration.get(
            "reference_condition",
            "clean",
        )
    )

    comparison = (
        compare_condition_confusions(
            conditions,
            reference_condition=(
                reference_condition
            ),
        )
    )
    comparison["experiment_name"] = str(
        configuration["experiment_name"]
    )

    output_content = configuration[
        "output"
    ]
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
            comparison,
            indent=2,
        ),
        encoding="utf-8",
    )

    create_confusion_figure(
        conditions,
        comparison,
        figure_path,
    )

    print("Channel confusion comparison")
    print("")

    for condition_name in conditions:
        result = comparison["conditions"][
            condition_name
        ]

        print(
            f"{condition_name.upper()}"
        )

        for class_name in class_names:
            error = result[
                "dominant_errors"
            ][class_name]

            print(
                f"  {class_name:5s} -> "
                f"{error['predicted_class']:5s}: "
                f"{error['rate']:.4f}"
            )

        print("")

    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
