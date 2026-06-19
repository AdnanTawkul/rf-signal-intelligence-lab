from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.evaluation.ssl_label_efficiency_analysis import (
    build_paired_change_matrix,
    held_out_method_metrics,
    load_json_mapping,
    pool_confusion_matrices,
    summarize_selection,
    validation_method_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse analysis arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SSL label-efficiency "
            "validation and held-out results."
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
    """Resolve a project-relative path."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_yaml_mapping(
    path: Path,
) -> dict[str, Any]:
    """Load one YAML mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Analysis configuration must "
            "be a mapping."
        )

    return content


def save_validation_figure(
    *,
    aggregate: dict[str, Any],
    fractions: list[dict[str, object]],
    methods: dict[str, str],
    output_path: Path,
) -> None:
    """Plot validation accuracy by label fraction."""
    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    percentages = [
        float(fraction["percentage"])
        for fraction in fractions
    ]

    for method, display_name in (
        methods.items()
    ):
        means = []
        standard_deviations = []

        for fraction in fractions:
            metrics = (
                validation_method_metrics(
                    aggregate,
                    fraction_identifier=str(
                        fraction["identifier"]
                    ),
                    method=method,
                )
            )
            means.append(
                float(
                    metrics[
                        "mean_validation_accuracy"
                    ]
                )
            )
            standard_deviations.append(
                float(
                    metrics[
                        "validation_accuracy_standard_deviation"
                    ]
                )
            )

        axis.errorbar(
            percentages,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=3,
            label=display_name,
        )

    axis.set_xscale("log")
    axis.set_xticks(
        percentages,
        labels=[
            str(fraction["display_name"])
            for fraction in fractions
        ],
    )
    axis.set_xlabel(
        "Labeled training fraction"
    )
    axis.set_ylabel(
        "Best validation accuracy"
    )
    axis.set_ylim(0.65, 1.0)
    axis.set_title(
        "SSL Label-Efficiency Validation"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_held_out_figure(
    *,
    aggregate: dict[str, Any],
    fractions: list[dict[str, object]],
    methods: dict[str, str],
    conditions: dict[str, str],
    output_path: Path,
) -> None:
    """Plot all four held-out conditions."""
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9),
        sharex=True,
    )

    percentages = [
        float(fraction["percentage"])
        for fraction in fractions
    ]

    for axis, (
        condition,
        condition_name,
    ) in zip(
        axes.flat,
        conditions.items(),
        strict=True,
    ):
        for method, display_name in (
            methods.items()
        ):
            means = []
            standard_deviations = []

            for fraction in fractions:
                metrics = (
                    held_out_method_metrics(
                        aggregate,
                        fraction_identifier=(
                            str(
                                fraction[
                                    "identifier"
                                ]
                            )
                        ),
                        condition=condition,
                        method=method,
                    )
                )
                means.append(
                    float(metrics["mean"])
                )
                standard_deviations.append(
                    float(
                        metrics[
                            "standard_deviation"
                        ]
                    )
                )

            axis.errorbar(
                percentages,
                means,
                yerr=standard_deviations,
                marker="o",
                capsize=3,
                label=display_name,
            )

        axis.set_xscale("log")
        axis.set_xticks(
            percentages,
            labels=[
                str(
                    fraction[
                        "display_name"
                    ]
                )
                for fraction in fractions
            ],
        )
        axis.set_ylim(0.25, 1.0)
        axis.set_title(condition_name)
        axis.set_xlabel(
            "Labeled training fraction"
        )
        axis.set_ylabel("Test accuracy")
        axis.grid(True, alpha=0.3)

    axes[0, 0].legend()
    figure.suptitle(
        "SSL Label Efficiency Across "
        "Held-Out Channel Conditions"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_paired_change_figure(
    *,
    aggregate: dict[str, Any],
    fractions: list[dict[str, object]],
    methods: dict[str, str],
    conditions: dict[str, str],
    output_path: Path,
) -> dict[str, object]:
    """Plot SSL-minus-random paired changes."""
    fraction_identifiers = [
        str(fraction["identifier"])
        for fraction in fractions
    ]

    row_labels, matrix = (
        build_paired_change_matrix(
            aggregate=aggregate,
            fractions=(
                fraction_identifiers
            ),
            conditions=list(
                conditions.keys()
            ),
            methods=list(methods.keys()),
        )
    )

    matrix_percentage_points = (
        100.0 * matrix
    )
    maximum = float(
        np.max(
            np.abs(
                matrix_percentage_points
            )
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )
    image = axis.imshow(
        matrix_percentage_points,
        aspect="auto",
        vmin=-maximum,
        vmax=maximum,
        cmap="coolwarm",
    )

    axis.set_xticks(
        np.arange(len(fractions)),
        labels=[
            str(fraction["display_name"])
            for fraction in fractions
        ],
    )
    axis.set_yticks(
        np.arange(len(row_labels)),
        labels=row_labels,
    )
    axis.set_xlabel(
        "Labeled training fraction"
    )
    axis.set_title(
        "Paired Accuracy Change Versus "
        "Random Initialization"
    )

    for row in range(matrix.shape[0]):
        for column in range(
            matrix.shape[1]
        ):
            axis.text(
                column,
                row,
                (
                    f"{matrix_percentage_points[row, column]:+.2f}"
                ),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label=(
            "Accuracy change "
            "(percentage points)"
        ),
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)

    return {
        "row_labels": row_labels,
        "fraction_identifiers": (
            fraction_identifiers
        ),
        "values": matrix.tolist(),
        "values_percentage_points": (
            matrix_percentage_points.tolist()
        ),
    }


def metrics_paths(
    *,
    root: Path,
    condition: str,
    fraction: str,
    method: str,
    seeds: list[int],
) -> list[Path]:
    """Resolve one five-seed metrics set."""
    paths = [
        (
            root
            / condition
            / fraction
            / method
            / f"seed_{seed}"
            / "metrics.json"
        )
        for seed in seeds
    ]

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    return paths


def save_confusion_figure(
    *,
    comparisons: list[dict[str, object]],
    metrics_root: Path,
    seeds: list[int],
    methods: dict[str, str],
    output_path: Path,
) -> dict[str, object]:
    """Plot selected pooled confusion matrices."""
    figure, axes = plt.subplots(
        len(comparisons),
        2,
        figsize=(11, 5 * len(comparisons)),
        squeeze=False,
    )

    summary = {}

    for row, comparison in enumerate(
        comparisons
    ):
        identifier = str(
            comparison["identifier"]
        )
        display_name = str(
            comparison["display_name"]
        )
        condition = str(
            comparison["condition"]
        )
        fraction = str(
            comparison["fraction"]
        )
        reference_method = str(
            comparison[
                "reference_method"
            ]
        )
        candidate_method = str(
            comparison[
                "candidate_method"
            ]
        )

        reference = pool_confusion_matrices(
            metrics_paths(
                root=metrics_root,
                condition=condition,
                fraction=fraction,
                method=reference_method,
                seeds=seeds,
            )
        )
        candidate = pool_confusion_matrices(
            metrics_paths(
                root=metrics_root,
                condition=condition,
                fraction=fraction,
                method=candidate_method,
                seeds=seeds,
            )
        )

        if (
            reference["class_names"]
            != candidate["class_names"]
        ):
            raise ValueError(
                "Confusion class names differ."
            )

        class_names = list(
            reference["class_names"]
        )

        for column, (
            method,
            pooled,
        ) in enumerate(
            (
                (
                    reference_method,
                    reference,
                ),
                (
                    candidate_method,
                    candidate,
                ),
            )
        ):
            matrix = np.asarray(
                pooled[
                    "normalized_confusion_matrix"
                ],
                dtype=np.float64,
            )
            axis = axes[row, column]
            method_display_name = (
                methods.get(
                    method,
                    method,
                )
            )
            image = axis.imshow(
                matrix,
                vmin=0.0,
                vmax=1.0,
            )

            axis.set_title(
                f"{display_name}\n"
                f"{method_display_name} | "
                f"{pooled['mean_accuracy']:.3f}"
            )
            axis.set_xlabel(
                "Predicted class"
            )
            axis.set_ylabel("True class")
            axis.set_xticks(
                np.arange(
                    len(class_names)
                ),
                labels=class_names,
            )
            axis.set_yticks(
                np.arange(
                    len(class_names)
                ),
                labels=class_names,
            )

            for matrix_row in range(
                len(class_names)
            ):
                for matrix_column in range(
                    len(class_names)
                ):
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{matrix[matrix_row, matrix_column]:.2f}",
                        ha="center",
                        va="center",
                    )

            figure.colorbar(
                image,
                ax=axis,
                fraction=0.046,
            )

        summary[identifier] = {
            "display_name": display_name,
            "condition": condition,
            "fraction": fraction,
            "reference_method": (
                reference_method
            ),
            "candidate_method": (
                candidate_method
            ),
            "reference": reference,
            "candidate": candidate,
        }

    figure.suptitle(
        "Selected SSL Confusion-Matrix "
        "Comparisons"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)

    return summary


def main() -> None:
    """Create the complete SSL analysis package."""
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml_mapping(
        config_path
    )

    validation_path = resolve_project_path(
        content["validation_aggregate"]
    )
    held_out_path = resolve_project_path(
        content["held_out_aggregate"]
    )
    metrics_root = resolve_project_path(
        content["held_out_metrics_root"]
    )

    validation = load_json_mapping(
        validation_path
    )
    held_out = load_json_mapping(
        held_out_path
    )

    if (
        validation.get(
            "completed_run_count"
        )
        != 75
    ):
        raise ValueError(
            "Expected 75 completed training runs."
        )

    if held_out.get("evaluation_count") != 300:
        raise ValueError(
            "Expected 300 held-out evaluations."
        )

    fractions = list(content["fractions"])
    methods = dict(content["methods"])
    conditions = dict(content["conditions"])
    seeds = [
        int(seed)
        for seed in content["seeds"]
    ]

    output = dict(content["output"])

    output_paths = {
        name: resolve_project_path(path)
        for name, path in output.items()
    }

    for path in output_paths.values():
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    save_validation_figure(
        aggregate=validation,
        fractions=fractions,
        methods=methods,
        output_path=output_paths[
            "validation_figure"
        ],
    )
    save_held_out_figure(
        aggregate=held_out,
        fractions=fractions,
        methods=methods,
        conditions=conditions,
        output_path=output_paths[
            "held_out_figure"
        ],
    )
    paired_matrix = (
        save_paired_change_figure(
            aggregate=held_out,
            fractions=fractions,
            methods=methods,
            conditions=conditions,
            output_path=output_paths[
                "paired_change_figure"
            ],
        )
    )
    confusion_summary = (
        save_confusion_figure(
            comparisons=list(
                content[
                    "confusion_comparisons"
                ]
            ),
            metrics_root=metrics_root,
            seeds=seeds,
            methods=methods,
            output_path=output_paths[
                "confusion_figure"
            ],
        )
    )

    selections = {}

    for name, specification in dict(
        content["selections"]
    ).items():
        summary = summarize_selection(
            validation_aggregate=validation,
            held_out_aggregate=held_out,
            fraction_identifier=str(
                specification["fraction"]
            ),
            method=str(
                specification["method"]
            ),
            conditions=list(
                conditions.keys()
            ),
        )
        selections[name] = asdict(summary)

    analysis_summary = {
        "format_version": 1,
        "experiment_name": content[
            "experiment_name"
        ],
        "validation_aggregate": (
            validation_path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        ),
        "held_out_aggregate": (
            held_out_path.relative_to(
                PROJECT_ROOT
            ).as_posix()
        ),
        "completed_training_runs": 75,
        "completed_test_evaluations": 300,
        "selections": selections,
        "paired_change_matrix": (
            paired_matrix
        ),
        "confusion_comparisons": (
            confusion_summary
        ),
        "figures": {
            name: path.relative_to(
                PROJECT_ROOT
            ).as_posix()
            for name, path in (
                output_paths.items()
            )
            if name != "summary_json"
        },
    }

    output_paths[
        "summary_json"
    ].write_text(
        json.dumps(
            analysis_summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Selected systems")
    print("=" * 88)

    for name, result in selections.items():
        print(
            f"{name:28s} | "
            f"{result['fraction_identifier']:15s} | "
            f"{result['method']:7s} | "
            f"validation="
            f"{result['validation_accuracy']:.4f} | "
            f"macro="
            f"{result['macro_condition_accuracy']:.4f} | "
            f"change="
            f"{result['macro_change_vs_random']:+.4f}"
        )

    print()
    print(
        "Summary: "
        f"{output_paths['summary_json']}"
    )

    for name in (
        "validation_figure",
        "held_out_figure",
        "paired_change_figure",
        "confusion_figure",
    ):
        print(
            f"{name}: {output_paths[name]}"
        )


if __name__ == "__main__":
    main()
