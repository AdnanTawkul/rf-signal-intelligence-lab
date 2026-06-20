from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.evaluation.calibration_visualization import (
    find_record,
    headline_summary,
    metric_change_matrix,
    reliability_curve,
    selective_accuracy_change_matrices,
    temperature_matrices,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create confidence-calibration "
            "analysis figures."
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
            "Visualization configuration "
            "must be a mapping."
        )

    return content


def load_json_mapping(
    path: Path,
) -> dict[str, Any]:
    content = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Analysis payload must be "
            "a mapping."
        )

    return content


def save_change_heatmap(
    *,
    matrix: np.ndarray,
    row_labels: list[str],
    column_labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    scale: float = 1.0,
) -> None:
    values = matrix * scale
    maximum = max(
        float(
            np.max(
                np.abs(values)
            )
        ),
        1e-12,
    )

    figure, axis = plt.subplots(
        figsize=(9.5, 9)
    )
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
    )

    axis.set_xticks(
        np.arange(
            len(column_labels)
        ),
        labels=column_labels,
    )
    axis.set_yticks(
        np.arange(
            len(row_labels)
        ),
        labels=row_labels,
    )
    axis.set_xlabel(
        "Held-out channel condition"
    )
    axis.set_ylabel(
        "Label fraction and initialization"
    )
    axis.set_title(title)

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                (
                    f"{values[
                        row_index,
                        column_index
                    ]:+.2f}"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )

    figure.colorbar(
        image,
        ax=axis,
        label=colorbar_label,
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_temperature_figure(
    *,
    aggregate: dict[str, Any],
    fractions: list[dict[str, Any]],
    methods: dict[str, str],
    output_path: Path,
) -> None:
    fraction_identifiers = [
        str(fraction["identifier"])
        for fraction in fractions
    ]
    percentages = [
        float(fraction["percentage"])
        for fraction in fractions
    ]
    method_identifiers = list(
        methods
    )

    means, standard_deviations = (
        temperature_matrices(
            aggregate,
            fractions=(
                fraction_identifiers
            ),
            methods=method_identifiers,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    for method_index, method in enumerate(
        method_identifiers
    ):
        axis.errorbar(
            percentages,
            means[method_index],
            yerr=(
                standard_deviations[
                    method_index
                ]
            ),
            marker="o",
            capsize=3,
            label=methods[method],
        )

    axis.axhline(
        1.0,
        linestyle="--",
        linewidth=1.2,
        label="No scaling",
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
    axis.set_xlabel(
        "Labeled training fraction"
    )
    axis.set_ylabel(
        "Validation-fitted temperature"
    )
    axis.set_title(
        "Validation-Fitted Scalar "
        "Temperatures"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_selective_accuracy_figure(
    *,
    records: list[dict[str, Any]],
    fractions: list[dict[str, Any]],
    conditions: dict[str, str],
    coverages: list[float],
    output_path: Path,
) -> None:
    fraction_identifiers = [
        str(fraction["identifier"])
        for fraction in fractions
    ]
    matrices = (
        selective_accuracy_change_matrices(
            records,
            fractions=(
                fraction_identifiers
            ),
            conditions=list(
                conditions
            ),
            coverages=coverages,
        )
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8.5),
        sharex=True,
        sharey=True,
    )

    for axis, (
        condition,
        condition_name,
    ) in zip(
        axes.flat,
        conditions.items(),
        strict=True,
    ):
        matrix = matrices[condition]

        for fraction_index, fraction in enumerate(
            fractions
        ):
            axis.plot(
                coverages,
                (
                    100.0
                    * matrix[
                        fraction_index
                    ]
                ),
                marker="o",
                label=str(
                    fraction[
                        "display_name"
                    ]
                ),
            )

        axis.axhline(
            0.0,
            linewidth=1.0,
        )
        axis.set_title(condition_name)
        axis.set_xlabel(
            "Retained coverage"
        )
        axis.set_ylabel(
            "Selective accuracy change "
            "(percentage points)"
        )
        axis.grid(True, alpha=0.3)

    axes[0, 0].legend(
        title="Label fraction"
    )
    figure.suptitle(
        "Selective Accuracy After "
        "Temperature Scaling"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def save_reliability_examples(
    *,
    records: list[dict[str, Any]],
    representative: dict[str, Any],
    conditions: dict[str, str],
    output_path: Path,
) -> None:
    selected_conditions = list(
        representative["conditions"]
    )

    figure, axes = plt.subplots(
        1,
        len(selected_conditions),
        figsize=(
            6 * len(selected_conditions),
            5,
        ),
        squeeze=False,
    )

    for axis, condition in zip(
        axes.flat,
        selected_conditions,
        strict=True,
    ):
        record = find_record(
            records,
            fraction_identifier=str(
                representative[
                    "fraction_identifier"
                ]
            ),
            method=str(
                representative["method"]
            ),
            seed=int(
                representative["seed"]
            ),
            condition=str(condition),
        )

        for variant, display_name in (
            (
                "baseline",
                "Uncalibrated",
            ),
            (
                "calibrated",
                "Temperature-scaled",
            ),
        ):
            curve = reliability_curve(
                record,
                variant=variant,
            )
            axis.plot(
                curve[
                    "mean_confidence"
                ],
                curve["accuracy"],
                marker="o",
                label=display_name,
            )

        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            linestyle="--",
            label="Ideal",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(
            "Mean confidence"
        )
        axis.set_ylabel(
            "Empirical accuracy"
        )
        axis.set_title(
            conditions[str(condition)]
        )
        axis.grid(True, alpha=0.3)

        baseline = record["baseline"]
        calibrated = record["calibrated"]

        axis.text(
            0.03,
            0.97,
            (
                "ECE: "
                f"{baseline[
                    'expected_calibration_error'
                ]:.3f}"
                " ? "
                f"{calibrated[
                    'expected_calibration_error'
                ]:.3f}\n"
                "NLL: "
                f"{baseline[
                    'negative_log_likelihood'
                ]:.3f}"
                " ? "
                f"{calibrated[
                    'negative_log_likelihood'
                ]:.3f}"
            ),
            transform=axis.transAxes,
            va="top",
        )

    axes[0, 0].legend()
    figure.suptitle(
        "Representative Reliability "
        "Diagrams"
    )
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml_mapping(
        config_path
    )

    input_path = resolve_project_path(
        content["input"][
            "aggregate_metrics"
        ]
    )
    payload = load_json_mapping(
        input_path
    )

    aggregate = payload["aggregate"]
    records = payload["records"]

    fractions = list(
        content["layout"]["fractions"]
    )
    methods = dict(
        content["layout"]["methods"]
    )
    conditions = dict(
        content["layout"]["conditions"]
    )
    coverages = [
        float(value)
        for value in content["layout"][
            "selective_coverages"
        ]
    ]

    rows = [
        (
            str(fraction["identifier"]),
            method,
        )
        for fraction in fractions
        for method in methods
    ]
    row_labels = [
        (
            f"{fraction['display_name']} / "
            f"{methods[method]}"
        )
        for fraction in fractions
        for method in methods
    ]

    output_content = content["output"]
    figure_directory = (
        resolve_project_path(
            output_content[
                "figure_directory"
            ]
        )
    )
    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {
        name: (
            figure_directory
            / str(filename)
        )
        for name, filename
        in dict(
            output_content["figures"]
        ).items()
    }

    nll_matrix = metric_change_matrix(
        aggregate,
        rows=rows,
        conditions=list(conditions),
        metric=(
            "negative_log_likelihood"
        ),
    )
    ece_matrix = metric_change_matrix(
        aggregate,
        rows=rows,
        conditions=list(conditions),
        metric=(
            "expected_calibration_error"
        ),
    )

    save_change_heatmap(
        matrix=nll_matrix,
        row_labels=row_labels,
        column_labels=list(
            conditions.values()
        ),
        title=(
            "Held-Out NLL Change After "
            "Validation-Fitted Calibration"
        ),
        colorbar_label=(
            "Calibrated minus baseline NLL"
        ),
        output_path=output_paths[
            "nll_transfer"
        ],
    )
    save_change_heatmap(
        matrix=ece_matrix,
        row_labels=row_labels,
        column_labels=list(
            conditions.values()
        ),
        title=(
            "Held-Out ECE Change After "
            "Validation-Fitted Calibration"
        ),
        colorbar_label=(
            "ECE change "
            "(percentage points)"
        ),
        output_path=output_paths[
            "ece_transfer"
        ],
        scale=100.0,
    )
    save_temperature_figure(
        aggregate=aggregate,
        fractions=fractions,
        methods=methods,
        output_path=output_paths[
            "temperatures"
        ],
    )
    save_selective_accuracy_figure(
        records=records,
        fractions=fractions,
        conditions=conditions,
        coverages=coverages,
        output_path=output_paths[
            "selective_accuracy"
        ],
    )
    save_reliability_examples(
        records=records,
        representative=dict(
            content[
                "representative"
            ]
        ),
        conditions=conditions,
        output_path=output_paths[
            "reliability_examples"
        ],
    )

    summary = headline_summary(
        records
    )
    summary["input_path"] = (
        input_path.resolve().as_posix()
    )
    summary["figures"] = {
        name: path.relative_to(
            PROJECT_ROOT
        ).as_posix()
        for name, path
        in output_paths.items()
    }

    summary_path = resolve_project_path(
        output_content[
            "summary_json"
        ]
    )
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Records summarized: "
        f"{summary['record_count']}"
    )
    print(
        "Accuracy preserved: "
        f"{summary[
            'accuracy_preserved_count'
        ]}"
    )

    for name, path in output_paths.items():
        print(f"{name}: {path}")

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
