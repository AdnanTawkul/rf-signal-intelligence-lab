from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from rfsil.evaluation.external_transfer import (
    AccuracySummary,
    ExternalSeedSweep,
    PairedAccuracyChange,
    compute_paired_accuracy_change,
    load_external_seed_sweep,
    summarize_accuracy,
    summarize_class_accuracy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate RadioML 2016 external-transfer "
            "results and generate comparison figures."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="External-transfer comparison YAML.",
    )
    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve one project-relative path."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def serialize_project_path(
    path: Path,
) -> str:
    """Serialize a path relative to the project when possible."""
    resolved = path.resolve()

    try:
        return resolved.relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_configuration(
    path: Path,
) -> dict[str, Any]:
    """Load and validate the top-level YAML mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Comparison configuration must be a mapping."
        )

    return content


def accuracy_summary_to_dict(
    summary: AccuracySummary,
) -> dict[str, object]:
    """Serialize one accuracy summary."""
    return {
        "mean": summary.mean,
        "standard_deviation": (
            summary.standard_deviation
        ),
        "minimum": summary.minimum,
        "maximum": summary.maximum,
        "per_seed": summary.per_seed.tolist(),
    }


def paired_change_to_dict(
    change: PairedAccuracyChange,
) -> dict[str, object]:
    """Serialize one paired accuracy comparison."""
    return {
        "seeds": list(change.seeds),
        "mean": change.mean,
        "standard_deviation": (
            change.standard_deviation
        ),
        "improved_seed_count": (
            change.improved_seed_count
        ),
        "per_seed": change.per_seed.tolist(),
    }


def validate_identifier_order(
    raw_value: object,
    *,
    available: set[str],
    name: str,
) -> tuple[str, ...]:
    """Validate an ordered list of model identifiers."""
    if (
        not isinstance(raw_value, list)
        or not raw_value
        or not all(
            isinstance(value, str)
            and value.strip()
            for value in raw_value
        )
    ):
        raise ValueError(
            f"{name} must be a non-empty list "
            "of model identifiers."
        )

    result = tuple(
        value.strip()
        for value in raw_value
    )

    if len(result) != len(set(result)):
        raise ValueError(
            f"{name} contains duplicate identifiers."
        )

    missing = set(result) - available

    if missing:
        raise ValueError(
            f"{name} references unknown models: "
            f"{sorted(missing)}"
        )

    return result


def load_model_sweeps(
    content: object,
) -> tuple[
    dict[str, ExternalSeedSweep],
    dict[str, str],
]:
    """Load all configured model evaluations."""
    if (
        not isinstance(content, dict)
        or not content
    ):
        raise ValueError(
            "models must be a non-empty mapping."
        )

    sweeps: dict[
        str,
        ExternalSeedSweep,
    ] = {}
    display_names: dict[str, str] = {}

    for identifier, raw_model in content.items():
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
        ):
            raise ValueError(
                "Model identifiers must be "
                "non-empty strings."
            )

        if not isinstance(raw_model, dict):
            raise ValueError(
                f"models.{identifier} must "
                "be a mapping."
            )

        display_name = str(
            raw_model["display_name"]
        ).strip()

        if not display_name:
            raise ValueError(
                f"models.{identifier}.display_name "
                "must not be empty."
            )

        aggregate_path = resolve_project_path(
            raw_model["aggregate_metrics"]
        )

        sweeps[identifier] = (
            load_external_seed_sweep(
                aggregate_path,
                name=display_name,
            )
        )
        display_names[identifier] = (
            display_name
        )

    reference = next(iter(sweeps.values()))

    for identifier, sweep in sweeps.items():
        if sweep.seeds != reference.seeds:
            raise ValueError(
                f"Model {identifier!r} uses "
                "different seeds."
            )

        if (
            sweep.class_names
            != reference.class_names
        ):
            raise ValueError(
                f"Model {identifier!r} uses "
                "different class names."
            )

        if (
            sweep.snr_values_db
            != reference.snr_values_db
        ):
            raise ValueError(
                f"Model {identifier!r} uses "
                "different SNR values."
            )

    return sweeps, display_names


def validate_shared_grid(
    raw_value: object,
) -> tuple[float, ...]:
    """Validate the common SNR grid."""
    if (
        not isinstance(raw_value, list)
        or not raw_value
    ):
        raise ValueError(
            "shared_snr_grid_db must be "
            "a non-empty list."
        )

    values: list[float] = []

    for raw_item in raw_value:
        if isinstance(raw_item, bool):
            raise ValueError(
                "Shared SNR values must be numeric."
            )

        try:
            value = float(raw_item)
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Shared SNR values must be numeric."
            ) from error

        if not np.isfinite(value):
            raise ValueError(
                "Shared SNR values must be finite."
            )

        values.append(value)

    if len(values) != len(set(values)):
        raise ValueError(
            "shared_snr_grid_db contains duplicates."
        )

    return tuple(values)


def create_overall_figure(
    *,
    sweeps: dict[str, ExternalSeedSweep],
    display_names: dict[str, str],
    display_order: tuple[str, ...],
    shared_grid: tuple[float, ...],
    output_path: Path,
) -> None:
    """Plot all-SNR and shared-grid accuracy."""
    all_summaries = [
        summarize_accuracy(
            sweeps[identifier]
        )
        for identifier in display_order
    ]
    shared_summaries = [
        summarize_accuracy(
            sweeps[identifier],
            snr_values_db=shared_grid,
        )
        for identifier in display_order
    ]

    positions = np.arange(
        len(display_order),
        dtype=np.float64,
    )
    width = 0.36

    figure, axis = plt.subplots(
        figsize=(12, 6)
    )

    axis.bar(
        positions - width / 2,
        [
            summary.mean
            for summary in all_summaries
        ],
        width,
        yerr=[
            summary.standard_deviation
            for summary in all_summaries
        ],
        capsize=4,
        label="All 20 SNR levels",
    )
    axis.bar(
        positions + width / 2,
        [
            summary.mean
            for summary in shared_summaries
        ],
        width,
        yerr=[
            summary.standard_deviation
            for summary in shared_summaries
        ],
        capsize=4,
        label="Shared six-SNR grid",
    )

    axis.set_xticks(
        positions,
        [
            display_names[identifier]
            for identifier in display_order
        ],
        rotation=18,
        ha="right",
    )
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(
        "RadioML 2016.10A External Transfer"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def create_snr_figure(
    *,
    sweeps: dict[str, ExternalSeedSweep],
    display_names: dict[str, str],
    detail_order: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot mean accuracy across RadioML SNR."""
    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    for identifier in detail_order:
        sweep = sweeps[identifier]

        mean = np.mean(
            sweep.snr_accuracy,
            axis=0,
        )
        standard_deviation = np.std(
            sweep.snr_accuracy,
            axis=0,
        )

        snr_values = np.asarray(
            sweep.snr_values_db,
            dtype=np.float64,
        )

        axis.plot(
            snr_values,
            mean,
            marker="o",
            label=display_names[identifier],
        )
        axis.fill_between(
            snr_values,
            mean - standard_deviation,
            mean + standard_deviation,
            alpha=0.18,
        )

    axis.axhline(
        0.25,
        linestyle="--",
        label="Four-class chance level",
    )
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(
        sweeps[
            detail_order[0]
        ].snr_values_db
    )
    axis.set_title(
        "External Accuracy by SNR"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def create_class_figure(
    *,
    sweeps: dict[str, ExternalSeedSweep],
    display_names: dict[str, str],
    detail_order: tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot mean class accuracy."""
    class_names = sweeps[
        detail_order[0]
    ].class_names

    positions = np.arange(
        len(class_names),
        dtype=np.float64,
    )
    width = 0.8 / len(detail_order)

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    for model_index, identifier in enumerate(
        detail_order
    ):
        sweep = sweeps[identifier]
        mean = np.mean(
            sweep.class_accuracy,
            axis=0,
        )
        standard_deviation = np.std(
            sweep.class_accuracy,
            axis=0,
        )

        offset = (
            model_index
            - (len(detail_order) - 1) / 2
        ) * width

        axis.bar(
            positions + offset,
            mean,
            width,
            yerr=standard_deviation,
            capsize=4,
            label=display_names[identifier],
        )

    axis.set_xticks(
        positions,
        class_names,
    )
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title(
        "RadioML Class Accuracy"
    )
    axis.grid(
        True,
        axis="y",
        alpha=0.3,
    )
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
    )
    plt.close(figure)


def main() -> None:
    """Generate the external-transfer summary package."""
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_configuration(
        config_path
    )

    experiment_name = str(
        content["experiment_name"]
    ).strip()

    if not experiment_name:
        raise ValueError(
            "experiment_name must not be empty."
        )

    shared_grid = validate_shared_grid(
        content["shared_snr_grid_db"]
    )

    sweeps, display_names = (
        load_model_sweeps(
            content["models"]
        )
    )

    available_models = set(sweeps)

    display_order = validate_identifier_order(
        content["display_order"],
        available=available_models,
        name="display_order",
    )
    detail_order = validate_identifier_order(
        content["detail_order"],
        available=available_models,
        name="detail_order",
    )

    model_summary: dict[
        str,
        dict[str, object],
    ] = {}

    for identifier, sweep in sweeps.items():
        class_summary = (
            summarize_class_accuracy(
                sweep
            )
        )

        model_summary[identifier] = {
            "display_name": (
                display_names[identifier]
            ),
            "seeds": list(sweep.seeds),
            "overall": (
                accuracy_summary_to_dict(
                    summarize_accuracy(
                        sweep
                    )
                )
            ),
            "shared_snr_grid": (
                accuracy_summary_to_dict(
                    summarize_accuracy(
                        sweep,
                        snr_values_db=(
                            shared_grid
                        ),
                    )
                )
            ),
            "class_accuracy": {
                class_name: (
                    accuracy_summary_to_dict(
                        summary
                    )
                )
                for class_name, summary
                in class_summary.items()
            },
            "snr_values_db": list(
                sweep.snr_values_db
            ),
            "accuracy_by_snr_mean": (
                np.mean(
                    sweep.snr_accuracy,
                    axis=0,
                ).tolist()
            ),
            "accuracy_by_snr_std": (
                np.std(
                    sweep.snr_accuracy,
                    axis=0,
                ).tolist()
            ),
        }

    raw_comparisons = content.get(
        "paired_comparisons",
        {},
    )

    if not isinstance(
        raw_comparisons,
        dict,
    ):
        raise ValueError(
            "paired_comparisons must be a mapping."
        )

    paired_summary: dict[
        str,
        dict[str, object],
    ] = {}

    for identifier, raw_comparison in (
        raw_comparisons.items()
    ):
        if not isinstance(
            raw_comparison,
            dict,
        ):
            raise ValueError(
                f"paired_comparisons.{identifier} "
                "must be a mapping."
            )

        reference_id = str(
            raw_comparison["reference"]
        )
        candidate_id = str(
            raw_comparison["candidate"]
        )

        if reference_id not in sweeps:
            raise ValueError(
                f"Unknown reference model: "
                f"{reference_id}."
            )

        if candidate_id not in sweeps:
            raise ValueError(
                f"Unknown candidate model: "
                f"{candidate_id}."
            )

        paired_summary[identifier] = {
            "display_name": str(
                raw_comparison[
                    "display_name"
                ]
            ),
            "reference": reference_id,
            "candidate": candidate_id,
            "overall": (
                paired_change_to_dict(
                    compute_paired_accuracy_change(
                        sweeps[reference_id],
                        sweeps[candidate_id],
                    )
                )
            ),
            "shared_snr_grid": (
                paired_change_to_dict(
                    compute_paired_accuracy_change(
                        sweeps[reference_id],
                        sweeps[candidate_id],
                        snr_values_db=(
                            shared_grid
                        ),
                    )
                )
            ),
        }

    raw_controls = content.get(
        "synthetic_controls",
        {},
    )

    if not isinstance(raw_controls, dict):
        raise ValueError(
            "synthetic_controls must be a mapping."
        )

    control_summary: dict[
        str,
        dict[str, object],
    ] = {}

    for identifier, raw_control in (
        raw_controls.items()
    ):
        if not isinstance(
            raw_control,
            dict,
        ):
            raise ValueError(
                f"synthetic_controls.{identifier} "
                "must be a mapping."
            )

        external_identifier = str(
            raw_control["external_model"]
        )

        if external_identifier not in sweeps:
            raise ValueError(
                f"Unknown external model: "
                f"{external_identifier}."
            )

        synthetic_path = resolve_project_path(
            raw_control["synthetic_metrics"]
        )
        synthetic = load_external_seed_sweep(
            synthetic_path,
            name=(
                f"{identifier} synthetic control"
            ),
        )
        external = sweeps[
            external_identifier
        ]

        control_summary[identifier] = {
            "display_name": str(
                raw_control["display_name"]
            ),
            "synthetic_metrics": (
                serialize_project_path(
                    synthetic_path
                )
            ),
            "external_model": (
                external_identifier
            ),
            "synthetic_shared_grid": (
                accuracy_summary_to_dict(
                    summarize_accuracy(
                        synthetic,
                        snr_values_db=(
                            shared_grid
                        ),
                    )
                )
            ),
            "external_shared_grid": (
                accuracy_summary_to_dict(
                    summarize_accuracy(
                        external,
                        snr_values_db=(
                            shared_grid
                        ),
                    )
                )
            ),
            "external_minus_synthetic": (
                paired_change_to_dict(
                    compute_paired_accuracy_change(
                        synthetic,
                        external,
                        snr_values_db=(
                            shared_grid
                        ),
                    )
                )
            ),
        }

    output_content = content["output"]

    if not isinstance(output_content, dict):
        raise ValueError(
            "output must be a mapping."
        )

    summary_path = resolve_project_path(
        output_content["summary_json"]
    )
    overall_figure_path = (
        resolve_project_path(
            output_content[
                "overall_figure"
            ]
        )
    )
    snr_figure_path = resolve_project_path(
        output_content["snr_figure"]
    )
    class_figure_path = (
        resolve_project_path(
            output_content["class_figure"]
        )
    )

    for path in (
        summary_path,
        overall_figure_path,
        snr_figure_path,
        class_figure_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    summary_content = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "shared_snr_grid_db": list(
            shared_grid
        ),
        "models": model_summary,
        "paired_comparisons": (
            paired_summary
        ),
        "synthetic_controls": (
            control_summary
        ),
        "model_selection": {
            "zero_shot_external_model": (
                "mixed_iq_baseline"
            ),
            "amplitude_calibrated_diagnostic": (
                "frozen_residual_scaled_x112"
            ),
            "maximum_synthetic_robustness": (
                "joint_residual_unscaled"
            ),
        },
        "confirmatory_status": {
            "unscaled_external_evaluations": (
                "initial zero-shot evaluation"
            ),
            "scaled_external_evaluations": (
                "post-hoc diagnostics using "
                "validation-selected input scales"
            ),
        },
    }

    summary_path.write_text(
        json.dumps(
            summary_content,
            indent=2,
        ),
        encoding="utf-8",
    )

    create_overall_figure(
        sweeps=sweeps,
        display_names=display_names,
        display_order=display_order,
        shared_grid=shared_grid,
        output_path=overall_figure_path,
    )
    create_snr_figure(
        sweeps=sweeps,
        display_names=display_names,
        detail_order=detail_order,
        output_path=snr_figure_path,
    )
    create_class_figure(
        sweeps=sweeps,
        display_names=display_names,
        detail_order=detail_order,
        output_path=class_figure_path,
    )

    print(
        "Model                       | "
        "All SNRs       | Shared grid"
    )
    print("-" * 75)

    for identifier in display_order:
        overall = model_summary[
            identifier
        ]["overall"]
        shared = model_summary[
            identifier
        ]["shared_snr_grid"]

        print(
            f"{display_names[identifier]:27s} | "
            f"{overall['mean']:.4f} "
            f"+/- "
            f"{overall['standard_deviation']:.4f} | "
            f"{shared['mean']:.4f} "
            f"+/- "
            f"{shared['standard_deviation']:.4f}"
        )

    print()
    print(f"Summary: {summary_path}")
    print(
        f"Overall figure: "
        f"{overall_figure_path}"
    )
    print(
        f"SNR figure: {snr_figure_path}"
    )
    print(
        f"Class figure: {class_figure_path}"
    )


if __name__ == "__main__":
    main()
