from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from rfsil.data.torch_dataset import NPZIQDataset
from rfsil.evaluation.channel_robustness import (
    SeedSweepMetrics,
    load_seed_sweep_metrics,
)
from rfsil.evaluation.confusion_robustness import (
    ConditionConfusionSummary,
    load_condition_confusions,
)
from rfsil.evaluation.equalizer_correction_analysis import (
    CorrectionStatisticsAccumulator,
    aggregate_seed_correction_statistics,
)
from rfsil.evaluation.mitigation_comparison import (
    compare_model_families,
)
from rfsil.models.model_factory import (
    create_model_from_checkpoint,
)
from rfsil.models.residual_equalizer_cnn import (
    ResidualEqualizerIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare original and multipath-mitigated "
            "RF classifiers."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    """Resolve one project-relative path."""
    path = Path(value)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Configuration must be a mapping."
        )

    return content


def load_model_results(
    *,
    model_name: str,
    content: dict[str, Any],
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
    class_names: tuple[str, ...],
) -> tuple[
    dict[str, SeedSweepMetrics],
    dict[str, ConditionConfusionSummary],
]:
    """Load metrics and predictions for one model family."""
    condition_content = content["conditions"]

    metrics: dict[str, SeedSweepMetrics] = {}
    confusions: dict[
        str,
        ConditionConfusionSummary,
    ] = {}

    for condition in conditions:
        paths = condition_content[condition]

        metrics[condition] = (
            load_seed_sweep_metrics(
                resolve_path(
                    paths["aggregate_metrics"]
                ),
                condition,
            )
        )
        confusions[condition] = (
            load_condition_confusions(
                condition=condition,
                predictions_directory=resolve_path(
                    paths[
                        "predictions_directory"
                    ]
                ),
                seeds=seeds,
                class_names=class_names,
            )
        )

    print(
        f"Loaded model family: {model_name}"
    )

    return metrics, confusions


def resolve_device(
    value: object,
) -> torch.device:
    """Resolve an analysis device."""
    device_name = str(value).strip().lower()

    if device_name == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device_name)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    return device


def analyze_equalizer_corrections(
    *,
    content: dict[str, Any],
    conditions: tuple[str, ...],
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Measure learned IQ corrections by condition and seed."""
    checkpoint_directory = resolve_path(
        content["checkpoint_directory"]
    )
    datasets = content["datasets"]

    if not isinstance(datasets, dict):
        raise ValueError(
            "correction_analysis.datasets "
            "must be a mapping."
        )

    device = resolve_device(
        content.get("device", "auto")
    )
    batch_size = int(
        content.get("batch_size", 128)
    )
    num_workers = int(
        content.get("num_workers", 0)
    )
    pin_memory = bool(
        content.get("pin_memory", True)
    )

    if batch_size <= 0:
        raise ValueError(
            "correction-analysis batch_size "
            "must be positive."
        )

    if num_workers < 0:
        raise ValueError(
            "correction-analysis num_workers "
            "must not be negative."
        )

    condition_results: dict[str, Any] = {}

    for condition in conditions:
        if condition not in datasets:
            raise ValueError(
                "Correction analysis is missing "
                f"dataset {condition!r}."
            )

        dataset = NPZIQDataset(
            resolve_path(datasets[condition])
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(
                pin_memory
                and device.type == "cuda"
            ),
        )

        per_seed: dict[
            int,
            dict[str, int | float],
        ] = {}

        for seed in seeds:
            checkpoint_path = (
                checkpoint_directory
                / f"seed_{seed}"
                / "best_model.pt"
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    checkpoint_path
                )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            model, _ = (
                create_model_from_checkpoint(
                    checkpoint
                )
            )

            if not isinstance(
                model,
                ResidualEqualizerIQCNN,
            ):
                raise TypeError(
                    "Correction analysis requires "
                    "ResidualEqualizerIQCNN "
                    "checkpoints."
                )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )
            model.to(device)
            model.eval()

            accumulator = (
                CorrectionStatisticsAccumulator()
            )

            with torch.inference_mode():
                for batch in loader:
                    inputs = batch["iq"].to(
                        device,
                        non_blocking=(
                            device.type == "cuda"
                        ),
                    )
                    corrections = (
                        model.equalizer
                        .predict_correction(inputs)
                    )

                    accumulator.update(
                        inputs,
                        corrections,
                    )

            per_seed[seed] = (
                accumulator.finalize()
            )

        condition_results[condition] = (
            aggregate_seed_correction_statistics(
                per_seed
            )
        )

        print(
            "Analyzed equalizer corrections:",
            condition,
        )

    return {
        "device": str(device),
        "checkpoint_directory": str(
            content["checkpoint_directory"]
        ),
        "conditions": condition_results,
    }


def create_overview_figure(
    summary: dict[str, Any],
    output_path: Path,
    original_name: str,
    mitigated_name: str,
    comparison_title: str,
) -> None:
    """Plot overall and class-specific mitigation effects."""
    conditions = summary["condition_order"]
    display_conditions = [
        condition.title()
        for condition in conditions
    ]
    indices = np.arange(len(conditions))

    original_means = np.asarray(
        [
            summary["conditions"][condition][
                "original"
            ]["overall_mean"]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    original_stds = np.asarray(
        [
            summary["conditions"][condition][
                "original"
            ][
                "overall_standard_deviation"
            ]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    mitigated_means = np.asarray(
        [
            summary["conditions"][condition][
                "mitigated"
            ]["overall_mean"]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    mitigated_stds = np.asarray(
        [
            summary["conditions"][condition][
                "mitigated"
            ][
                "overall_standard_deviation"
            ]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    improvements = np.asarray(
        [
            summary["conditions"][condition][
                "improvement"
            ]["overall_mean"]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    improvement_stds = np.asarray(
        [
            summary["conditions"][condition][
                "improvement"
            ][
                "overall_standard_deviation"
            ]
            for condition in conditions
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
    )

    axes[0, 0].errorbar(
        indices,
        original_means,
        yerr=original_stds,
        marker="o",
        capsize=4,
        label=original_name,
    )
    axes[0, 0].errorbar(
        indices,
        mitigated_means,
        yerr=mitigated_stds,
        marker="o",
        capsize=4,
        label=mitigated_name,
    )
    axes[0, 0].set_xticks(
        indices,
        display_conditions,
    )
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title(
        "Overall Accuracy by Channel Condition"
    )
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].bar(
        indices,
        improvements,
        yerr=improvement_stds,
        capsize=4,
    )
    axes[0, 1].axhline(
        0.0,
        linewidth=1.0,
    )
    axes[0, 1].set_xticks(
        indices,
        display_conditions,
    )
    axes[0, 1].set_ylabel(
        "Accuracy change"
    )
    axes[0, 1].set_title(
        "Paired Improvement from Mitigation"
    )
    axes[0, 1].grid(
        axis="y",
        alpha=0.3,
    )

    class_names = summary["class_names"]

    for class_name in class_names:
        class_values = [
            summary["conditions"][condition][
                "improvement"
            ]["class_accuracy"][class_name]
            for condition in conditions
        ]

        axes[1, 0].plot(
            indices,
            class_values,
            marker="o",
            label=class_name,
        )

    axes[1, 0].axhline(
        0.0,
        linewidth=1.0,
    )
    axes[1, 0].set_xticks(
        indices,
        display_conditions,
    )
    axes[1, 0].set_ylabel(
        "Class accuracy change"
    )
    axes[1, 0].set_title(
        "Per-Class Mitigation Effect"
    )
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    severe = summary["conditions"]["severe"]
    snr_values = np.asarray(
        summary["snr_values_db"],
        dtype=np.float64,
    )
    original_snr = np.asarray(
        [
            severe["original"][
                "accuracy_by_snr_mean"
            ][str(snr)]
            for snr in summary["snr_values_db"]
        ],
        dtype=np.float64,
    )
    mitigated_snr = np.asarray(
        [
            severe["mitigated"][
                "accuracy_by_snr_mean"
            ][str(snr)]
            for snr in summary["snr_values_db"]
        ],
        dtype=np.float64,
    )

    axes[1, 1].plot(
        snr_values,
        original_snr,
        marker="o",
        label=original_name,
    )
    axes[1, 1].plot(
        snr_values,
        mitigated_snr,
        marker="o",
        label=mitigated_name,
    )
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title(
        "Severe Multipath Accuracy by SNR"
    )
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    figure.suptitle(comparison_title)
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


def create_confusion_figure(
    summary: dict[str, Any],
    output_path: Path,
    original_name: str,
    mitigated_name: str,
    comparison_title: str,
) -> None:
    """Compare moderate and severe pooled confusion matrices."""
    class_names = summary["class_names"]
    selected_conditions = (
        "moderate",
        "severe",
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(11, 10),
        constrained_layout=True,
    )

    image = None

    for row, condition in enumerate(
        selected_conditions
    ):
        condition_result = summary[
            "conditions"
        ][condition]

        for column, (
            model_key,
            model_name,
        ) in enumerate(
            (
                ("original", original_name),
                ("mitigated", mitigated_name),
            )
        ):
            matrix = np.asarray(
                condition_result[
                    "pooled_normalized_confusion"
                ][model_key],
                dtype=np.float64,
            )
            axis = axes[row, column]

            image = axis.imshow(
                matrix,
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_title(
                f"{condition.title()} - "
                f"{model_name}"
            )
            axis.set_xlabel(
                "Predicted class"
            )
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

            for true_index in range(
                len(class_names)
            ):
                for predicted_index in range(
                    len(class_names)
                ):
                    axis.text(
                        predicted_index,
                        true_index,
                        (
                            f"{matrix[
                                true_index,
                                predicted_index
                            ]:.2f}"
                        ),
                        ha="center",
                        va="center",
                        fontsize=9,
                    )

    if image is None:
        raise RuntimeError(
            "No confusion matrix was plotted."
        )

    figure.colorbar(
        image,
        ax=list(axes.ravel()),
        fraction=0.025,
        pad=0.02,
        label="Row-normalized frequency",
    )
    figure.suptitle(
        f"{comparison_title}: "
        "Pooled Five-Seed Confusion Matrices"
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
    """Run the direct mitigation comparison."""
    arguments = parse_arguments()
    configuration = load_yaml(
        resolve_path(arguments.config)
    )

    conditions = tuple(
        str(value)
        for value in configuration[
            "condition_order"
        ]
    )
    seeds = tuple(
        int(value)
        for value in configuration["seeds"]
    )
    class_names = tuple(
        str(value)
        for value in configuration[
            "class_names"
        ]
    )

    model_content = configuration["models"]
    original_content = model_content[
        "original"
    ]
    mitigated_content = model_content[
        "mitigated"
    ]

    original_metrics, original_confusions = (
        load_model_results(
            model_name="original",
            content=original_content,
            conditions=conditions,
            seeds=seeds,
            class_names=class_names,
        )
    )
    mitigated_metrics, mitigated_confusions = (
        load_model_results(
            model_name="mitigated",
            content=mitigated_content,
            conditions=conditions,
            seeds=seeds,
            class_names=class_names,
        )
    )

    summary = compare_model_families(
        original_metrics=original_metrics,
        mitigated_metrics=mitigated_metrics,
        original_confusions=(
            original_confusions
        ),
        mitigated_confusions=(
            mitigated_confusions
        ),
        condition_order=conditions,
        target_class=str(
            configuration["target_class"]
        ),
    )
    summary["experiment_name"] = str(
        configuration["experiment_name"]
    )

    comparison_title = str(
        configuration.get(
            "comparison_title",
            configuration["experiment_name"],
        )
    )
    summary["comparison_title"] = (
        comparison_title
    )

    correction_content = configuration.get(
        "correction_analysis"
    )

    if correction_content is not None:
        if not isinstance(
            correction_content,
            dict,
        ):
            raise ValueError(
                "correction_analysis must be "
                "a mapping."
            )

        summary["correction_analysis"] = (
            analyze_equalizer_corrections(
                content=correction_content,
                conditions=conditions,
                seeds=seeds,
            )
        )

    output_content = configuration["output"]
    summary_path = resolve_path(
        output_content["summary_json"]
    )
    overview_path = resolve_path(
        output_content[
            "overview_figure_path"
        ]
    )
    confusion_path = resolve_path(
        output_content[
            "confusion_figure_path"
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
        ),
        encoding="utf-8",
    )

    original_name = str(
        original_content["display_name"]
    )
    mitigated_name = str(
        mitigated_content["display_name"]
    )

    create_overview_figure(
        summary,
        overview_path,
        original_name,
        mitigated_name,
        comparison_title,
    )
    create_confusion_figure(
        summary,
        confusion_path,
        original_name,
        mitigated_name,
        comparison_title,
    )

    print(comparison_title)
    print("")

    for condition in conditions:
        result = summary["conditions"][
            condition
        ]
        improvement = result[
            "improvement"
        ]

        print(
            f"{condition:8s} | "
            f"original="
            f"{result['original']['overall_mean']:.4f} "
            f"| mitigated="
            f"{result['mitigated']['overall_mean']:.4f} "
            f"| change="
            f"{improvement['overall_mean']:+.4f} "
            f"+/- "
            f"{improvement[
                'overall_standard_deviation'
            ]:.4f}"
        )

    print("")
    print(
        f"Misclassification reduction toward "
        f"{summary['target_class']}"
    )

    for condition in (
        "moderate",
        "severe",
    ):
        print(condition.upper())

        errors = summary["conditions"][
            condition
        ]["misclassification_to_target"]

        for class_name, values in (
            errors.items()
        ):
            print(
                f"  {class_name:5s} | "
                f"{values['original_rate']:.4f} "
                f"-> "
                f"{values['mitigated_rate']:.4f} "
                f"| reduction="
                f"{values[
                    'absolute_reduction'
                ]:+.4f}"
            )

    correction_analysis = summary.get(
        "correction_analysis"
    )

    if correction_analysis is not None:
        print("")
        print("Equalizer correction statistics")

        for condition in conditions:
            aggregate = correction_analysis[
                "conditions"
            ][condition]["aggregate"]

            mean_absolute = aggregate[
                "mean_absolute_correction"
            ]
            relative_rms = aggregate[
                "relative_correction_rms"
            ]

            print(
                f"{condition:8s} | "
                f"mean abs="
                f"{mean_absolute['mean']:.4f} "
                f"+/- "
                f"{mean_absolute[
                    'standard_deviation'
                ]:.4f} | "
                f"relative RMS="
                f"{relative_rms['mean']:.4f} "
                f"+/- "
                f"{relative_rms[
                    'standard_deviation'
                ]:.4f}"
            )

    print("")
    print(f"Summary: {summary_path}")
    print(f"Overview: {overview_path}")
    print(f"Confusions: {confusion_path}")


if __name__ == "__main__":
    main()
