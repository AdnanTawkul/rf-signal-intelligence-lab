from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)
from rfsil.evaluation.class_snr_seed import (
    ClassSNRSeedAggregate,
    ClassSNRSeedResult,
    aggregate_class_snr_seed_results,
)
from rfsil.evaluation.classification import collect_predictions
from rfsil.evaluation.error_analysis import (
    compute_class_snr_analysis,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare two checkpoint groups using class-by-SNR accuracy."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/compare_batchnorm_groupnorm_class_snr_v1.yaml"
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
        raise ValueError(
            "Comparison configuration must be a YAML mapping."
        )

    return content


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> BaselineIQCNN:
    """Reconstruct a trained model from checkpoint metadata."""
    model_content = checkpoint["model_configuration"]

    configuration = BaselineCNNConfig(
        in_channels=int(model_content["in_channels"]),
        num_classes=int(model_content["num_classes"]),
        channels=tuple(
            int(value)
            for value in model_content["channels"]
        ),
        kernel_size=int(model_content["kernel_size"]),
        dropout=float(model_content["dropout"]),
        normalize_input_rms=bool(
            model_content.get("normalize_input_rms", False)
        ),
        normalization=str(
            model_content.get("normalization", "batch")
        ),
        group_norm_groups=int(
            model_content.get("group_norm_groups", 8)
        ),
    )

    model = BaselineIQCNN(configuration)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_checkpoint_group(
    checkpoint_directory: Path,
    seeds: list[int],
    data_loader: object,
    device: torch.device,
) -> tuple[
    list[str],
    ClassSNRSeedAggregate,
]:
    """Evaluate one checkpoint family by class and SNR."""
    class_names: list[str] | None = None
    seed_results: list[ClassSNRSeedResult] = []

    for seed in seeds:
        checkpoint_path = (
            checkpoint_directory
            / f"seed_{seed}"
            / "best_model.pt"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        current_class_names = [
            str(name).upper()
            for name in checkpoint["class_names"]
        ]

        if class_names is None:
            class_names = current_class_names
        elif class_names != current_class_names:
            raise ValueError(
                "All checkpoints must use identical class names."
            )

        model = build_model_from_checkpoint(
            checkpoint,
            device,
        )
        predictions = collect_predictions(
            model=model,
            data_loader=data_loader,
            device=device,
        )

        analysis = compute_class_snr_analysis(
            labels=predictions.labels,
            predictions=predictions.predictions,
            snr_db=predictions.snr_db,
            num_classes=len(current_class_names),
        )

        seed_results.append(
            ClassSNRSeedResult(
                seed=seed,
                snr_values_db=analysis.snr_values_db,
                accuracy=analysis.accuracy,
            )
        )

        print(
            f"{checkpoint_directory.name} seed {seed}: complete"
        )

    if class_names is None:
        raise RuntimeError("No checkpoints were evaluated.")

    return (
        class_names,
        aggregate_class_snr_seed_results(seed_results),
    )


def annotate_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    signed: bool,
) -> None:
    """Write numeric values into a heatmap."""
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]

            label = (
                "N/A"
                if np.isnan(value)
                else (
                    f"{float(value):+.2f}"
                    if signed
                    else f"{float(value):.2f}"
                )
            )

            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
            )


def configure_axis(
    axis: plt.Axes,
    class_names: list[str],
    snr_values_db: np.ndarray,
    title: str,
) -> None:
    """Configure class and SNR tick labels."""
    axis.set_title(title)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("True modulation class")
    axis.set_xticks(
        np.arange(len(snr_values_db)),
        labels=[
            f"{float(value):.0f}"
            for value in snr_values_db
        ],
    )
    axis.set_yticks(
        np.arange(len(class_names)),
        labels=class_names,
    )


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    content = load_yaml(config_path)

    seeds = [int(seed) for seed in content["seeds"]]

    if not seeds:
        raise ValueError("At least one seed is required.")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed values must be unique.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    evaluation_content = content["evaluation"]
    dataset = NPZIQDataset(
        resolve_project_path(content["test_path"])
    )
    data_loader = create_data_loader(
        dataset,
        DataLoaderConfig(
            batch_size=int(evaluation_content["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation_content["num_workers"]),
            pin_memory=(
                bool(evaluation_content["pin_memory"])
                and torch.cuda.is_available()
            ),
            seed=2026,
        ),
    )

    model_content = content["models"]
    batchnorm_directory = resolve_project_path(
        model_content["batchnorm"]["checkpoint_directory"]
    )
    groupnorm_directory = resolve_project_path(
        model_content["groupnorm"]["checkpoint_directory"]
    )

    print(f"Device: {device}")
    print(f"Test examples: {len(dataset)}")

    batchnorm_classes, batchnorm = evaluate_checkpoint_group(
        checkpoint_directory=batchnorm_directory,
        seeds=seeds,
        data_loader=data_loader,
        device=device,
    )
    groupnorm_classes, groupnorm = evaluate_checkpoint_group(
        checkpoint_directory=groupnorm_directory,
        seeds=seeds,
        data_loader=data_loader,
        device=device,
    )

    if batchnorm_classes != groupnorm_classes:
        raise ValueError(
            "BatchNorm and GroupNorm class names do not match."
        )

    if not np.allclose(
        batchnorm.snr_values_db,
        groupnorm.snr_values_db,
    ):
        raise ValueError(
            "BatchNorm and GroupNorm SNR values do not match."
        )

    difference = (
        groupnorm.accuracy_mean
        - batchnorm.accuracy_mean
    ).astype(np.float32)

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    figure_path = resolve_project_path(
        output_content["figure_path"]
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = output_directory / "comparison.json"

    report = {
        "format_version": 1,
        "experiment_name": str(content["experiment_name"]),
        "seeds": seeds,
        "class_names": batchnorm_classes,
        "snr_values_db": [
            float(value)
            for value in batchnorm.snr_values_db
        ],
        "batchnorm_accuracy_mean": (
            batchnorm.accuracy_mean.tolist()
        ),
        "batchnorm_accuracy_std": (
            batchnorm.accuracy_std.tolist()
        ),
        "groupnorm_accuracy_mean": (
            groupnorm.accuracy_mean.tolist()
        ),
        "groupnorm_accuracy_std": (
            groupnorm.accuracy_std.tolist()
        ),
        "groupnorm_minus_batchnorm": difference.tolist(),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 6),
    )

    batchnorm_image = axes[0].imshow(
        batchnorm.accuracy_mean,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    configure_axis(
        axes[0],
        batchnorm_classes,
        batchnorm.snr_values_db,
        "BatchNorm Mean Accuracy",
    )
    annotate_heatmap(
        axes[0],
        batchnorm.accuracy_mean,
        signed=False,
    )
    figure.colorbar(
        batchnorm_image,
        ax=axes[0],
        label="Accuracy",
    )

    groupnorm_image = axes[1].imshow(
        groupnorm.accuracy_mean,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    configure_axis(
        axes[1],
        groupnorm_classes,
        groupnorm.snr_values_db,
        "GroupNorm Mean Accuracy",
    )
    annotate_heatmap(
        axes[1],
        groupnorm.accuracy_mean,
        signed=False,
    )
    figure.colorbar(
        groupnorm_image,
        ax=axes[1],
        label="Accuracy",
    )

    maximum_difference = max(
        float(np.nanmax(np.abs(difference))),
        0.01,
    )
    difference_image = axes[2].imshow(
        difference,
        vmin=-maximum_difference,
        vmax=maximum_difference,
        aspect="auto",
        cmap="coolwarm",
    )
    configure_axis(
        axes[2],
        groupnorm_classes,
        groupnorm.snr_values_db,
        "GroupNorm Minus BatchNorm",
    )
    annotate_heatmap(
        axes[2],
        difference,
        signed=True,
    )
    figure.colorbar(
        difference_image,
        ax=axes[2],
        label="Accuracy difference",
    )

    figure.suptitle(
        "Five-Seed Class-by-SNR Normalization Comparison"
    )
    figure.tight_layout()
    figure.savefig(
        figure_path,
        dpi=160,
    )
    plt.close(figure)

    eight_psk_index = batchnorm_classes.index("8PSK")

    print("")
    print("8PSK accuracy by SNR")
    for snr_value, batch_value, group_value, delta in zip(
        batchnorm.snr_values_db,
        batchnorm.accuracy_mean[eight_psk_index],
        groupnorm.accuracy_mean[eight_psk_index],
        difference[eight_psk_index],
        strict=True,
    ):
        print(
            f"{float(snr_value):.1f} dB | "
            f"BatchNorm={float(batch_value):.4f} | "
            f"GroupNorm={float(group_value):.4f} | "
            f"change={float(delta):+.4f}"
        )

    finite_difference = np.where(
        np.isnan(difference),
        0.0,
        difference,
    )
    largest_gain_index = np.unravel_index(
        int(np.argmax(finite_difference)),
        difference.shape,
    )
    largest_loss_index = np.unravel_index(
        int(np.argmin(finite_difference)),
        difference.shape,
    )

    print("")
    print(
        "Largest GroupNorm gain: "
        f"{batchnorm_classes[largest_gain_index[0]]} at "
        f"{float(batchnorm.snr_values_db[largest_gain_index[1]]):.1f} dB, "
        f"change={float(difference[largest_gain_index]):+.4f}"
    )
    print(
        "Largest GroupNorm loss: "
        f"{batchnorm_classes[largest_loss_index[0]]} at "
        f"{float(batchnorm.snr_values_db[largest_loss_index[1]]):.1f} dB, "
        f"change={float(difference[largest_loss_index]):+.4f}"
    )
    print(f"Report: {report_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
