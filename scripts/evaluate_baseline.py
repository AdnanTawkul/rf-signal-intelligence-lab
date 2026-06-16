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
from rfsil.evaluation.classification import (
    collect_predictions,
    evaluate_predictions,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RF modulation classifier.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/evaluate_baseline_v1.yaml"),
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    path = Path(path_value)

    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError("Evaluation configuration must be a YAML mapping.")

    return content


def plot_confusion_matrix(
    normalized_confusion: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a normalized confusion-matrix figure."""
    figure, axis = plt.subplots(figsize=(7, 6))

    image = axis.imshow(
        normalized_confusion,
        vmin=0.0,
        vmax=1.0,
    )

    axis.set_title("Baseline CNN Normalized Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(
        np.arange(len(class_names)),
        labels=class_names,
    )
    axis.set_yticks(
        np.arange(len(class_names)),
        labels=class_names,
    )

    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = float(normalized_confusion[row, column])
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        label="Fraction of true class",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_accuracy_by_snr(
    snr_values_db: np.ndarray,
    snr_accuracy: np.ndarray,
    output_path: Path,
) -> None:
    """Save classification accuracy as a function of SNR."""
    figure, axis = plt.subplots(figsize=(8, 5))

    axis.plot(
        snr_values_db,
        snr_accuracy,
        marker="o",
    )
    axis.set_title("Baseline CNN Accuracy by SNR")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(snr_values_db)
    axis.grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    content = load_yaml(config_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = resolve_project_path(
        content["checkpoint_path"]
    )
    test_path = resolve_project_path(
        content["test_path"]
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model_content = checkpoint["model_configuration"]
    model_configuration = BaselineCNNConfig(
        in_channels=int(model_content["in_channels"]),
        num_classes=int(model_content["num_classes"]),
        channels=tuple(
            int(value)
            for value in model_content["channels"]
        ),
        kernel_size=int(model_content["kernel_size"]),
        dropout=float(model_content["dropout"]),
    )

    class_names = [
        str(name).upper()
        for name in checkpoint["class_names"]
    ]

    model = BaselineIQCNN(model_configuration)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)

    evaluation_content = content["evaluation"]
    dataset = NPZIQDataset(test_path)
    loader = create_data_loader(
        dataset,
        DataLoaderConfig(
            batch_size=int(evaluation_content["batch_size"]),
            shuffle=False,
            num_workers=int(evaluation_content["num_workers"]),
            pin_memory=(
                bool(evaluation_content["pin_memory"])
                and torch.cuda.is_available()
            ),
            seed=int(checkpoint["seed"]),
        ),
    )

    prediction_results = collect_predictions(
        model=model,
        data_loader=loader,
        device=device,
    )

    evaluation = evaluate_predictions(
        labels=prediction_results.labels,
        predictions=prediction_results.predictions,
        snr_db=prediction_results.snr_db,
        num_classes=model_configuration.num_classes,
    )

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    confusion_path = resolve_project_path(
        output_content["confusion_matrix_figure"]
    )
    accuracy_by_snr_path = resolve_project_path(
        output_content["accuracy_by_snr_figure"]
    )
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    accuracy_by_snr_path.parent.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        evaluation.normalized_confusion_matrix,
        class_names,
        confusion_path,
    )
    plot_accuracy_by_snr(
        evaluation.snr_values_db,
        evaluation.snr_accuracy,
        accuracy_by_snr_path,
    )

    metrics_path = output_directory / "metrics.json"
    predictions_path = output_directory / "predictions.npz"

    metrics = {
        "format_version": 1,
        "experiment_name": str(content["experiment_name"]),
        "checkpoint_path": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
        "test_path": str(
            test_path.relative_to(PROJECT_ROOT)
        ),
        "example_count": evaluation.example_count,
        "overall_accuracy": evaluation.accuracy,
        "class_names": class_names,
        "class_accuracy": {
            class_name: float(accuracy)
            for class_name, accuracy in zip(
                class_names,
                evaluation.class_accuracy,
                strict=True,
            )
        },
        "accuracy_by_snr": {
            str(float(snr_value)): float(accuracy)
            for snr_value, accuracy in zip(
                evaluation.snr_values_db,
                evaluation.snr_accuracy,
                strict=True,
            )
        },
        "confusion_matrix": (
            evaluation.confusion_matrix.tolist()
        ),
        "normalized_confusion_matrix": (
            evaluation.normalized_confusion_matrix.tolist()
        ),
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    np.savez_compressed(
        predictions_path,
        labels=prediction_results.labels,
        predictions=prediction_results.predictions,
        snr_db=prediction_results.snr_db,
    )

    print(f"Device: {device}")
    print(f"Test examples: {evaluation.example_count}")
    print(f"Overall test accuracy: {evaluation.accuracy:.4f}")

    for class_name, accuracy in zip(
        class_names,
        evaluation.class_accuracy,
        strict=True,
    ):
        print(
            f"Class accuracy {class_name}: "
            f"{float(accuracy):.4f}"
        )

    for snr_value, accuracy in zip(
        evaluation.snr_values_db,
        evaluation.snr_accuracy,
        strict=True,
    ):
        print(
            f"Accuracy at {float(snr_value):.1f} dB: "
            f"{float(accuracy):.4f}"
        )

    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Accuracy by SNR: {accuracy_by_snr_path}")


if __name__ == "__main__":
    main()
