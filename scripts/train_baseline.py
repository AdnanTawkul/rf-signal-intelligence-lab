from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn
from torch.optim import AdamW

from rfsil.data.synthetic import MODULATION_CLASSES
from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
    count_trainable_parameters,
)
from rfsil.training.engine import (
    run_evaluation_epoch,
    run_training_epoch,
    set_global_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the baseline RF modulation CNN.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_baseline_smoke.yaml"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed stored in the YAML configuration.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Override the experiment name.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override the result output directory.",
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
        help="Override the training-curve output path.",
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(content, dict):
        raise ValueError("Training configuration must be a YAML mapping.")

    return content


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    content = load_yaml(config_path)

    training_content = content["training"]
    model_content = content["model"]
    dataset_content = content["dataset"]
    output_content = content["output"]

    experiment_name = (
        arguments.experiment_name
        if arguments.experiment_name is not None
        else str(content["experiment_name"])
    )
    seed = (
        int(arguments.seed)
        if arguments.seed is not None
        else int(content["seed"])
    )

    epochs = int(training_content["epochs"])
    batch_size = int(training_content["batch_size"])
    learning_rate = float(training_content["learning_rate"])
    weight_decay = float(training_content["weight_decay"])
    num_workers = int(training_content["num_workers"])
    pin_memory = bool(training_content["pin_memory"])

    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    set_global_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_dataset = NPZIQDataset(
        resolve_project_path(dataset_content["train_path"])
    )
    validation_dataset = NPZIQDataset(
        resolve_project_path(dataset_content["validation_path"])
    )

    train_loader = create_data_loader(
        train_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            seed=seed,
        ),
    )
    validation_loader = create_data_loader(
        validation_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            seed=seed,
        ),
    )

    model_configuration = BaselineCNNConfig(
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

    model = BaselineIQCNN(model_configuration).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_directory_value = (
        arguments.output_directory
        if arguments.output_directory is not None
        else Path(str(output_content["directory"]))
    )
    figure_path_value = (
        arguments.figure_path
        if arguments.figure_path is not None
        else Path(str(output_content["figure_path"]))
    )

    output_directory = resolve_project_path(
        output_directory_value
    )
    figure_path = resolve_project_path(
        figure_path_value
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_validation_accuracy = float("-inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None

    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")
    print(
        "Trainable parameters: "
        f"{count_trainable_parameters(model)}"
    )

    for epoch in range(1, epochs + 1):
        train_metrics = run_training_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )
        validation_metrics = run_evaluation_epoch(
            model=model,
            data_loader=validation_loader,
            loss_function=loss_function,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics.loss:.4f} | "
            f"train acc {train_metrics.accuracy:.3f} | "
            f"val loss {validation_metrics.loss:.4f} | "
            f"val acc {validation_metrics.accuracy:.3f}"
        )

        if validation_metrics.accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_metrics.accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(
                model.state_dict()
            )

    if best_model_state is None:
        raise RuntimeError("Training completed without a model checkpoint.")

    checkpoint_path = output_directory / "best_model.pt"
    history_path = output_directory / "history.json"
    summary_path = output_directory / "summary.json"

    torch.save(
        {
            "format_version": 1,
            "experiment_name": experiment_name,
            "model_configuration": asdict(model_configuration),
            "model_state_dict": best_model_state,
            "class_names": [
                modulation.value
                for modulation in MODULATION_CLASSES
            ],
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "seed": seed,
        },
        checkpoint_path,
    )

    history_path.write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    final_record = history[-1]

    summary = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_validation_accuracy,
        "final_train_loss": float(final_record["train_loss"]),
        "final_train_accuracy": float(final_record["train_accuracy"]),
        "final_validation_loss": float(
            final_record["validation_loss"]
        ),
        "final_validation_accuracy": float(
            final_record["validation_accuracy"]
        ),
        "checkpoint_path": str(
            checkpoint_path.relative_to(PROJECT_ROOT)
        ),
        "history_path": str(
            history_path.relative_to(PROJECT_ROOT)
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    epoch_values = [
        int(record["epoch"])
        for record in history
    ]
    train_losses = [
        float(record["train_loss"])
        for record in history
    ]
    validation_losses = [
        float(record["validation_loss"])
        for record in history
    ]
    train_accuracies = [
        float(record["train_accuracy"])
        for record in history
    ]
    validation_accuracies = [
        float(record["validation_accuracy"])
        for record in history
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epoch_values,
        train_losses,
        marker="o",
        label="Train",
    )
    axes[0].plot(
        epoch_values,
        validation_losses,
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epoch_values,
        train_accuracies,
        marker="o",
        label="Train",
    )
    axes[1].plot(
        epoch_values,
        validation_accuracies,
        marker="o",
        label="Validation",
    )
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle(f"{experiment_name} | seed={seed}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
