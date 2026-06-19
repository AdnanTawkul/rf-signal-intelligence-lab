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

from rfsil.data.stratified_subset import create_class_snr_stratified_subset
from rfsil.data.synthetic import MODULATION_CLASSES
from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)
from rfsil.models.baseline_cnn import (
    count_trainable_parameters,
)
from rfsil.models.model_factory import (
    create_model_from_mapping,
)
from rfsil.training.backbone_initialization import (
    initialize_frozen_backbone_from_checkpoint,
)
from rfsil.training.budget import resolve_training_budget
from rfsil.training.engine import (
    run_evaluation_epoch,
    run_training_epoch,
    set_global_seed,
)
from rfsil.training.initialization import (
    initialize_encoder_from_ssl_checkpoint,
)
from rfsil.training.losses import ClassSNRWeightedCrossEntropyLoss

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

    epochs_value = training_content.get(
        "epochs"
    )
    target_optimizer_steps_value = (
        training_content.get(
            "target_optimizer_steps"
        )
    )
    require_exact_optimizer_steps = (
        training_content.get(
            "require_exact_optimizer_steps",
            True,
        )
    )
    drop_last = training_content.get(
        "drop_last",
        False,
    )

    batch_size = int(training_content["batch_size"])
    learning_rate = float(training_content["learning_rate"])
    weight_decay = float(training_content["weight_decay"])
    num_workers = int(training_content["num_workers"])
    pin_memory = bool(training_content["pin_memory"])

    if not isinstance(drop_last, bool):
        raise ValueError(
            "training.drop_last must be a boolean."
        )

    if not isinstance(
        require_exact_optimizer_steps,
        bool,
    ):
        raise ValueError(
            "training.require_exact_optimizer_steps "
            "must be a boolean."
        )

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

    labeled_subset_metadata: (
        dict[str, int] | None
    ) = None

    examples_per_class_snr_value = (
        training_content.get(
            "examples_per_class_snr"
        )
    )

    if examples_per_class_snr_value is not None:
        subset_seed_value = training_content.get(
            "subset_seed",
            seed,
        )
        full_training_example_count = len(
            train_dataset
        )

        train_dataset = (
            create_class_snr_stratified_subset(
                dataset=train_dataset,
                examples_per_stratum=(
                    examples_per_class_snr_value
                ),
                seed=subset_seed_value,
            )
        )

        labeled_subset_metadata = {
            "strategy": "class_snr_stratified",
            "examples_per_class_snr": int(
                examples_per_class_snr_value
            ),
            "subset_seed": int(
                subset_seed_value
            ),
            "full_training_examples": int(
                full_training_example_count
            ),
            "selected_training_examples": int(
                len(train_dataset)
            ),
        }


    training_budget = resolve_training_budget(
        example_count=len(train_dataset),
        batch_size=batch_size,
        epochs=epochs_value,
        target_optimizer_steps=(
            target_optimizer_steps_value
        ),
        drop_last=drop_last,
        require_exact=(
            require_exact_optimizer_steps
        ),
    )
    training_budget_metadata = asdict(
        training_budget
    )
    epochs = training_budget.epochs

    train_loader = create_data_loader(
        train_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
            seed=seed,
        ),
    )
    if (
        len(train_loader)
        != training_budget.steps_per_epoch
    ):
        raise RuntimeError(
            "Resolved training steps do not "
            "match the DataLoader length."
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

    (
        model,
        model_configuration,
    ) = create_model_from_mapping(
        model_content
    )

    initialization_content = content.get(
        "initialization"
    )
    initialization_metadata: (
        dict[str, object] | None
    ) = None

    if initialization_content is not None:
        if not isinstance(
            initialization_content,
            dict,
        ):
            raise ValueError(
                "initialization must be "
                "a YAML mapping."
            )

        initialization_type = str(
            initialization_content.get(
                "type",
                "ssl_encoder",
            )
        ).strip().lower()

        if initialization_type == "ssl_encoder":
            checkpoint_value = (
                initialization_content.get(
                    "encoder_checkpoint_path"
                )
            )

            if checkpoint_value is None:
                raise ValueError(
                    "initialization."
                    "encoder_checkpoint_path "
                    "is required."
                )

            initialization_path = (
                resolve_project_path(
                    str(checkpoint_value)
                )
            )

            imported_initialization = (
                initialize_encoder_from_ssl_checkpoint(
                    model,
                    initialization_path,
                )
            )

            resolved_path = (
                initialization_path.resolve()
            )

            try:
                serialized_path = (
                    resolved_path
                    .relative_to(
                        PROJECT_ROOT.resolve()
                    )
                    .as_posix()
                )
            except ValueError:
                serialized_path = (
                    resolved_path.as_posix()
                )

            initialization_metadata = {
                "type": "ssl_encoder",
                "method": (
                    imported_initialization.method
                ),
                "experiment_name": (
                    imported_initialization
                    .experiment_name
                ),
                "seed": (
                    imported_initialization.seed
                ),
                "best_epoch": (
                    imported_initialization
                    .best_epoch
                ),
                "encoder_checkpoint_path": (
                    serialized_path
                ),
                "classifier_initialized_fresh": True,
            }

        elif initialization_type == (
            "frozen_supervised_backbone"
        ):
            checkpoint_value = (
                initialization_content.get(
                    "checkpoint_path"
                )
            )

            if checkpoint_value is None:
                raise ValueError(
                    "initialization.checkpoint_path "
                    "is required."
                )

            initialization_path = (
                resolve_project_path(
                    str(checkpoint_value)
                )
            )

            imported_initialization = (
                initialize_frozen_backbone_from_checkpoint(
                    model,
                    initialization_path,
                )
            )

            resolved_path = (
                initialization_path.resolve()
            )

            try:
                serialized_path = (
                    resolved_path
                    .relative_to(
                        PROJECT_ROOT.resolve()
                    )
                    .as_posix()
                )
            except ValueError:
                serialized_path = (
                    resolved_path.as_posix()
                )

            initialization_metadata = {
                "type": (
                    "frozen_supervised_backbone"
                ),
                "method": (
                    "supervised_baseline_backbone"
                ),
                "experiment_name": (
                    imported_initialization
                    .experiment_name
                ),
                "seed": (
                    imported_initialization.seed
                ),
                "best_epoch": (
                    imported_initialization
                    .best_epoch
                ),
                "checkpoint_path": (
                    serialized_path
                ),
                "backbone_frozen": True,
                "classifier_initialized_fresh": False,
            }

        else:
            raise ValueError(
                "Unsupported initialization type: "
                f"{initialization_type!r}."
            )

    model = model.to(device)
    loss_function = nn.CrossEntropyLoss()

    targeted_weighting_content = training_content.get(
        "targeted_weighting",
        {},
    )

    if targeted_weighting_content is None:
        targeted_weighting_content = {}

    if not isinstance(targeted_weighting_content, dict):
        raise ValueError(
            "training.targeted_weighting must be a mapping."
        )

    enabled_value = targeted_weighting_content.get(
        "enabled",
        False,
    )

    if not isinstance(enabled_value, bool):
        raise ValueError(
            "training.targeted_weighting.enabled must be a boolean."
        )

    metadata_loss_function: nn.Module | None = None
    targeted_weighting_configuration: dict[str, object] = {
        "enabled": enabled_value,
    }

    if enabled_value:
        required_fields = (
            "target_class_index",
            "target_snr_values_db",
            "target_weight",
        )
        missing_fields = [
            field
            for field in required_fields
            if field not in targeted_weighting_content
        ]

        if missing_fields:
            raise ValueError(
                "Missing targeted-weighting fields: "
                + ", ".join(missing_fields)
            )

        target_snr_values = targeted_weighting_content[
            "target_snr_values_db"
        ]

        if not isinstance(target_snr_values, list):
            raise ValueError(
                "target_snr_values_db must be a YAML list."
            )

        targeted_loss = ClassSNRWeightedCrossEntropyLoss(
            target_class_index=targeted_weighting_content[
                "target_class_index"
            ],
            target_snr_values_db=tuple(
                float(value)
                for value in target_snr_values
            ),
            target_weight=float(
                targeted_weighting_content["target_weight"]
            ),
            snr_tolerance=float(
                targeted_weighting_content.get(
                    "snr_tolerance",
                    1e-4,
                )
            ),
        )

        metadata_loss_function = targeted_loss
        targeted_weighting_configuration = {
            "enabled": True,
            "target_class_index": (
                targeted_loss.target_class_index
            ),
            "target_snr_values_db": list(
                targeted_loss.target_snr_values_db
            ),
            "target_weight": targeted_loss.target_weight,
            "snr_tolerance": targeted_loss.snr_tolerance,
        }

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError(
            "The model has no trainable parameters."
        )

    optimizer = AdamW(
        trainable_parameters,
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

    if labeled_subset_metadata is not None:
        print(
            "Full training examples: "
            f"{labeled_subset_metadata['full_training_examples']}"
        )
        print(
            "Examples per class-SNR stratum: "
            f"{labeled_subset_metadata['examples_per_class_snr']}"
        )
        print(
            "Labeled-subset seed: "
            f"{labeled_subset_metadata['subset_seed']}"
        )
    print(f"Validation examples: {len(validation_dataset)}")
    print(
        "Training batches per epoch: "
        f"{training_budget.steps_per_epoch}"
    )
    print(
        "Training epochs: "
        f"{training_budget.epochs}"
    )
    print(
        "Target optimizer steps: "
        f"{training_budget.target_optimizer_steps}"
    )
    print(
        "Actual optimizer steps: "
        f"{training_budget.actual_optimizer_steps}"
    )
    print(
        "Exact target match: "
        f"{training_budget.exact_match}"
    )
    print(
        "Trainable parameters: "
        f"{count_trainable_parameters(model)}"
    )

    if initialization_metadata is None:
        print("Initialization: random")
    elif (
        initialization_metadata["type"]
        == "ssl_encoder"
    ):
        print(
            "Initialization: "
            f"{initialization_metadata['method']} "
            "SSL encoder"
        )
        print(
            "Initialization checkpoint: "
            f"{initialization_metadata[
                'encoder_checkpoint_path'
            ]}"
        )
        print(
            "Classifier initialized fresh: yes"
        )
    else:
        print(
            "Initialization: frozen supervised "
            "backbone"
        )
        print(
            "Initialization checkpoint: "
            f"{initialization_metadata[
                'checkpoint_path'
            ]}"
        )
        print("Backbone frozen: yes")
        print(
            "Classifier initialized fresh: no"
        )

    for epoch in range(1, epochs + 1):
        train_metrics = run_training_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            metadata_loss_function=metadata_loss_function,
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
            "training_loss_configuration": targeted_weighting_configuration,
            "model_configuration": asdict(model_configuration),
            "model_state_dict": best_model_state,
            "class_names": [
                modulation.value
                for modulation in MODULATION_CLASSES
            ],
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_validation_accuracy,
            "seed": seed,
            "initialization": initialization_metadata,
            "labeled_subset": labeled_subset_metadata,
            "training_budget": training_budget_metadata,
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
        "training_loss_configuration": targeted_weighting_configuration,
        "seed": seed,
        "initialization": initialization_metadata,
        "labeled_subset": labeled_subset_metadata,
        "training_budget": training_budget_metadata,
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
