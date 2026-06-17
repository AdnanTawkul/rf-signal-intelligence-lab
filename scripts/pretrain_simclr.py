from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import AdamW

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
from rfsil.ssl.augmentations import (
    IQAugmentationConfig,
    RandomIQAugmentation,
)
from rfsil.ssl.contrastive import (
    ProjectionHeadConfig,
    SimCLRModel,
)
from rfsil.ssl.training import (
    ContrastiveEpochMetrics,
    run_contrastive_evaluation_epoch,
    run_contrastive_training_epoch,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain an RF IQ encoder using SimCLR "
            "contrastive learning."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/pretrain_simclr_v1.yaml"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional epoch override for smoke tests."
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_value)

    return (
        path
        if path.is_absolute()
        else PROJECT_ROOT / path
    )


def serialize_project_path(path: Path) -> str:
    """Return a portable project-relative path when possible."""
    resolved_path = path.resolve()
    resolved_root = PROJECT_ROOT.resolve()

    try:
        return resolved_path.relative_to(
            resolved_root
        ).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "SimCLR configuration must be a YAML mapping."
        )

    return content


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_device_generator(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    """Create a seeded generator on the selected device."""
    generator = torch.Generator(
        device=device.type,
    )
    generator.manual_seed(seed)

    return generator


def create_encoder_configuration(
    content: dict[str, Any],
) -> BaselineCNNConfig:
    """Create the GroupNorm CNN encoder configuration."""
    return BaselineCNNConfig(
        in_channels=int(
            content.get("in_channels", 2)
        ),
        num_classes=int(
            content.get("num_classes", 4)
        ),
        channels=tuple(
            int(value)
            for value in content.get(
                "channels",
                [32, 64, 128],
            )
        ),
        kernel_size=int(
            content.get("kernel_size", 7)
        ),
        dropout=float(
            content.get("dropout", 0.2)
        ),
        normalize_input_rms=bool(
            content.get(
                "normalize_input_rms",
                False,
            )
        ),
        normalization=str(
            content.get(
                "normalization",
                "group",
            )
        ),
        group_norm_groups=int(
            content.get(
                "group_norm_groups",
                8,
            )
        ),
    )


def create_projection_configuration(
    content: dict[str, Any],
) -> ProjectionHeadConfig:
    """Create the projection-head configuration."""
    return ProjectionHeadConfig(
        hidden_dimension=int(
            content.get(
                "hidden_dimension",
                256,
            )
        ),
        output_dimension=int(
            content.get(
                "output_dimension",
                128,
            )
        ),
    )


def create_augmentation_configuration(
    content: dict[str, Any],
) -> IQAugmentationConfig:
    """Create the IQ augmentation configuration."""
    return IQAugmentationConfig(
        phase_rotation_probability=float(
            content.get(
                "phase_rotation_probability",
                0.8,
            )
        ),
        max_phase_rotation_rad=float(
            content.get(
                "max_phase_rotation_rad",
                np.pi,
            )
        ),
        amplitude_scale_probability=float(
            content.get(
                "amplitude_scale_probability",
                0.5,
            )
        ),
        amplitude_scale_min=float(
            content.get(
                "amplitude_scale_min",
                0.8,
            )
        ),
        amplitude_scale_max=float(
            content.get(
                "amplitude_scale_max",
                1.25,
            )
        ),
        time_shift_probability=float(
            content.get(
                "time_shift_probability",
                0.5,
            )
        ),
        max_time_shift_samples=int(
            content.get(
                "max_time_shift_samples",
                64,
            )
        ),
        awgn_probability=float(
            content.get(
                "awgn_probability",
                0.5,
            )
        ),
        awgn_snr_db_min=float(
            content.get(
                "awgn_snr_db_min",
                18.0,
            )
        ),
        awgn_snr_db_max=float(
            content.get(
                "awgn_snr_db_max",
                30.0,
            )
        ),
    )


def validate_contrastive_batching(
    example_count: int,
    batch_size: int,
    split_name: str,
) -> None:
    """Prevent a final one-example contrastive batch."""
    if batch_size < 2:
        raise ValueError(
            "Contrastive batch size must be at least two."
        )

    if example_count < 2:
        raise ValueError(
            f"{split_name} split must contain at least "
            "two examples."
        )

    if example_count % batch_size == 1:
        raise ValueError(
            f"{split_name} split would produce a final "
            "one-example batch. Select another batch size."
        )


def state_dict_to_cpu(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Copy one module state dictionary to CPU."""
    return {
        key: value.detach().cpu().clone()
        for key, value
        in module.state_dict().items()
    }


def metrics_to_record(
    epoch: int,
    training: ContrastiveEpochMetrics,
    validation: ContrastiveEpochMetrics,
) -> dict[str, float | int]:
    """Serialize one epoch of metrics."""
    return {
        "epoch": epoch,
        "training_loss": training.loss,
        "training_positive_cosine_similarity": (
            training.positive_cosine_similarity
        ),
        "training_projection_standard_deviation": (
            training.projection_standard_deviation
        ),
        "validation_loss": validation.loss,
        "validation_positive_cosine_similarity": (
            validation.positive_cosine_similarity
        ),
        "validation_projection_standard_deviation": (
            validation.projection_standard_deviation
        ),
    }


def plot_history(
    history: list[dict[str, float | int]],
    figure_path: Path,
) -> None:
    """Plot SimCLR loss and representation diagnostics."""
    epochs = [
        int(record["epoch"])
        for record in history
    ]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 11),
        sharex=True,
    )

    axes[0].plot(
        epochs,
        [
            float(record["training_loss"])
            for record in history
        ],
        label="Training",
    )
    axes[0].plot(
        epochs,
        [
            float(record["validation_loss"])
            for record in history
        ],
        label="Validation",
    )
    axes[0].set_ylabel("NT-Xent loss")
    axes[0].set_title("SimCLR pretraining")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        [
            float(
                record[
                    "training_positive_cosine_similarity"
                ]
            )
            for record in history
        ],
        label="Training",
    )
    axes[1].plot(
        epochs,
        [
            float(
                record[
                    "validation_positive_cosine_similarity"
                ]
            )
            for record in history
        ],
        label="Validation",
    )
    axes[1].set_ylabel("Positive cosine similarity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        epochs,
        [
            float(
                record[
                    "training_projection_standard_deviation"
                ]
            )
            for record in history
        ],
        label="Training",
    )
    axes[2].plot(
        epochs,
        [
            float(
                record[
                    "validation_projection_standard_deviation"
                ]
            )
            for record in history
        ],
        label="Validation",
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean projection std.")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()

    figure_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        figure_path,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml(config_path)

    experiment_name = str(
        content["experiment_name"]
    )
    seed = int(
        content.get("seed", 2026)
    )

    set_reproducible_seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset_content = content["dataset"]
    training_content = content["training"]
    output_content = content["output"]

    epochs = (
        int(arguments.epochs)
        if arguments.epochs is not None
        else int(training_content["epochs"])
    )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    batch_size = int(
        training_content["batch_size"]
    )
    learning_rate = float(
        training_content["learning_rate"]
    )
    weight_decay = float(
        training_content["weight_decay"]
    )
    temperature = float(
        training_content["temperature"]
    )

    train_dataset = NPZIQDataset(
        resolve_project_path(
            dataset_content["train_path"]
        )
    )
    validation_dataset = NPZIQDataset(
        resolve_project_path(
            dataset_content[
                "validation_path"
            ]
        )
    )

    validate_contrastive_batching(
        len(train_dataset),
        batch_size,
        "Training",
    )
    validate_contrastive_batching(
        len(validation_dataset),
        batch_size,
        "Validation",
    )

    pin_memory = (
        bool(
            training_content.get(
                "pin_memory",
                True,
            )
        )
        and torch.cuda.is_available()
    )
    num_workers = int(
        training_content.get(
            "num_workers",
            0,
        )
    )

    train_loader = create_data_loader(
        train_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        ),
    )
    validation_loader = create_data_loader(
        validation_dataset,
        DataLoaderConfig(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        ),
    )

    encoder_configuration = (
        create_encoder_configuration(
            content["encoder"]
        )
    )
    projection_configuration = (
        create_projection_configuration(
            content["projection_head"]
        )
    )
    augmentation_configuration = (
        create_augmentation_configuration(
            content["augmentation"]
        )
    )

    encoder = BaselineIQCNN(
        encoder_configuration
    )

    for parameter in (
        encoder.classifier.parameters()
    ):
        parameter.requires_grad_(False)

    model = SimCLRModel(
        encoder=encoder,
        projection_configuration=(
            projection_configuration
        ),
    ).to(device)

    augmentation = RandomIQAugmentation(
        augmentation_configuration
    ).to(device)

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_directory_value = (
        arguments.output_directory
        if arguments.output_directory is not None
        else Path(
            str(output_content["directory"])
        )
    )
    figure_path_value = (
        arguments.figure_path
        if arguments.figure_path is not None
        else Path(
            str(output_content["figure_path"])
        )
    )

    output_directory = resolve_project_path(
        output_directory_value
    )
    figure_path = resolve_project_path(
        figure_path_value
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    history: list[
        dict[str, float | int]
    ] = []

    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_metrics: (
        ContrastiveEpochMetrics | None
    ) = None
    best_model_state: (
        dict[str, torch.Tensor] | None
    ) = None
    best_encoder_state: (
        dict[str, torch.Tensor] | None
    ) = None
    best_projection_state: (
        dict[str, torch.Tensor] | None
    ) = None

    print(f"Experiment: {experiment_name}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(
        f"Training examples: {len(train_dataset)}"
    )
    print(
        "Validation examples: "
        f"{len(validation_dataset)}"
    )
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(
        "Trainable parameters: "
        f"{count_trainable_parameters(model)}"
    )
    print(
        "Labels used during pretraining: no"
    )

    for epoch in range(1, epochs + 1):
        training_generator = (
            create_device_generator(
                device,
                seed + epoch,
            )
        )
        validation_generator = (
            create_device_generator(
                device,
                seed + 1_000_000,
            )
        )

        training_metrics = (
            run_contrastive_training_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                augmentation=augmentation,
                device=device,
                temperature=temperature,
                generator=training_generator,
            )
        )

        validation_metrics = (
            run_contrastive_evaluation_epoch(
                model=model,
                data_loader=validation_loader,
                augmentation=augmentation,
                device=device,
                temperature=temperature,
                generator=validation_generator,
            )
        )

        history.append(
            metrics_to_record(
                epoch,
                training_metrics,
                validation_metrics,
            )
        )

        if (
            validation_metrics.loss
            < best_validation_loss
        ):
            best_epoch = epoch
            best_validation_loss = (
                validation_metrics.loss
            )
            best_validation_metrics = (
                validation_metrics
            )
            best_model_state = (
                state_dict_to_cpu(model)
            )
            best_encoder_state = (
                state_dict_to_cpu(
                    model.encoder
                )
            )
            best_projection_state = (
                state_dict_to_cpu(
                    model.projection_head
                )
            )

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss "
            f"{training_metrics.loss:.4f} | "
            f"val loss "
            f"{validation_metrics.loss:.4f} | "
            f"train positive "
            f"{training_metrics.positive_cosine_similarity:.3f} | "
            f"val positive "
            f"{validation_metrics.positive_cosine_similarity:.3f} | "
            f"train std "
            f"{training_metrics.projection_standard_deviation:.4f} | "
            f"val std "
            f"{validation_metrics.projection_standard_deviation:.4f}"
        )

    if (
        best_model_state is None
        or best_encoder_state is None
        or best_projection_state is None
        or best_validation_metrics is None
    ):
        raise RuntimeError(
            "Pretraining completed without a checkpoint."
        )

    checkpoint_path = (
        output_directory / "best_model.pt"
    )
    history_path = (
        output_directory / "history.json"
    )
    summary_path = (
        output_directory / "summary.json"
    )

    training_configuration = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    torch.save(
        {
            "format_version": 1,
            "experiment_name": experiment_name,
            "method": "simclr",
            "seed": seed,
            "encoder_configuration": asdict(
                encoder_configuration
            ),
            "projection_configuration": asdict(
                projection_configuration
            ),
            "augmentation_configuration": asdict(
                augmentation_configuration
            ),
            "training_configuration": (
                training_configuration
            ),
            "model_state_dict": best_model_state,
            "encoder_state_dict": (
                best_encoder_state
            ),
            "projection_head_state_dict": (
                best_projection_state
            ),
            "best_epoch": best_epoch,
            "best_validation_loss": (
                best_validation_loss
            ),
            "best_validation_positive_cosine_similarity": (
                best_validation_metrics
                .positive_cosine_similarity
            ),
            "best_validation_projection_standard_deviation": (
                best_validation_metrics
                .projection_standard_deviation
            ),
        },
        checkpoint_path,
    )

    history_path.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_record = history[-1]

    summary = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "method": "simclr",
        "seed": seed,
        "device": str(device),
        "train_examples": len(train_dataset),
        "validation_examples": len(
            validation_dataset
        ),
        "trainable_parameters": (
            count_trainable_parameters(model)
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_validation_positive_cosine_similarity": (
            best_validation_metrics
            .positive_cosine_similarity
        ),
        "best_validation_projection_standard_deviation": (
            best_validation_metrics
            .projection_standard_deviation
        ),
        "final_training_loss": float(
            final_record["training_loss"]
        ),
        "final_validation_loss": float(
            final_record["validation_loss"]
        ),
        "checkpoint": serialize_project_path(
            checkpoint_path
        ),
        "history": serialize_project_path(
            history_path
        ),
        "figure": serialize_project_path(
            figure_path
        ),
        "encoder_configuration": asdict(
            encoder_configuration
        ),
        "projection_configuration": asdict(
            projection_configuration
        ),
        "augmentation_configuration": asdict(
            augmentation_configuration
        ),
        "training_configuration": (
            training_configuration
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_history(
        history,
        figure_path,
    )

    print("")
    print(f"Best epoch: {best_epoch}")
    print(
        "Best validation loss: "
        f"{best_validation_loss:.4f}"
    )
    print(
        "Best validation positive similarity: "
        f"{best_validation_metrics.positive_cosine_similarity:.4f}"
    )
    print(
        "Best validation projection std: "
        f"{best_validation_metrics.projection_standard_deviation:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
