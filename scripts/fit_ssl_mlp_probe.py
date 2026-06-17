from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)
from rfsil.models.mlp_probe import (
    FrozenMLPProbeConfig,
    fit_frozen_mlp_probe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a validation-selected nonlinear classifier "
            "on frozen SSL encoder embeddings."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
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
            "MLP probe configuration must be a YAML mapping."
        )

    return content


def reconstruct_encoder(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[BaselineIQCNN, BaselineCNNConfig]:
    """Reconstruct a frozen encoder from an SSL checkpoint."""
    configuration_content = checkpoint.get(
        "encoder_configuration"
    )

    if not isinstance(configuration_content, dict):
        raise ValueError(
            "SSL checkpoint encoder_configuration "
            "must be a mapping."
        )

    configuration = BaselineCNNConfig(
        in_channels=int(
            configuration_content["in_channels"]
        ),
        num_classes=int(
            configuration_content["num_classes"]
        ),
        channels=tuple(
            int(value)
            for value in configuration_content["channels"]
        ),
        kernel_size=int(
            configuration_content["kernel_size"]
        ),
        dropout=float(
            configuration_content["dropout"]
        ),
        normalize_input_rms=bool(
            configuration_content.get(
                "normalize_input_rms",
                False,
            )
        ),
        normalization=str(
            configuration_content.get(
                "normalization",
                "group",
            )
        ),
        group_norm_groups=int(
            configuration_content.get(
                "group_norm_groups",
                8,
            )
        ),
    )

    encoder_state = checkpoint.get(
        "encoder_state_dict"
    )

    if not isinstance(encoder_state, dict):
        raise ValueError(
            "SSL checkpoint encoder_state_dict "
            "must be a mapping."
        )

    model = BaselineIQCNN(configuration)
    model.load_state_dict(encoder_state)
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, configuration


def extract_embeddings(
    model: BaselineIQCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[Float32Array, Int64Array]:
    """Extract frozen embeddings and class labels."""
    embedding_batches: list[Float32Array] = []
    label_batches: list[Int64Array] = []

    model.eval()

    with torch.inference_mode():
        for batch in data_loader:
            inputs = batch["iq"].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
            labels = batch["label"].to(
                device=device,
                dtype=torch.int64,
                non_blocking=True,
            )

            embeddings = model.extract_features(inputs)

            embedding_batches.append(
                embeddings.cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )
            )
            label_batches.append(
                labels.cpu().numpy().astype(
                    np.int64,
                    copy=False,
                )
            )

    if not embedding_batches:
        raise ValueError(
            "DataLoader produced no examples."
        )

    return (
        np.concatenate(
            embedding_batches,
            axis=0,
        ),
        np.concatenate(
            label_batches,
            axis=0,
        ),
    )


def state_dict_to_cpu(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Copy a model state dictionary to CPU."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(
        arguments.config
    )
    content = load_yaml(config_path)

    experiment_name = str(
        content["experiment_name"]
    )
    source_checkpoint_path = resolve_project_path(
        content["source_ssl_checkpoint_path"]
    )

    checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "SSL checkpoint must contain a mapping."
        )

    source_ssl_method = checkpoint.get("method")

    if source_ssl_method not in {"simclr", "vicreg"}:
        raise ValueError(
            "Source checkpoint must be marked "
            "as SimCLR or VICReg."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    encoder, encoder_configuration = (
        reconstruct_encoder(
            checkpoint,
            device,
        )
    )

    class_names = tuple(
        str(value)
        for value in content["class_names"]
    )

    if (
        len(class_names)
        != encoder_configuration.num_classes
    ):
        raise ValueError(
            "class_names count must match num_classes."
        )

    seed = int(
        content["probe"].get(
            "seed",
            checkpoint.get("seed", 2026),
        )
    )

    extraction_content = content["extraction"]

    loader_configuration = DataLoaderConfig(
        batch_size=int(
            extraction_content["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            extraction_content.get(
                "num_workers",
                0,
            )
        ),
        pin_memory=(
            bool(
                extraction_content.get(
                    "pin_memory",
                    True,
                )
            )
            and torch.cuda.is_available()
        ),
        seed=seed,
    )

    dataset_content = content["dataset"]

    train_dataset = NPZIQDataset(
        resolve_project_path(
            dataset_content["train_path"]
        )
    )
    validation_dataset = NPZIQDataset(
        resolve_project_path(
            dataset_content["validation_path"]
        )
    )

    train_loader = create_data_loader(
        train_dataset,
        loader_configuration,
    )
    validation_loader = create_data_loader(
        validation_dataset,
        loader_configuration,
    )

    print(f"Experiment: {experiment_name}")
    print(f"SSL method: {source_ssl_method}")
    print(f"Device: {device}")
    print(
        "Source SSL checkpoint: "
        f"{source_checkpoint_path}"
    )
    print(
        "Source SSL best epoch: "
        f"{checkpoint.get('best_epoch')}"
    )
    print(
        f"Training examples: {len(train_dataset)}"
    )
    print(
        "Validation examples: "
        f"{len(validation_dataset)}"
    )
    print("Encoder frozen: yes")
    print("Test split accessed: no")

    train_embeddings, train_labels = (
        extract_embeddings(
            encoder,
            train_loader,
            device,
        )
    )
    validation_embeddings, validation_labels = (
        extract_embeddings(
            encoder,
            validation_loader,
            device,
        )
    )

    probe_content = content["probe"]

    probe_configuration = FrozenMLPProbeConfig(
        hidden_dimension=int(
            probe_content["hidden_dimension"]
        ),
        dropout=float(
            probe_content["dropout"]
        ),
        epochs=int(
            probe_content["epochs"]
        ),
        batch_size=int(
            probe_content["batch_size"]
        ),
        learning_rate=float(
            probe_content["learning_rate"]
        ),
        weight_decay=float(
            probe_content["weight_decay"]
        ),
        seed=seed,
    )

    fit_result = fit_frozen_mlp_probe(
        train_features=train_embeddings,
        train_labels=train_labels,
        validation_features=validation_embeddings,
        validation_labels=validation_labels,
        num_classes=encoder_configuration.num_classes,
        configuration=probe_configuration,
        device=device,
    )

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_directory
        / str(
            output_content.get(
                "checkpoint_name",
                "best_model.pt",
            )
        )
    )
    summary_path = (
        output_directory
        / str(
            output_content.get(
                "summary_name",
                "summary.json",
            )
        )
    )
    history_path = (
        output_directory
        / str(
            output_content.get(
                "history_name",
                "history.json",
            )
        )
    )

    history_path.write_text(
        json.dumps(
            list(fit_result.history),
            indent=2,
        ),
        encoding="utf-8",
    )

    checkpoint_content = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "model_type": "frozen_ssl_mlp_probe",
        "source_ssl_method": source_ssl_method,
        "source_ssl_checkpoint": (
            serialize_project_path(
                source_checkpoint_path
            )
        ),
        "source_ssl_best_epoch": checkpoint.get(
            "best_epoch"
        ),
        "source_ssl_seed": checkpoint.get("seed"),
        "encoder_configuration": asdict(
            encoder_configuration
        ),
        "encoder_state_dict": state_dict_to_cpu(
            encoder
        ),
        "probe_configuration": asdict(
            probe_configuration
        ),
        "probe_state_dict": state_dict_to_cpu(
            fit_result.model
        ),
        "class_names": list(class_names),
        "embedding_dimension": int(
            train_embeddings.shape[1]
        ),
        "best_probe_epoch": fit_result.best_epoch,
        "training_accuracy": (
            fit_result.training_accuracy
        ),
        "validation_accuracy": (
            fit_result.validation_accuracy
        ),
        "best_validation_loss": (
            fit_result.best_validation_loss
        ),
    }

    torch.save(
        checkpoint_content,
        checkpoint_path,
    )

    summary = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "model_type": "frozen_ssl_mlp_probe",
        "source_ssl_method": source_ssl_method,
        "source_ssl_checkpoint": (
            serialize_project_path(
                source_checkpoint_path
            )
        ),
        "source_ssl_best_epoch": checkpoint.get(
            "best_epoch"
        ),
        "train_examples": int(
            len(train_labels)
        ),
        "validation_examples": int(
            len(validation_labels)
        ),
        "embedding_dimension": int(
            train_embeddings.shape[1]
        ),
        "probe_configuration": asdict(
            probe_configuration
        ),
        "best_probe_epoch": fit_result.best_epoch,
        "training_accuracy": (
            fit_result.training_accuracy
        ),
        "validation_accuracy": (
            fit_result.validation_accuracy
        ),
        "best_validation_loss": (
            fit_result.best_validation_loss
        ),
        "checkpoint": serialize_project_path(
            checkpoint_path
        ),
        "history": serialize_project_path(
            history_path
        ),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(
        "Embedding dimension: "
        f"{train_embeddings.shape[1]}"
    )
    print(
        "Best probe epoch: "
        f"{fit_result.best_epoch}"
    )
    print(
        "Restored training accuracy: "
        f"{fit_result.training_accuracy:.4f}"
    )
    print(
        "Selected validation accuracy: "
        f"{fit_result.validation_accuracy:.4f}"
    )
    print(
        "Selected validation loss: "
        f"{fit_result.best_validation_loss:.4f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
