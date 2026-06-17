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
from rfsil.models.frozen_head import (
    FrozenLinearHeadFit,
    fit_frozen_linear_head,
)
from rfsil.models.head_refit import (
    apply_linear_head_parameters,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Fit a validation-selected linear classifier "
            "on frozen SimCLR encoder embeddings."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/fit_simclr_linear_probe_v1.yaml"
        ),
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
            "Linear-probe configuration must be "
            "a YAML mapping."
        )

    return content


def reconstruct_encoder(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[BaselineIQCNN, BaselineCNNConfig]:
    """Reconstruct the frozen CNN encoder from an SSL checkpoint."""
    model_content = checkpoint.get(
        "encoder_configuration"
    )

    if not isinstance(model_content, dict):
        raise ValueError(
            "SSL checkpoint encoder_configuration "
            "must be a mapping."
        )

    configuration = BaselineCNNConfig(
        in_channels=int(
            model_content["in_channels"]
        ),
        num_classes=int(
            model_content["num_classes"]
        ),
        channels=tuple(
            int(value)
            for value in model_content["channels"]
        ),
        kernel_size=int(
            model_content["kernel_size"]
        ),
        dropout=float(
            model_content["dropout"]
        ),
        normalize_input_rms=bool(
            model_content.get(
                "normalize_input_rms",
                False,
            )
        ),
        normalization=str(
            model_content.get(
                "normalization",
                "group",
            )
        ),
        group_norm_groups=int(
            model_content.get(
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
    """Extract frozen embeddings and labels."""
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

            embeddings = model.extract_features(
                inputs
            )

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


def evaluate_accuracy(
    model: BaselineIQCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate the native linear probe on one split."""
    correct_count = 0
    example_count = 0

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

            predictions = torch.argmax(
                model(inputs),
                dim=1,
            )

            correct_count += int(
                torch.count_nonzero(
                    predictions == labels
                ).item()
            )
            example_count += int(
                labels.shape[0]
            )

    if example_count == 0:
        raise ValueError(
            "DataLoader produced no examples."
        )

    return correct_count / example_count


def serialize_candidate_results(
    result: FrozenLinearHeadFit,
) -> list[dict[str, float]]:
    """Serialize validation accuracy for each regularization value."""
    return [
        {
            "regularization_c": float(candidate),
            "validation_accuracy": float(accuracy),
        }
        for candidate, accuracy
        in result.candidate_accuracies
    ]


def state_dict_to_cpu(
    model: BaselineIQCNN,
) -> dict[str, torch.Tensor]:
    """Copy model state to CPU."""
    return {
        key: value.detach().cpu().clone()
        for key, value
        in model.state_dict().items()
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

    if checkpoint.get("method") != "simclr":
        raise ValueError(
            "Source checkpoint is not marked as SimCLR."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, model_configuration = reconstruct_encoder(
        checkpoint,
        device,
    )

    seed = int(
        checkpoint.get("seed", 2026)
    )

    class_names = tuple(
        str(value)
        for value in content["class_names"]
    )

    if (
        len(class_names)
        != model_configuration.num_classes
    ):
        raise ValueError(
            "class_names count must match num_classes."
        )

    dataset_content = content["dataset"]
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
            model,
            train_loader,
            device,
        )
    )
    validation_embeddings, validation_labels = (
        extract_embeddings(
            model,
            validation_loader,
            device,
        )
    )

    probe_content = content["linear_probe"]

    fit_result = fit_frozen_linear_head(
        train_features=train_embeddings,
        train_labels=train_labels,
        validation_features=(
            validation_embeddings
        ),
        validation_labels=validation_labels,
        regularization_candidates=tuple(
            float(value)
            for value in probe_content[
                "regularization_candidates"
            ]
        ),
        max_iter=int(
            probe_content.get(
                "max_iter",
                5000,
            )
        ),
        random_state=int(
            probe_content.get(
                "random_state",
                seed,
            )
        ),
    )

    expected_classes = np.arange(
        model_configuration.num_classes,
        dtype=np.int64,
    )

    if not np.array_equal(
        fit_result.classes,
        expected_classes,
    ):
        raise ValueError(
            "Linear-probe classes do not match "
            "the model output ordering."
        )

    apply_linear_head_parameters(
        model,
        fit_result.parameters,
    )

    deployed_validation_accuracy = (
        evaluate_accuracy(
            model,
            validation_loader,
            device,
        )
    )

    if not np.isclose(
        deployed_validation_accuracy,
        fit_result.validation_accuracy,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Native PyTorch probe accuracy does not "
            "match the selected sklearn accuracy."
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

    probe_metadata = {
        "method": (
            "frozen_standardized_multinomial_"
            "logistic_regression"
        ),
        "encoder_frozen": True,
        "selected_regularization_c": float(
            fit_result.regularization_c
        ),
        "validation_accuracy": float(
            deployed_validation_accuracy
        ),
        "candidate_results": (
            serialize_candidate_results(
                fit_result
            )
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
    }

    torch.save(
        {
            "format_version": 1,
            "experiment_name": experiment_name,
            "model_configuration": asdict(
                model_configuration
            ),
            "model_state_dict": state_dict_to_cpu(
                model
            ),
            "class_names": list(class_names),
            "seed": seed,
            "source_ssl_checkpoint": (
                serialize_project_path(
                    source_checkpoint_path
                )
            ),
            "source_ssl_best_epoch": checkpoint.get(
                "best_epoch"
            ),
            "source_ssl_best_validation_loss": (
                checkpoint.get(
                    "best_validation_loss"
                )
            ),
            "linear_probe": probe_metadata,
        },
        checkpoint_path,
    )

    summary = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "seed": seed,
        "source_ssl_checkpoint": (
            serialize_project_path(
                source_checkpoint_path
            )
        ),
        "source_ssl_best_epoch": checkpoint.get(
            "best_epoch"
        ),
        "output_checkpoint": (
            serialize_project_path(
                checkpoint_path
            )
        ),
        **probe_metadata,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print("Candidate validation results")

    for candidate, accuracy in (
        fit_result.candidate_accuracies
    ):
        marker = (
            " <- selected"
            if np.isclose(
                candidate,
                fit_result.regularization_c,
            )
            else ""
        )

        print(
            f"C={candidate:g} | "
            f"validation accuracy={accuracy:.4f}"
            f"{marker}"
        )

    print("")
    print(
        "Selected validation accuracy: "
        f"{deployed_validation_accuracy:.4f}"
    )
    print(
        "Selected regularization C: "
        f"{fit_result.regularization_c:g}"
    )
    print(
        "Embedding dimension: "
        f"{train_embeddings.shape[1]}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
