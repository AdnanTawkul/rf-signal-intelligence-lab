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
            "Refit a frozen CNN classifier head and save a native "
            "PyTorch checkpoint."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/refit_groupnorm_head_v1.yaml"
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
        relative_path = resolved_path.relative_to(
            resolved_root
        )
    except ValueError:
        return resolved_path.as_posix()

    return relative_path.as_posix()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration mapping."""
    content = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(content, dict):
        raise ValueError(
            "Refit configuration must be a YAML mapping."
        )

    return content


def reconstruct_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[BaselineIQCNN, BaselineCNNConfig]:
    """Reconstruct a CNN from checkpoint metadata."""
    model_content = checkpoint.get(
        "model_configuration"
    )

    if not isinstance(model_content, dict):
        raise ValueError(
            "Checkpoint model_configuration must be a mapping."
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
                "batch",
            )
        ),
        group_norm_groups=int(
            model_content.get(
                "group_norm_groups",
                8,
            )
        ),
    )

    model = BaselineIQCNN(configuration)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    return model, configuration


def extract_embeddings(
    model: BaselineIQCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[
    Float32Array,
    Int64Array,
    Int64Array,
]:
    """Extract embeddings, labels, and current predictions."""
    embedding_batches: list[Float32Array] = []
    label_batches: list[Int64Array] = []
    prediction_batches: list[Int64Array] = []

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
            logits = model.classifier(
                embeddings
            )
            predictions = torch.argmax(
                logits,
                dim=1,
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
            prediction_batches.append(
                predictions.cpu().numpy().astype(
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
        np.concatenate(
            prediction_batches,
            axis=0,
        ),
    )


def evaluate_model_accuracy(
    model: BaselineIQCNN,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy end to end."""
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
    """Serialize candidate validation results."""
    return [
        {
            "regularization_c": float(
                candidate
            ),
            "validation_accuracy": float(
                accuracy
            ),
        }
        for candidate, accuracy
        in result.candidate_accuracies
    ]


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
        content["source_checkpoint_path"]
    )

    checkpoint = torch.load(
        source_checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Source checkpoint must contain a mapping."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, model_configuration = reconstruct_model(
        checkpoint,
        device,
    )

    seed = int(
        checkpoint.get("seed", 2026)
    )

    dataset_content = content["dataset"]
    evaluation_content = content["evaluation"]

    loader_configuration = DataLoaderConfig(
        batch_size=int(
            evaluation_content["batch_size"]
        ),
        shuffle=False,
        num_workers=int(
            evaluation_content["num_workers"]
        ),
        pin_memory=(
            bool(
                evaluation_content["pin_memory"]
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
            dataset_content[
                "validation_path"
            ]
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
        f"Source checkpoint: "
        f"{source_checkpoint_path}"
    )
    print(
        f"Train examples: {len(train_dataset)}"
    )
    print(
        "Validation examples: "
        f"{len(validation_dataset)}"
    )

    (
        train_embeddings,
        train_labels,
        _,
    ) = extract_embeddings(
        model,
        train_loader,
        device,
    )
    (
        validation_embeddings,
        validation_labels,
        original_validation_predictions,
    ) = extract_embeddings(
        model,
        validation_loader,
        device,
    )

    original_validation_accuracy = float(
        np.mean(
            original_validation_predictions
            == validation_labels
        )
    )

    refit_content = content["refit"]
    candidates = tuple(
        float(value)
        for value in refit_content[
            "regularization_candidates"
        ]
    )

    fit_result = fit_frozen_linear_head(
        train_features=train_embeddings,
        train_labels=train_labels,
        validation_features=(
            validation_embeddings
        ),
        validation_labels=validation_labels,
        regularization_candidates=candidates,
        max_iter=int(
            refit_content.get(
                "max_iter",
                5000,
            )
        ),
        random_state=int(
            refit_content.get(
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
            "Refitted classifier classes do not match "
            "the model output ordering."
        )

    apply_linear_head_parameters(
        model,
        fit_result.parameters,
    )

    deployed_validation_accuracy = (
        evaluate_model_accuracy(
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
            "Native PyTorch checkpoint accuracy does "
            "not match the selected refit accuracy."
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

    state_dict_cpu = {
        key: value.detach().cpu()
        for key, value
        in model.state_dict().items()
    }

    class_names = [
        str(name)
        for name in checkpoint["class_names"]
    ]

    refit_metadata = {
        "method": (
            "standardized_multinomial_"
            "logistic_regression"
        ),
        "encoder_frozen": True,
        "selected_regularization_c": float(
            fit_result.regularization_c
        ),
        "original_validation_accuracy": (
            original_validation_accuracy
        ),
        "refit_validation_accuracy": float(
            deployed_validation_accuracy
        ),
        "validation_accuracy_change": float(
            deployed_validation_accuracy
            - original_validation_accuracy
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
    }

    torch.save(
        {
            "format_version": 1,
            "experiment_name": experiment_name,
            "model_configuration": asdict(
                model_configuration
            ),
            "model_state_dict": state_dict_cpu,
            "class_names": class_names,
            "seed": seed,
            "source_checkpoint": serialize_project_path(
                source_checkpoint_path
            ),
            "source_best_epoch": checkpoint.get(
                "best_epoch"
            ),
            "source_best_validation_accuracy": (
                checkpoint.get(
                    "best_validation_accuracy"
                )
            ),
            "head_refit": refit_metadata,
        },
        checkpoint_path,
    )

    summary = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "seed": seed,
        "source_checkpoint": serialize_project_path(
            source_checkpoint_path
        ),
        "output_checkpoint": serialize_project_path(
            checkpoint_path
        ),
        "embedding_dimension": int(
            train_embeddings.shape[1]
        ),
        **refit_metadata,
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
        "Original validation accuracy: "
        f"{original_validation_accuracy:.4f}"
    )
    print(
        "Refitted validation accuracy: "
        f"{deployed_validation_accuracy:.4f}"
    )
    print(
        "Validation change: "
        f"{deployed_validation_accuracy - original_validation_accuracy:+.4f}"
    )
    print(
        "Selected regularization C: "
        f"{fit_result.regularization_c:g}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
