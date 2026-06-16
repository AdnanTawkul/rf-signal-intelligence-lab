from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
from rfsil.evaluation.seed_test import (
    SeedTestResult,
    aggregate_seed_test_results,
)
from rfsil.models.baseline_cnn import (
    BaselineCNNConfig,
    BaselineIQCNN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all baseline seed-sweep checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/evaluate_seed_sweep_v1.yaml"
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
            "Seed evaluation configuration must be a mapping."
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

    return model


def main() -> None:
    arguments = parse_arguments()
    config_path = resolve_project_path(arguments.config)
    content = load_yaml(config_path)

    experiment_name = str(content["experiment_name"])
    checkpoint_directory = resolve_project_path(
        content["checkpoint_directory"]
    )
    test_path = resolve_project_path(content["test_path"])
    seeds = [int(seed) for seed in content["seeds"]]

    if not seeds:
        raise ValueError("At least one seed is required.")

    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed values must be unique.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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
            seed=2026,
        ),
    )

    results: list[SeedTestResult] = []
    class_names: list[str] | None = None

    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Test examples: {len(dataset)}")
    print(f"Seeds: {seeds}")

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
            data_loader=loader,
            device=device,
        )

        evaluation = evaluate_predictions(
            labels=predictions.labels,
            predictions=predictions.predictions,
            snr_db=predictions.snr_db,
            num_classes=len(current_class_names),
        )

        result = SeedTestResult(
            seed=seed,
            overall_accuracy=evaluation.accuracy,
            class_accuracy=evaluation.class_accuracy,
            snr_values_db=evaluation.snr_values_db,
            snr_accuracy=evaluation.snr_accuracy,
        )
        results.append(result)

        print(
            f"Seed {seed}: "
            f"test accuracy={evaluation.accuracy:.4f}"
        )

    if class_names is None:
        raise RuntimeError("No checkpoints were evaluated.")

    aggregate = aggregate_seed_test_results(results)

    output_content = content["output"]
    output_directory = resolve_project_path(
        output_content["directory"]
    )
    figure_path = resolve_project_path(
        output_content["figure_path"]
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    aggregate_path = output_directory / "aggregate_metrics.json"

    aggregate_content = {
        "format_version": 1,
        "experiment_name": experiment_name,
        "test_path": str(
            test_path.relative_to(PROJECT_ROOT)
        ),
        "class_names": class_names,
        "runs": [
            {
                "seed": result.seed,
                "overall_accuracy": result.overall_accuracy,
                "class_accuracy": {
                    class_name: float(accuracy)
                    for class_name, accuracy in zip(
                        class_names,
                        result.class_accuracy,
                        strict=True,
                    )
                },
                "accuracy_by_snr": {
                    str(float(snr_value)): float(accuracy)
                    for snr_value, accuracy in zip(
                        result.snr_values_db,
                        result.snr_accuracy,
                        strict=True,
                    )
                },
            }
            for result in results
        ],
        "aggregate": {
            "overall_mean": aggregate.overall_mean,
            "overall_std": aggregate.overall_std,
            "overall_minimum": aggregate.overall_minimum,
            "overall_maximum": aggregate.overall_maximum,
            "class_accuracy_mean": {
                class_name: float(accuracy)
                for class_name, accuracy in zip(
                    class_names,
                    aggregate.class_accuracy_mean,
                    strict=True,
                )
            },
            "class_accuracy_std": {
                class_name: float(accuracy)
                for class_name, accuracy in zip(
                    class_names,
                    aggregate.class_accuracy_std,
                    strict=True,
                )
            },
            "accuracy_by_snr_mean": {
                str(float(snr_value)): float(accuracy)
                for snr_value, accuracy in zip(
                    aggregate.snr_values_db,
                    aggregate.snr_accuracy_mean,
                    strict=True,
                )
            },
            "accuracy_by_snr_std": {
                str(float(snr_value)): float(accuracy)
                for snr_value, accuracy in zip(
                    aggregate.snr_values_db,
                    aggregate.snr_accuracy_std,
                    strict=True,
                )
            },
        },
    }

    aggregate_path.write_text(
        json.dumps(aggregate_content, indent=2),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        [str(seed) for seed in aggregate.seeds],
        aggregate.overall_accuracy,
    )
    axes[0].axhline(
        aggregate.overall_mean,
        linestyle="--",
        label=f"Mean {aggregate.overall_mean:.3f}",
    )
    axes[0].set_title("Held-Out Test Accuracy by Training Seed")
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    lower_bound = (
        aggregate.snr_accuracy_mean
        - aggregate.snr_accuracy_std
    )
    upper_bound = (
        aggregate.snr_accuracy_mean
        + aggregate.snr_accuracy_std
    )

    axes[1].plot(
        aggregate.snr_values_db,
        aggregate.snr_accuracy_mean,
        marker="o",
        label="Mean test accuracy",
    )
    axes[1].fill_between(
        aggregate.snr_values_db,
        lower_bound,
        upper_bound,
        alpha=0.25,
        label="1 standard deviation",
    )
    axes[1].set_title("Mean Test Accuracy by SNR")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(aggregate.snr_values_db)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("Baseline CNN Five-Seed Test Evaluation")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)

    print("")
    print("Five-seed held-out test results")
    print(f"Mean test accuracy: {aggregate.overall_mean:.4f}")
    print(
        "Test accuracy standard deviation: "
        f"{aggregate.overall_std:.4f}"
    )
    print(
        "Test accuracy range: "
        f"{aggregate.overall_minimum:.4f} to "
        f"{aggregate.overall_maximum:.4f}"
    )

    for class_name, mean, standard_deviation in zip(
        class_names,
        aggregate.class_accuracy_mean,
        aggregate.class_accuracy_std,
        strict=True,
    ):
        print(
            f"{class_name}: "
            f"{float(mean):.4f} ? "
            f"{float(standard_deviation):.4f}"
        )

    for snr_value, mean, standard_deviation in zip(
        aggregate.snr_values_db,
        aggregate.snr_accuracy_mean,
        aggregate.snr_accuracy_std,
        strict=True,
    ):
        print(
            f"{float(snr_value):.1f} dB: "
            f"{float(mean):.4f} ? "
            f"{float(standard_deviation):.4f}"
        )

    print(f"Aggregate metrics: {aggregate_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
