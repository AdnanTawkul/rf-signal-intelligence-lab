from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from rfsil.data.torch_dataset import (
    DataLoaderConfig,
    NPZIQDataset,
    create_data_loader,
)
from rfsil.evaluation.calibration_artifacts import (
    save_calibration_artifact,
)
from rfsil.evaluation.classification import (
    PredictionResults,
    collect_calibration_predictions,
    evaluate_predictions,
)
from rfsil.evaluation.prediction_artifacts import (
    save_prediction_results,
)
from rfsil.evaluation.ssl_label_efficiency import (
    TestCondition,
    build_metrics_path,
    load_completed_test_evaluation,
    serialize_path,
    validate_checkpoint_metadata,
    validate_test_conditions,
)
from rfsil.models.model_factory import (
    create_model_from_checkpoint,
)
from rfsil.training.ssl_sweep_execution import (
    SweepRunExpectation,
    load_run_expectation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse evaluation and filtering arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SSL label-efficiency "
            "checkpoints on held-out datasets."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )
    parser.add_argument(
        "--fraction",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
    )

    return parser.parse_args()


def resolve_project_path(
    value: str | Path,
) -> Path:
    """Resolve a project-relative path."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_mapping(
    path: Path,
) -> dict[str, Any]:
    """Load a JSON or YAML mapping."""
    if path.suffix.lower() == ".json":
        content = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    else:
        content = yaml.safe_load(
            path.read_text(
                encoding="utf-8"
            )
        )

    if not isinstance(content, dict):
        raise ValueError(
            f"Configuration must be a mapping: "
            f"{path}"
        )

    return content


def is_selected(
    expectation: SweepRunExpectation,
    *,
    fractions: set[str],
    methods: set[str],
    seeds: set[int],
) -> bool:
    """Apply checkpoint filters."""
    if (
        fractions
        and expectation.fraction_identifier
        not in fractions
    ):
        return False

    if (
        methods
        and expectation.method not in methods
    ):
        return False

    return not (
        seeds
        and expectation.seed not in seeds
    )


def create_model(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Reconstruct a trained classifier."""
    model, _ = create_model_from_checkpoint(
        checkpoint
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    return model


def write_execution_manifest(
    *,
    path: Path,
    source_manifest: Path,
    records: list[dict[str, object]],
    selected_checkpoint_count: int,
    selected_condition_count: int,
    resume: bool,
) -> None:
    """Persist evaluation progress."""
    content = {
        "format_version": 1,
        "source_training_manifest": (
            source_manifest.resolve().as_posix()
        ),
        "resume": resume,
        "selected_checkpoint_count": (
            selected_checkpoint_count
        ),
        "selected_condition_count": (
            selected_condition_count
        ),
        "planned_evaluation_count": (
            selected_checkpoint_count
            * selected_condition_count
        ),
        "processed_evaluation_count": (
            len(records)
        ),
        "completed_evaluation_count": sum(
            record["status"] == "completed"
            for record in records
        ),
        "skipped_evaluation_count": sum(
            record["status"] == "skipped"
            for record in records
        ),
        "evaluations": records,
    }

    path.write_text(
        json.dumps(
            content,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run or resume the held-out evaluation matrix."""
    arguments = parse_arguments()

    if (
        arguments.max_runs is not None
        and arguments.max_runs <= 0
    ):
        raise ValueError(
            "--max-runs must be positive."
        )

    config_path = resolve_project_path(
        arguments.config
    )
    content = load_mapping(config_path)

    training_manifest_path = (
        resolve_project_path(
            content["training_manifest"]
        )
    )
    training_manifest = load_mapping(
        training_manifest_path
    )

    raw_runs = training_manifest.get(
        "runs"
    )

    if not isinstance(raw_runs, list):
        raise ValueError(
            "Training manifest runs must "
            "be a list."
        )

    expectations = [
        load_run_expectation(
            record,
            project_root=PROJECT_ROOT,
        )
        for record in raw_runs
    ]

    selected_expectations = [
        expectation
        for expectation in expectations
        if is_selected(
            expectation,
            fractions=set(
                arguments.fraction
            ),
            methods=set(arguments.method),
            seeds=set(arguments.seed),
        )
    ]

    if arguments.max_runs is not None:
        selected_expectations = (
            selected_expectations[
                : arguments.max_runs
            ]
        )

    if not selected_expectations:
        raise ValueError(
            "No checkpoints match the filters."
        )

    conditions = validate_test_conditions(
        content["conditions"],
        project_root=PROJECT_ROOT,
    )

    requested_conditions = set(
        arguments.condition
    )

    selected_conditions = tuple(
        condition
        for condition in conditions
        if (
            not requested_conditions
            or condition.identifier
            in requested_conditions
        )
    )

    if not selected_conditions:
        raise ValueError(
            "No conditions match the filters."
        )

    evaluation_content = content.get(
        "evaluation"
    )

    if not isinstance(
        evaluation_content,
        dict,
    ):
        raise ValueError(
            "evaluation must be a mapping."
        )

    batch_size = int(
        evaluation_content["batch_size"]
    )
    num_workers = int(
        evaluation_content["num_workers"]
    )
    pin_memory = bool(
        evaluation_content["pin_memory"]
    )
    save_predictions = bool(
        evaluation_content.get(
            "save_predictions",
            False,
        )
    )
    save_calibration_predictions = bool(
        evaluation_content.get(
            "save_calibration_predictions",
            False,
        )
    )
    input_scale = float(
        evaluation_content.get(
            "input_scale",
            1.0,
        )
    )

    if (
        not math.isfinite(input_scale)
        or input_scale <= 0.0
    ):
        raise ValueError(
            "input_scale must be positive "
            "and finite."
        )

    output_root = resolve_project_path(
        content["output"]["directory"]
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    execution_manifest_path = (
        output_root
        / "execution_manifest.json"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loaders = {}

    for condition in selected_conditions:
        dataset = NPZIQDataset(
            condition.test_path
        )
        loader = create_data_loader(
            dataset,
            DataLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(
                    pin_memory
                    and torch.cuda.is_available()
                ),
                seed=2026,
            ),
        )
        loaders[condition.identifier] = (
            dataset,
            loader,
        )

    records: list[dict[str, object]] = []

    print(
        "Selected checkpoints: "
        f"{len(selected_expectations)}"
    )
    print(
        "Selected conditions: "
        f"{len(selected_conditions)}"
    )
    print(
        "Planned evaluations: "
        f"{len(selected_expectations) * len(selected_conditions)}"
    )
    print(f"Device: {device}")
    print(f"Resume enabled: {arguments.resume}")

    for run_index, expectation in enumerate(
        selected_expectations,
        start=1,
    ):
        pending_conditions: list[
            TestCondition
        ] = []
        completed_by_condition = {}

        for condition in selected_conditions:
            metrics_path = build_metrics_path(
                output_root=output_root,
                expectation=expectation,
                condition=condition,
            )
            completed = (
                load_completed_test_evaluation(
                    metrics_path=metrics_path,
                    expectation=expectation,
                    condition=condition,
                    project_root=PROJECT_ROOT,
                )
            )
            completed_by_condition[
                condition.identifier
            ] = completed

            if completed is None:
                pending_conditions.append(
                    condition
                )
            elif not arguments.resume:
                raise FileExistsError(
                    "Validated metrics already exist. "
                    "Use --resume to skip them."
                )

        model = None
        checkpoint = None
        class_names = None

        if pending_conditions:
            checkpoint_path = (
                expectation.output_directory
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
            validate_checkpoint_metadata(
                checkpoint,
                expectation,
            )
            class_names = [
                str(name).upper()
                for name in checkpoint[
                    "class_names"
                ]
            ]
            model = create_model(
                checkpoint,
                device,
            )

        print()
        print("=" * 80)
        print(
            f"Checkpoint "
            f"{run_index}/"
            f"{len(selected_expectations)} | "
            f"{expectation.fraction_identifier} | "
            f"{expectation.method} | "
            f"seed={expectation.seed}"
        )
        print("=" * 80)

        for condition in selected_conditions:
            completed = completed_by_condition[
                condition.identifier
            ]

            if completed is not None:
                status = "skipped"
                accuracy = (
                    completed.overall_accuracy
                )
                metrics_path = (
                    completed.metrics_path
                )
            else:
                if (
                    model is None
                    or checkpoint is None
                    or class_names is None
                ):
                    raise RuntimeError(
                        "Model was not loaded for "
                        "a pending evaluation."
                    )

                dataset, loader = loaders[
                    condition.identifier
                ]
                calibration_predictions = (
                    collect_calibration_predictions(
                        model=model,
                        data_loader=loader,
                        device=device,
                        input_scale=input_scale,
                        class_names=tuple(
                            class_names
                        ),
                    )
                )

                if (
                    calibration_predictions.snr_db
                    is None
                ):
                    raise RuntimeError(
                        "Evaluation predictions do "
                        "not contain SNR values."
                    )

                predictions = PredictionResults(
                    labels=(
                        calibration_predictions
                        .labels
                    ),
                    predictions=(
                        calibration_predictions
                        .predictions
                    ),
                    snr_db=(
                        calibration_predictions
                        .snr_db
                        .astype(
                            "float32",
                            copy=False,
                        )
                    ),
                )
                evaluation = (
                    evaluate_predictions(
                        labels=predictions.labels,
                        predictions=(
                            predictions.predictions
                        ),
                        snr_db=predictions.snr_db,
                        num_classes=len(
                            class_names
                        ),
                    )
                )

                metrics_path = build_metrics_path(
                    output_root=output_root,
                    expectation=expectation,
                    condition=condition,
                )
                metrics_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                checkpoint_path = (
                    expectation.output_directory
                    / "best_model.pt"
                )

                metrics = {
                    "format_version": 1,
                    "experiment_name": (
                        str(
                            content[
                                "experiment_name"
                            ]
                        )
                    ),
                    "fraction_identifier": (
                        expectation.fraction_identifier
                    ),
                    "method": expectation.method,
                    "seed": expectation.seed,
                    "condition": (
                        condition.identifier
                    ),
                    "condition_display_name": (
                        condition.display_name
                    ),
                    "checkpoint_path": (
                        serialize_path(
                            checkpoint_path,
                            project_root=(
                                PROJECT_ROOT
                            ),
                        )
                    ),
                    "test_path": serialize_path(
                        condition.test_path,
                        project_root=PROJECT_ROOT,
                    ),
                    "input_scale": input_scale,
                    "example_count": (
                        evaluation.example_count
                    ),
                    "overall_accuracy": (
                        evaluation.accuracy
                    ),
                    "class_names": class_names,
                    "class_accuracy": {
                        name: float(value)
                        for name, value in zip(
                            class_names,
                            evaluation.class_accuracy,
                            strict=True,
                        )
                    },
                    "accuracy_by_snr": {
                        str(float(snr)): (
                            float(value)
                        )
                        for snr, value in zip(
                            evaluation.snr_values_db,
                            evaluation.snr_accuracy,
                            strict=True,
                        )
                    },
                    "confusion_matrix": (
                        evaluation.confusion_matrix.tolist()
                    ),
                    "normalized_confusion_matrix": (
                        evaluation
                        .normalized_confusion_matrix
                        .tolist()
                    ),
                    "labeled_subset": (
                        checkpoint[
                            "labeled_subset"
                        ]
                    ),
                    "training_budget": (
                        checkpoint[
                            "training_budget"
                        ]
                    ),
                }

                metrics_path.write_text(
                    json.dumps(
                        metrics,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if save_predictions:
                    save_prediction_results(
                        metrics_path.parent
                        / "predictions.npz",
                        predictions,
                    )

                if save_calibration_predictions:
                    save_calibration_artifact(
                        metrics_path.parent
                        / (
                            "calibration_"
                            "predictions.npz"
                        ),
                        calibration_predictions,
                    )

                accuracy = (
                    evaluation.accuracy
                )
                status = "completed"

            records.append(
                {
                    "fraction_identifier": (
                        expectation.fraction_identifier
                    ),
                    "method": (
                        expectation.method
                    ),
                    "seed": expectation.seed,
                    "condition": (
                        condition.identifier
                    ),
                    "overall_accuracy": (
                        accuracy
                    ),
                    "metrics_path": (
                        metrics_path.resolve()
                        .as_posix()
                    ),
                    "status": status,
                }
            )

            write_execution_manifest(
                path=execution_manifest_path,
                source_manifest=(
                    training_manifest_path
                ),
                records=records,
                selected_checkpoint_count=(
                    len(
                        selected_expectations
                    )
                ),
                selected_condition_count=(
                    len(selected_conditions)
                ),
                resume=arguments.resume,
            )

            print(
                f"{condition.identifier:8s} | "
                f"accuracy={accuracy:.4f} | "
                f"{status}"
            )

        del model
        del checkpoint

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print(
        "Execution manifest: "
        f"{execution_manifest_path}"
    )


if __name__ == "__main__":
    main()
