from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
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
from rfsil.evaluation.calibration_backfill import (
    build_calibration_artifact_path,
    load_valid_calibration_artifact,
    validate_artifact_accuracy,
)
from rfsil.evaluation.classification import (
    collect_calibration_predictions,
)
from rfsil.evaluation.ssl_label_efficiency import (
    TestCondition,
    build_metrics_path,
    load_completed_test_evaluation,
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
    """Parse backfill filters."""
    parser = argparse.ArgumentParser(
        description=(
            "Backfill calibration-ready logits "
            "for completed SSL evaluations."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
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


def expected_example_count(
    metrics: dict[str, Any],
) -> int:
    """Read the completed evaluation size."""
    if "example_count" in metrics:
        count = int(metrics["example_count"])

        if count <= 0:
            raise ValueError(
                "Completed metrics contain an "
                "invalid example_count."
            )

        return count

    confusion = np.asarray(
        metrics["confusion_matrix"],
        dtype=np.int64,
    )

    if (
        confusion.ndim != 2
        or confusion.shape[0]
        != confusion.shape[1]
    ):
        raise ValueError(
            "Completed confusion matrix must "
            "be square."
        )

    count = int(confusion.sum())

    if count <= 0:
        raise ValueError(
            "Completed confusion matrix is empty."
        )

    return count


def expected_class_names(
    metrics: dict[str, Any],
) -> tuple[str, ...]:
    """Read the class order from completed metrics."""
    raw_names = metrics.get("class_names")

    if (
        not isinstance(raw_names, list)
        or not raw_names
    ):
        raise ValueError(
            "Completed metrics must contain "
            "class_names."
        )

    names = tuple(
        str(name).strip()
        for name in raw_names
    )

    if any(not name for name in names):
        raise ValueError(
            "Completed metrics contain an empty "
            "class name."
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Completed metrics class names "
            "must be unique."
        )

    return names


def write_manifest(
    *,
    path: Path,
    config_path: Path,
    records: list[dict[str, object]],
) -> None:
    """Persist backfill progress."""
    content = {
        "format_version": 1,
        "config_path": (
            config_path.resolve().as_posix()
        ),
        "processed_count": len(records),
        "backfilled_count": sum(
            record["status"] == "backfilled"
            for record in records
        ),
        "skipped_count": sum(
            record["status"] == "skipped"
            for record in records
        ),
        "records": records,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(
            content,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Backfill missing calibration artifacts."""
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
    raw_runs = training_manifest.get("runs")

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

    backfill_content = content.get(
        "calibration_backfill",
        {},
    )

    if not isinstance(
        backfill_content,
        dict,
    ):
        raise ValueError(
            "calibration_backfill must be "
            "a mapping."
        )

    artifact_filename = str(
        backfill_content.get(
            "artifact_filename",
            "calibration_predictions.npz",
        )
    )

    output_root = resolve_project_path(
        content["output"]["directory"]
    )
    manifest_path = resolve_project_path(
        backfill_content.get(
            "manifest_path",
            (
                output_root
                / (
                    "calibration_"
                    "backfill_manifest.json"
                )
            ),
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    loaders: dict[
        str,
        tuple[NPZIQDataset, Any],
    ] = {}

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
    print(f"Device: {device}")
    print(f"Overwrite: {arguments.overwrite}")

    for run_index, expectation in enumerate(
        selected_expectations,
        start=1,
    ):
        tasks: list[
            tuple[
                TestCondition,
                Path,
                Path,
                dict[str, Any],
                float,
                int,
                tuple[str, ...],
            ]
        ] = []

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

            if completed is None:
                raise FileNotFoundError(
                    "Completed evaluation metrics "
                    "are required before backfill: "
                    f"{metrics_path}"
                )

            metrics = load_mapping(
                metrics_path
            )
            class_names = (
                expected_class_names(metrics)
            )
            example_count = (
                expected_example_count(metrics)
            )
            artifact_path = (
                build_calibration_artifact_path(
                    metrics_path,
                    filename=(
                        artifact_filename
                    ),
                )
            )

            existing = (
                load_valid_calibration_artifact(
                    artifact_path,
                    expected_example_count=(
                        example_count
                    ),
                    expected_class_names=(
                        class_names
                    ),
                )
            )

            if (
                existing is not None
                and not arguments.overwrite
            ):
                validate_artifact_accuracy(
                    existing,
                    expected_accuracy=(
                        completed
                        .overall_accuracy
                    ),
                    absolute_tolerance=1e-7,
                )

                records.append(
                    {
                        "fraction_identifier": (
                            expectation
                            .fraction_identifier
                        ),
                        "method": (
                            expectation.method
                        ),
                        "seed": expectation.seed,
                        "condition": (
                            condition.identifier
                        ),
                        "artifact_path": (
                            artifact_path
                            .resolve()
                            .as_posix()
                        ),
                        "status": "skipped",
                    }
                )
                continue

            tasks.append(
                (
                    condition,
                    metrics_path,
                    artifact_path,
                    metrics,
                    completed.overall_accuracy,
                    example_count,
                    class_names,
                )
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

        if not tasks:
            print(
                "All requested artifacts are "
                "already complete."
            )
            write_manifest(
                path=manifest_path,
                config_path=config_path,
                records=records,
            )
            continue

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
        checkpoint_class_names = tuple(
            str(name).upper()
            for name in checkpoint[
                "class_names"
            ]
        )
        model = create_model(
            checkpoint,
            device,
        )

        for (
            condition,
            _metrics_path,
            artifact_path,
            _metrics,
            completed_accuracy,
            example_count,
            class_names,
        ) in tasks:
            if (
                checkpoint_class_names
                != class_names
            ):
                raise ValueError(
                    "Checkpoint class names do "
                    "not match completed metrics."
                )

            _dataset, loader = loaders[
                condition.identifier
            ]
            artifact = (
                collect_calibration_predictions(
                    model=model,
                    data_loader=loader,
                    device=device,
                    input_scale=input_scale,
                    class_names=(
                        checkpoint_class_names
                    ),
                )
            )

            if (
                artifact.example_count
                != example_count
            ):
                raise ValueError(
                    "Generated artifact example "
                    "count does not match metrics."
                )

            accuracy = (
                validate_artifact_accuracy(
                    artifact,
                    expected_accuracy=(
                        completed_accuracy
                    ),
                    absolute_tolerance=1e-7,
                )
            )

            save_calibration_artifact(
                artifact_path,
                artifact,
            )

            records.append(
                {
                    "fraction_identifier": (
                        expectation
                        .fraction_identifier
                    ),
                    "method": (
                        expectation.method
                    ),
                    "seed": expectation.seed,
                    "condition": (
                        condition.identifier
                    ),
                    "accuracy": accuracy,
                    "artifact_path": (
                        artifact_path
                        .resolve()
                        .as_posix()
                    ),
                    "status": "backfilled",
                }
            )

            write_manifest(
                path=manifest_path,
                config_path=config_path,
                records=records,
            )

            print(
                f"{condition.identifier:8s} | "
                f"accuracy={accuracy:.4f} | "
                "backfilled"
            )

        del model
        del checkpoint

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest(
        path=manifest_path,
        config_path=config_path,
        records=records,
    )

    print()
    print(
        "Backfill manifest: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()
