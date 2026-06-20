from __future__ import annotations

import argparse
import json
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
    load_calibration_artifact,
    save_calibration_artifact,
)
from rfsil.evaluation.classification import (
    collect_calibration_predictions,
)
from rfsil.evaluation.ssl_label_efficiency import (
    validate_checkpoint_metadata,
)
from rfsil.evaluation.temperature_workflow import (
    fit_temperature_for_artifact,
    load_temperature_summary,
    save_temperature_summary,
)
from rfsil.models.model_factory import (
    create_model_from_checkpoint,
)
from rfsil.training.ssl_sweep_execution import (
    load_run_expectation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export SSL validation logits and "
            "fit one temperature per checkpoint."
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
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_mapping(
    path: Path,
) -> dict[str, Any]:
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
            f"Expected a mapping: {path}"
        )

    return content


def selected(
    record: dict[str, Any],
    *,
    fractions: set[str],
    methods: set[str],
    seeds: set[int],
) -> bool:
    if (
        fractions
        and record["fraction_identifier"]
        not in fractions
    ):
        return False

    if (
        methods
        and record["method"]
        not in methods
    ):
        return False

    return not (
        seeds
        and int(record["training_seed"])
        not in seeds
    )


def write_manifest(
    path: Path,
    *,
    config_path: Path,
    records: list[dict[str, Any]],
) -> None:
    content = {
        "format_version": 1,
        "config_path": (
            config_path.resolve().as_posix()
        ),
        "processed_count": len(records),
        "fitted_count": sum(
            record["status"] == "fitted"
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

    manifest_path = resolve_project_path(
        content["training_manifest"]
    )
    manifest = load_mapping(
        manifest_path
    )
    raw_runs = manifest.get("runs")

    if not isinstance(raw_runs, list):
        raise ValueError(
            "Training manifest runs must "
            "be a list."
        )

    runs = [
        record
        for record in raw_runs
        if selected(
            record,
            fractions=set(
                arguments.fraction
            ),
            methods=set(arguments.method),
            seeds=set(arguments.seed),
        )
    ]

    if arguments.max_runs is not None:
        runs = runs[: arguments.max_runs]

    if not runs:
        raise ValueError(
            "No training runs match "
            "the requested filters."
        )

    evaluation = content["evaluation"]
    scaling = content[
        "temperature_scaling"
    ]
    output = content["output"]

    batch_size = int(
        evaluation["batch_size"]
    )
    num_workers = int(
        evaluation["num_workers"]
    )
    pin_memory = bool(
        evaluation["pin_memory"]
    )
    bin_count = int(
        scaling["bin_count"]
    )
    bounds = tuple(
        float(value)
        for value in scaling[
            "temperature_bounds"
        ]
    )
    tolerance = float(
        scaling[
            "optimization_tolerance"
        ]
    )
    max_iterations = int(
        scaling["max_iterations"]
    )

    artifact_filename = str(
        output["artifact_filename"]
    )
    summary_filename = str(
        output["summary_filename"]
    )
    execution_manifest_path = (
        resolve_project_path(
            output["manifest_path"]
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    records: list[dict[str, Any]] = []

    print(f"Selected runs: {len(runs)}")
    print(f"Device: {device}")
    print(f"Overwrite: {arguments.overwrite}")

    for index, record in enumerate(
        runs,
        start=1,
    ):
        expectation = load_run_expectation(
            record,
            project_root=PROJECT_ROOT,
        )

        checkpoint_path = (
            expectation.output_directory
            / "best_model.pt"
        )
        artifact_path = (
            expectation.output_directory
            / artifact_filename
        )
        summary_path = (
            expectation.output_directory
            / summary_filename
        )

        print()
        print("=" * 80)
        print(
            f"Run {index}/{len(runs)} | "
            f"{expectation.fraction_identifier} | "
            f"{expectation.method} | "
            f"seed={expectation.seed}"
        )
        print("=" * 80)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        validate_checkpoint_metadata(
            checkpoint,
            expectation,
        )

        class_names = tuple(
            str(name).upper()
            for name in checkpoint[
                "class_names"
            ]
        )
        expected_accuracy = float(
            checkpoint[
                "best_validation_accuracy"
            ]
        )

        if (
            artifact_path.is_file()
            and summary_path.is_file()
            and not arguments.overwrite
        ):
            artifact = (
                load_calibration_artifact(
                    artifact_path
                )
            )
            summary = (
                load_temperature_summary(
                    summary_path
                )
            )
            accuracy = float(
                np.mean(
                    artifact.labels
                    == artifact.predictions
                )
            )

            if artifact.class_names != class_names:
                raise ValueError(
                    "Existing validation artifact "
                    "has incorrect class names."
                )

            if not np.isclose(
                accuracy,
                expected_accuracy,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    "Existing validation artifact "
                    "accuracy does not match "
                    "the checkpoint."
                )

            records.append(
                {
                    "fraction_identifier": (
                        expectation
                        .fraction_identifier
                    ),
                    "method": expectation.method,
                    "seed": expectation.seed,
                    "temperature": (
                        summary[
                            "temperature_scaling"
                        ]["temperature"]
                    ),
                    "status": "skipped",
                }
            )

            print(
                "Validated existing artifact "
                "and temperature; skipping."
            )
            continue

        generated_config_path = Path(
            record[
                "generated_config_path"
            ]
        )
        generated_config = load_mapping(
            generated_config_path
        )
        validation_path = (
            resolve_project_path(
                generated_config[
                    "dataset"
                ]["validation_path"]
            )
        )

        dataset = NPZIQDataset(
            validation_path
        )
        loader = create_data_loader(
            dataset,
            DataLoaderConfig(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(
                    pin_memory
                    and device.type == "cuda"
                ),
                seed=expectation.seed,
            ),
        )

        model, _ = (
            create_model_from_checkpoint(
                checkpoint
            )
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.to(device)
        model.eval()

        artifact = (
            collect_calibration_predictions(
                model=model,
                data_loader=loader,
                device=device,
                class_names=class_names,
            )
        )

        accuracy = float(
            np.mean(
                artifact.labels
                == artifact.predictions
            )
        )

        if not np.isclose(
            accuracy,
            expected_accuracy,
            rtol=0.0,
            atol=1e-8,
        ):
            raise RuntimeError(
                "Regenerated validation accuracy "
                "does not match the checkpoint: "
                f"generated={accuracy:.10f}, "
                f"checkpoint="
                f"{expected_accuracy:.10f}."
            )

        summary = fit_temperature_for_artifact(
            artifact,
            bin_count=bin_count,
            temperature_bounds=(
                bounds
            ),
            optimization_tolerance=(
                tolerance
            ),
            max_iterations=max_iterations,
        )
        summary["metadata"] = {
            "fraction_identifier": (
                expectation
                .fraction_identifier
            ),
            "method": expectation.method,
            "seed": expectation.seed,
            "checkpoint_path": (
                checkpoint_path
                .resolve()
                .as_posix()
            ),
            "validation_path": (
                validation_path
                .resolve()
                .as_posix()
            ),
            "validation_accuracy": (
                accuracy
            ),
        }

        save_calibration_artifact(
            artifact_path,
            artifact,
        )
        save_temperature_summary(
            summary_path,
            summary,
        )

        temperature = float(
            summary[
                "temperature_scaling"
            ]["temperature"]
        )

        records.append(
            {
                "fraction_identifier": (
                    expectation
                    .fraction_identifier
                ),
                "method": expectation.method,
                "seed": expectation.seed,
                "validation_accuracy": (
                    accuracy
                ),
                "temperature": temperature,
                "baseline_nll": (
                    summary[
                        "temperature_scaling"
                    ]["baseline_nll"]
                ),
                "calibrated_nll": (
                    summary[
                        "temperature_scaling"
                    ]["calibrated_nll"]
                ),
                "status": "fitted",
            }
        )

        write_manifest(
            execution_manifest_path,
            config_path=config_path,
            records=records,
        )

        print(
            f"Validation accuracy: "
            f"{accuracy:.4f}"
        )
        print(
            f"Temperature: "
            f"{temperature:.6f}"
        )
        print(
            "NLL: "
            f"{summary[
                'temperature_scaling'
            ]['baseline_nll']:.6f}"
            " -> "
            f"{summary[
                'temperature_scaling'
            ]['calibrated_nll']:.6f}"
        )

        del model
        del checkpoint

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_manifest(
        execution_manifest_path,
        config_path=config_path,
        records=records,
    )

    print()
    print(
        "Execution manifest: "
        f"{execution_manifest_path}"
    )


if __name__ == "__main__":
    main()
