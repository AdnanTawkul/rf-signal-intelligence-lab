from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from rfsil.training.ssl_sweep_execution import (
    load_completed_run,
    load_run_expectation,
)


def make_manifest_record(
    tmp_path: Path,
) -> dict[str, object]:
    config_path = (
        tmp_path / "generated.yaml"
    )
    output_directory = (
        tmp_path / "run"
    )

    config = {
        "experiment_name": (
            "test_labels_005pct_random_seed_2029"
        ),
        "seed": 2029,
        "training": {
            "batch_size": 128,
            "examples_per_class_snr": 10,
            "subset_seed": 2029,
            "target_optimizer_steps": 1320,
            "require_exact_optimizer_steps": True,
            "drop_last": False,
        },
        "output": {
            "directory": str(
                output_directory
            ),
            "figure_path": str(
                tmp_path / "figure.png"
            ),
        },
    }

    config_path.write_text(
        yaml.safe_dump(
            config,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return {
        "fraction_identifier": (
            "labels_005pct"
        ),
        "fraction_display_name": "5%",
        "examples_per_class_snr": 10,
        "selected_training_examples": 280,
        "method": "random",
        "training_seed": 2029,
        "subset_seed": 2029,
        "generated_config_path": str(
            config_path
        ),
        "output_directory": str(
            output_directory
        ),
        "training_budget": {
            "example_count": 280,
            "batch_size": 128,
            "drop_last": False,
            "steps_per_epoch": 3,
            "target_optimizer_steps": 1320,
            "epochs": 440,
            "actual_optimizer_steps": 1320,
            "exact_match": True,
        },
    }


def write_completed_run(
    tmp_path: Path,
    record: dict[str, object],
) -> None:
    output_directory = Path(
        str(record["output_directory"])
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_directory / "best_model.pt"
    )
    history_path = (
        output_directory / "history.json"
    )

    checkpoint_path.write_bytes(
        b"checkpoint"
    )
    history_path.write_text(
        "[]",
        encoding="utf-8",
    )

    config = yaml.safe_load(
        Path(
            str(
                record[
                    "generated_config_path"
                ]
            )
        ).read_text(encoding="utf-8")
    )

    summary = {
        "experiment_name": (
            config["experiment_name"]
        ),
        "seed": record["training_seed"],
        "labeled_subset": {
            "examples_per_class_snr": (
                record[
                    "examples_per_class_snr"
                ]
            ),
            "subset_seed": (
                record["subset_seed"]
            ),
            "selected_training_examples": (
                record[
                    "selected_training_examples"
                ]
            ),
        },
        "training_budget": (
            record["training_budget"]
        ),
        "best_epoch": 100,
        "best_validation_accuracy": 0.75,
        "checkpoint_path": str(
            checkpoint_path
        ),
        "history_path": str(
            history_path
        ),
    }

    (
        output_directory / "summary.json"
    ).write_text(
        json.dumps(summary),
        encoding="utf-8",
    )


def test_loads_run_expectation(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        tmp_path
    )

    expectation = load_run_expectation(
        record,
        project_root=tmp_path,
    )

    assert expectation.seed == 2029
    assert expectation.subset_seed == 2029
    assert (
        expectation.selected_training_examples
        == 280
    )
    assert (
        expectation.training_budget[
            "actual_optimizer_steps"
        ]
        == 1320
    )


def test_rejects_wrong_generated_seed(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        tmp_path
    )
    config_path = Path(
        str(record["generated_config_path"])
    )
    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )
    config["seed"] = 2030
    config_path.write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="top-level seed",
    ):
        load_run_expectation(
            record,
            project_root=tmp_path,
        )


def test_missing_summary_is_incomplete(
    tmp_path: Path,
) -> None:
    expectation = load_run_expectation(
        make_manifest_record(tmp_path),
        project_root=tmp_path,
    )

    assert (
        load_completed_run(
            expectation,
            project_root=tmp_path,
        )
        is None
    )


def test_loads_valid_completed_run(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        tmp_path
    )
    write_completed_run(
        tmp_path,
        record,
    )
    expectation = load_run_expectation(
        record,
        project_root=tmp_path,
    )

    completed = load_completed_run(
        expectation,
        project_root=tmp_path,
    )

    assert completed is not None
    assert (
        completed.best_validation_accuracy
        == pytest.approx(0.75)
    )
    assert completed.best_epoch == 100

    serialized = asdict(completed)
    assert serialized["seed"] == 2029


def test_rejects_wrong_completed_budget(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        tmp_path
    )
    write_completed_run(
        tmp_path,
        record,
    )

    summary_path = (
        Path(str(record["output_directory"]))
        / "summary.json"
    )
    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )
    summary["training_budget"][
        "actual_optimizer_steps"
    ] = 1319
    summary_path.write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    expectation = load_run_expectation(
        record,
        project_root=tmp_path,
    )

    with pytest.raises(
        ValueError,
        match="training budget",
    ):
        load_completed_run(
            expectation,
            project_root=tmp_path,
        )


def test_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    record = make_manifest_record(
        tmp_path
    )
    write_completed_run(
        tmp_path,
        record,
    )

    (
        Path(str(record["output_directory"]))
        / "best_model.pt"
    ).unlink()

    expectation = load_run_expectation(
        record,
        project_root=tmp_path,
    )

    with pytest.raises(FileNotFoundError):
        load_completed_run(
            expectation,
            project_root=tmp_path,
        )
