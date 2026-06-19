from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfsil.evaluation.ssl_label_efficiency import (
    TestCondition,
    build_metrics_path,
    load_completed_test_evaluation,
    validate_checkpoint_metadata,
    validate_test_conditions,
)
from rfsil.training.ssl_sweep_execution import (
    SweepRunExpectation,
)


def make_expectation(
    tmp_path: Path,
) -> SweepRunExpectation:
    return SweepRunExpectation(
        fraction_identifier="labels_005pct",
        method="vicreg",
        seed=2029,
        subset_seed=2029,
        examples_per_class_snr=10,
        selected_training_examples=280,
        experiment_name="test_run",
        training_budget={
            "example_count": 280,
            "batch_size": 128,
            "drop_last": False,
            "steps_per_epoch": 3,
            "target_optimizer_steps": 1320,
            "epochs": 440,
            "actual_optimizer_steps": 1320,
            "exact_match": True,
        },
        generated_config_path=(
            tmp_path / "generated.yaml"
        ),
        output_directory=(
            tmp_path / "training"
        ),
    )


def make_condition(
    tmp_path: Path,
) -> TestCondition:
    test_path = tmp_path / "test.npz"
    test_path.write_bytes(b"dataset")

    return TestCondition(
        identifier="clean",
        display_name="Clean",
        test_path=test_path,
    )


def test_validates_conditions(
    tmp_path: Path,
) -> None:
    test_path = tmp_path / "test.npz"
    test_path.write_bytes(b"dataset")

    conditions = validate_test_conditions(
        {
            "clean": {
                "display_name": "Clean",
                "test_path": str(test_path),
            }
        },
        project_root=tmp_path,
    )

    assert len(conditions) == 1
    assert conditions[0].identifier == "clean"


def test_rejects_missing_condition_dataset(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError
    ):
        validate_test_conditions(
            {
                "clean": {
                    "display_name": "Clean",
                    "test_path": "missing.npz",
                }
            },
            project_root=tmp_path,
        )


def test_builds_deterministic_metrics_path(
    tmp_path: Path,
) -> None:
    path = build_metrics_path(
        output_root=tmp_path / "results",
        expectation=make_expectation(
            tmp_path
        ),
        condition=make_condition(
            tmp_path
        ),
    )

    assert path == (
        tmp_path
        / "results"
        / "clean"
        / "labels_005pct"
        / "vicreg"
        / "seed_2029"
        / "metrics.json"
    )


def test_validates_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    expectation = make_expectation(
        tmp_path
    )

    validate_checkpoint_metadata(
        {
            "seed": 2029,
            "labeled_subset": {
                "examples_per_class_snr": 10,
                "subset_seed": 2029,
                "selected_training_examples": 280,
            },
            "training_budget": (
                expectation.training_budget
            ),
        },
        expectation,
    )


def test_rejects_wrong_checkpoint_seed(
    tmp_path: Path,
) -> None:
    expectation = make_expectation(
        tmp_path
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint seed",
    ):
        validate_checkpoint_metadata(
            {
                "seed": 2030,
                "labeled_subset": {
                    "examples_per_class_snr": 10,
                    "subset_seed": 2029,
                    "selected_training_examples": 280,
                },
                "training_budget": (
                    expectation.training_budget
                ),
            },
            expectation,
        )


def test_missing_metrics_is_incomplete(
    tmp_path: Path,
) -> None:
    result = load_completed_test_evaluation(
        metrics_path=(
            tmp_path / "metrics.json"
        ),
        expectation=make_expectation(
            tmp_path
        ),
        condition=make_condition(
            tmp_path
        ),
        project_root=tmp_path,
    )

    assert result is None


def test_loads_valid_completed_metrics(
    tmp_path: Path,
) -> None:
    expectation = make_expectation(
        tmp_path
    )
    condition = make_condition(
        tmp_path
    )

    expectation.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = (
        expectation.output_directory
        / "best_model.pt"
    )
    checkpoint_path.write_bytes(
        b"checkpoint"
    )

    metrics_path = (
        tmp_path / "metrics.json"
    )
    metrics_path.write_text(
        json.dumps(
            {
                "fraction_identifier": (
                    expectation
                    .fraction_identifier
                ),
                "method": expectation.method,
                "seed": expectation.seed,
                "condition": (
                    condition.identifier
                ),
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "test_path": str(
                    condition.test_path
                ),
                "overall_accuracy": 0.75,
                "class_names": [
                    "BPSK",
                    "QPSK",
                ],
                "confusion_matrix": [
                    [10, 1],
                    [2, 9],
                ],
                "normalized_confusion_matrix": [
                    [0.9, 0.1],
                    [0.2, 0.8],
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = (
        load_completed_test_evaluation(
            metrics_path=metrics_path,
            expectation=expectation,
            condition=condition,
            project_root=tmp_path,
        )
    )

    assert completed is not None
    assert (
        completed.overall_accuracy
        == pytest.approx(0.75)
    )


def test_rejects_invalid_accuracy(
    tmp_path: Path,
) -> None:
    expectation = make_expectation(
        tmp_path
    )
    condition = make_condition(
        tmp_path
    )

    expectation.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = (
        expectation.output_directory
        / "best_model.pt"
    )
    checkpoint_path.write_bytes(
        b"checkpoint"
    )

    metrics_path = (
        tmp_path / "metrics.json"
    )
    metrics_path.write_text(
        json.dumps(
            {
                "fraction_identifier": (
                    expectation
                    .fraction_identifier
                ),
                "method": expectation.method,
                "seed": expectation.seed,
                "condition": "clean",
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "test_path": str(
                    condition.test_path
                ),
                "overall_accuracy": 1.5,
                "class_names": ["BPSK"],
                "confusion_matrix": [[1]],
                "normalized_confusion_matrix": [
                    [1.0]
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="between zero and one",
    ):
        load_completed_test_evaluation(
            metrics_path=metrics_path,
            expectation=expectation,
            condition=condition,
            project_root=tmp_path,
        )
