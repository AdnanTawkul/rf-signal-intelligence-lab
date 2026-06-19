from __future__ import annotations

from pathlib import Path

import pytest

from rfsil.training.ssl_label_efficiency import (
    LabelFraction,
    build_run_plan,
    validate_label_fractions,
    validate_seeds,
)


def make_template() -> dict[str, object]:
    return {
        "experiment_name": "template",
        "seed": 2026,
        "dataset": {
            "train_path": "train.npz",
            "validation_path": "validation.npz",
        },
        "model": {
            "normalization": "group",
        },
        "training": {
            "epochs": 330,
            "seed": 9999,
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_workers": 0,
            "pin_memory": True,
            "examples_per_class_snr": 10,
            "subset_seed": 2026,
        },
        "initialization": None,
        "output": {
            "directory": "old",
            "figure_path": "old.png",
        },
    }


def test_validates_seed_list() -> None:
    assert validate_seeds(
        [
            2026,
            2027,
        ]
    ) == (
        2026,
        2027,
    )


def test_rejects_duplicate_seeds() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        validate_seeds(
            [
                2026,
                2026,
            ]
        )


def test_validates_label_fractions() -> None:
    fractions = validate_label_fractions(
        [
            {
                "identifier": "labels_005pct",
                "display_name": "5%",
                "examples_per_class_snr": 10,
            }
        ]
    )

    assert fractions == (
        LabelFraction(
            identifier="labels_005pct",
            display_name="5%",
            examples_per_class_snr=10,
        ),
    )


@pytest.mark.parametrize(
    (
        "examples_per_class_snr",
        "expected_examples",
        "expected_steps",
        "expected_epochs",
    ),
    (
        (2, 56, 1, 1320),
        (10, 280, 3, 440),
        (20, 560, 5, 264),
        (50, 1400, 11, 120),
        (200, 5600, 44, 30),
    ),
)
def test_builds_exact_fraction_budget(
    tmp_path: Path,
    examples_per_class_snr: int,
    expected_examples: int,
    expected_steps: int,
    expected_epochs: int,
) -> None:
    fraction = LabelFraction(
        identifier=(
            f"labels_{examples_per_class_snr}"
        ),
        display_name="fraction",
        examples_per_class_snr=(
            examples_per_class_snr
        ),
    )

    plan = build_run_plan(
        template=make_template(),
        method="random",
        seed=2029,
        fraction=fraction,
        stratum_count=28,
        batch_size=128,
        target_optimizer_steps=1320,
        drop_last=False,
        require_exact=True,
        experiment_prefix="test",
        output_root=tmp_path / "results",
        figure_root=tmp_path / "figures",
        project_root=tmp_path,
    )

    assert (
        plan.selected_training_examples
        == expected_examples
    )
    assert (
        plan.budget.steps_per_epoch
        == expected_steps
    )
    assert plan.budget.epochs == expected_epochs
    assert (
        plan.budget.actual_optimizer_steps
        == 1320
    )


def test_sets_real_training_and_subset_seed(
    tmp_path: Path,
) -> None:
    plan = build_run_plan(
        template=make_template(),
        method="simclr",
        seed=2030,
        fraction=LabelFraction(
            identifier="labels_005pct",
            display_name="5%",
            examples_per_class_snr=10,
        ),
        stratum_count=28,
        batch_size=128,
        target_optimizer_steps=1320,
        drop_last=False,
        require_exact=True,
        experiment_prefix="test",
        output_root=tmp_path / "results",
        figure_root=tmp_path / "figures",
        project_root=tmp_path,
    )

    training = plan.content["training"]

    assert plan.content["seed"] == 2030
    assert training["subset_seed"] == 2030
    assert "seed" not in training
    assert "epochs" not in training
    assert (
        training["target_optimizer_steps"]
        == 1320
    )
