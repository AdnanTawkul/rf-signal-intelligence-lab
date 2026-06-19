from __future__ import annotations

from pathlib import Path

import yaml

from rfsil.training.seed_sweep_config import (
    expand_seed_templates,
    prepare_seed_training_config,
)


def test_nested_seed_templates_are_expanded() -> None:
    content = {
        "seed": 2026,
        "initialization": {
            "type": "frozen_supervised_backbone",
            "checkpoint_path": (
                "results/baseline/"
                "seed_{seed}/best_model.pt"
            ),
        },
        "output": {
            "directory": (
                "results/experiment_{seed}"
            ),
        },
    }

    expanded = expand_seed_templates(
        content,
        2029,
    )

    assert expanded["initialization"][
        "checkpoint_path"
    ] == (
        "results/baseline/"
        "seed_2029/best_model.pt"
    )
    assert expanded["output"][
        "directory"
    ] == "results/experiment_2029"


def test_non_template_values_are_preserved() -> None:
    content = {
        "epochs": 30,
        "enabled": True,
        "channels": [32, 64, 128],
        "name": "baseline",
    }

    expanded = expand_seed_templates(
        content,
        2027,
    )

    assert expanded == content


def test_prepared_config_is_self_contained(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path / "seed_2030.yaml"
    )

    result = prepare_seed_training_config(
        base_content={
            "seed": 2026,
            "initialization": {
                "checkpoint_path": (
                    "results/source/"
                    "seed_{seed}/best_model.pt"
                ),
            },
        },
        seed=2030,
        output_path=output_path,
    )

    assert result["seed"] == 2030
    assert result["initialization"][
        "checkpoint_path"
    ] == (
        "results/source/"
        "seed_2030/best_model.pt"
    )

    stored = yaml.safe_load(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored == result


def test_base_content_is_not_modified(
    tmp_path: Path,
) -> None:
    base = {
        "seed": 2026,
        "path": "seed_{seed}",
    }

    prepare_seed_training_config(
        base_content=base,
        seed=2028,
        output_path=tmp_path / "config.yaml",
    )

    assert base == {
        "seed": 2026,
        "path": "seed_{seed}",
    }
