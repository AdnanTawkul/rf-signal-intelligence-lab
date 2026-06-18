from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfsil.evaluation.channel_robustness import (
    compare_seed_sweep_conditions,
    load_seed_sweep_metrics,
)


def write_aggregate(
    path: Path,
    accuracies: tuple[float, float],
    class_names: tuple[str, ...] = (
        "BPSK",
        "QPSK",
    ),
    snr_values: tuple[float, ...] = (
        -4.0,
        0.0,
    ),
) -> None:
    """Write a minimal valid aggregate file."""
    runs = []

    for seed, accuracy in zip(
        (2026, 2027),
        accuracies,
        strict=True,
    ):
        runs.append(
            {
                "seed": seed,
                "overall_accuracy": accuracy,
                "class_accuracy": {
                    class_name: accuracy
                    for class_name in class_names
                },
                "accuracy_by_snr": {
                    str(snr): accuracy
                    for snr in snr_values
                },
            }
        )

    content = {
        "format_version": 1,
        "experiment_name": "test",
        "test_path": "test.npz",
        "class_names": list(class_names),
        "runs": runs,
        "aggregate": {},
    }

    path.write_text(
        json.dumps(content),
        encoding="utf-8",
    )


def test_load_seed_sweep_metrics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(
        path,
        (0.9, 0.8),
    )

    metrics = load_seed_sweep_metrics(
        path,
        "clean",
    )

    assert metrics.condition == "clean"
    assert metrics.seeds == (2026, 2027)
    assert metrics.class_names == (
        "BPSK",
        "QPSK",
    )
    assert metrics.snr_values_db == (
        -4.0,
        0.0,
    )
    assert metrics.overall_accuracy.shape == (2,)
    assert metrics.class_accuracy.shape == (2, 2)
    assert metrics.accuracy_by_snr.shape == (2, 2)


def test_comparison_computes_paired_drop(
    tmp_path: Path,
) -> None:
    clean_path = tmp_path / "clean.json"
    mild_path = tmp_path / "mild.json"

    write_aggregate(
        clean_path,
        (0.9, 0.8),
    )
    write_aggregate(
        mild_path,
        (0.7, 0.6),
    )

    clean = load_seed_sweep_metrics(
        clean_path,
        "clean",
    )
    mild = load_seed_sweep_metrics(
        mild_path,
        "mild",
    )

    result = compare_seed_sweep_conditions(
        {
            "clean": clean,
            "mild": mild,
        }
    )

    drop = result["conditions"]["mild"][
        "paired_drop_from_reference"
    ]

    assert drop["mean"] == pytest.approx(0.2)
    assert drop["standard_deviation"] == pytest.approx(
        0.0
    )


def test_mismatched_seeds_are_rejected(
    tmp_path: Path,
) -> None:
    clean_path = tmp_path / "clean.json"
    mild_path = tmp_path / "mild.json"

    write_aggregate(
        clean_path,
        (0.9, 0.8),
    )
    write_aggregate(
        mild_path,
        (0.7, 0.6),
    )

    content = json.loads(
        mild_path.read_text(encoding="utf-8")
    )
    content["runs"][1]["seed"] = 2030
    mild_path.write_text(
        json.dumps(content),
        encoding="utf-8",
    )

    clean = load_seed_sweep_metrics(
        clean_path,
        "clean",
    )
    mild = load_seed_sweep_metrics(
        mild_path,
        "mild",
    )

    with pytest.raises(ValueError):
        compare_seed_sweep_conditions(
            {
                "clean": clean,
                "mild": mild,
            }
        )


def test_mismatched_classes_are_rejected(
    tmp_path: Path,
) -> None:
    clean_path = tmp_path / "clean.json"
    mild_path = tmp_path / "mild.json"

    write_aggregate(
        clean_path,
        (0.9, 0.8),
    )
    write_aggregate(
        mild_path,
        (0.7, 0.6),
        class_names=("BPSK", "8PSK"),
    )

    clean = load_seed_sweep_metrics(
        clean_path,
        "clean",
    )
    mild = load_seed_sweep_metrics(
        mild_path,
        "mild",
    )

    with pytest.raises(ValueError):
        compare_seed_sweep_conditions(
            {
                "clean": clean,
                "mild": mild,
            }
        )


def test_mismatched_snr_values_are_rejected(
    tmp_path: Path,
) -> None:
    clean_path = tmp_path / "clean.json"
    mild_path = tmp_path / "mild.json"

    write_aggregate(
        clean_path,
        (0.9, 0.8),
    )
    write_aggregate(
        mild_path,
        (0.7, 0.6),
        snr_values=(-4.0, 4.0),
    )

    clean = load_seed_sweep_metrics(
        clean_path,
        "clean",
    )
    mild = load_seed_sweep_metrics(
        mild_path,
        "mild",
    )

    with pytest.raises(ValueError):
        compare_seed_sweep_conditions(
            {
                "clean": clean,
                "mild": mild,
            }
        )


def test_invalid_accuracy_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(
        path,
        (0.9, 1.5),
    )

    with pytest.raises(ValueError):
        load_seed_sweep_metrics(
            path,
            "clean",
        )


def test_missing_reference_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(
        path,
        (0.9, 0.8),
    )

    metrics = load_seed_sweep_metrics(
        path,
        "mild",
    )

    with pytest.raises(ValueError):
        compare_seed_sweep_conditions(
            {
                "mild": metrics,
            },
            reference_condition="clean",
        )
