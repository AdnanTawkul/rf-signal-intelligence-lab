from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.external_transfer import (
    compute_paired_accuracy_change,
    load_external_seed_sweep,
    summarize_accuracy,
    summarize_class_accuracy,
)


def write_aggregate(
    path: Path,
    *,
    offset: float = 0.0,
    seeds: tuple[int, ...] = (
        2026,
        2027,
    ),
) -> None:
    class_names = [
        "BPSK",
        "QPSK",
    ]

    runs = []

    for run_index, seed in enumerate(
        seeds
    ):
        base = (
            0.5
            + run_index * 0.1
            + offset
        )

        runs.append(
            {
                "seed": seed,
                "overall_accuracy": base,
                "class_accuracy": {
                    "BPSK": base + 0.1,
                    "QPSK": base - 0.1,
                },
                "accuracy_by_snr": {
                    "-4.0": base - 0.1,
                    "0.0": base,
                    "4.0": base + 0.1,
                },
            }
        )

    path.write_text(
        json.dumps(
            {
                "experiment_name": (
                    "test_experiment"
                ),
                "class_names": class_names,
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def test_loads_and_sorts_seed_runs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"

    write_aggregate(
        path,
        seeds=(
            2027,
            2026,
        ),
    )

    sweep = load_external_seed_sweep(
        path
    )

    assert sweep.seeds == (
        2026,
        2027,
    )
    assert sweep.class_names == (
        "BPSK",
        "QPSK",
    )
    assert sweep.snr_values_db == (
        -4.0,
        0.0,
        4.0,
    )
    assert sweep.overall_accuracy.shape == (
        2,
    )
    assert sweep.class_accuracy.shape == (
        2,
        2,
    )
    assert sweep.snr_accuracy.shape == (
        2,
        3,
    )


def test_summarizes_overall_accuracy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(path)

    sweep = load_external_seed_sweep(
        path
    )
    summary = summarize_accuracy(
        sweep
    )

    assert summary.mean == pytest.approx(
        0.55
    )
    assert (
        summary.standard_deviation
        == pytest.approx(0.05)
    )
    np.testing.assert_allclose(
        summary.per_seed,
        np.asarray(
            [
                0.5,
                0.6,
            ]
        ),
    )


def test_summarizes_selected_snr_grid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(path)

    sweep = load_external_seed_sweep(
        path
    )
    summary = summarize_accuracy(
        sweep,
        snr_values_db=(
            -4.0,
            4.0,
        ),
    )

    assert summary.mean == pytest.approx(
        0.55
    )
    np.testing.assert_allclose(
        summary.per_seed,
        np.asarray(
            [
                0.5,
                0.6,
            ]
        ),
    )


def test_summarizes_class_accuracy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(path)

    sweep = load_external_seed_sweep(
        path
    )
    summary = summarize_class_accuracy(
        sweep
    )

    assert summary["BPSK"].mean == (
        pytest.approx(0.65)
    )
    assert summary["QPSK"].mean == (
        pytest.approx(0.45)
    )


def test_computes_paired_change(
    tmp_path: Path,
) -> None:
    reference_path = (
        tmp_path / "reference.json"
    )
    candidate_path = (
        tmp_path / "candidate.json"
    )

    write_aggregate(
        reference_path
    )
    write_aggregate(
        candidate_path,
        offset=0.02,
    )

    reference = (
        load_external_seed_sweep(
            reference_path
        )
    )
    candidate = (
        load_external_seed_sweep(
            candidate_path
        )
    )

    change = (
        compute_paired_accuracy_change(
            reference,
            candidate,
        )
    )

    assert change.mean == pytest.approx(
        0.02
    )
    assert (
        change.standard_deviation
        == pytest.approx(0.0)
    )
    assert change.improved_seed_count == 2
    np.testing.assert_allclose(
        change.per_seed,
        np.asarray(
            [
                0.02,
                0.02,
            ]
        ),
        atol=1e-12,
    )


def test_rejects_mismatched_paired_seeds(
    tmp_path: Path,
) -> None:
    reference_path = (
        tmp_path / "reference.json"
    )
    candidate_path = (
        tmp_path / "candidate.json"
    )

    write_aggregate(
        reference_path,
        seeds=(
            2026,
            2027,
        ),
    )
    write_aggregate(
        candidate_path,
        seeds=(
            2026,
            2028,
        ),
    )

    reference = (
        load_external_seed_sweep(
            reference_path
        )
    )
    candidate = (
        load_external_seed_sweep(
            candidate_path
        )
    )

    with pytest.raises(
        ValueError,
        match="identical seeds",
    ):
        compute_paired_accuracy_change(
            reference,
            candidate,
        )


def test_rejects_missing_requested_snr(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aggregate.json"
    write_aggregate(path)

    sweep = load_external_seed_sweep(
        path
    )

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        summarize_accuracy(
            sweep,
            snr_values_db=(
                -4.0,
                8.0,
            ),
        )
