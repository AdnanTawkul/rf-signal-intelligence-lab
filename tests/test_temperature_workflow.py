from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.calibration_artifacts import (
    build_calibration_artifact,
)
from rfsil.evaluation.temperature_workflow import (
    fit_temperature_for_artifact,
    load_temperature_summary,
    save_temperature_summary,
    validate_temperature_summary,
)


def create_artifact():
    return build_calibration_artifact(
        labels=[0, 0, 1, 1],
        logits=[
            [8.0, 0.0],
            [7.0, 0.0],
            [0.0, 8.0],
            [8.0, 0.0],
        ],
        snr_db=[
            -4.0,
            0.0,
            4.0,
            8.0,
        ],
        class_names=(
            "BPSK",
            "QPSK",
        ),
    )


def test_fits_temperature_and_preserves_accuracy(
) -> None:
    summary = fit_temperature_for_artifact(
        create_artifact(),
        bin_count=5,
    )

    scaling = summary[
        "temperature_scaling"
    ]

    assert scaling["temperature"] > 1.0
    assert (
        scaling["calibrated_nll"]
        < scaling["baseline_nll"]
    )
    assert summary[
        "accuracy_preserved"
    ]
    assert (
        summary[
            "baseline_calibration"
        ]["accuracy"]
        == summary[
            "calibrated_calibration"
        ]["accuracy"]
    )


def test_summary_contains_class_metadata() -> None:
    summary = fit_temperature_for_artifact(
        create_artifact()
    )

    assert summary["prediction_count"] == 4
    assert summary["class_count"] == 2
    assert summary["class_names"] == [
        "BPSK",
        "QPSK",
    ]


def test_round_trip(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "nested"
        / "temperature.json"
    )
    summary = fit_temperature_for_artifact(
        create_artifact()
    )

    result = save_temperature_summary(
        path,
        summary,
    )
    loaded = load_temperature_summary(
        path
    )

    assert result == path
    assert (
        loaded["temperature_scaling"][
            "temperature"
        ]
        == pytest.approx(
            summary[
                "temperature_scaling"
            ]["temperature"]
        )
    )


def test_rejects_unsupported_version() -> None:
    summary = fit_temperature_for_artifact(
        create_artifact()
    )
    summary["format_version"] = 999

    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        validate_temperature_summary(
            summary
        )


def test_rejects_nonpositive_temperature() -> None:
    summary = fit_temperature_for_artifact(
        create_artifact()
    )
    summary["temperature_scaling"][
        "temperature"
    ] = 0.0

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        validate_temperature_summary(
            summary
        )


def test_rejects_missing_accuracy_confirmation(
) -> None:
    summary = fit_temperature_for_artifact(
        create_artifact()
    )
    summary["accuracy_preserved"] = False

    with pytest.raises(
        ValueError,
        match="accuracy preservation",
    ):
        validate_temperature_summary(
            summary
        )


def test_load_rejects_invalid_json_structure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(["not", "a", "mapping"]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mapping",
    ):
        load_temperature_summary(path)


def test_calibrated_probabilities_keep_predictions(
) -> None:
    artifact = create_artifact()
    summary = fit_temperature_for_artifact(
        artifact
    )

    assert summary[
        "baseline_calibration"
    ]["accuracy"] == pytest.approx(
        np.mean(
            artifact.labels
            == artifact.predictions
        )
    )
