from __future__ import annotations

from pathlib import Path

import pytest

from rfsil.evaluation.calibration_artifacts import (
    build_calibration_artifact,
    save_calibration_artifact,
)
from rfsil.evaluation.calibration_backfill import (
    build_calibration_artifact_path,
    compute_artifact_accuracy,
    load_valid_calibration_artifact,
    validate_artifact_accuracy,
)


def create_artifact():
    return build_calibration_artifact(
        labels=[0, 1, 1],
        logits=[
            [3.0, 1.0],
            [0.0, 2.0],
            [2.0, 1.0],
        ],
        class_names=("BPSK", "QPSK"),
    )


def test_builds_path_beside_metrics() -> None:
    path = build_calibration_artifact_path(
        Path("results/run/metrics.json")
    )

    assert path == Path(
        "results/run/"
        "calibration_predictions.npz"
    )


def test_supports_custom_filename() -> None:
    path = build_calibration_artifact_path(
        "results/run/metrics.json",
        filename="logits_v1.npz",
    )

    assert path.name == "logits_v1.npz"


def test_rejects_nested_filename() -> None:
    with pytest.raises(
        ValueError,
        match="local .npz",
    ):
        build_calibration_artifact_path(
            "metrics.json",
            filename="nested/output.npz",
        )


def test_missing_artifact_returns_none(
    tmp_path: Path,
) -> None:
    result = (
        load_valid_calibration_artifact(
            tmp_path / "missing.npz"
        )
    )

    assert result is None


def test_loads_matching_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.npz"
    save_calibration_artifact(
        path,
        create_artifact(),
    )

    loaded = (
        load_valid_calibration_artifact(
            path,
            expected_example_count=3,
            expected_class_names=(
                "BPSK",
                "QPSK",
            ),
        )
    )

    assert loaded is not None
    assert loaded.example_count == 3


def test_rejects_example_count_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.npz"
    save_calibration_artifact(
        path,
        create_artifact(),
    )

    with pytest.raises(
        ValueError,
        match="example count",
    ):
        load_valid_calibration_artifact(
            path,
            expected_example_count=4,
        )


def test_rejects_class_name_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.npz"
    save_calibration_artifact(
        path,
        create_artifact(),
    )

    with pytest.raises(
        ValueError,
        match="class names",
    ):
        load_valid_calibration_artifact(
            path,
            expected_class_names=(
                "QPSK",
                "BPSK",
            ),
        )


def test_validates_matching_accuracy() -> None:
    artifact = create_artifact()

    assert compute_artifact_accuracy(
        artifact
    ) == pytest.approx(2.0 / 3.0)

    result = validate_artifact_accuracy(
        artifact,
        expected_accuracy=2.0 / 3.0,
    )

    assert result == pytest.approx(
        2.0 / 3.0
    )


def test_rejects_accuracy_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        validate_artifact_accuracy(
            create_artifact(),
            expected_accuracy=1.0,
        )
