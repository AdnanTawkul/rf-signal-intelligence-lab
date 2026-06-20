from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.calibration_artifacts import (
    CALIBRATION_ARTIFACT_VERSION,
    build_calibration_artifact,
    load_calibration_artifact,
    save_calibration_artifact,
    validate_calibration_artifact,
)


def example_logits() -> np.ndarray:
    return np.asarray(
        [
            [3.0, 1.0, -1.0],
            [-2.0, 0.5, 2.5],
            [0.0, 4.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_builds_artifact_from_logits() -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
        snr_db=[-4.0, 0.0, 8.0],
        class_names=(
            "bpsk",
            "qpsk",
            "8psk",
        ),
    )

    assert artifact.example_count == 3
    assert artifact.class_count == 3
    assert artifact.predictions.tolist() == [
        0,
        2,
        1,
    ]

    np.testing.assert_allclose(
        artifact.probabilities.sum(axis=1),
        np.ones(3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_round_trip_with_metadata(
    tmp_path: Path,
) -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
        snr_db=[-4.0, 0.0, 8.0],
        class_names=(
            "bpsk",
            "qpsk",
            "8psk",
        ),
    )
    path = tmp_path / "predictions.npz"

    save_calibration_artifact(
        path,
        artifact,
    )
    loaded = load_calibration_artifact(
        path
    )

    np.testing.assert_array_equal(
        loaded.labels,
        artifact.labels,
    )
    np.testing.assert_array_equal(
        loaded.predictions,
        artifact.predictions,
    )
    np.testing.assert_allclose(
        loaded.logits,
        artifact.logits,
    )
    np.testing.assert_allclose(
        loaded.probabilities,
        artifact.probabilities,
    )
    np.testing.assert_allclose(
        loaded.snr_db,
        artifact.snr_db,
    )
    assert (
        loaded.class_names
        == artifact.class_names
    )


def test_round_trip_without_optional_metadata(
    tmp_path: Path,
) -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
    )
    path = tmp_path / "predictions.npz"

    save_calibration_artifact(
        path,
        artifact,
    )
    loaded = load_calibration_artifact(
        path
    )

    assert loaded.snr_db is None
    assert loaded.class_names is None


def test_save_creates_parent_directory(
    tmp_path: Path,
) -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
    )
    path = (
        tmp_path
        / "nested"
        / "directory"
        / "predictions.npz"
    )

    result = save_calibration_artifact(
        path,
        artifact,
    )

    assert result == path
    assert path.is_file()


def test_rejects_boolean_labels() -> None:
    with pytest.raises(
        ValueError,
        match="integers",
    ):
        build_calibration_artifact(
            labels=[True, False, True],
            logits=example_logits(),
        )


def test_rejects_label_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="3 examples",
    ):
        build_calibration_artifact(
            labels=[0, 1],
            logits=example_logits(),
        )


def test_rejects_out_of_range_labels() -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        build_calibration_artifact(
            labels=[0, 3, 1],
            logits=example_logits(),
        )


def test_rejects_nonfinite_logits() -> None:
    logits = example_logits()
    logits[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        build_calibration_artifact(
            labels=[0, 2, 1],
            logits=logits,
        )


def test_rejects_wrong_class_name_count() -> None:
    with pytest.raises(
        ValueError,
        match="class count",
    ):
        build_calibration_artifact(
            labels=[0, 2, 1],
            logits=example_logits(),
            class_names=("a", "b"),
        )


def test_rejects_duplicate_class_names() -> None:
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        build_calibration_artifact(
            labels=[0, 2, 1],
            logits=example_logits(),
            class_names=("a", "a", "b"),
        )


def test_rejects_prediction_mismatch() -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
    )

    with pytest.raises(
        ValueError,
        match="predictions",
    ):
        validate_calibration_artifact(
            labels=artifact.labels,
            predictions=[1, 2, 1],
            logits=artifact.logits,
            probabilities=(
                artifact.probabilities
            ),
        )


def test_rejects_probability_logit_mismatch(
) -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
    )
    probabilities = (
        artifact.probabilities.copy()
    )
    probabilities[0] = np.asarray(
        [0.2, 0.7, 0.1],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="inconsistent with logits",
    ):
        validate_calibration_artifact(
            labels=artifact.labels,
            predictions=[1, 2, 1],
            logits=artifact.logits,
            probabilities=probabilities,
        )


def test_rejects_probability_rows_not_summing_to_one(
) -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
    )
    probabilities = (
        artifact.probabilities.copy()
    )
    probabilities[0] *= 0.5

    with pytest.raises(
        ValueError,
        match="sum to one",
    ):
        validate_calibration_artifact(
            labels=artifact.labels,
            predictions=artifact.predictions,
            logits=artifact.logits,
            probabilities=probabilities,
        )


def test_rejects_legacy_artifact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.npz"

    np.savez_compressed(
        path,
        labels=np.asarray(
            [0, 1],
            dtype=np.int64,
        ),
        predictions=np.asarray(
            [0, 1],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [0.0, 4.0],
            dtype=np.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Legacy prediction artifact",
    ):
        load_calibration_artifact(path)


def test_rejects_missing_required_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.npz"

    np.savez_compressed(
        path,
        format_version=np.asarray(
            CALIBRATION_ARTIFACT_VERSION,
            dtype=np.int64,
        ),
        labels=np.asarray(
            [0],
            dtype=np.int64,
        ),
        predictions=np.asarray(
            [0],
            dtype=np.int64,
        ),
        logits=np.asarray(
            [[2.0, 1.0]],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="logits and probabilities",
    ):
        load_calibration_artifact(path)


def test_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    artifact = build_calibration_artifact(
        labels=[0, 2, 1],
        logits=example_logits(),
    )
    path = tmp_path / "version.npz"

    np.savez_compressed(
        path,
        format_version=np.asarray(
            999,
            dtype=np.int64,
        ),
        labels=artifact.labels,
        predictions=artifact.predictions,
        logits=artifact.logits,
        probabilities=(
            artifact.probabilities
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        load_calibration_artifact(path)
