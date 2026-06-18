from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.classification import (
    PredictionResults,
)
from rfsil.evaluation.prediction_artifacts import (
    load_prediction_results,
    save_prediction_results,
)


def create_results() -> PredictionResults:
    """Create a valid prediction fixture."""
    return PredictionResults(
        labels=np.asarray(
            [0, 1, 2, 3],
            dtype=np.int64,
        ),
        predictions=np.asarray(
            [0, 2, 2, 3],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [-4.0, 0.0, 4.0, 8.0],
            dtype=np.float32,
        ),
    )


def test_round_trip_preserves_arrays(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.npz"

    save_prediction_results(
        path,
        create_results(),
    )
    loaded = load_prediction_results(path)

    np.testing.assert_array_equal(
        loaded.labels,
        create_results().labels,
    )
    np.testing.assert_array_equal(
        loaded.predictions,
        create_results().predictions,
    )
    np.testing.assert_array_equal(
        loaded.snr_db,
        create_results().snr_db,
    )


def test_save_creates_parent_directory(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "nested"
        / "seed_2026.npz"
    )

    result = save_prediction_results(
        path,
        create_results(),
    )

    assert result == path
    assert path.is_file()


def test_standard_schema_is_written(
    tmp_path: Path,
) -> None:
    path = tmp_path / "predictions.npz"

    save_prediction_results(
        path,
        create_results(),
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as content:
        assert set(content.files) == {
            "labels",
            "predictions",
            "snr_db",
        }
        assert content["labels"].dtype == np.int64
        assert (
            content["predictions"].dtype
            == np.int64
        )
        assert content["snr_db"].dtype == np.float32


def test_missing_file_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_prediction_results(
            tmp_path / "missing.npz"
        )


def test_missing_key_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.npz"

    np.savez_compressed(
        path,
        labels=np.asarray([0]),
    )

    with pytest.raises(ValueError):
        load_prediction_results(path)


def test_mismatched_shapes_are_rejected(
    tmp_path: Path,
) -> None:
    results = PredictionResults(
        labels=np.asarray([0, 1]),
        predictions=np.asarray([0]),
        snr_db=np.asarray(
            [0.0, 4.0],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        save_prediction_results(
            tmp_path / "invalid.npz",
            results,
        )


def test_empty_arrays_are_rejected(
    tmp_path: Path,
) -> None:
    results = PredictionResults(
        labels=np.asarray([], dtype=np.int64),
        predictions=np.asarray(
            [],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        save_prediction_results(
            tmp_path / "invalid.npz",
            results,
        )


def test_nonfinite_snr_is_rejected(
    tmp_path: Path,
) -> None:
    results = PredictionResults(
        labels=np.asarray([0], dtype=np.int64),
        predictions=np.asarray(
            [0],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [np.nan],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        save_prediction_results(
            tmp_path / "invalid.npz",
            results,
        )
