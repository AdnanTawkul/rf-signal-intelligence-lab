from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfsil.deployment.iq_io import (
    load_iq_file,
)


def test_loads_complex_npy_vector(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signal.npy"
    samples = np.asarray(
        [1.0 + 2.0j, 3.0 + 4.0j],
        dtype=np.complex64,
    )
    np.save(path, samples)

    loaded = load_iq_file(path)

    assert loaded.iq.shape == (1, 2, 2)
    assert loaded.iq.dtype == np.float32
    np.testing.assert_array_equal(
        loaded.iq[0, 0],
        [1.0, 3.0],
    )
    np.testing.assert_array_equal(
        loaded.iq[0, 1],
        [2.0, 4.0],
    )


def test_loads_complex_npy_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch.npy"
    samples = np.ones(
        (3, 16),
        dtype=np.complex64,
    )
    np.save(path, samples)

    loaded = load_iq_file(path)

    assert loaded.iq.shape == (3, 2, 16)
    np.testing.assert_array_equal(
        loaded.sample_indices,
        np.asarray(
            [0, 1, 2],
            dtype=np.int64,
        ),
    )


def test_loads_real_single_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signal.npy"
    samples = np.zeros(
        (2, 32),
        dtype=np.float64,
    )
    np.save(path, samples)

    loaded = load_iq_file(
        path,
        expected_sample_count=32,
    )

    assert loaded.iq.shape == (1, 2, 32)
    assert loaded.sample_count == 32
    assert loaded.channel_count == 2


def test_loads_real_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch.npy"
    samples = np.zeros(
        (4, 2, 64),
        dtype=np.float32,
    )
    np.save(path, samples)

    loaded = load_iq_file(path)

    assert loaded.batch_size == 4
    assert loaded.iq.shape == (4, 2, 64)


def test_loads_npz_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.npz"

    np.savez_compressed(
        path,
        iq=np.zeros(
            (3, 2, 16),
            dtype=np.float32,
        ),
        labels=np.asarray(
            [0, 1, 2],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [-4.0, 0.0, 4.0],
            dtype=np.float32,
        ),
    )

    loaded = load_iq_file(path)

    assert loaded.array_key == "iq"

    np.testing.assert_array_equal(
        loaded.labels,
        [0, 1, 2],
    )
    np.testing.assert_array_equal(
        loaded.snr_db,
        [-4.0, 0.0, 4.0],
    )


def test_selects_one_npz_sample(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.npz"

    iq = np.arange(
        3 * 2 * 8,
        dtype=np.float32,
    ).reshape(3, 2, 8)

    np.savez_compressed(
        path,
        iq=iq,
        labels=np.asarray(
            [1, 2, 3],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [-2.0, 0.0, 2.0],
            dtype=np.float32,
        ),
    )

    loaded = load_iq_file(
        path,
        sample_index=1,
    )

    assert loaded.iq.shape == (1, 2, 8)

    np.testing.assert_array_equal(
        loaded.iq[0],
        iq[1],
    )
    np.testing.assert_array_equal(
        loaded.sample_indices,
        [1],
    )
    np.testing.assert_array_equal(
        loaded.labels,
        [2],
    )
    np.testing.assert_array_equal(
        loaded.snr_db,
        [0.0],
    )


def test_selects_one_npy_sample(
    tmp_path: Path,
) -> None:
    path = tmp_path / "batch.npy"
    iq = np.zeros(
        (5, 2, 8),
        dtype=np.float32,
    )
    np.save(path, iq)

    loaded = load_iq_file(
        path,
        sample_index=4,
    )

    assert loaded.iq.shape == (1, 2, 8)
    np.testing.assert_array_equal(
        loaded.sample_indices,
        [4],
    )
    assert loaded.labels is None
    assert loaded.snr_db is None


def test_supports_custom_npz_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "custom.npz"

    np.savez_compressed(
        path,
        signals=np.zeros(
            (2, 2, 8),
            dtype=np.float32,
        ),
    )

    loaded = load_iq_file(
        path,
        array_key="signals",
    )

    assert loaded.array_key == "signals"
    assert loaded.batch_size == 2


def test_missing_file_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_iq_file(
            tmp_path / "missing.npy"
        )


def test_missing_npz_key_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.npz"

    np.savez_compressed(
        path,
        signals=np.zeros(
            (2, 2, 8),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        KeyError,
        match="Available keys",
    ):
        load_iq_file(path)


def test_unsupported_extension_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signal.txt"
    path.write_text(
        "not IQ",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=".npy or .npz",
    ):
        load_iq_file(path)


def test_invalid_real_layout_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signal.npy"

    np.save(
        path,
        np.zeros(
            (64, 2),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Real IQ input",
    ):
        load_iq_file(path)


def test_nonfinite_iq_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signal.npy"
    iq = np.zeros(
        (2, 16),
        dtype=np.float32,
    )
    iq[0, 0] = np.nan
    np.save(path, iq)

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        load_iq_file(path)


def test_wrong_sample_count_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "signal.npy"

    np.save(
        path,
        np.zeros(
            (2, 15),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        ValueError,
        match="expected value of 16",
    ):
        load_iq_file(
            path,
            expected_sample_count=16,
        )


@pytest.mark.parametrize(
    "sample_index",
    (
        -1,
        2,
        True,
        1.5,
    ),
)
def test_invalid_sample_index_is_rejected(
    tmp_path: Path,
    sample_index: object,
) -> None:
    path = tmp_path / "batch.npy"

    np.save(
        path,
        np.zeros(
            (2, 2, 16),
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        (ValueError, IndexError),
    ):
        load_iq_file(
            path,
            sample_index=sample_index,
        )


def test_mismatched_metadata_length_is_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.npz"

    np.savez_compressed(
        path,
        iq=np.zeros(
            (3, 2, 16),
            dtype=np.float32,
        ),
        labels=np.asarray(
            [0, 1],
            dtype=np.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="batch size",
    ):
        load_iq_file(path)
