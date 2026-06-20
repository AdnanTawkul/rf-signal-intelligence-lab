from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rfsil.evaluation.iq_feature_artifacts import (
    IQ_FEATURE_ARTIFACT_FORMAT_VERSION,
    IQChannelFeatureArtifact,
    load_iq_channel_feature_artifact,
    save_iq_channel_feature_artifact,
)


def create_artifact(
    *,
    example_count: int = 4,
) -> IQChannelFeatureArtifact:
    return IQChannelFeatureArtifact(
        values=np.arange(
            example_count * 3,
            dtype=np.float64,
        ).reshape(example_count, 3),
        feature_names=(
            "feature_a",
            "feature_b",
            "feature_c",
        ),
        labels=np.arange(
            example_count,
            dtype=np.int64,
        ) % 4,
        snr_db=np.linspace(
            -4.0,
            20.0,
            example_count,
        ),
        frequency_offset_hz=np.linspace(
            -100.0,
            100.0,
            example_count,
        ),
        phase_offset_rad=np.linspace(
            -1.0,
            1.0,
            example_count,
        ),
        amplitude_scale=np.ones(
            example_count,
        ),
        time_shift_samples=np.arange(
            example_count,
            dtype=np.int64,
        )
        - 2,
        rayleigh_fading=np.zeros(
            example_count,
            dtype=np.bool_,
        ),
        example_seed=np.arange(
            100,
            100 + example_count,
            dtype=np.uint32,
        ),
        condition="clean",
        source_dataset=(
            "data/example/test.npz"
        ),
    )


def test_artifact_properties() -> None:
    artifact = create_artifact()

    assert artifact.example_count == 4
    assert artifact.feature_count == 3
    assert (
        artifact.format_version
        == IQ_FEATURE_ARTIFACT_FORMAT_VERSION
    )


def test_artifact_summary() -> None:
    summary = create_artifact().summary()

    assert summary["condition"] == "clean"
    assert summary["example_count"] == 4
    assert summary["feature_count"] == 3
    assert summary["feature_names"] == [
        "feature_a",
        "feature_b",
        "feature_c",
    ]


def test_round_trip(
    tmp_path: Path,
) -> None:
    original = create_artifact()
    path = (
        tmp_path
        / "nested"
        / "features.npz"
    )

    saved_path = (
        save_iq_channel_feature_artifact(
            original,
            path,
        )
    )
    loaded = (
        load_iq_channel_feature_artifact(
            saved_path
        )
    )

    assert saved_path == path
    assert path.is_file()
    assert (
        loaded.feature_names
        == original.feature_names
    )
    assert (
        loaded.condition
        == original.condition
    )
    assert (
        loaded.source_dataset
        == original.source_dataset
    )
    np.testing.assert_array_equal(
        loaded.values,
        original.values,
    )
    np.testing.assert_array_equal(
        loaded.labels,
        original.labels,
    )
    np.testing.assert_array_equal(
        loaded.example_seed,
        original.example_seed,
    )


def test_saved_archive_does_not_require_pickle(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "features.npz"
    )
    save_iq_channel_feature_artifact(
        create_artifact(),
        path,
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as archive:
        assert "values" in archive.files
        assert (
            archive["feature_names"].dtype.kind
            == "U"
        )


@pytest.mark.parametrize(
    "values",
    (
        np.ones(4),
        np.ones((2, 2, 2)),
        np.empty((0, 3)),
        np.empty((4, 0)),
        np.full(
            (4, 3),
            float("nan"),
        ),
    ),
)
def test_rejects_invalid_values(
    values: object,
) -> None:
    base = create_artifact()

    with pytest.raises(ValueError):
        IQChannelFeatureArtifact(
            values=values,
            feature_names=base.feature_names,
            labels=base.labels,
            snr_db=base.snr_db,
            frequency_offset_hz=(
                base.frequency_offset_hz
            ),
            phase_offset_rad=(
                base.phase_offset_rad
            ),
            amplitude_scale=(
                base.amplitude_scale
            ),
            time_shift_samples=(
                base.time_shift_samples
            ),
            rayleigh_fading=(
                base.rayleigh_fading
            ),
            example_seed=base.example_seed,
            condition=base.condition,
            source_dataset=(
                base.source_dataset
            ),
        )


@pytest.mark.parametrize(
    "feature_names",
    (
        ("a", "b"),
        ("a", "a", "c"),
        ("a", "", "c"),
    ),
)
def test_rejects_invalid_feature_names(
    feature_names: tuple[str, ...],
) -> None:
    base = create_artifact()

    with pytest.raises(ValueError):
        IQChannelFeatureArtifact(
            values=base.values,
            feature_names=feature_names,
            labels=base.labels,
            snr_db=base.snr_db,
            frequency_offset_hz=(
                base.frequency_offset_hz
            ),
            phase_offset_rad=(
                base.phase_offset_rad
            ),
            amplitude_scale=(
                base.amplitude_scale
            ),
            time_shift_samples=(
                base.time_shift_samples
            ),
            rayleigh_fading=(
                base.rayleigh_fading
            ),
            example_seed=base.example_seed,
            condition=base.condition,
            source_dataset=(
                base.source_dataset
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "labels",
        "snr_db",
        "frequency_offset_hz",
        "phase_offset_rad",
        "amplitude_scale",
        "time_shift_samples",
        "rayleigh_fading",
        "example_seed",
    ),
)
def test_rejects_metadata_length_mismatch(
    field: str,
) -> None:
    base = create_artifact()
    arguments = {
        "values": base.values,
        "feature_names": (
            base.feature_names
        ),
        "labels": base.labels,
        "snr_db": base.snr_db,
        "frequency_offset_hz": (
            base.frequency_offset_hz
        ),
        "phase_offset_rad": (
            base.phase_offset_rad
        ),
        "amplitude_scale": (
            base.amplitude_scale
        ),
        "time_shift_samples": (
            base.time_shift_samples
        ),
        "rayleigh_fading": (
            base.rayleigh_fading
        ),
        "example_seed": base.example_seed,
        "condition": base.condition,
        "source_dataset": (
            base.source_dataset
        ),
    }
    arguments[field] = np.asarray(
        arguments[field]
    )[:-1]

    with pytest.raises(ValueError):
        IQChannelFeatureArtifact(
            **arguments,
        )


def test_rejects_negative_labels() -> None:
    base = create_artifact()
    labels = base.labels.copy()
    labels[0] = -1

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        IQChannelFeatureArtifact(
            values=base.values,
            feature_names=base.feature_names,
            labels=labels,
            snr_db=base.snr_db,
            frequency_offset_hz=(
                base.frequency_offset_hz
            ),
            phase_offset_rad=(
                base.phase_offset_rad
            ),
            amplitude_scale=(
                base.amplitude_scale
            ),
            time_shift_samples=(
                base.time_shift_samples
            ),
            rayleigh_fading=(
                base.rayleigh_fading
            ),
            example_seed=base.example_seed,
            condition=base.condition,
            source_dataset=(
                base.source_dataset
            ),
        )


def test_rejects_non_boolean_fading() -> None:
    base = create_artifact()

    with pytest.raises(
        ValueError,
        match="boolean",
    ):
        IQChannelFeatureArtifact(
            values=base.values,
            feature_names=base.feature_names,
            labels=base.labels,
            snr_db=base.snr_db,
            frequency_offset_hz=(
                base.frequency_offset_hz
            ),
            phase_offset_rad=(
                base.phase_offset_rad
            ),
            amplitude_scale=(
                base.amplitude_scale
            ),
            time_shift_samples=(
                base.time_shift_samples
            ),
            rayleigh_fading=np.zeros(
                base.example_count,
                dtype=np.int64,
            ),
            example_seed=base.example_seed,
            condition=base.condition,
            source_dataset=(
                base.source_dataset
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("condition", ""),
        ("condition", 5),
        ("source_dataset", ""),
        ("source_dataset", None),
    ),
)
def test_rejects_invalid_strings(
    field: str,
    value: object,
) -> None:
    base = create_artifact()
    arguments = {
        "values": base.values,
        "feature_names": (
            base.feature_names
        ),
        "labels": base.labels,
        "snr_db": base.snr_db,
        "frequency_offset_hz": (
            base.frequency_offset_hz
        ),
        "phase_offset_rad": (
            base.phase_offset_rad
        ),
        "amplitude_scale": (
            base.amplitude_scale
        ),
        "time_shift_samples": (
            base.time_shift_samples
        ),
        "rayleigh_fading": (
            base.rayleigh_fading
        ),
        "example_seed": base.example_seed,
        "condition": base.condition,
        "source_dataset": (
            base.source_dataset
        ),
    }
    arguments[field] = value

    with pytest.raises(ValueError):
        IQChannelFeatureArtifact(
            **arguments,
        )


def test_load_rejects_missing_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.npz"

    np.savez_compressed(
        path,
        values=np.ones((2, 2)),
    )

    with pytest.raises(
        ValueError,
        match="missing keys",
    ):
        load_iq_channel_feature_artifact(
            path
        )
