from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from rfsil.deployment.shift_detector import (
    IQShiftDetectorArtifact,
    load_iq_shift_detector,
    save_iq_shift_detector,
    select_shift_threshold,
)
from rfsil.evaluation.iq_channel_features import (
    compute_iq_channel_features,
)


def example_iq(
    *,
    batch_size: int = 4,
    sample_count: int = 64,
) -> np.ndarray:
    generator = np.random.default_rng(
        2026
    )

    return generator.normal(
        size=(
            batch_size,
            2,
            sample_count,
        )
    ).astype(np.float32)


def example_artifact() -> (
    IQShiftDetectorArtifact
):
    iq = example_iq()
    features = (
        compute_iq_channel_features(iq)
    )
    feature_count = len(
        features.feature_names
    )

    return IQShiftDetectorArtifact(
        format_version=1,
        artifact_name="test_detector",
        expected_sample_count=64,
        feature_names=(
            features.feature_names
        ),
        feature_mean=np.zeros(
            feature_count,
            dtype=np.float64,
        ),
        feature_scale=np.ones(
            feature_count,
            dtype=np.float64,
        ),
        coefficients=np.linspace(
            -0.5,
            0.5,
            num=feature_count,
            dtype=np.float64,
        ),
        intercept=0.1,
        l2_strength=0.1,
        threshold=0.0,
        target_tpr=0.95,
        development_auroc=0.9,
        development_average_precision=(
            0.95
        ),
        development_fpr_at_target_tpr=(
            0.5
        ),
        development_clean_mean=-0.3,
        development_clean_std=0.2,
        development_shifted_mean=0.4,
        development_shifted_std=0.6,
        autocorrelation_lags=(
            1,
            2,
            4,
            8,
        ),
        occupancy_fraction=0.9,
        epsilon=1e-12,
        provenance={
            "experiment": "test",
        },
    )


def test_selects_threshold_at_target_tpr(
) -> None:
    clean = np.asarray(
        [-3.0, -2.0, -1.0, 0.0],
        dtype=np.float64,
    )
    shifted = np.asarray(
        [-1.0, 1.0, 2.0, 3.0],
        dtype=np.float64,
    )

    selection = select_shift_threshold(
        clean,
        shifted,
        target_tpr=0.75,
    )

    assert selection.threshold == (
        pytest.approx(1.0)
    )
    assert selection.achieved_tpr == (
        pytest.approx(0.75)
    )
    assert selection.achieved_fpr == (
        pytest.approx(0.0)
    )


def test_assesses_iq_batch() -> None:
    artifact = example_artifact()
    assessment = artifact.assess_iq(
        example_iq()
    )

    assert assessment.batch_size == 4
    assert assessment.scores.shape == (
        4,
    )
    assert assessment.shift_like.shape == (
        4,
    )
    assert (
        assessment.feature_values.shape
        == (
            4,
            artifact.feature_count,
        )
    )


def test_feature_score_matches_model() -> None:
    artifact = example_artifact()
    features = (
        compute_iq_channel_features(
            example_iq()
        )
    )

    expected = (
        artifact.detector
        .decision_function(
            features.values
        )
    )
    actual = artifact.score_features(
        features.values,
        feature_names=(
            features.feature_names
        ),
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_round_trip_json(
    tmp_path: Path,
) -> None:
    artifact = example_artifact()
    path = tmp_path / "detector.json"

    save_iq_shift_detector(
        artifact,
        path,
    )
    loaded = load_iq_shift_detector(
        path
    )

    assert (
        loaded.artifact_name
        == artifact.artifact_name
    )
    assert (
        loaded.feature_names
        == artifact.feature_names
    )
    np.testing.assert_allclose(
        loaded.coefficients,
        artifact.coefficients,
    )


def test_assessment_to_dict() -> None:
    artifact = example_artifact()
    payload = artifact.assess_iq(
        example_iq(batch_size=1)
    ).to_dict()

    assert payload["batch_size"] == 1
    assert len(payload["records"]) == 1
    assert (
        len(
            payload["records"][0][
                "features"
            ]
        )
        == artifact.feature_count
    )


def test_rejects_wrong_sample_count() -> None:
    artifact = example_artifact()

    with pytest.raises(
        ValueError,
        match="sample count",
    ):
        artifact.assess_iq(
            example_iq(
                sample_count=32
            )
        )


def test_rejects_feature_order_mismatch(
) -> None:
    artifact = example_artifact()
    values = np.ones(
        (
            2,
            artifact.feature_count,
        ),
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="Feature names",
    ):
        artifact.score_features(
            values,
            feature_names=tuple(
                reversed(
                    artifact.feature_names
                )
            ),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("format_version", 2),
        ("expected_sample_count", 0),
        ("feature_scale", [0.0] * 21),
        ("target_tpr", 1.1),
        ("occupancy_fraction", 1.0),
        ("epsilon", 0.0),
    ),
)
def test_rejects_invalid_artifact(
    field: str,
    value: object,
) -> None:
    payload = (
        example_artifact().to_dict()
    )

    field_locations = {
        "format_version": (
            payload,
            "format_version",
        ),
        "expected_sample_count": (
            payload["input"],
            "expected_sample_count",
        ),
        "feature_scale": (
            payload["model"],
            "feature_scale",
        ),
        "target_tpr": (
            payload["decision"],
            "target_tpr",
        ),
        "occupancy_fraction": (
            payload["feature_extraction"],
            "occupancy_fraction",
        ),
        "epsilon": (
            payload["feature_extraction"],
            "epsilon",
        ),
    }
    section, key = field_locations[field]
    section[key] = value

    with pytest.raises(ValueError):
        IQShiftDetectorArtifact.from_dict(
            deepcopy(payload)
        )


def test_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError
    ):
        load_iq_shift_detector(
            tmp_path / "missing.json"
        )
