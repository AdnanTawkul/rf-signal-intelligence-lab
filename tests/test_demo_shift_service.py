from __future__ import annotations

import numpy as np
import pytest

from rfsil.demo.shift_service import (
    assess_single_window_shift,
    build_shift_assessment_document,
)
from rfsil.deployment.shift_detector import (
    IQShiftDetectorArtifact,
)
from rfsil.evaluation.iq_channel_features import (
    compute_iq_channel_features,
)


def example_iq(
    *,
    batch_size: int = 1,
) -> np.ndarray:
    generator = np.random.default_rng(
        2026
    )

    return generator.normal(
        size=(
            batch_size,
            2,
            64,
        )
    ).astype(np.float32)


def example_artifact() -> (
    IQShiftDetectorArtifact
):
    features = compute_iq_channel_features(
        example_iq()
    )
    feature_count = len(
        features.feature_names
    )

    return IQShiftDetectorArtifact(
        format_version=1,
        artifact_name="demo_detector",
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
            -0.25,
            0.25,
            num=feature_count,
            dtype=np.float64,
        ),
        intercept=0.05,
        l2_strength=0.1,
        threshold=0.0,
        target_tpr=0.95,
        development_auroc=0.9,
        development_average_precision=(
            0.94
        ),
        development_fpr_at_target_tpr=(
            0.55
        ),
        development_clean_mean=-0.4,
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
            "test": True,
        },
    )


def test_assesses_single_window() -> None:
    assessment = (
        assess_single_window_shift(
            example_artifact(),
            example_iq(),
            top_feature_count=5,
        )
    )

    assert assessment.status in {
        "shift_like",
        "within_clean_reference",
    }
    assert len(
        assessment.feature_contributions
    ) == 5
    assert assessment.margin == (
        pytest.approx(
            assessment.score
            - assessment.threshold
        )
    )


def test_contributions_reconstruct_score(
) -> None:
    artifact = example_artifact()
    iq = example_iq()
    assessment = (
        assess_single_window_shift(
            artifact,
            iq,
            top_feature_count=(
                artifact.feature_count
            ),
        )
    )

    reconstructed = (
        artifact.intercept
        + sum(
            item.contribution
            for item
            in assessment.feature_contributions
        )
    )

    assert reconstructed == pytest.approx(
        assessment.score,
        abs=1e-10,
    )


def test_builds_public_document() -> None:
    assessment = (
        assess_single_window_shift(
            example_artifact(),
            example_iq(),
        )
    )

    document = (
        build_shift_assessment_document(
            assessment,
            source_name=(
                "../private/validation.npz"
            ),
            sample_position=3,
            sample_index=101,
        )
    )

    assert document["input"][
        "source_name"
    ] == "validation.npz"
    assert document["input"][
        "sample_position"
    ] == 3
    assert document["input"][
        "sample_index"
    ] == 101
    assert (
        "private"
        not in str(document)
    )


def test_rejects_multiple_windows() -> None:
    with pytest.raises(
        ValueError,
        match="exactly one",
    ):
        assess_single_window_shift(
            example_artifact(),
            example_iq(batch_size=2),
        )


@pytest.mark.parametrize(
    "value",
    (
        0,
        -1,
        1.5,
        True,
    ),
)
def test_rejects_invalid_top_feature_count(
    value: object,
) -> None:
    with pytest.raises(ValueError):
        assess_single_window_shift(
            example_artifact(),
            example_iq(),
            top_feature_count=value,
        )


def test_document_to_json_compatible() -> None:
    assessment = (
        assess_single_window_shift(
            example_artifact(),
            example_iq(),
        )
    )

    document = (
        build_shift_assessment_document(
            assessment,
            source_name="signal.npy",
            sample_position=0,
        )
    )

    assert document[
        "analysis_type"
    ] == "iq_channel_shift_assessment"
    assert isinstance(
        document["assessment"][
            "shift_like"
        ],
        bool,
    )
