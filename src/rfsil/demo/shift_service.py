from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from rfsil.deployment.shift_detector import (
    IQShiftDetectorArtifact,
)


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """One standardized linear-model contribution."""

    feature_name: str
    raw_value: float
    standardized_value: float
    coefficient: float
    contribution: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible contribution data."""
        return {
            "feature_name": self.feature_name,
            "raw_value": self.raw_value,
            "standardized_value": (
                self.standardized_value
            ),
            "coefficient": self.coefficient,
            "contribution": self.contribution,
        }


@dataclass(frozen=True, slots=True)
class DemoShiftAssessment:
    """GUI-ready assessment for one IQ window."""

    artifact_name: str
    score: float
    threshold: float
    margin: float
    shift_like: bool
    status: str
    target_tpr: float
    development_auroc: float
    development_fpr_at_target_tpr: float
    development_clean_mean: float
    development_clean_std: float
    development_shifted_mean: float
    development_shifted_std: float
    feature_contributions: tuple[
        FeatureContribution,
        ...,
    ]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible assessment data."""
        return {
            "artifact_name": self.artifact_name,
            "score": self.score,
            "threshold": self.threshold,
            "margin": self.margin,
            "shift_like": self.shift_like,
            "status": self.status,
            "decision_rule": (
                "score >= threshold"
            ),
            "development_reference": {
                "target_tpr": self.target_tpr,
                "auroc": self.development_auroc,
                "fpr_at_target_tpr": (
                    self
                    .development_fpr_at_target_tpr
                ),
                "clean_score_mean": (
                    self.development_clean_mean
                ),
                "clean_score_std": (
                    self.development_clean_std
                ),
                "shifted_score_mean": (
                    self.development_shifted_mean
                ),
                "shifted_score_std": (
                    self.development_shifted_std
                ),
            },
            "feature_contributions": [
                contribution.to_dict()
                for contribution
                in self.feature_contributions
            ],
            "interpretation": (
                "The detector score is not a "
                "probability or channel-severity "
                "estimate. It is a binary "
                "development-selected shift score."
            ),
        }

    def to_json(self) -> str:
        """Serialize the assessment."""
        return json.dumps(
            self.to_dict(),
            indent=2,
        ) + "\n"


def _positive_integer(
    value: object,
    *,
    name: str,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return int(value)


def assess_single_window_shift(
    artifact: IQShiftDetectorArtifact,
    iq: object,
    *,
    top_feature_count: object = 10,
) -> DemoShiftAssessment:
    """Assess one window and explain its score."""
    if not isinstance(
        artifact,
        IQShiftDetectorArtifact,
    ):
        raise TypeError(
            "artifact must be an "
            "IQShiftDetectorArtifact."
        )

    validated_top_count = _positive_integer(
        top_feature_count,
        name="top_feature_count",
    )

    raw = np.asarray(iq)

    if raw.ndim == 2:
        batch_size = 1
    elif raw.ndim == 3:
        batch_size = int(raw.shape[0])
    else:
        raise ValueError(
            "iq must have shape [2, samples] "
            "or [1, 2, samples]."
        )

    if batch_size != 1:
        raise ValueError(
            "The GUI shift assessment requires "
            "exactly one IQ window."
        )

    assessment = artifact.assess_iq(iq)

    if assessment.batch_size != 1:
        raise RuntimeError(
            "Detector returned an unexpected "
            "assessment batch size."
        )

    raw_features = assessment.feature_values[0]
    standardized = (
        raw_features
        - artifact.feature_mean
    ) / artifact.feature_scale
    contribution_values = (
        standardized
        * artifact.coefficients
    )

    reconstructed_score = float(
        np.sum(contribution_values)
        + artifact.intercept
    )
    score = float(
        assessment.scores[0]
    )

    if not np.isclose(
        score,
        reconstructed_score,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError(
            "Feature contributions do not "
            "reconstruct the detector score."
        )

    contributions = [
        FeatureContribution(
            feature_name=feature_name,
            raw_value=float(
                raw_features[index]
            ),
            standardized_value=float(
                standardized[index]
            ),
            coefficient=float(
                artifact.coefficients[index]
            ),
            contribution=float(
                contribution_values[index]
            ),
        )
        for index, feature_name
        in enumerate(
            artifact.feature_names
        )
    ]

    contributions.sort(
        key=lambda item: abs(
            item.contribution
        ),
        reverse=True,
    )

    top_contributions = tuple(
        contributions[
            :min(
                validated_top_count,
                len(contributions),
            )
        ]
    )
    shift_like = bool(
        assessment.shift_like[0]
    )
    margin = (
        score - artifact.threshold
    )

    return DemoShiftAssessment(
        artifact_name=artifact.artifact_name,
        score=score,
        threshold=artifact.threshold,
        margin=margin,
        shift_like=shift_like,
        status=(
            "shift_like"
            if shift_like
            else "within_clean_reference"
        ),
        target_tpr=artifact.target_tpr,
        development_auroc=(
            artifact.development_auroc
        ),
        development_fpr_at_target_tpr=(
            artifact
            .development_fpr_at_target_tpr
        ),
        development_clean_mean=(
            artifact.development_clean_mean
        ),
        development_clean_std=(
            artifact.development_clean_std
        ),
        development_shifted_mean=(
            artifact.development_shifted_mean
        ),
        development_shifted_std=(
            artifact.development_shifted_std
        ),
        feature_contributions=(
            top_contributions
        ),
    )


def build_shift_assessment_document(
    assessment: DemoShiftAssessment,
    *,
    source_name: str,
    sample_position: object,
    sample_index: object | None = None,
) -> dict[str, object]:
    """Create a public, machine-safe export document."""
    if not isinstance(
        assessment,
        DemoShiftAssessment,
    ):
        raise TypeError(
            "assessment must be a "
            "DemoShiftAssessment."
        )

    safe_source_name = Path(
        str(source_name)
    ).name

    if not safe_source_name:
        raise ValueError(
            "source_name must not be empty."
        )

    if (
        isinstance(
            sample_position,
            (bool, np.bool_),
        )
        or not isinstance(
            sample_position,
            Integral,
        )
        or int(sample_position) < 0
    ):
        raise ValueError(
            "sample_position must be a "
            "non-negative integer."
        )

    normalized_sample_index = None

    if sample_index is not None:
        if (
            isinstance(
                sample_index,
                (bool, np.bool_),
            )
            or not isinstance(
                sample_index,
                Integral,
            )
            or int(sample_index) < 0
        ):
            raise ValueError(
                "sample_index must be a "
                "non-negative integer."
            )

        normalized_sample_index = int(
            sample_index
        )

    return {
        "format_version": 1,
        "analysis_type": (
            "iq_channel_shift_assessment"
        ),
        "input": {
            "source_name": safe_source_name,
            "sample_position": int(
                sample_position
            ),
            "sample_index": (
                normalized_sample_index
            ),
        },
        "assessment": assessment.to_dict(),
    }


__all__ = [
    "DemoShiftAssessment",
    "FeatureContribution",
    "assess_single_window_shift",
    "build_shift_assessment_document",
]
