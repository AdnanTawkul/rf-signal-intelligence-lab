from __future__ import annotations

import numpy as np
import pytest

from rfsil.evaluation.channel_robustness import (
    SeedSweepMetrics,
)
from rfsil.evaluation.confusion_robustness import (
    ConditionConfusionSummary,
)
from rfsil.evaluation.mitigation_comparison import (
    compare_model_families,
)


def create_metrics(
    condition: str,
    accuracies: tuple[float, float],
) -> SeedSweepMetrics:
    """Create synthetic two-seed metrics."""
    values = np.asarray(
        accuracies,
        dtype=np.float64,
    )

    return SeedSweepMetrics(
        condition=condition,
        seeds=(2026, 2027),
        class_names=("A", "B"),
        snr_values_db=(-4.0, 0.0),
        overall_accuracy=values,
        class_accuracy=np.stack(
            (values, values),
            axis=1,
        ),
        accuracy_by_snr=np.stack(
            (values, values),
            axis=1,
        ),
    )


def create_confusion(
    condition: str,
    matrices: tuple[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ],
) -> ConditionConfusionSummary:
    """Create synthetic paired confusion results."""
    confusion = np.asarray(
        matrices,
        dtype=np.int64,
    )
    normalized = (
        confusion
        / confusion.sum(
            axis=2,
            keepdims=True,
        )
    )

    return ConditionConfusionSummary(
        condition=condition,
        seeds=(2026, 2027),
        class_names=("A", "B"),
        labels=np.asarray(
            [0, 0, 1, 1],
            dtype=np.int64,
        ),
        snr_db=np.asarray(
            [-4.0, 0.0, -4.0, 0.0],
            dtype=np.float32,
        ),
        overall_accuracy=np.asarray(
            [0.5, 0.5],
            dtype=np.float64,
        ),
        confusion_matrices=confusion,
        normalized_confusion_matrices=(
            normalized.astype(np.float64)
        ),
    )


def create_inputs():
    """Create one-condition comparison inputs."""
    original_metrics = {
        "clean": create_metrics(
            "clean",
            (0.8, 0.9),
        )
    }
    mitigated_metrics = {
        "clean": create_metrics(
            "clean",
            (0.9, 0.9),
        )
    }

    original_confusions = {
        "clean": create_confusion(
            "clean",
            (
                ((8, 2), (4, 6)),
                ((8, 2), (4, 6)),
            ),
        )
    }
    mitigated_confusions = {
        "clean": create_confusion(
            "clean",
            (
                ((9, 1), (2, 8)),
                ((9, 1), (2, 8)),
            ),
        )
    }

    return (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    )


def test_overall_paired_improvement() -> None:
    (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    ) = create_inputs()

    result = compare_model_families(
        original_metrics=original_metrics,
        mitigated_metrics=mitigated_metrics,
        original_confusions=(
            original_confusions
        ),
        mitigated_confusions=(
            mitigated_confusions
        ),
        condition_order=("clean",),
        target_class="B",
    )

    improvement = result["conditions"][
        "clean"
    ]["improvement"]

    assert improvement[
        "overall_mean"
    ] == pytest.approx(0.05)


def test_target_error_reduction() -> None:
    (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    ) = create_inputs()

    result = compare_model_families(
        original_metrics=original_metrics,
        mitigated_metrics=mitigated_metrics,
        original_confusions=(
            original_confusions
        ),
        mitigated_confusions=(
            mitigated_confusions
        ),
        condition_order=("clean",),
        target_class="B",
    )

    values = result["conditions"]["clean"][
        "misclassification_to_target"
    ]["A"]

    assert values[
        "original_rate"
    ] == pytest.approx(0.2)
    assert values[
        "mitigated_rate"
    ] == pytest.approx(0.1)
    assert values[
        "absolute_reduction"
    ] == pytest.approx(0.1)


def test_missing_condition_is_rejected() -> None:
    (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    ) = create_inputs()

    with pytest.raises(ValueError):
        compare_model_families(
            original_metrics=original_metrics,
            mitigated_metrics=mitigated_metrics,
            original_confusions=(
                original_confusions
            ),
            mitigated_confusions=(
                mitigated_confusions
            ),
            condition_order=(
                "clean",
                "severe",
            ),
            target_class="B",
        )


def test_unknown_target_is_rejected() -> None:
    (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    ) = create_inputs()

    with pytest.raises(ValueError):
        compare_model_families(
            original_metrics=original_metrics,
            mitigated_metrics=mitigated_metrics,
            original_confusions=(
                original_confusions
            ),
            mitigated_confusions=(
                mitigated_confusions
            ),
            condition_order=("clean",),
            target_class="C",
        )


def test_mismatched_seeds_are_rejected() -> None:
    (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    ) = create_inputs()

    mismatched = mitigated_metrics["clean"]

    mitigated_metrics["clean"] = (
        SeedSweepMetrics(
            condition="clean",
            seeds=(2026, 2030),
            class_names=mismatched.class_names,
            snr_values_db=(
                mismatched.snr_values_db
            ),
            overall_accuracy=(
                mismatched.overall_accuracy
            ),
            class_accuracy=(
                mismatched.class_accuracy
            ),
            accuracy_by_snr=(
                mismatched.accuracy_by_snr
            ),
        )
    )

    with pytest.raises(ValueError):
        compare_model_families(
            original_metrics=original_metrics,
            mitigated_metrics=mitigated_metrics,
            original_confusions=(
                original_confusions
            ),
            mitigated_confusions=(
                mitigated_confusions
            ),
            condition_order=("clean",),
            target_class="B",
        )


def test_duplicate_condition_is_rejected() -> None:
    (
        original_metrics,
        mitigated_metrics,
        original_confusions,
        mitigated_confusions,
    ) = create_inputs()

    with pytest.raises(ValueError):
        compare_model_families(
            original_metrics=original_metrics,
            mitigated_metrics=mitigated_metrics,
            original_confusions=(
                original_confusions
            ),
            mitigated_confusions=(
                mitigated_confusions
            ),
            condition_order=(
                "clean",
                "clean",
            ),
            target_class="B",
        )
