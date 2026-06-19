from __future__ import annotations

import pytest
import torch

from rfsil.evaluation.equalizer_correction_analysis import (
    CorrectionStatisticsAccumulator,
    aggregate_seed_correction_statistics,
)


def test_known_correction_statistics() -> None:
    inputs = torch.ones(2, 2, 4)
    corrections = torch.full_like(
        inputs,
        0.5,
    )

    accumulator = (
        CorrectionStatisticsAccumulator()
    )
    accumulator.update(
        inputs,
        corrections,
    )

    result = accumulator.finalize()

    assert result["example_count"] == 2
    assert result["value_count"] == 16
    assert result[
        "mean_absolute_correction"
    ] == pytest.approx(0.5)
    assert result[
        "correction_rms"
    ] == pytest.approx(0.5)
    assert result[
        "maximum_absolute_correction"
    ] == pytest.approx(0.5)
    assert result["input_rms"] == pytest.approx(
        1.0
    )
    assert result[
        "corrected_rms"
    ] == pytest.approx(1.5)
    assert result[
        "relative_correction_rms"
    ] == pytest.approx(0.5)


def test_multiple_batches_are_combined() -> None:
    accumulator = (
        CorrectionStatisticsAccumulator()
    )

    accumulator.update(
        torch.ones(1, 2, 4),
        torch.zeros(1, 2, 4),
    )
    accumulator.update(
        torch.ones(1, 2, 4),
        torch.ones(1, 2, 4),
    )

    result = accumulator.finalize()

    assert result["example_count"] == 2
    assert result[
        "mean_absolute_correction"
    ] == pytest.approx(0.5)
    assert result[
        "correction_rms"
    ] == pytest.approx(2.0**-0.5)


def test_shape_mismatch_is_rejected() -> None:
    accumulator = (
        CorrectionStatisticsAccumulator()
    )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        accumulator.update(
            torch.randn(2, 2, 32),
            torch.randn(2, 2, 31),
        )


def test_empty_accumulator_is_rejected() -> None:
    accumulator = (
        CorrectionStatisticsAccumulator()
    )

    with pytest.raises(
        ValueError,
        match="No correction batches",
    ):
        accumulator.finalize()


def test_seed_statistics_are_aggregated() -> None:
    per_seed = {
        2026: {
            "example_count": 10,
            "value_count": 40,
            "mean_absolute_correction": 0.2,
            "correction_rms": 0.3,
            "maximum_absolute_correction": 1.0,
            "input_rms": 1.0,
            "corrected_rms": 1.1,
            "relative_correction_rms": 0.3,
        },
        2027: {
            "example_count": 10,
            "value_count": 40,
            "mean_absolute_correction": 0.4,
            "correction_rms": 0.5,
            "maximum_absolute_correction": 1.4,
            "input_rms": 1.0,
            "corrected_rms": 1.2,
            "relative_correction_rms": 0.5,
        },
    }

    result = (
        aggregate_seed_correction_statistics(
            per_seed
        )
    )

    assert result["aggregate"][
        "mean_absolute_correction"
    ]["mean"] == pytest.approx(0.3)

    assert result["aggregate"][
        "correction_rms"
    ]["mean"] == pytest.approx(0.4)

    assert result["aggregate"][
        "relative_correction_rms"
    ]["standard_deviation"] == pytest.approx(
        0.1
    )


def test_empty_seed_statistics_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        aggregate_seed_correction_statistics({})
