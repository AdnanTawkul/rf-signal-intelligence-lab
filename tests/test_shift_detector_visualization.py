from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from rfsil.evaluation.shift_detector_comparison import (
    SHIFT_DETECTOR_SYSTEMS,
)
from rfsil.evaluation.shift_detector_visualization import (
    FIGURE_FILENAMES,
    FUSION_BASELINE_SYSTEMS,
    create_detector_comparison_figures,
    load_detector_comparison_visualization_data,
    parse_detector_comparison_visualization_data,
)


def create_payload() -> dict:
    conditions = [
        "mild",
        "moderate",
        "severe",
    ]
    groups = []

    for condition_index, condition in (
        enumerate(conditions)
    ):
        for system_index, system in (
            enumerate(
                SHIFT_DETECTOR_SYSTEMS
            )
        ):
            groups.append(
                {
                    "condition": condition,
                    "system_name": system,
                    "run_count": 75,
                    "metrics": {
                        "auroc": {
                            "mean": (
                                0.6
                                + 0.05
                                * condition_index
                                + 0.02
                                * system_index
                            ),
                            "std": 0.01,
                            "minimum": 0.5,
                            "maximum": 1.0,
                        },
                        "average_precision": {
                            "mean": 0.7,
                            "std": 0.01,
                            "minimum": 0.5,
                            "maximum": 1.0,
                        },
                        "fpr_at_target_tpr": {
                            "mean": (
                                0.8
                                - 0.05
                                * condition_index
                                - 0.02
                                * system_index
                            ),
                            "std": 0.02,
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "threshold_counts": {},
                }
            )

    fusion_changes = []

    for condition_index, condition in (
        enumerate(conditions)
    ):
        for baseline_index, baseline in (
            enumerate(
                FUSION_BASELINE_SYSTEMS
            )
        ):
            fusion_changes.append(
                {
                    "condition": condition,
                    "baseline_system": (
                        baseline
                    ),
                    "fusion_system": (
                        "iq_energy_fusion"
                    ),
                    "run_count": 75,
                    "fusion_minus_baseline": {
                        "auroc": {
                            "mean": (
                                0.01
                                * (
                                    condition_index
                                    + baseline_index
                                )
                            ),
                            "std": 0.005,
                            "minimum": -0.1,
                            "maximum": 0.1,
                        },
                        "average_precision": {
                            "mean": 0.0,
                            "std": 0.005,
                            "minimum": -0.1,
                            "maximum": 0.1,
                        },
                        "fpr_at_target_tpr": {
                            "mean": (
                                -0.02
                                * (
                                    condition_index
                                    + baseline_index
                                )
                            ),
                            "std": 0.01,
                            "minimum": -0.5,
                            "maximum": 0.5,
                        },
                    },
                    "improvement_counts": {},
                }
            )

    return {
        "format_version": 1,
        "aggregate": {
            "conditions": conditions,
            "systems": list(
                SHIFT_DETECTOR_SYSTEMS
            ),
            "system_condition_groups": (
                groups
            ),
            "fusion_paired_changes": (
                fusion_changes
            ),
        },
    }


def test_parses_expected_shapes() -> None:
    data = (
        parse_detector_comparison_visualization_data(
            create_payload()
        )
    )

    assert data.auroc_mean.shape == (
        3,
        4,
    )
    assert data.fpr95_mean.shape == (
        3,
        4,
    )
    assert (
        data.fusion_auroc_change_mean.shape
        == (3, 3)
    )


def test_preserves_predefined_order() -> None:
    data = (
        parse_detector_comparison_visualization_data(
            create_payload()
        )
    )

    assert data.conditions == (
        "mild",
        "moderate",
        "severe",
    )
    assert data.systems == tuple(
        SHIFT_DETECTOR_SYSTEMS
    )
    assert data.fusion_baselines == (
        FUSION_BASELINE_SYSTEMS
    )


def test_loads_json_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(create_payload()),
        encoding="utf-8",
    )

    data = (
        load_detector_comparison_visualization_data(
            path
        )
    )

    assert data.auroc_mean[
        0,
        0,
    ] == pytest.approx(0.6)


def test_rejects_missing_detector_group() -> None:
    payload = create_payload()
    payload["aggregate"][
        "system_condition_groups"
    ].pop()

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        parse_detector_comparison_visualization_data(
            payload
        )


def test_rejects_duplicate_detector_group() -> None:
    payload = create_payload()
    groups = payload["aggregate"][
        "system_condition_groups"
    ]
    groups.append(
        deepcopy(groups[0])
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        parse_detector_comparison_visualization_data(
            payload
        )


def test_rejects_missing_fusion_change() -> None:
    payload = create_payload()
    payload["aggregate"][
        "fusion_paired_changes"
    ].pop()

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        parse_detector_comparison_visualization_data(
            payload
        )


def test_rejects_non_finite_metric() -> None:
    payload = create_payload()
    payload["aggregate"][
        "system_condition_groups"
    ][0]["metrics"]["auroc"][
        "mean"
    ] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        parse_detector_comparison_visualization_data(
            payload
        )


def test_creates_non_empty_figures(
    tmp_path: Path,
) -> None:
    data = (
        parse_detector_comparison_visualization_data(
            create_payload()
        )
    )

    paths = create_detector_comparison_figures(
        data,
        tmp_path,
        dpi=100,
    )

    assert set(paths) == set(
        FIGURE_FILENAMES
    )

    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0


@pytest.mark.parametrize(
    "dpi",
    (
        0,
        -1,
        1.5,
        True,
    ),
)
def test_rejects_invalid_dpi(
    dpi: object,
    tmp_path: Path,
) -> None:
    data = (
        parse_detector_comparison_visualization_data(
            create_payload()
        )
    )

    with pytest.raises(
        ValueError,
        match="dpi",
    ):
        create_detector_comparison_figures(
            data,
            tmp_path,
            dpi=dpi,
        )
