from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from rfsil.evaluation.shift_detector_comparison import (
    SHIFT_DETECTOR_SYSTEMS,
)

Float64Array = NDArray[np.float64]

FUSION_BASELINE_SYSTEMS = (
    "lag8",
    "all_iq_linear",
    "output_energy",
)

SYSTEM_DISPLAY_NAMES = {
    "lag8": "Lag-8 autocorrelation",
    "all_iq_linear": "All-IQ linear",
    "output_energy": "Output energy",
    "iq_energy_fusion": "IQ + energy fusion",
}

FIGURE_FILENAMES = {
    "auroc": (
        "channel_shift_detector_auroc_v1.png"
    ),
    "fpr95": (
        "channel_shift_detector_fpr95_v1.png"
    ),
    "fusion_auroc_change": (
        "channel_shift_fusion_auroc_change_v1.png"
    ),
    "fusion_fpr95_change": (
        "channel_shift_fusion_fpr95_change_v1.png"
    ),
}


@dataclass(frozen=True, slots=True)
class DetectorComparisonVisualizationData:
    """Validated detector-comparison plot data."""

    conditions: tuple[str, ...]
    systems: tuple[str, ...]
    auroc_mean: Float64Array
    auroc_std: Float64Array
    fpr95_mean: Float64Array
    fpr95_std: Float64Array
    fusion_baselines: tuple[str, ...]
    fusion_auroc_change_mean: Float64Array
    fusion_auroc_change_std: Float64Array
    fusion_fpr95_change_mean: Float64Array
    fusion_fpr95_change_std: Float64Array

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""
        return {
            "conditions": list(
                self.conditions
            ),
            "systems": list(self.systems),
            "fusion_baselines": list(
                self.fusion_baselines
            ),
            "condition_count": len(
                self.conditions
            ),
            "system_count": len(
                self.systems
            ),
        }


def _validate_string_sequence(
    value: object,
    *,
    name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(
            f"{name} must be a sequence."
        )

    try:
        raw_values = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"{name} must be a sequence."
        ) from error

    if not raw_values:
        raise ValueError(
            f"{name} must not be empty."
        )

    values = []

    for raw_value in raw_values:
        if (
            not isinstance(raw_value, str)
            or not raw_value.strip()
        ):
            raise ValueError(
                f"Every {name} entry must be "
                "a non-empty string."
            )

        values.append(
            raw_value.strip()
        )

    if len(set(values)) != len(values):
        raise ValueError(
            f"{name} entries must be unique."
        )

    return tuple(values)


def _validate_summary_value(
    value: object,
    *,
    name: str,
    non_negative: bool = False,
) -> float:
    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"{name} must be numeric."
        ) from error

    if not math.isfinite(number):
        raise ValueError(
            f"{name} must be finite."
        )

    if non_negative and number < 0.0:
        raise ValueError(
            f"{name} must be non-negative."
        )

    return number


def _extract_metric_summary(
    group: dict[str, Any],
    *,
    metric_name: str,
) -> tuple[float, float]:
    metrics = group.get("metrics")

    if not isinstance(metrics, dict):
        raise ValueError(
            "Detector group metrics must "
            "be a mapping."
        )

    summary = metrics.get(metric_name)

    if not isinstance(summary, dict):
        raise ValueError(
            f"Missing metric summary: "
            f"{metric_name}."
        )

    mean = _validate_summary_value(
        summary.get("mean"),
        name=f"{metric_name} mean",
    )
    standard_deviation = (
        _validate_summary_value(
            summary.get("std"),
            name=f"{metric_name} std",
            non_negative=True,
        )
    )

    return mean, standard_deviation


def parse_detector_comparison_visualization_data(
    payload: object,
) -> DetectorComparisonVisualizationData:
    """Parse aggregate comparison JSON data."""
    if not isinstance(payload, dict):
        raise ValueError(
            "Comparison payload must be "
            "a mapping."
        )

    aggregate = payload.get("aggregate")

    if not isinstance(aggregate, dict):
        raise ValueError(
            "Comparison payload must contain "
            "an aggregate mapping."
        )

    conditions = _validate_string_sequence(
        aggregate.get("conditions"),
        name="conditions",
    )
    systems = _validate_string_sequence(
        aggregate.get("systems"),
        name="systems",
    )

    if systems != tuple(
        SHIFT_DETECTOR_SYSTEMS
    ):
        raise ValueError(
            "Comparison systems do not match "
            "the predefined system order."
        )

    raw_groups = aggregate.get(
        "system_condition_groups"
    )

    if not isinstance(raw_groups, list):
        raise ValueError(
            "system_condition_groups must "
            "be a list."
        )

    group_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ValueError(
                "Every detector group must "
                "be a mapping."
            )

        condition = raw_group.get(
            "condition"
        )
        system = raw_group.get(
            "system_name"
        )

        if (
            condition not in conditions
            or system not in systems
        ):
            raise ValueError(
                "Detector group contains an "
                "unknown condition or system."
            )

        key = (
            str(condition),
            str(system),
        )

        if key in group_lookup:
            raise ValueError(
                "Duplicate detector group: "
                f"{key}."
            )

        group_lookup[key] = raw_group

    expected_group_count = (
        len(conditions)
        * len(systems)
    )

    if len(group_lookup) != (
        expected_group_count
    ):
        raise ValueError(
            "Detector comparison is missing "
            "condition/system groups."
        )

    shape = (
        len(conditions),
        len(systems),
    )
    auroc_mean = np.empty(
        shape,
        dtype=np.float64,
    )
    auroc_std = np.empty(
        shape,
        dtype=np.float64,
    )
    fpr95_mean = np.empty(
        shape,
        dtype=np.float64,
    )
    fpr95_std = np.empty(
        shape,
        dtype=np.float64,
    )

    for condition_index, condition in (
        enumerate(conditions)
    ):
        for system_index, system in (
            enumerate(systems)
        ):
            group = group_lookup[
                (
                    condition,
                    system,
                )
            ]

            (
                auroc_mean[
                    condition_index,
                    system_index,
                ],
                auroc_std[
                    condition_index,
                    system_index,
                ],
            ) = _extract_metric_summary(
                group,
                metric_name="auroc",
            )

            (
                fpr95_mean[
                    condition_index,
                    system_index,
                ],
                fpr95_std[
                    condition_index,
                    system_index,
                ],
            ) = _extract_metric_summary(
                group,
                metric_name=(
                    "fpr_at_target_tpr"
                ),
            )

    raw_changes = aggregate.get(
        "fusion_paired_changes"
    )

    if not isinstance(raw_changes, list):
        raise ValueError(
            "fusion_paired_changes must "
            "be a list."
        )

    change_lookup: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for raw_change in raw_changes:
        if not isinstance(
            raw_change,
            dict,
        ):
            raise ValueError(
                "Every fusion change must "
                "be a mapping."
            )

        condition = raw_change.get(
            "condition"
        )
        baseline = raw_change.get(
            "baseline_system"
        )

        if (
            condition not in conditions
            or baseline
            not in FUSION_BASELINE_SYSTEMS
        ):
            raise ValueError(
                "Fusion change contains an "
                "unknown condition or baseline."
            )

        key = (
            str(condition),
            str(baseline),
        )

        if key in change_lookup:
            raise ValueError(
                "Duplicate fusion change: "
                f"{key}."
            )

        change_lookup[key] = raw_change

    expected_change_count = (
        len(conditions)
        * len(FUSION_BASELINE_SYSTEMS)
    )

    if len(change_lookup) != (
        expected_change_count
    ):
        raise ValueError(
            "Fusion comparison is missing "
            "condition/baseline entries."
        )

    change_shape = (
        len(conditions),
        len(FUSION_BASELINE_SYSTEMS),
    )

    auroc_change_mean = np.empty(
        change_shape,
        dtype=np.float64,
    )
    auroc_change_std = np.empty(
        change_shape,
        dtype=np.float64,
    )
    fpr95_change_mean = np.empty(
        change_shape,
        dtype=np.float64,
    )
    fpr95_change_std = np.empty(
        change_shape,
        dtype=np.float64,
    )

    for condition_index, condition in (
        enumerate(conditions)
    ):
        for baseline_index, baseline in (
            enumerate(
                FUSION_BASELINE_SYSTEMS
            )
        ):
            change = change_lookup[
                (
                    condition,
                    baseline,
                )
            ]
            differences = change.get(
                "fusion_minus_baseline"
            )

            if not isinstance(
                differences,
                dict,
            ):
                raise ValueError(
                    "Fusion change must contain "
                    "fusion_minus_baseline."
                )

            auroc_summary = differences.get(
                "auroc"
            )
            fpr_summary = differences.get(
                "fpr_at_target_tpr"
            )

            if (
                not isinstance(
                    auroc_summary,
                    dict,
                )
                or not isinstance(
                    fpr_summary,
                    dict,
                )
            ):
                raise ValueError(
                    "Fusion metric summaries "
                    "are missing."
                )

            auroc_change_mean[
                condition_index,
                baseline_index,
            ] = _validate_summary_value(
                auroc_summary.get("mean"),
                name="fusion AUROC mean",
            )
            auroc_change_std[
                condition_index,
                baseline_index,
            ] = _validate_summary_value(
                auroc_summary.get("std"),
                name="fusion AUROC std",
                non_negative=True,
            )
            fpr95_change_mean[
                condition_index,
                baseline_index,
            ] = _validate_summary_value(
                fpr_summary.get("mean"),
                name="fusion FPR95 mean",
            )
            fpr95_change_std[
                condition_index,
                baseline_index,
            ] = _validate_summary_value(
                fpr_summary.get("std"),
                name="fusion FPR95 std",
                non_negative=True,
            )

    return DetectorComparisonVisualizationData(
        conditions=conditions,
        systems=systems,
        auroc_mean=np.ascontiguousarray(
            auroc_mean
        ),
        auroc_std=np.ascontiguousarray(
            auroc_std
        ),
        fpr95_mean=np.ascontiguousarray(
            fpr95_mean
        ),
        fpr95_std=np.ascontiguousarray(
            fpr95_std
        ),
        fusion_baselines=(
            FUSION_BASELINE_SYSTEMS
        ),
        fusion_auroc_change_mean=(
            np.ascontiguousarray(
                auroc_change_mean
            )
        ),
        fusion_auroc_change_std=(
            np.ascontiguousarray(
                auroc_change_std
            )
        ),
        fusion_fpr95_change_mean=(
            np.ascontiguousarray(
                fpr95_change_mean
            )
        ),
        fusion_fpr95_change_std=(
            np.ascontiguousarray(
                fpr95_change_std
            )
        ),
    )


def load_detector_comparison_visualization_data(
    input_path: str | Path,
) -> DetectorComparisonVisualizationData:
    """Load visualization data from JSON."""
    path = Path(input_path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    return (
        parse_detector_comparison_visualization_data(
            payload
        )
    )


def _validate_dpi(
    value: object,
) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            "dpi must be a positive integer."
        )

    return int(value)


def _system_labels(
    systems: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        SYSTEM_DISPLAY_NAMES.get(
            system,
            system,
        )
        for system in systems
    )


def _save_grouped_bar_figure(
    *,
    matrix: Float64Array,
    error_matrix: Float64Array,
    conditions: tuple[str, ...],
    series_names: tuple[str, ...],
    title: str,
    ylabel: str,
    output_path: Path,
    dpi: int,
    reference_value: float | None = None,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10.5, 6.0)
    )

    condition_positions = np.arange(
        len(conditions),
        dtype=np.float64,
    )
    width = 0.8 / len(series_names)

    for series_index, series_name in (
        enumerate(series_names)
    ):
        positions = (
            condition_positions
            - 0.4
            + 0.5 * width
            + series_index * width
        )

        axis.bar(
            positions,
            matrix[:, series_index],
            width=width,
            yerr=error_matrix[
                :,
                series_index,
            ],
            capsize=3,
            label=series_name,
        )

    if reference_value is not None:
        axis.axhline(
            reference_value,
            linewidth=1.0,
            linestyle="--",
        )

    axis.set_xticks(
        condition_positions,
        labels=[
            condition.capitalize()
            for condition in conditions
        ],
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(
        axis="y",
        alpha=0.25,
    )
    axis.legend(
        frameon=False,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_detector_comparison_figures(
    data: DetectorComparisonVisualizationData,
    output_directory: str | Path,
    *,
    dpi: object = 180,
) -> dict[str, Path]:
    """Create the four detector-comparison figures."""
    if not isinstance(
        data,
        DetectorComparisonVisualizationData,
    ):
        raise TypeError(
            "data must be a "
            "DetectorComparisonVisualizationData."
        )

    validated_dpi = _validate_dpi(dpi)
    directory = Path(output_directory)
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_paths = {
        key: directory / filename
        for key, filename
        in FIGURE_FILENAMES.items()
    }

    _save_grouped_bar_figure(
        matrix=data.auroc_mean,
        error_matrix=data.auroc_std,
        conditions=data.conditions,
        series_names=_system_labels(
            data.systems
        ),
        title=(
            "Channel-shift detection AUROC "
            "across 75 checkpoints"
        ),
        ylabel="AUROC (higher is better)",
        output_path=output_paths[
            "auroc"
        ],
        dpi=validated_dpi,
        reference_value=0.5,
    )

    _save_grouped_bar_figure(
        matrix=data.fpr95_mean,
        error_matrix=data.fpr95_std,
        conditions=data.conditions,
        series_names=_system_labels(
            data.systems
        ),
        title=(
            "False-positive rate at 95% "
            "shift recall"
        ),
        ylabel=(
            "FPR at 95% TPR "
            "(lower is better)"
        ),
        output_path=output_paths[
            "fpr95"
        ],
        dpi=validated_dpi,
    )

    _save_grouped_bar_figure(
        matrix=(
            data.fusion_auroc_change_mean
        ),
        error_matrix=(
            data.fusion_auroc_change_std
        ),
        conditions=data.conditions,
        series_names=_system_labels(
            data.fusion_baselines
        ),
        title=(
            "Fusion AUROC change relative "
            "to each baseline"
        ),
        ylabel=(
            "Fusion minus baseline AUROC"
        ),
        output_path=output_paths[
            "fusion_auroc_change"
        ],
        dpi=validated_dpi,
        reference_value=0.0,
    )

    _save_grouped_bar_figure(
        matrix=(
            data.fusion_fpr95_change_mean
        ),
        error_matrix=(
            data.fusion_fpr95_change_std
        ),
        conditions=data.conditions,
        series_names=_system_labels(
            data.fusion_baselines
        ),
        title=(
            "Fusion FPR95 change relative "
            "to each baseline"
        ),
        ylabel=(
            "Fusion minus baseline FPR95 "
            "(negative is better)"
        ),
        output_path=output_paths[
            "fusion_fpr95_change"
        ],
        dpi=validated_dpi,
        reference_value=0.0,
    )

    for name, path in output_paths.items():
        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            raise RuntimeError(
                "Visualization output is "
                f"missing or empty: {name}."
            )

    return output_paths


__all__ = [
    "FIGURE_FILENAMES",
    "FUSION_BASELINE_SYSTEMS",
    "SYSTEM_DISPLAY_NAMES",
    "DetectorComparisonVisualizationData",
    "create_detector_comparison_figures",
    "load_detector_comparison_visualization_data",
    "parse_detector_comparison_visualization_data",
]
